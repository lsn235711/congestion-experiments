"""
Point estimators for the policy gradient V'(p).

Three estimators for switchback experiments (Section 2):
  - Model-free  (tau_lambda, Eq. 5)
  - Idle-time-based  (tau_pi0, Eq. 6)
  - Weighted direct effect  (tau_WDE, Eq. 8)

WDE for user-level randomization (Section 4, Eq. 23-24).

Non-stationary generalizations with kernel windows (Section 5, Eq. 27-30).
"""

from __future__ import annotations
import numpy as np
from congestion_experiments.simulator import (
    SummaryStats,
    WindowedSummaryStats,
    ExperimentLog,
    compute_summary_stats,
    compute_windowed_summary_stats,
)


# ── helpers ────────────────────────────────────────────────────────────────

def _safe_div(a, b, default=0.0):
    """Return a/b, or *default* when b ≈ 0."""
    return a / b if abs(b) > 1e-30 else default


# ── switchback estimators (Section 2) ──────────────────────────────────────

def model_free_estimator(stats: SummaryStats) -> float:
    r"""Model-free estimator  τ̂_λ  (Eq. 5).

    .. math::
        \hat\tau_{\bar\lambda}
        = \frac{1}{2\zeta}\!\left(\frac{N_+}{T_+} - \frac{N_-}{T_-}\right).
    """
    zeta = stats.zeta
    return _safe_div(
        _safe_div(stats.N_plus, stats.T_plus) -
        _safe_div(stats.N_minus, stats.T_minus),
        2 * zeta,
    )


def idle_time_estimator(stats: SummaryStats) -> float:
    r"""Idle-time-based estimator  τ̂_{π₀}  (Eq. 6).

    .. math::
        \hat\tau_{\pi_0}
        = -\frac{\mu}{2\zeta}\!\left(
              \frac{T_{0,+}}{T_+} - \frac{T_{0,-}}{T_-}
          \right).
    """
    zeta = stats.zeta
    mu = stats.mu
    return -mu / (2 * zeta) * (
        _safe_div(stats.T0_plus, stats.T_plus) -
        _safe_div(stats.T0_minus, stats.T_minus)
    )


def _delta_k_switchback(stats: SummaryStats, k: int) -> float:
    r"""Finite-difference estimator Δ̂_k for switchbacks (Eq. 7).

    .. math::
        \hat\Delta_k = \frac{1}{\zeta}
            \frac{N_{k,+}/T_{k,+} - N_{k,-}/T_{k,-}}
                 {N_{k,+}/T_{k,+} + N_{k,-}/T_{k,-}}.
    """
    r_plus = _safe_div(stats.N_k_plus[k], stats.T_k_plus[k])
    r_minus = _safe_div(stats.N_k_minus[k], stats.T_k_minus[k])
    denom = r_plus + r_minus
    if abs(denom) < 1e-30:
        return 0.0
    return (r_plus - r_minus) / (stats.zeta * denom)


def wde_estimator(stats: SummaryStats) -> float:
    r"""Weighted direct-effect estimator  τ̂_{WDE} for switchbacks (Eq. 8).

    .. math::
        \hat\tau_{\mathrm{WDE}}
        = \mu \frac{T_0}{T}
          \sum_{k=0}^{K-1} \hat\Delta_k
          \sum_{i=k+1}^{K} \frac{T_i}{T}.
    """
    K = stats.K
    T = stats.T_total
    mu = stats.mu
    T_k = stats.T_k

    pi_hat = T_k / T  # estimated stationary probabilities

    result = 0.0
    for k in range(K):  # k = 0, ..., K-1
        delta_k = _delta_k_switchback(stats, k)
        tail = np.sum(pi_hat[k + 1: K + 1])
        result += delta_k * tail

    return mu * pi_hat[0] * result


# ── user-level estimators (Section 4) ──────────────────────────────────────

def _delta_k_user_level(stats: SummaryStats, k: int) -> float:
    r"""Finite-difference estimator Δ̂_k for user-level randomization (Eq. 23).

    .. math::
        \hat\Delta_k = \frac{1}{\zeta}
            \frac{N_{k,+} - N_{k,-}}{N_{k,+} + N_{k,-}}.
    """
    n_plus = stats.N_k_plus[k]
    n_minus = stats.N_k_minus[k]
    denom = n_plus + n_minus
    if denom == 0:
        return 0.0
    return (n_plus - n_minus) / (stats.zeta * denom)


def wde_estimator_user_level(stats: SummaryStats) -> float:
    r"""WDE estimator under user-level randomization (Eq. 24).

    .. math::
        \hat\tau_{\mathrm{WDE}}
        = \mu \frac{T_0}{T}
          \sum_{k=0}^{K-1} \hat\Delta_k
          \sum_{i=k+1}^{K} \frac{T_i}{T}.

    The only difference from the switchback version is the definition of
    Δ̂_k (counts-based rather than rate-based).
    """
    K = stats.K
    T = stats.T_total
    mu = stats.mu
    T_k = stats.T_k

    pi_hat = T_k / T

    result = 0.0
    for k in range(K):
        delta_k = _delta_k_user_level(stats, k)
        tail = np.sum(pi_hat[k + 1: K + 1])
        result += delta_k * tail

    return mu * pi_hat[0] * result


# ── non-stationary estimators (Section 5) ──────────────────────────────────

def wde_estimator_nonstationary(
    log: ExperimentLog,
    kernel_length: float,
) -> float:
    r"""Windowed WDE estimator for switchbacks in non-stationary settings
    (Eq. 28).

    Partitions the time horizon into windows of length *kernel_length* and
    constructs a local WDE within each window, then averages.

    .. math::
        \hat\tau_{\mathrm{WDE}}(s)
        = \mu \frac{s}{T} \sum_w
          \left[\frac{T_{0,w}}{s}
                \sum_{k=0}^{K-1} \hat\Delta_{k,w}
                \sum_{i=k+1}^{K} \frac{T_{i,w}}{s}
          \right].
    """
    ws = compute_windowed_summary_stats(log, kernel_length)
    T = ws.T_total
    mu = ws.mu
    s = ws.s

    total = 0.0
    for win in ws.windows:
        local_T = win.T_total
        if local_T < 1e-30:
            continue
        K_w = win.K
        T_k_w = win.T_k
        pi_hat_w = T_k_w / local_T

        local_wde = 0.0
        for k in range(K_w):
            delta_k = _delta_k_switchback(win, k)
            tail = np.sum(pi_hat_w[k + 1: K_w + 1])
            local_wde += delta_k * tail

        local_wde *= mu * pi_hat_w[0]
        total += local_wde * local_T

    return total / T


def wde_estimator_nonstationary_user_level(
    log: ExperimentLog,
    kernel_length: float,
) -> float:
    r"""Windowed WDE estimator for user-level randomization in
    non-stationary settings (Eq. 30).

    Same as :func:`wde_estimator_nonstationary` but uses the user-level
    version of Δ̂_k within each window.
    """
    ws = compute_windowed_summary_stats(log, kernel_length)
    T = ws.T_total
    mu = ws.mu

    total = 0.0
    for win in ws.windows:
        local_T = win.T_total
        if local_T < 1e-30:
            continue
        K_w = win.K
        T_k_w = win.T_k
        pi_hat_w = T_k_w / local_T

        local_wde = 0.0
        for k in range(K_w):
            delta_k = _delta_k_user_level(win, k)
            tail = np.sum(pi_hat_w[k + 1: K_w + 1])
            local_wde += delta_k * tail

        local_wde *= mu * pi_hat_w[0]
        total += local_wde * local_T

    return total / T


# ── convenience: operate directly on logs ──────────────────────────────────

def estimate_all(log: ExperimentLog, user_level: bool = False):
    """Compute all applicable estimators and return a dict.

    Parameters
    ----------
    log : ExperimentLog
    user_level : bool
        If True, use user-level versions of the estimators.

    Returns
    -------
    dict with keys 'model_free', 'idle_time', 'wde' and their values.
    """
    stats = compute_summary_stats(log)
    if user_level:
        return {
            "wde": wde_estimator_user_level(stats),
        }
    return {
        "model_free": model_free_estimator(stats),
        "idle_time": idle_time_estimator(stats),
        "wde": wde_estimator(stats),
    }
