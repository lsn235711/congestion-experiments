"""
Plug-in variance estimators and confidence intervals (Section 3.1).

Variance formulae (Eqs. 17-18):
    σ̂²_λ   = (1 − π̂₀)μ + 2μπ̂₀ [Σ_{k=1}^{K-1} Ŝ_k Ŝ_{k+1}/π̂_k
                                  − (1−π̂₀) Σ_{k=1}^{K} Ŝ²_k/π̂_k]
    σ̂²_π₀  = 2μ π̂₀² Σ_{k=1}^{K} Ŝ²_k / π̂_k
    σ̂²_WDE = μ π̂₀² Σ_{k=1}^{K} Ŝ²_k / π̂_k

where  Ŝ_k = Σ_{j=k}^{K} π̂_j  and  π̂_k = T_k / T.

Confidence intervals (Eq. 21):
    τ̂ ± z_{α/2} · σ̂ / √(T ζ²)
"""

from __future__ import annotations
import numpy as np
from scipy.stats import norm as _norm
from congestion_experiments.simulator import SummaryStats


def _pi_and_S(stats: SummaryStats):
    """Return π̂_k and Ŝ_k arrays."""
    K = stats.K
    T = stats.T_total
    T_k = stats.T_k

    pi_hat = T_k / T  # shape (K+1,)
    # S_k = sum_{j=k}^{K} pi_hat[j]
    S = np.zeros(K + 1)
    S[K] = pi_hat[K]
    for k in range(K - 1, -1, -1):
        S[k] = S[k + 1] + pi_hat[k]

    return pi_hat, S


def variance_model_free(stats: SummaryStats) -> float:
    r"""Plug-in variance estimator σ̂²_λ (Eq. 17).

    .. math::
        \hat\sigma^2_{\bar\lambda}
        = (1-\hat\pi_0)\mu
          + 2\mu\hat\pi_0\!\left(
              \sum_{k=1}^{K-1}\frac{\hat S_k \hat S_{k+1}}{\hat\pi_k}
              - (1-\hat\pi_0)\sum_{k=1}^{K}\frac{\hat S_k^2}{\hat\pi_k}
          \right).
    """
    pi_hat, S = _pi_and_S(stats)
    K = stats.K
    mu = stats.mu
    pi0 = pi_hat[0]

    term1 = (1 - pi0) * mu

    sum_cross = 0.0
    sum_sq = 0.0
    for k in range(1, K + 1):
        if pi_hat[k] < 1e-30:
            continue
        sum_sq += S[k] ** 2 / pi_hat[k]
        if k <= K - 1:
            sum_cross += S[k] * S[k + 1] / pi_hat[k]

    term2 = 2 * mu * pi0 * (sum_cross - (1 - pi0) * sum_sq)

    return term1 + term2


def variance_idle_time(stats: SummaryStats) -> float:
    r"""Plug-in variance estimator σ̂²_{π₀} (Eq. 18).

    .. math::
        \hat\sigma^2_{\pi_0}
        = 2\mu\hat\pi_0^2 \sum_{k=1}^{K}\frac{\hat S_k^2}{\hat\pi_k}.
    """
    pi_hat, S = _pi_and_S(stats)
    K = stats.K
    mu = stats.mu
    pi0 = pi_hat[0]

    total = 0.0
    for k in range(1, K + 1):
        if pi_hat[k] < 1e-30:
            continue
        total += S[k] ** 2 / pi_hat[k]

    return 2 * mu * pi0 ** 2 * total


def variance_wde(stats: SummaryStats) -> float:
    r"""Plug-in variance estimator σ̂²_{WDE} (Eq. 18).

    .. math::
        \hat\sigma^2_{\mathrm{WDE}}
        = \mu\hat\pi_0^2 \sum_{k=1}^{K}\frac{\hat S_k^2}{\hat\pi_k}.
    """
    pi_hat, S = _pi_and_S(stats)
    K = stats.K
    mu = stats.mu
    pi0 = pi_hat[0]

    total = 0.0
    for k in range(1, K + 1):
        if pi_hat[k] < 1e-30:
            continue
        total += S[k] ** 2 / pi_hat[k]

    return mu * pi0 ** 2 * total


def confidence_interval(
    estimate: float,
    var_estimate: float,
    T: float,
    zeta: float,
    alpha: float = 0.05,
) -> tuple[float, float]:
    r"""Construct a (1−α) confidence interval for V'(p) (Eq. 21).

    .. math::
        \hat\tau \;\pm\; z_{\alpha/2}\,
        \frac{\hat\sigma}{\sqrt{T\,\zeta^2}}.

    Parameters
    ----------
    estimate : float
        Point estimate (any of the three estimators).
    var_estimate : float
        Corresponding variance estimate (σ̂²).
    T : float
        Experiment duration.
    zeta : float
        Price perturbation magnitude.
    alpha : float
        Significance level (default 0.05 for 95% CI).

    Returns
    -------
    (lower, upper) : tuple of float
    """
    z = _norm.ppf(1 - alpha / 2)
    se = np.sqrt(var_estimate / (T * zeta ** 2))
    return (estimate - z * se, estimate + z * se)


def compute_all_variances(stats: SummaryStats) -> dict:
    """Compute all three variance estimates and return a dict."""
    return {
        "model_free": variance_model_free(stats),
        "idle_time": variance_idle_time(stats),
        "wde": variance_wde(stats),
    }
