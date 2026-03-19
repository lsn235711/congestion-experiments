"""
Queueing-system simulator for switchback and user-level experiments.

The system is a single-server queue with state-dependent Poisson arrivals
at rate lambda_k(p) when the queue length is k and the price is p, and
exponential service at rate mu.  See Section 2 of the paper.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Optional, List


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class ExperimentLog:
    """Raw event-level log produced by the simulator.

    Attributes
    ----------
    times : ndarray, shape (n_events,)
        Absolute times of every event (arrival or departure).
    queue_lengths : ndarray, shape (n_events,)
        Queue length *after* the event.
    event_types : ndarray, shape (n_events,)
        +1 for an arrival, -1 for a departure.
    prices : ndarray, shape (n_events,)
        Price that was active when the event occurred.
    treatments : ndarray, shape (n_events,)
        +1 if treated (p + zeta), -1 if control (p - zeta).
        For user-level randomization each arrival gets its own treatment;
        departures inherit the treatment of the customer being served.
    T : float
        Total experiment horizon.
    p : float
        Reference price.
    zeta : float
        Price perturbation.
    mu : float
        Service rate.
    """
    times: np.ndarray
    queue_lengths: np.ndarray
    event_types: np.ndarray
    prices: np.ndarray
    treatments: np.ndarray
    T: float
    p: float
    zeta: float
    mu: float


@dataclass
class SummaryStats:
    """Sufficient statistics needed by all estimators.

    The summary is indexed by queue-length state k ∈ {0, …, K} and
    treatment arm w ∈ {+1, −1}.

    Attributes
    ----------
    K : int
        Maximum observed queue length.
    T_total : float
        Total experiment duration.
    T_plus : float
        Time under treatment (+).
    T_minus : float
        Time under control (−).
    T_k : ndarray, shape (K+1,)
        Total time the queue length is k.
    T_k_plus : ndarray, shape (K+1,)
        Time the queue is k *and* treatment is active.
    T_k_minus : ndarray, shape (K+1,)
        Time the queue is k *and* control is active.
    N_plus : int
        Number of arrivals under treatment.
    N_minus : int
        Number of arrivals under control.
    N_k_plus : ndarray, shape (K+1,)
        Arrivals when queue is k and treatment is active.
    N_k_minus : ndarray, shape (K+1,)
        Arrivals when queue is k and control is active.
    T0_plus : float
        Idle time (queue == 0) under treatment.
    T0_minus : float
        Idle time (queue == 0) under control.
    p : float
    zeta : float
    mu : float
    """
    K: int
    T_total: float
    T_plus: float
    T_minus: float
    T_k: np.ndarray
    T_k_plus: np.ndarray
    T_k_minus: np.ndarray
    N_plus: int
    N_minus: int
    N_k_plus: np.ndarray
    N_k_minus: np.ndarray
    T0_plus: float
    T0_minus: float
    p: float
    zeta: float
    mu: float


@dataclass
class WindowedSummaryStats:
    """Per-window summary statistics for non-stationary estimators.

    Attributes
    ----------
    windows : list[SummaryStats]
        One SummaryStats per window of length *s*.
    s : float
        Window (kernel) length.
    T_total : float
    p : float
    zeta : float
    mu : float
    """
    windows: List[SummaryStats]
    s: float
    T_total: float
    p: float
    zeta: float
    mu: float


# ---------------------------------------------------------------------------
# Core queue simulator
# ---------------------------------------------------------------------------

class QueueSimulator:
    """Gillespie-style exact simulator for a single-server queue.

    Parameters
    ----------
    lambda_k : callable(k, p) -> float
        State-dependent arrival rate when queue length is k and price is p.
        Can also be callable(k, p, t) -> float for non-stationary settings
        (set ``time_varying=True``).
    mu : float
        Service rate (exponential).
    K : int or None
        Hard cap on the queue length.  Arrivals are rejected when the queue
        is at capacity.  If None, no hard cap (but lambda_k should go to 0
        for large k to keep the queue stable).
    time_varying : bool
        If True, lambda_k is called as lambda_k(k, p, t).
    """

    def __init__(
        self,
        lambda_k: Callable,
        mu: float,
        K: Optional[int] = None,
        time_varying: bool = False,
    ):
        self.lambda_k = lambda_k
        self.mu = mu
        self.K = K
        self.time_varying = time_varying

    def _arrival_rate(self, k: int, p: float, t: float = 0.0) -> float:
        if self.K is not None and k >= self.K:
            return 0.0
        if self.time_varying:
            return max(self.lambda_k(k, p, t), 0.0)
        return max(self.lambda_k(k, p), 0.0)

    def _run(
        self,
        T: float,
        price_schedule: Callable,
        rng: np.random.Generator,
        q0: int = 0,
    ) -> ExperimentLog:
        """Internal simulation loop.

        Parameters
        ----------
        T : float
            Time horizon.
        price_schedule : callable(t, queue_length) -> (price, treatment)
            Returns the current price and treatment indicator at time t.
        rng : Generator
        q0 : int
            Initial queue length.
        """
        times_list: list = []
        ql_list: list = []
        et_list: list = []
        price_list: list = []
        treat_list: list = []

        t = 0.0
        q = q0
        p_cur, w_cur = price_schedule(t, q)

        while t < T:
            lam = self._arrival_rate(q, p_cur, t)
            dep_rate = self.mu if q > 0 else 0.0
            total_rate = lam + dep_rate

            if total_rate <= 0:
                # System is stuck (queue empty and no arrivals). Fast-forward.
                # Check if price changes at some point; for simplicity jump to T.
                break

            dt = rng.exponential(1.0 / total_rate)
            t_new = t + dt

            if t_new >= T:
                break

            t = t_new
            # Determine event type
            if rng.random() < lam / total_rate:
                # Arrival
                q += 1
                p_ev, w_ev = price_schedule(t, q - 1)  # price at moment of arrival
                times_list.append(t)
                ql_list.append(q)
                et_list.append(1)
                price_list.append(p_ev)
                treat_list.append(w_ev)
            else:
                # Departure
                q -= 1
                times_list.append(t)
                ql_list.append(q)
                et_list.append(-1)
                price_list.append(p_cur)
                treat_list.append(w_cur)

            # Update price/treatment (may change at boundaries)
            p_cur, w_cur = price_schedule(t, q)

        return ExperimentLog(
            times=np.array(times_list, dtype=np.float64),
            queue_lengths=np.array(ql_list, dtype=np.int64),
            event_types=np.array(et_list, dtype=np.int64),
            prices=np.array(price_list, dtype=np.float64),
            treatments=np.array(treat_list, dtype=np.int64),
            T=T,
            p=0.0,   # filled by caller
            zeta=0.0, # filled by caller
            mu=self.mu,
        )


# ---------------------------------------------------------------------------
# Experiment wrappers
# ---------------------------------------------------------------------------

def simulate_interval_switchback(
    lambda_k: Callable,
    mu: float,
    p: float,
    zeta: float,
    T: float,
    interval_length: float,
    K: Optional[int] = None,
    seed: int = 42,
    q0: int = 0,
    time_varying: bool = False,
) -> ExperimentLog:
    """Run an interval switchback experiment.

    Treatment is assigned via a completely randomized design: exactly
    floor(n/2) intervals receive treatment (+1) and the rest receive
    control (−1), with the assignment uniformly permuted at random.

    Parameters
    ----------
    lambda_k : callable(k, p) or callable(k, p, t)
    mu, p, zeta, T, K : model parameters
    interval_length : float
        Length of each switchback interval.
    seed : int
    q0 : int
        Initial queue length.
    time_varying : bool
    """
    rng = np.random.default_rng(seed)
    sim = QueueSimulator(lambda_k, mu, K=K, time_varying=time_varying)

    n_intervals = int(np.ceil(T / interval_length))
    # Completely randomized design: fix the number of treated/control
    # intervals, then randomly permute.
    n_treat = n_intervals // 2
    treatments = np.array([1] * n_treat + [-1] * (n_intervals - n_treat))
    rng.shuffle(treatments)

    def price_schedule(t, q):
        idx = min(int(t / interval_length), n_intervals - 1)
        w = treatments[idx]
        return p + w * zeta, w

    log = sim._run(T, price_schedule, rng, q0=q0)
    log.p = p
    log.zeta = zeta
    return log


def simulate_regenerative_switchback(
    lambda_k: Callable,
    mu: float,
    p: float,
    zeta: float,
    T: float,
    k_r: int = 0,
    K: Optional[int] = None,
    seed: int = 42,
    q0: int = 0,
) -> ExperimentLog:
    """Run a regenerative switchback experiment.

    The price is re-randomized every time the queue hits state k_r
    (Section 2.2).

    Parameters
    ----------
    k_r : int
        Regeneration state (default 0 = empty queue).
    """
    rng = np.random.default_rng(seed)
    sim = QueueSimulator(lambda_k, mu, K=K)

    current_w = rng.choice([-1, 1])

    # We need a mutable container so the closure can update it
    state = {"w": current_w, "prev_q": q0}

    def price_schedule(t, q):
        # Re-randomize when queue reaches k_r
        if q == k_r and state["prev_q"] != k_r:
            state["w"] = rng.choice([-1, 1])
        state["prev_q"] = q
        return p + state["w"] * zeta, state["w"]

    log = sim._run(T, price_schedule, rng, q0=q0)
    log.p = p
    log.zeta = zeta
    return log


def simulate_user_level(
    lambda_k: Callable,
    mu: float,
    p: float,
    zeta: float,
    T: float,
    K: Optional[int] = None,
    seed: int = 42,
    q0: int = 0,
    time_varying: bool = False,
) -> ExperimentLog:
    """Run a user-level randomized experiment.

    Each arriving customer independently receives price p+ζ or p−ζ with
    equal probability (Section 4, Eq. 22).  The system-level effective
    arrival rate in state k is (λ_k(p+ζ) + λ_k(p−ζ))/2, and each arrival
    is assigned to treatment (+1) with probability
    λ_k(p+ζ)/(λ_k(p+ζ)+λ_k(p−ζ)).  Treatment is recorded per arrival;
    departures get treatment = 0.
    """
    rng = np.random.default_rng(seed)
    K_cap = K

    times_list: list = []
    ql_list: list = []
    et_list: list = []
    price_list: list = []
    treat_list: list = []

    t = 0.0
    q = q0

    def _lam(k, price, time):
        if K_cap is not None and k >= K_cap:
            return 0.0
        if time_varying:
            return max(lambda_k(k, price, time), 0.0)
        return max(lambda_k(k, price), 0.0)

    while t < T:
        lam_plus = _lam(q, p + zeta, t)
        lam_minus = _lam(q, p - zeta, t)
        lam_eff = (lam_plus + lam_minus) / 2.0
        dep_rate = mu if q > 0 else 0.0
        total_rate = lam_eff + dep_rate

        if total_rate <= 0:
            break

        dt = rng.exponential(1.0 / total_rate)
        t_new = t + dt
        if t_new >= T:
            break
        t = t_new

        if rng.random() < lam_eff / total_rate:
            # Arrival — assign treatment proportional to arrival rates
            prob_plus = lam_plus / (lam_plus + lam_minus) if (lam_plus + lam_minus) > 0 else 0.5
            w = 1 if rng.random() < prob_plus else -1
            q += 1
            times_list.append(t)
            ql_list.append(q)
            et_list.append(1)
            price_list.append(p + w * zeta)
            treat_list.append(w)
        else:
            # Departure
            q -= 1
            times_list.append(t)
            ql_list.append(q)
            et_list.append(-1)
            price_list.append(p)
            treat_list.append(0)

    return ExperimentLog(
        times=np.array(times_list, dtype=np.float64),
        queue_lengths=np.array(ql_list, dtype=np.int64),
        event_types=np.array(et_list, dtype=np.int64),
        prices=np.array(price_list, dtype=np.float64),
        treatments=np.array(treat_list, dtype=np.int64),
        T=T,
        p=p,
        zeta=zeta,
        mu=mu,
    )


# ---------------------------------------------------------------------------
# Summary statistics extraction
# ---------------------------------------------------------------------------

def compute_summary_stats(log: ExperimentLog) -> SummaryStats:
    """Extract sufficient statistics from an experiment log.

    Works for both switchback experiments and user-level randomization.
    """
    T = log.T
    mu = log.mu
    p = log.p
    zeta = log.zeta

    # Reconstruct queue length trajectory for time-weighted statistics.
    # Insert a sentinel at time 0 and time T.
    n = len(log.times)
    if n == 0:
        return SummaryStats(
            K=0, T_total=T, T_plus=T/2, T_minus=T/2,
            T_k=np.array([T]), T_k_plus=np.array([T/2]),
            T_k_minus=np.array([T/2]),
            N_plus=0, N_minus=0,
            N_k_plus=np.array([0]), N_k_minus=np.array([0]),
            T0_plus=T/2, T0_minus=T/2,
            p=p, zeta=zeta, mu=mu,
        )

    # Queue lengths *before* each event
    # ql_list[i] is queue length after event i.  Queue length before event 0
    # is obtained from the first event type.
    ql_after = log.queue_lengths
    ql_before = np.empty(n, dtype=np.int64)
    ql_before[0] = ql_after[0] - log.event_types[0]
    ql_before[1:] = ql_after[:-1]

    # Determine treatment active in each *interval* between events.
    # For switchback: the treatment is the same for the whole interval.
    # For user-level: we need the "system treatment" which we approximate
    # as 0 (the system runs at price p). But for the purpose of the
    # estimator statistics we track per-arrival treatment.

    # We'll compute time-in-state per treatment for switchback designs.
    # For user-level designs, T_k,+ and T_k,- are not meaningful in the
    # switchback sense; instead we only need per-arrival counts.

    is_user_level = np.any(log.treatments == 0)

    K_max = int(np.max(ql_after)) if n > 0 else 0
    K_max = max(K_max, int(np.max(ql_before)))

    # Time intervals
    t_events = np.concatenate([[0.0], log.times, [T]])
    # Queue length in each interval
    q_intervals = np.empty(n + 1, dtype=np.int64)
    q_intervals[0] = ql_before[0]
    q_intervals[1:] = ql_after
    dt = np.diff(t_events)

    if is_user_level:
        # User-level: system runs at price p, no meaningful T_k,+/T_k,-.
        # But we still compute T_k for the WDE estimator.
        T_k = np.zeros(K_max + 1)
        for i in range(len(dt)):
            k = q_intervals[i]
            if 0 <= k <= K_max:
                T_k[k] += dt[i]

        # Per-arrival counts by queue length and treatment
        arrivals = log.event_types == 1
        N_k_plus = np.zeros(K_max + 1, dtype=np.int64)
        N_k_minus = np.zeros(K_max + 1, dtype=np.int64)
        for i in range(n):
            if log.event_types[i] == 1:
                k = ql_before[i]
                if log.treatments[i] == 1:
                    N_k_plus[k] += 1
                elif log.treatments[i] == -1:
                    N_k_minus[k] += 1

        return SummaryStats(
            K=K_max,
            T_total=T,
            T_plus=T / 2,   # by design, half the arrivals get each treatment
            T_minus=T / 2,
            T_k=T_k,
            T_k_plus=T_k / 2,  # approximate (system is at price p throughout)
            T_k_minus=T_k / 2,
            N_plus=int(np.sum(N_k_plus)),
            N_minus=int(np.sum(N_k_minus)),
            N_k_plus=N_k_plus,
            N_k_minus=N_k_minus,
            T0_plus=T_k[0] / 2,
            T0_minus=T_k[0] / 2,
            p=p, zeta=zeta, mu=mu,
        )

    # Switchback case: treatment is constant between price switches
    # Determine treatment for each interval.
    # The treatment in the interval [t_events[i], t_events[i+1]) is the
    # treatment at the start of the interval.
    w_intervals = np.empty(n + 1, dtype=np.int64)
    # First interval uses the treatment of the first event
    w_intervals[0] = log.treatments[0] if n > 0 else 1
    w_intervals[1:] = log.treatments

    # But for switchbacks the treatment between events is constant from
    # the last event's treatment.  Actually, the treatment at time t is
    # determined by the price schedule. Between events, the price doesn't
    # change (in switchback). We approximate: the treatment in interval
    # [event i, event i+1) is the treatment recorded at event i.
    # For the first interval [0, event_0), we use the treatment of event 0.

    T_k = np.zeros(K_max + 1)
    T_k_plus = np.zeros(K_max + 1)
    T_k_minus = np.zeros(K_max + 1)
    T_plus = 0.0
    T_minus = 0.0

    for i in range(len(dt)):
        k = q_intervals[i]
        w = w_intervals[i]
        d = dt[i]
        if 0 <= k <= K_max:
            T_k[k] += d
            if w == 1:
                T_k_plus[k] += d
                T_plus += d
            else:
                T_k_minus[k] += d
                T_minus += d

    # Arrival counts
    N_k_plus = np.zeros(K_max + 1, dtype=np.int64)
    N_k_minus = np.zeros(K_max + 1, dtype=np.int64)
    for i in range(n):
        if log.event_types[i] == 1:
            k = ql_before[i]
            if log.treatments[i] == 1:
                N_k_plus[k] += 1
            else:
                N_k_minus[k] += 1

    return SummaryStats(
        K=K_max,
        T_total=T,
        T_plus=T_plus,
        T_minus=T_minus,
        T_k=T_k,
        T_k_plus=T_k_plus,
        T_k_minus=T_k_minus,
        N_plus=int(np.sum(N_k_plus)),
        N_minus=int(np.sum(N_k_minus)),
        N_k_plus=N_k_plus,
        N_k_minus=N_k_minus,
        T0_plus=T_k_plus[0],
        T0_minus=T_k_minus[0],
        p=p, zeta=zeta, mu=mu,
    )


def compute_windowed_summary_stats(
    log: ExperimentLog, s: float
) -> WindowedSummaryStats:
    """Partition the experiment into windows of length *s* and compute
    per-window summary statistics (Section 5, Eq. 27-30)."""
    T = log.T
    n_windows = int(np.ceil(T / s))
    windows: List[SummaryStats] = []

    for w_idx in range(n_windows):
        t_start = w_idx * s
        t_end = min((w_idx + 1) * s, T)

        # Select events in this window
        mask = (log.times >= t_start) & (log.times < t_end)
        if not np.any(mask):
            # Empty window – create a minimal SummaryStats
            dur = t_end - t_start
            windows.append(SummaryStats(
                K=0, T_total=dur, T_plus=dur/2, T_minus=dur/2,
                T_k=np.array([dur]), T_k_plus=np.array([dur/2]),
                T_k_minus=np.array([dur/2]),
                N_plus=0, N_minus=0,
                N_k_plus=np.array([0]), N_k_minus=np.array([0]),
                T0_plus=dur/2, T0_minus=dur/2,
                p=log.p, zeta=log.zeta, mu=log.mu,
            ))
            continue

        # Build a sub-log
        sub_times = log.times[mask] - t_start
        sub_ql = log.queue_lengths[mask]
        sub_et = log.event_types[mask]
        sub_prices = log.prices[mask]
        sub_treats = log.treatments[mask]

        sub_log = ExperimentLog(
            times=sub_times,
            queue_lengths=sub_ql,
            event_types=sub_et,
            prices=sub_prices,
            treatments=sub_treats,
            T=t_end - t_start,
            p=log.p,
            zeta=log.zeta,
            mu=log.mu,
        )
        windows.append(compute_summary_stats(sub_log))

    return WindowedSummaryStats(
        windows=windows,
        s=s,
        T_total=T,
        p=log.p,
        zeta=log.zeta,
        mu=log.mu,
    )
