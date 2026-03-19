"""
Non-stationary queueing environment driven by real hospital arrival data.

This module implements the non-stationary simulation environment used
in Figures 9-11 of the paper. The base arrival rates come from
half-hourly emergency-department data (`data0.csv`), and are scaled
across four simulated weeks to create a realistic non-stationary
pattern.

Usage
-----
The main entry point is :func:`make_nonstationary_lambda`, which returns
a callable ``lambda_k(k, p, t)`` suitable for the ``time_varying=True``
mode of :class:`congestion_experiments.simulator.QueueSimulator`.

    >>> from nonstationary_simulator import make_nonstationary_lambda, DEFAULT_PARAMS
    >>> lambda_k = make_nonstationary_lambda()
    >>> # arrival rate at queue length 3, price 1.0, time 10.5
    >>> lambda_k(3, 1.0, 10.5)
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Callable, Optional


# ---------------------------------------------------------------------------
# Default parameters (matching the paper, Figures 9-11)
# ---------------------------------------------------------------------------

@dataclass
class NonStationaryParams:
    """Parameters for the non-stationary queueing environment.

    Attributes
    ----------
    mu : float
        Service rate.
    p : float
        Reference (baseline) price.
    K : int
        Maximum queue length (hard cap).
    zeta : float
        Price perturbation half-width.
    block_length : float
        Length of each half-hourly block (hours) in the arrival data.
    T_total : float
        Total simulation horizon (hours). Default is 4 weeks = 24*7*4.
    week_scalers : tuple of float
        Multiplicative scalers applied to each week of data.
        Default (0.9, 1.0, 1.1, 1.2) creates a slow upward trend over
        four weeks.
    """
    mu: float = 2.0
    p: float = 1.0
    K: int = 30
    zeta: float = 0.1
    block_length: float = 0.5
    T_total: float = 24.0 * 7 * 4   # 4 weeks
    week_scalers: tuple = (0.9, 1.0, 1.1, 1.2)


DEFAULT_PARAMS = NonStationaryParams()


# ---------------------------------------------------------------------------
# Arrival-rate helpers
# ---------------------------------------------------------------------------

def _lambda_fun(p: float, k: int) -> float:
    """State-and-price-dependent arrival-rate multiplier.

    Corresponds to Eq. (31) in the paper::

        lambda_k(p) = 2 * (2 - p) / (1 + k)

    Parameters
    ----------
    p : float
        Price.
    k : int
        Queue-length state (0-indexed).

    Returns
    -------
    float
    """
    return 2.0 * (2.0 - p) / (1.0 + k)


def _build_lambda_multipliers(
    p: float, zeta: float, K: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build state-dependent multiplier vectors for treatment/control/average.

    Returns arrays of length K+1 (states 0..K-1 plus an absorbing state K
    with rate 0). The three vectors correspond to the treatment (+),
    control (-), and average arrival-rate multipliers.
    """
    p_plus = p + zeta
    p_minus = p - zeta

    lam_plus = np.array([_lambda_fun(p_plus, k) for k in range(K)] + [0.0])
    lam_minus = np.array([_lambda_fun(p_minus, k) for k in range(K)] + [0.0])
    lam_avg = (lam_plus + lam_minus) / 2.0

    return lam_plus, lam_minus, lam_avg


# ---------------------------------------------------------------------------
# Load and process the arrival-rate data
# ---------------------------------------------------------------------------

def load_arrival_rates(
    csv_path: Optional[str] = None,
    week_scalers: tuple = (0.9, 1.0, 1.1, 1.2),
) -> np.ndarray:
    """Load ``data0.csv`` and build the full arrival-rate trajectory.

    The CSV contains one week (7 days) of half-hourly arrival rates.
    The rates are doubled (as in the paper) and then replicated four
    times with the given ``week_scalers`` to produce a four-week
    trajectory.

    Parameters
    ----------
    csv_path : str or None
        Path to ``data0.csv``.  If None, looks in the same directory as
        this module.
    week_scalers : tuple of float
        Multiplicative scalers for each week.

    Returns
    -------
    lambda_t_full : ndarray, shape (n_blocks_total,)
        Base arrival rate for each half-hourly block across all weeks.
    """
    if csv_path is None:
        csv_path = os.path.join(os.path.dirname(__file__), "data0.csv")

    df = pd.read_csv(csv_path, header=None,
                     names=["time", "lambda", "day", "dat_num"])

    # Compute absolute time and double the rates (matching the paper)
    df["day_time"] = df["time"] + df["dat_num"] * 24
    df["lambda"] = df["lambda"] * 2

    # Sort by absolute time to get one-week trajectory
    lambda_t = df.sort_values("day_time")["lambda"].values

    # Tile across weeks with scaling
    lambda_t_full = np.concatenate(
        [scaler * lambda_t for scaler in week_scalers]
    )

    return lambda_t_full


# ---------------------------------------------------------------------------
# Main factory: create a non-stationary lambda_k(k, p, t)
# ---------------------------------------------------------------------------

def make_nonstationary_lambda(
    csv_path: Optional[str] = None,
    params: Optional[NonStationaryParams] = None,
) -> Callable[[int, float, float], float]:
    """Build a time-varying arrival rate function from real data.

    The returned callable has signature ``lambda_k(k, p, t)`` and is
    compatible with the ``time_varying=True`` mode of the package's
    :func:`simulate_interval_switchback` and :func:`simulate_user_level`.

    The arrival rate at queue-length *k*, price *p*, and time *t* is::

        lambda_k(k, p, t) = multiplier(k, p) * base_rate(t)

    where ``multiplier(k, p) = 2*(2-p)/(1+k)`` and ``base_rate(t)``
    is the half-hourly rate from the hospital data.

    Parameters
    ----------
    csv_path : str or None
        Path to ``data0.csv``.
    params : NonStationaryParams or None
        Environment parameters.  Uses ``DEFAULT_PARAMS`` if None.

    Returns
    -------
    lambda_k : callable(k, p, t) -> float
    """
    if params is None:
        params = DEFAULT_PARAMS

    lambda_t_full = load_arrival_rates(
        csv_path=csv_path,
        week_scalers=params.week_scalers,
        block_length=params.block_length,
    )
    num_blocks = len(lambda_t_full)
    block_length = params.block_length
    K = params.K

    def lambda_k(k: int, p: float, t: float) -> float:
        """Non-stationary, state-dependent arrival rate.

        Parameters
        ----------
        k : int
            Current queue length.
        p : float
            Current price.
        t : float
            Current time (hours).

        Returns
        -------
        float
            Arrival rate (non-negative).
        """
        if k >= K:
            return 0.0

        # Look up the base arrival rate for this time block
        block_idx = int(t / block_length)
        block_idx = min(block_idx, num_blocks - 1)
        base_rate = lambda_t_full[block_idx]

        # State-and-price-dependent multiplier: 2*(2-p)/(1+k)
        multiplier = max(0.0, 2.0 * (2.0 - p) / (1.0 + k))

        return multiplier * base_rate

    return lambda_k


# ---------------------------------------------------------------------------
# Convenience: expose multiplier vectors for direct use
# ---------------------------------------------------------------------------

def get_multiplier_vectors(
    params: Optional[NonStationaryParams] = None,
) -> dict:
    """Return the treatment / control / average multiplier vectors.

    Useful for manual inspection or for replicating the full simulation
    logic at a lower level.

    Returns
    -------
    dict with keys 'plus', 'minus', 'average', each an ndarray of
    length K+1.
    """
    if params is None:
        params = DEFAULT_PARAMS
    lam_plus, lam_minus, lam_avg = _build_lambda_multipliers(
        params.p, params.zeta, params.K
    )
    return {"plus": lam_plus, "minus": lam_minus, "average": lam_avg}


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    lam_fn = make_nonstationary_lambda()
    params = DEFAULT_PARAMS
    print(f"Parameters: mu={params.mu}, p={params.p}, K={params.K}, "
          f"zeta={params.zeta}, T_total={params.T_total}")
    print(f"lambda_k(0, 1.0, 0.0) = {lam_fn(0, 1.0, 0.0):.4f}")
    print(f"lambda_k(5, 1.0, 12.0) = {lam_fn(5, 1.0, 12.0):.4f}")
    print(f"lambda_k(0, 1.0, 200.0) = {lam_fn(0, 1.0, 200.0):.4f}")

    # Plot the base arrival rates
    rates = load_arrival_rates()
    print(f"Total blocks: {len(rates)}, T_total = {len(rates)*0.5:.0f} hours")
