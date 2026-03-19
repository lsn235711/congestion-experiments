"""
congestion_experiments – Estimators for experimenting under stochastic congestion.

Implements the estimators from:
  Li, Johari, Kuang, and Wager,
  "Experimenting under Stochastic Congestion", Management Science.

Modules
-------
simulator : Simulate queueing systems under switchback or user-level experiments.
estimators : Model-free, idle-time, and weighted-direct-effect estimators.
variance : Plug-in variance estimators and confidence intervals.
"""

from congestion_experiments.simulator import (
    QueueSimulator,
    simulate_interval_switchback,
    simulate_regenerative_switchback,
    simulate_user_level,
)
from congestion_experiments.estimators import (
    model_free_estimator,
    idle_time_estimator,
    wde_estimator,
    wde_estimator_user_level,
    wde_estimator_nonstationary,
    wde_estimator_nonstationary_user_level,
)
from congestion_experiments.variance import (
    variance_model_free,
    variance_idle_time,
    variance_wde,
    confidence_interval,
)
