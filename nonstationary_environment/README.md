# Non-Stationary Environment Simulator

This folder contains a non-stationary queueing environment driven by real hospital emergency-department arrival data, corresponding to Figures 9-11 of the paper.

## Files

- **`nonstationary_simulator.py`** — Python module that loads the arrival data and exposes a time-varying arrival rate function `lambda_k(k, p, t)` compatible with the `congestion_experiments` simulator.
- **`data0.csv`** — Half-hourly arrival rate data for one week (7 days, 336 rows).

## About the Data

`data0.csv` contains half-hourly arrival rate data originally from the **SEEStat database** maintained by The Harold and Inge Marcus Center for Service Enterprise Engineering (SEE) at the Technion. The authors of the paper requested access to the SEE terminal and computed summary statistics of emergency department records to produce this file.

To request access to the original data, visit: http://seeserver.iem.technion.ac.il/see-terminal/

### CSV Format

The file has no header. The four columns are:

| Column | Description |
|--------|-------------|
| `time` | Time of day (hours, in 0.5-hour increments from 0.5 to 24.0) |
| `lambda` | Half-hourly arrival rate |
| `day` | Day of the week (Sun, Mon, Tue, Wed, Thu, Fri, Sat) |
| `dat_num` | Day index (0 = Sunday, 1 = Monday, ..., 6 = Saturday) |

### Processing

Following the paper, the simulator:

1. Doubles all arrival rates (`lambda *= 2`).
2. Sorts by absolute time (`time + dat_num * 24`) to produce a one-week trajectory.
3. Replicates the week four times with scaling factors `(0.9, 1.0, 1.1, 1.2)` to simulate a gradual upward trend over four weeks (`T_total = 672 hours`).

## Usage

```python
from nonstationary_environment.nonstationary_simulator import (
    make_nonstationary_lambda,
    DEFAULT_PARAMS,
)

# Build the time-varying arrival rate function
lambda_k = make_nonstationary_lambda()

# Query: arrival rate at queue length 3, price 1.0, time 12.0 hours
rate = lambda_k(3, 1.0, 12.0)

# Use with the congestion_experiments simulator
from congestion_experiments import simulate_user_level

log = simulate_user_level(
    lambda_k=lambda_k,
    mu=DEFAULT_PARAMS.mu,
    p=DEFAULT_PARAMS.p,
    zeta=DEFAULT_PARAMS.zeta,
    T=DEFAULT_PARAMS.T_total,
    K=DEFAULT_PARAMS.K,
    seed=42,
    time_varying=True,
)
```

## Default Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `mu` | 2.0 | Service rate |
| `p` | 1.0 | Reference price |
| `K` | 30 | Maximum queue length |
| `zeta` | 0.1 | Price perturbation |
| `block_length` | 0.5 | Duration of each data block (hours) |
| `T_total` | 672.0 | Total simulation horizon (4 weeks in hours) |
