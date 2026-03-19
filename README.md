# Experimenting under Stochastic Congestion

This repository provides a Python implementation of the experimental designs and estimators from Li, Johari, Kuang, and Wager (2026), "Experimenting under Stochastic Congestion."

The paper studies how to estimate the causal effect of price changes on throughput in queueing systems, where interference arises naturally through congestion: one customer's treatment affects the queue state seen by future customers. The code implements the proposed estimators and experimental designs, and includes a demo notebook that reproduces simulation studies comparing their performance.

## Repository structure

The `congestion_experiments/` package contains the core library:

- `simulator.py` — Gillespie-style exact simulator for a single-server queue with state-dependent arrival rates. Implements three experimental designs: interval switchback (with completely randomized treatment assignment), regenerative switchback, and user-randomized experiments. Also includes windowed summary statistics for non-stationary settings.
- `estimators.py` — Point estimators for the policy gradient V'(p): the model-free estimator, the idle-time-based estimator, and the weighted direct-effect (WDE) estimator, for both switchback and user-randomized designs. Includes non-stationary extensions with kernel windowing.
- `variance.py` — Plug-in variance estimators and confidence interval construction for all three estimators.
- `__init__.py` — Convenience imports.

The root directory contains:

- `demo_congestion_experiments.ipynb` — A self-contained Jupyter notebook that walks through usage examples, defines illustrative queueing models (zero-deflated M/M/1 and power-law joining), computes true derivatives via numerical differentiation, and runs simulation studies comparing estimators across designs and settings (stationary and non-stationary).
- `setup.py` — Package installation configuration.


## Quick start

Clone the repo and open the notebook:

```bash
git clone https://github.com/lsn235711/congestion-experiments.git
cd congestion-experiments
jupyter notebook demo_congestion_experiments.ipynb
```

No installation step is needed — the notebook adds the package to the Python path automatically. The only prerequisites are Python >= 3.9, NumPy, SciPy, and Matplotlib.

## Code generation disclosure

All code in this repository was generated with the assistance of Claude (Anthropic). The implementation follows the methodology described in the referenced paper.

## Reference

Shuangning Li, Ramesh Johari, Xu Kuang, and Stefan Wager. "Experimenting under Stochastic Congestion." Forthcoming, *Management Science*, 2026. Available at [https://arxiv.org/abs/2302.12093](https://arxiv.org/abs/2302.12093).
