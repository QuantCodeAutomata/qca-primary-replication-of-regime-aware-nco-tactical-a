# Regime-Aware NCO Tactical Asset Allocation — Indonesian Multi-Asset

![Python](https://img.shields.io/badge/Python-3.11%2B-blue)
![Framework](https://img.shields.io/badge/Framework-HMM%20%2B%20NCO-green)
![Assets](https://img.shields.io/badge/Assets-5%20Indonesian%20Multi--Asset-orange)
![Backtest](https://img.shields.io/badge/Backtest-2021--2025-yellow)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## Overview

This repository implements a **Regime-Aware Nested Cluster Optimization (NCO)** tactical asset allocation (TAA) strategy applied to a five-asset Indonesian multi-asset universe. The system combines:

1. **Hidden Markov Model (HMM)** regime detection — A three-state Gaussian HMM identifies market regimes (Bullish, Sideways, Bearish) from rolling macroeconomic and return features. Regime labels are assigned by ranking latent states on their mean equity-return feature.

2. **Nested Cluster Optimization (NCO)** — Within each detected regime, assets are clustered using average-linkage hierarchical clustering on an angular distance matrix derived from a Ledoit-Wolf shrinkage covariance estimate. Cluster count is selected by silhouette score. Intra-cluster and inter-cluster minimum-variance portfolios are then combined to yield the final weight vector.

3. **Adaptive Rebalancing** — Weights are updated only when the portfolio drift (L1 distance from target) exceeds a threshold δ, reducing unnecessary turnover and transaction costs.

4. **Walk-Forward Backtest** — A fully out-of-sample backtest runs from January 2021 to August 2025 (56 months), trained on data from January 2019 to December 2020.

The strategy is benchmarked against Equal-Weighting (EW), Buy-and-Hold (B&H), Global Minimum Variance (MinVar), and Standard NCO (without regime conditioning).

---

## Repository Structure

```
project/
├── data/
│   ├── __init__.py
│   ├── fetch.py              # Data download, cleaning, and feature engineering
│   └── raw/                  # Raw data cache (git-ignored)
├── src/
│   ├── __init__.py
│   ├── hmm_regime.py         # Rolling GaussianHMM regime detection
│   ├── nco.py                # NCO portfolio optimizer
│   ├── backtest.py           # Walk-forward backtest engine
│   ├── metrics.py            # Performance metrics (Sharpe, MDD, CVaR, JK test)
│   └── benchmarks.py         # EW, B&H, MinVar, Standard NCO benchmarks
├── exp/
│   ├── __init__.py
│   ├── exp_1_regime_nco.py   # Experiment 1: Primary NCO replication
│   ├── exp_2_benchmarks.py   # Experiment 2: Benchmark comparison
│   ├── exp_3_rebalancing.py  # Experiment 3: Rebalancing rule analysis
│   └── exp_4_robustness.py   # Experiment 4: Robustness / sensitivity analysis
├── tests/
│   ├── __init__.py
│   └── ...                   # Unit tests
├── results/
│   ├── RESULTS.md            # Summary of experiment outputs
│   └── *.csv                 # Output tables (tracked by git)
├── .gitignore
├── requirements.txt
├── README.md
└── AGENTS.md
```

---

## Data

### Asset Universe (N = 5)

| Symbol     | Description                          | Source                          |
|------------|--------------------------------------|---------------------------------|
| `LQ45`     | Indonesian equity index (LQ45)       | Yahoo Finance `^JKLQ45`         |
| `BTC_IDR`  | Bitcoin priced in Indonesian Rupiah  | Yahoo Finance `BTC-IDR`         |
| `CNY_IDR`  | Chinese Yuan / Indonesian Rupiah FX  | Yahoo Finance `CNYIDR=X`        |
| `Gold_IDR` | Gold price in Indonesian Rupiah      | Yahoo Finance `GC=F` × `USDIDR=X` |
| `SBN`      | Indonesian government bonds (proxy)  | Yahoo Finance `EIDO` (proxy)    |

> **Gold conversion note:** Gold_IDR is computed as `XAU/USD × USD/IDR` (i.e., `GC=F` × `USDIDR=X`). This correctly converts the COMEX gold futures price into Indonesian Rupiah. The CNY/IDR rate is used only as the FX asset in the portfolio — it does **not** enter the gold conversion formula.

### Date Ranges

| Split      | Period                          | Months |
|------------|---------------------------------|--------|
| Full data  | 2019-01-02 → 2025-08-31        | ~80    |
| Training   | 2019-01-02 → 2020-12-31        | 24     |
| Backtest   | 2021-01-01 → 2025-08-31        | ~56    |

All returns are **monthly log returns**, computed from month-end closing prices.

### HMM Features

At each month *t*, a rolling window of *W* = 24 months is used to compute:

| Feature      | Description                                           |
|--------------|-------------------------------------------------------|
| `mean_eq`    | Mean LQ45 log return over the trailing 24-month window |
| `vol_eq`     | Sample std of LQ45 log return over trailing window    |
| `mean_bond`  | Mean SBN log return over trailing window              |
| `delta_fx`   | CNY/IDR log return at month *t* (single observation)  |

---

## Installation

```bash
# 1. Clone the repository
git clone <repo-url>
cd project

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

**Python 3.11+** is recommended. All data is sourced from Yahoo Finance via `yfinance` — no additional API keys are required.

---

## Usage

### Fetch and inspect data

```bash
python data/fetch.py
```

This downloads all tickers, computes Gold_IDR, resamples to monthly returns, and builds HMM features. Shapes and date ranges are printed to stdout.

### Run experiments

Each experiment is a self-contained script. Run them in order for a full replication:

```bash
# Experiment 1 — Primary regime-aware NCO backtest
python exp/exp_1_regime_nco.py

# Experiment 2 — Benchmark comparison (EW, B&H, MinVar, Standard NCO)
python exp/exp_2_benchmarks.py

# Experiment 3 — Rebalancing rule sensitivity (threshold δ, cost c)
python exp/exp_3_rebalancing.py

# Experiment 4 — Robustness analysis (HMM states K, window W, asset subsets)
python exp/exp_4_robustness.py
```

Results (tables, NAV curves, weight charts) are saved to `results/`.

### Run tests

```bash
pytest tests/ -v
```

---

## Results

See [`results/RESULTS.md`](results/RESULTS.md) for experiment outputs. Summary metrics include:

| Metric             | Description                                              |
|--------------------|----------------------------------------------------------|
| Annualized Return  | Geometric mean annual return                             |
| Annualized Sharpe  | Excess return / annualized volatility (Rf = 0.479%/mo)  |
| Sortino Ratio      | Downside-deviation-adjusted return                       |
| Max Drawdown (MDD) | Peak-to-trough NAV decline                               |
| Calmar Ratio       | Annual return / MDD                                      |
| CVaR 95%           | Expected shortfall at 5% tail                            |
| Jobson-Korkie test | Statistical significance of Sharpe ratio differences     |

---

## Framework Details

```
┌─────────────────────────────────────────────────────────┐
│              Regime-Aware NCO TAA Pipeline               │
├────────────┬────────────────┬────────────────────────────┤
│  DATA LAYER│  REGIME LAYER  │     PORTFOLIO LAYER         │
│            │                │                             │
│  yfinance  │  GaussianHMM   │  Ledoit-Wolf shrinkage      │
│  5 assets  │  K=3 states    │  Avg-linkage clustering     │
│  monthly   │  W=24 rolling  │  Silhouette M∈{2,3,4}       │
│  log ret.  │  4 features    │  Min-Var intra + inter      │
│            │                │  Adaptive rebal. δ=0.05     │
└────────────┴────────────────┴────────────────────────────┘
```

### Key Hyperparameters

| Parameter       | Value         | Description                               |
|-----------------|---------------|-------------------------------------------|
| K               | 3             | HMM latent states (Bullish/Sideways/Bear) |
| W               | 24 months     | Rolling HMM training window               |
| M               | 2, 3, or 4   | Clusters, chosen by silhouette score      |
| Shrinkage       | Ledoit-Wolf   | Covariance estimator                      |
| Linkage         | Average       | Hierarchical clustering linkage method    |
| α (intra alloc) | min(1, T_k/10)| Regime-sample-weighted intra allocation   |
| c               | 10 bps        | Per-trade transaction cost                |
| δ               | 0.05          | Adaptive rebalancing drift threshold      |
| Risk-free rate  | 0.479%/month  | Indonesian deposit rate proxy             |

---

## Citation

If you use this code, please cite the original NCO methodology:

> López de Prado, M. (2016). *Building Diversified Portfolios that Outperform Out-of-Sample*. Journal of Portfolio Management, 42(4), 59–69.

And the HMM regime-switching literature:

> Hamilton, J. D. (1989). *A New Approach to the Economic Analysis of Nonstationary Time Series and the Business Cycle*. Econometrica, 57(2), 357–384.
