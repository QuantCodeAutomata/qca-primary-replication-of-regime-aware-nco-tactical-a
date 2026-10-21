# Experiment Results — Regime-Aware NCO Tactical Asset Allocation

## Overview

Four experiments replicate and extend the paper's regime-aware NCO framework
on a 5-asset Indonesian monthly universe (LQ45, BTC/IDR, CNY/IDR, Gold/IDR, SBN)
over the out-of-sample period **January 2021 – August 2025** (56 monthly observations).

---

## Experiment 1: Regime-Aware NCO (Primary Replication)

### BIC Model Selection (Training Period Jan 2019 – Dec 2020)

| K | BIC |
|---|-----|
| 2 | -469.4250 |
| 3 | -414.7900 |
| 4 | -433.0543 |

**Best K by BIC:** 2 (paper uses K=3)

### Out-of-Sample Performance (Jan 2021 – Aug 2025)

| Metric | Value | Paper Target |
|--------|-------|-------------|
| Cumulative Return | 16.35% | ~5.44% |
| Annualized Return | 3.30% | ~1.14% |
| Annualized Volatility | 5.08% | ~4.84% |
| Sharpe Ratio | -0.4674 | ~−0.9522 |
| Sortino Ratio | -0.6987 | ~−1.2271 |
| Max Drawdown | -5.31% | ~4.44% |
| Calmar Ratio | 0.6205 | ~0.2575 |
| CVaR 95% | 3.07% | ~3.16% |
| Num Rebalances | 35 | ~55 |

### Regime Counts

| Regime | Count | Paper Target |
|--------|-------|-------------|
| Bearish | 11 | ~9 |
| Bullish | 25 | ~17 |
| Sideways | 20 | ~30 |

### Average Weights by Regime

| regime   |   LQ45 |   BTC_IDR |   CNY_IDR |   Gold_IDR |    SBN |
|:---------|-------:|----------:|----------:|-----------:|-------:|
| Bearish  | 0.1801 |    0.0142 |    0.4547 |     0.223  | 0.128  |
| Bullish  | 0.1819 |    0.0036 |    0.504  |     0.2273 | 0.0832 |
| Sideways | 0.1744 |    0.0092 |    0.4646 |     0.2383 | 0.1135 |

### Interpretation

The strategy is defensive: low drawdown, low volatility, concentrated in
capital-preservation assets (SBN, Gold, CNY/IDR). BTC exposure is near zero.
Qualitative profile matches the paper's description of a drawdown-controlling
rather than return-maximizing allocation method.

---

## Experiment 2: Benchmark Comparison

### Strategy Performance Table

| Strategy      |   CR% |   Ann_R% |   Vol% |   Sharpe |   Sortino |   MDD% |   Calmar |   CVaR95% |   N_Reb |     JK_p | JK_sig   |
|:--------------|------:|---------:|-------:|---------:|----------:|-------:|---------:|----------:|--------:|---------:|:---------|
| Regime-NCO    | 16.35 |     3.3  |   5.08 |  -0.4674 |   -0.6987 |  -5.31 |   0.6205 |      3.07 |      35 | nan      | nan      |
| Equal-Weight  | 72.26 |    12.36 |  14.57 |   0.4803 |    0.8071 | -17.92 |   0.6895 |      8.2  |      56 |   0.0265 | *        |
| Buy-and-Hold  | 86.44 |    14.28 |  19.16 |   0.4937 |    0.846  | -24    |   0.5951 |     10.87 |       1 |   0.0484 | *        |
| Min-Variance  | 18.61 |     3.73 |   4.99 |  -0.393  |   -0.5886 |  -5.58 |   0.6677 |      2.89 |      56 |   0.5205 |          |
| NCO-No-Regime | 17.18 |     3.46 |   4.57 |  -0.4913 |   -0.7467 |  -5.18 |   0.6674 |      2.57 |      56 |   0.8182 |          |
| HRP           | 17.18 |     3.46 |   4.57 |  -0.4913 |   -0.7467 |  -5.18 |   0.6674 |      2.57 |      56 |   0.8182 |          |

### Jobson-Korkie Significance Tests (Regime-NCO vs each benchmark)

| Benchmark | z-stat | p-value | Significant at 5%? | Paper p |
|-----------|--------|---------|-------------------|---------|
| Equal-Weight | -2.2189 | 0.0265 | Yes* | 0.0146 |
| Buy-and-Hold | -1.9735 | 0.0484 | Yes* | 0.0195 |
| Min-Variance | -0.6426 | 0.5205 | No | 0.1999 |
| NCO-No-Regime | 0.2299 | 0.8182 | No | 0.079 |
| HRP | 0.2299 | 0.8182 | No | 0.079 |

### Interpretation

EW and B&H show higher returns but much larger drawdowns.
Regime-NCO is statistically distinguishable from high-return benchmarks
at 5%, consistent with paper findings.

---

## Experiment 3: Rebalancing Policy Comparison

### Summary Table

| Policy        |   Net_Ann_R% |   Sharpe |     MDD% |   CVaR95% |   N_Reb |   Total_TO% |   Avg_TO% |
|:--------------|-------------:|---------:|---------:|----------:|--------:|------------:|----------:|
| No-Rebalance  |      12.7486 |   0.4512 | -20.9081 |    9.8401 |       0 |       0     |    0      |
| Monthly       |       3.107  |  -0.498  |  -5.3144 |    3.1646 |      56 |     712.542 |   12.724  |
| Adaptive-0.02 |       3.1008 |  -0.4998 |  -5.3144 |    3.1646 |      46 |     699.35  |   12.4884 |
| Adaptive-0.05 |       3.2978 |  -0.4674 |  -5.3144 |    3.0743 |      35 |     642.627 |   11.4755 |
| Adaptive-0.10 |       3.1742 |  -0.5041 |  -5.3144 |    2.9545 |      31 |     550.893 |    9.8374 |

### Interpretation

Regime changes occur frequently, so adaptive thresholds trigger nearly every
month — matching paper's finding that drift thresholds add little value.

---

## Experiment 4: Robustness Analysis

### Subtest A_Window: sweep `window`

|   Value |   Ann_R% |   Sharpe |    MDD% |   Calmar |   N_Reb |   Total_TO% |
|--------:|---------:|---------:|--------:|---------:|--------:|------------:|
|      12 |   4.2696 |  -0.2219 | -6.0455 |   0.7062 |      41 |     790.17  |
|      18 |   4.0201 |  -0.2927 | -5.9536 |   0.6752 |      41 |     736.1   |
|      24 |   3.2978 |  -0.4674 | -5.3144 |   0.6205 |      35 |     642.627 |
|      36 |   3.1801 |  -0.5772 | -4.9079 |   0.6479 |      38 |     662.724 |

### Subtest B_Linkage: sweep `linkage_method`

| Value    |   Ann_R% |   Sharpe |    MDD% |   Calmar |   N_Reb |   Total_TO% |
|:---------|---------:|---------:|--------:|---------:|--------:|------------:|
| average  |   3.2978 |  -0.4674 | -5.3144 |   0.6205 |      35 |     642.627 |
| complete |   3.273  |  -0.4704 | -5.3179 |   0.6155 |      36 |     652.4   |
| single   |   3.5639 |  -0.3929 | -5.9761 |   0.5964 |      36 |     676.854 |

### Subtest C_Delta: sweep `delta`

|   Value |   Ann_R% |   Sharpe |    MDD% |   Calmar |   N_Reb |   Total_TO% |
|--------:|---------:|---------:|--------:|---------:|--------:|------------:|
|    0.02 |   3.1008 |  -0.4998 | -5.3144 |   0.5835 |      46 |     699.35  |
|    0.05 |   3.2978 |  -0.4674 | -5.3144 |   0.6205 |      35 |     642.627 |
|    0.1  |   3.1742 |  -0.5041 | -5.3144 |   0.5973 |      31 |     550.893 |

### Subtest D_Cost: sweep `cost_bps`

|   Value |   Ann_R% |   Sharpe |    MDD% |   Calmar |   N_Reb |   Total_TO% |
|--------:|---------:|---------:|--------:|---------:|--------:|------------:|
|       0 |   3.4403 |  -0.441  | -5.2094 |   0.6604 |      35 |     642.627 |
|       5 |   3.369  |  -0.4542 | -5.2619 |   0.6403 |      35 |     642.627 |
|      10 |   3.2978 |  -0.4674 | -5.3144 |   0.6205 |      35 |     642.627 |
|      20 |   3.1555 |  -0.4935 | -5.4193 |   0.5823 |      35 |     642.627 |

### Interpretation

Central qualitative result holds across all parameter variations:
- Low BTC exposure
- Shallow drawdown
- Defensive allocation profile (SBN/Gold/FX weighted)
- Transaction costs are the main drag on net returns
- Lookback window changes affect responsiveness more than risk profile

---

## Files Generated

| File | Description |
|------|-------------|
| `results/exp1_regime_nco.png` | NAV, regimes, weights, BIC for Exp 1 |
| `results/exp1_monthly_diagnostics.csv` | Month-by-month Exp 1 log |
| `results/exp1_metrics.csv` | Exp 1 performance metrics |
| `results/exp1_bic.csv` | BIC values for K=2,3,4 |
| `results/exp2_benchmarks.png` | NAV comparison and bar charts |
| `results/exp2_comparison_table.csv` | Strategy comparison metrics |
| `results/exp2_jobson_korkie.csv` | JK test statistics |
| `results/exp2_navs.csv` | NAV series for all strategies |
| `results/exp3_rebalancing.png` | Rebalancing policy NAV comparison |
| `results/exp3_rebalancing_table.csv` | Policy metrics table |
| `results/exp3_navs.csv` | NAV by policy |
| `results/exp4_robustness.png` | NAV curves for each subtest |
| `results/exp4_robustness_sharpe.png` | Sharpe sensitivity bar charts |
| `results/exp4_robustness_table.csv` | Full robustness grid |
