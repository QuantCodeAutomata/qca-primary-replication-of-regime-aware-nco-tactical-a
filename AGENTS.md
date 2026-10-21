# Regime-Aware NCO TAA — Project Memory

## Architecture
- data/fetch.py: Data fetching using yfinance and Massive API; produces monthly returns DataFrame
- src/hmm_regime.py: GaussianHMM with rolling window; regime labeling by equity return mean
- src/nco.py: NCO with Ledoit-Wolf shrinkage, avg-linkage clustering, silhouette selection, min-var optimization
- src/backtest.py: Walk-forward backtest engine; adaptive rebalancing logic; NAV accounting
- src/metrics.py: Performance metrics: Sharpe, Sortino, MDD, Calmar, CVaR95, Jobson-Korkie test
- src/benchmarks.py: EW, B&H, MinVar, Standard NCO (no regime)
- exp/exp_1_regime_nco.py: Primary replication (exp_1)
- exp/exp_2_benchmarks.py: Benchmark comparison (exp_2)
- exp/exp_3_rebalancing.py: Rebalancing rule evaluation (exp_3)
- exp/exp_4_robustness.py: Sensitivity analysis (exp_4)

## Key Parameters
- Assets: SBN, LQ45, Gold_IDR, CNYIDR, BTC_IDR (N=5)
- Training: 2019-01-02 to 2020-12-31
- Backtest: 2021-01-01 to 2025-08-31 (56 months)
- HMM: K=3, rolling W=24 months, features=(mean_eq, vol_eq, mean_bond, delta_fx)
- NCO: avg-linkage, silhouette M in {2,3,4}, LW shrinkage, alpha=min(1, T_k/10)
- Cost: c=10bps, delta=0.05 for adaptive rebalancing
- Risk-free: 0.479%/month

## Data Sources
- LQ45: Yahoo Finance ^JKLQ45
- BTC/IDR: Yahoo Finance BTC-IDR
- CNY/IDR: Yahoo Finance CNYIDR=X
- Gold IDR: GC=F × USDIDR=X from Yahoo Finance
- SBN: EIDO or proxy via Yahoo Finance (^JKBOND or alternative)
- USD/IDR for conversion: USDIDR=X from Yahoo Finance

## Implementation Notes
- Use simple returns (exp(log_return)-1) for NAV compounding
- HMM state labeling: rank by mean equity-return feature (highest=Bullish, middle=Sideways, lowest=Bearish)
- BIC from hmmlearn model.bic(X) method (available in hmmlearn 0.3.3)
- Silhouette computed on precomputed angular distance matrix
- Ledoit-Wolf: sklearn.covariance.LedoitWolf
- Avg-linkage clustering: scipy.cluster.hierarchy.linkage with method='average'
- select_clusters: searches M in {2, 3, 4}; if all assets equidistant (random/uncorrelated data) silhouette loop falls through to M=2 fallback with single-cluster labels; nco_weights degrades gracefully to equal weights
- regime_mask in NCO: which periods in the rolling window share the SAME labeled_state as the last period; created from per-period HMM fit (not from rolling_hmm_decode output)
- rolling_hmm_decode is imported in backtest.py but per-period HMM is re-fit in regime_nco_backtest to obtain the full window decoding needed for regime_mask construction
- Transaction cost model: net_ret = gross_ret - (cost_bps/10000 * turnover); turnover = sum(|w_target - w_pretrade|); pretrade weights are market-drifted from previous period
- Walk-forward execution: pretrade_w for period t = drift(executed_w_{t-1}, r_{t-1}); t=0 starts from cash (zeros)
- should_rebalance 'adaptive': checks max(|pretrade - target|) > delta for drift trigger; regime_change when prev_regime != '' and current != prev

## Package Versions (verified working)
- hmmlearn==0.3.3
- scikit-learn==1.5.1
- scipy==1.14.1
- pandas==2.2.2
- numpy==1.26.4
