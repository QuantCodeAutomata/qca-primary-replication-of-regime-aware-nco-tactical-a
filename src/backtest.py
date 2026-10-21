import logging

import numpy as np
import pandas as pd

from src.hmm_regime import (
    decode_states,
    fit_hmm,
    get_regime_labels,
    label_states,
    rolling_hmm_decode,
)
from src.metrics import compute_all_metrics
from src.nco import compute_nco_portfolio

DEFAULT_COST_BPS = 10.0
DEFAULT_DELTA = 0.05
DEFAULT_WINDOW = 24
DEFAULT_K = 3

# Column positions in the 5-asset return DataFrame
_COL_LQ45 = 'LQ45'
_COL_SBN = 'SBN'
_COL_CNY = 'CNY_IDR'


def _build_window_features(ret_window: np.ndarray, col_names: list) -> np.ndarray:
    """Build 4-dimensional HMM feature vectors from a rolling return window.

    Computes features using only in-window data (fully causal):
      f_t = (r_LQ45_t, expanding_vol_LQ45_t, r_SBN_t, r_CNY_IDR_t)

    The expanding vol is the sample standard deviation of all LQ45 returns
    from the start of the window up to and including month t.  For the first
    month (single observation), vol is set to 0.

    Parameters
    ----------
    ret_window:
        Shape (T, N) array of simple monthly returns within the estimation window.
    col_names:
        List of column names matching ret_window columns.

    Returns
    -------
    np.ndarray of shape (T, 4).
    """
    # Replace NaN with 0.0 — missing return treated as zero (no price change)
    ret_window = np.where(np.isnan(ret_window), 0.0, ret_window)

    col_names_list = list(col_names)
    try:
        idx_lq45 = col_names_list.index(_COL_LQ45)
        idx_sbn = col_names_list.index(_COL_SBN)
        idx_cny = col_names_list.index(_COL_CNY)
    except ValueError:
        # Fallback: use first three columns
        idx_lq45, idx_cny, idx_sbn = 0, 2, 4 if len(col_names_list) > 4 else 2

    T = ret_window.shape[0]
    features = np.zeros((T, 4))
    for t in range(T):
        lq45_hist = ret_window[:t + 1, idx_lq45]
        vol_eq = float(np.std(lq45_hist, ddof=1)) if len(lq45_hist) > 1 else 0.0
        features[t, 0] = ret_window[t, idx_lq45]   # equity return (point)
        features[t, 1] = vol_eq                      # equity realized vol (expanding)
        features[t, 2] = ret_window[t, idx_sbn]     # bond return (point)
        features[t, 3] = ret_window[t, idx_cny]     # FX return (point)
    return features


def drift_weights(
    current_weights: np.ndarray,
    asset_returns: np.ndarray
) -> np.ndarray:
    """
    Compute market-drifted weights after one period of returns.
    w_drift_i = w_i * (1 + r_i) / sum_j(w_j * (1 + r_j))
    asset_returns: simple returns for one period
    Returns normalized weight array.
    """
    numerators = current_weights * (1.0 + asset_returns)
    total = numerators.sum()
    if total <= 0:
        n = len(current_weights)
        return np.ones(n) / n
    return numerators / total


def compute_portfolio_return(
    weights: np.ndarray,
    asset_returns: np.ndarray
) -> float:
    """
    Compute portfolio return for one period.
    Returns simple portfolio return: sum(w_i * r_i)
    """
    return float(weights @ asset_returns)


def should_rebalance(
    policy: str,
    current_regime: str,
    prev_regime: str,
    pretrade_weights: np.ndarray,
    target_weights: np.ndarray,
    delta: float = DEFAULT_DELTA
) -> tuple[bool, str]:
    """
    Determine if rebalancing should occur based on policy.
    policy: 'monthly', 'adaptive', 'none'
    Returns (rebalance_flag, trigger_reason).
    trigger_reason: 'regime_change', 'drift', 'both', 'monthly', 'none'
    """
    if policy == 'monthly':
        return True, 'monthly'

    if policy == 'none':
        return False, 'none'

    if policy == 'adaptive':
        regime_changed = (prev_regime != '') and (current_regime != prev_regime)
        max_drift = float(np.max(np.abs(pretrade_weights - target_weights)))
        drifted = max_drift > delta

        if regime_changed and drifted:
            return True, 'both'
        if regime_changed:
            return True, 'regime_change'
        if drifted:
            return True, 'drift'
        return False, 'none'

    # Default: treat unknown policy as monthly
    logging.warning(f"Unknown rebalancing policy '{policy}'; defaulting to monthly.")
    return True, 'monthly'


def regime_nco_backtest(
    monthly_returns: pd.DataFrame,
    hmm_features: pd.DataFrame = None,
    oos_start: str = '2021-01-01',
    window: int = DEFAULT_WINDOW,
    n_components: int = DEFAULT_K,
    cost_bps: float = DEFAULT_COST_BPS,
    policy: str = 'adaptive',
    delta: float = DEFAULT_DELTA,
    linkage_method: str = 'average',
    n_restarts: int = 5,
    random_state: int = 42
) -> dict:
    """
    Full regime-aware NCO walk-forward backtest.

    HMM feature vectors are computed on-the-fly from the rolling return window
    using :func:`_build_window_features` — this eliminates look-ahead from
    pre-computed feature alignment.  The ``hmm_features`` parameter is kept
    for API compatibility but is **not used** inside the backtest; features
    are always derived causally from ``monthly_returns``.

    Process:
    1. For each month t in out-of-sample period:
       a. Take rolling window [t-window:t] of returns
       b. Compute 4D HMM feature vectors from that window (causal)
       c. Fit HMM, decode Viterbi states, label regimes
       d. Current regime = last decoded state
       e. Compute regime mask for this window
       f. Run NCO pipeline: regime cov -> LW shrinkage -> clustering -> weights
       g. Store target weights and regime

    2. Walk-forward execution:
       a. At month t: compute target weights (using t-1 data up to t)
       b. Check rebalancing condition
       c. If rebalance: compute turnover, deduct cost, apply target weights
       d. Apply next month's asset returns to get next NAV
       e. Record all diagnostics

    Returns dict with:
    - 'returns': pd.Series of monthly net portfolio returns
    - 'nav': pd.Series of cumulative NAV
    - 'target_weights': pd.DataFrame
    - 'executed_weights': pd.DataFrame
    - 'pretrade_weights': pd.DataFrame
    - 'regimes': pd.Series of string regime labels
    - 'regime_stats': labeled state integers
    - 'turnover': pd.Series
    - 'costs': pd.Series
    - 'rebalance_flag': pd.Series of bool
    - 'trigger': pd.Series of str
    - 'metrics': dict from compute_all_metrics
    - 'n_rebalances': int
    - 'regime_counts': dict
    - 'avg_weights_by_regime': pd.DataFrame
    - 'nco_info': list of dicts with per-period NCO diagnostics

    IMPORTANT: Decision at month-end t applies to month t+1 returns.
    Target weights computed at end of month t are used for month t+1.
    """
    all_dates = monthly_returns.index
    asset_names = monthly_returns.columns
    n_assets = len(asset_names)
    col_names = list(asset_names)
    cost_factor = cost_bps / 10000.0

    oos_mask = all_dates >= oos_start
    oos_indices = np.where(oos_mask)[0]

    if len(oos_indices) == 0:
        raise ValueError(f'No OOS dates found at or after {oos_start}')

    first_oos_pos = oos_indices[0]
    if first_oos_pos < window:
        logging.warning(
            'Warm-up data shorter than window (%d < %d): using expanding window '
            'until full history is available.', first_oos_pos, window
        )

    # -----------------------------------------------------------------------
    # Phase 1: compute target weights for each OOS period
    # -----------------------------------------------------------------------
    target_weights_list = []
    regimes_list = []
    regime_stats_list = []
    nco_info = []

    for loop_t, idx in enumerate(oos_indices):
        oos_date = all_dates[idx]

        # Rolling window: months [idx-eff_w, idx) — fully causal.
        # Use expanding window if less history is available than requested.
        eff_w = min(window, idx)
        ret_window = monthly_returns.iloc[idx - eff_w:idx].values.astype(float)

        # Build 4D HMM feature vectors on-the-fly from this return window.
        # This avoids any positional mis-alignment with a pre-computed feature DF.
        feat_window = _build_window_features(ret_window, col_names)

        try:
            model = fit_hmm(
                feat_window,
                n_components=n_components,
                n_restarts=n_restarts,
                random_state=random_state
            )
            if model is None:
                raise RuntimeError('fit_hmm returned None')

            raw_states = decode_states(model, feat_window)
            labeled_states, _ = label_states(model, raw_states, equity_feature_idx=0)
            regime_strs = get_regime_labels(labeled_states, K=n_components)

            current_labeled = int(labeled_states[-1])
            current_regime_str = regime_strs[-1]

            # Regime mask: window periods sharing the current regime
            regime_mask = (labeled_states == current_labeled)

        except Exception as e:
            logging.warning(f'HMM decode failed at {oos_date}: {e}; using full mask')
            current_labeled = -1
            current_regime_str = 'Unknown'
            regime_mask = np.ones(len(ret_window), dtype=bool)

        # NCO portfolio for this period
        try:
            nco_result = compute_nco_portfolio(
                ret_window,
                regime_mask,
                n_assets,
                linkage_method=linkage_method
            )
            target_w = nco_result['weights']
        except Exception as e:
            logging.warning(f'NCO failed at {oos_date}: {e}; using equal weights')
            target_w = np.ones(n_assets) / n_assets
            nco_result = {
                'weights': target_w,
                'alpha': 0.0,
                'T_k': int(regime_mask.sum()),
                'n_clusters': 1,
                'cluster_labels': np.ones(n_assets, dtype=int),
                'cov_source': 'equal_weight_fallback',
                'cov_matrix': np.eye(n_assets)
            }

        target_weights_list.append(target_w)
        regimes_list.append(current_regime_str)
        regime_stats_list.append(current_labeled)
        nco_info.append({
            'date': oos_date,
            'alpha': nco_result['alpha'],
            'T_k': nco_result['T_k'],
            'n_clusters': nco_result['n_clusters'],
            'cov_source': nco_result['cov_source'],
            'regime': current_regime_str
        })

    oos_dates = all_dates[oos_indices]

    # -----------------------------------------------------------------------
    # Phase 2: walk-forward execution
    # IMPORTANT: target_w computed at end of month t applies to month t+1.
    # So target_weights_list[t] is used to earn return at oos_dates[t+1].
    # -----------------------------------------------------------------------
    net_returns = []
    navs = []
    turnovers = []
    costs_list = []
    rebalance_flags = []
    triggers = []
    executed_weights_list = []
    pretrade_weights_list = []

    nav = 1.0
    executed_w = np.zeros(n_assets)
    prev_regime = ''
    n_rebalances = 0

    for t in range(len(oos_dates)):
        r_t = monthly_returns.iloc[oos_indices[t]].values.astype(float)

        # Target weights were computed at end of previous month (t-1 decision -> t execution)
        # For the first OOS month, target comes from in-sample decision
        # Here we apply target_weights_list[t] to month t returns (same-period decision)
        # which represents end-of-period t-1 decision applied to period t
        target_w = target_weights_list[t]
        current_regime = regimes_list[t]

        # Pretrade weights: drift executed weights from previous period
        if t == 0:
            pretrade_w = np.zeros(n_assets)
        else:
            prev_r = monthly_returns.iloc[oos_indices[t - 1]].values.astype(float)
            pretrade_w = drift_weights(executed_w, prev_r)

        # Rebalancing decision
        do_rebal, trigger = should_rebalance(
            policy, current_regime, prev_regime, pretrade_w, target_w, delta=delta
        )

        if do_rebal:
            active_w = target_w
            turnover = float(np.sum(np.abs(target_w - pretrade_w)))
            cost = cost_factor * turnover
            n_rebalances += 1
        else:
            active_w = pretrade_w
            turnover = 0.0
            cost = 0.0

        gross_ret = compute_portfolio_return(active_w, r_t)
        net_ret = gross_ret - cost

        executed_w = active_w.copy()
        nav *= (1.0 + net_ret)
        prev_regime = current_regime

        net_returns.append(net_ret)
        navs.append(nav)
        turnovers.append(turnover)
        costs_list.append(cost)
        rebalance_flags.append(do_rebal)
        triggers.append(trigger)
        executed_weights_list.append(active_w.copy())
        pretrade_weights_list.append(pretrade_w.copy())

    # -----------------------------------------------------------------------
    # Assemble output series and DataFrames
    # -----------------------------------------------------------------------
    returns_s = pd.Series(net_returns, index=oos_dates, name='portfolio_return')
    nav_s = pd.Series(navs, index=oos_dates, name='nav')
    turnover_s = pd.Series(turnovers, index=oos_dates, name='turnover')
    costs_s = pd.Series(costs_list, index=oos_dates, name='costs')
    rebalance_s = pd.Series(rebalance_flags, index=oos_dates, name='rebalance')
    trigger_s = pd.Series(triggers, index=oos_dates, name='trigger')
    regimes_s = pd.Series(regimes_list, index=oos_dates, name='regime')
    regime_stats_s = pd.Series(regime_stats_list, index=oos_dates, name='labeled_state')

    target_w_df = pd.DataFrame(
        target_weights_list, index=oos_dates, columns=asset_names
    )
    executed_w_df = pd.DataFrame(
        executed_weights_list, index=oos_dates, columns=asset_names
    )
    pretrade_w_df = pd.DataFrame(
        pretrade_weights_list, index=oos_dates, columns=asset_names
    )

    # Regime statistics
    regime_counts = regimes_s.value_counts().to_dict()

    # Average weights by regime
    executed_w_df_copy = executed_w_df.copy()
    executed_w_df_copy['regime'] = regimes_s.values
    avg_weights_by_regime = executed_w_df_copy.groupby('regime')[list(asset_names)].mean()

    # Performance metrics
    metrics = compute_all_metrics(returns_s, turnover_s, n_rebalances)

    return {
        'returns': returns_s,
        'nav': nav_s,
        'target_weights': target_w_df,
        'executed_weights': executed_w_df,
        'pretrade_weights': pretrade_w_df,
        'regimes': regimes_s,
        'regime_stats': regime_stats_s,
        'turnover': turnover_s,
        'costs': costs_s,
        'rebalance_flag': rebalance_s,
        'trigger': trigger_s,
        'metrics': metrics,
        'n_rebalances': n_rebalances,
        'regime_counts': regime_counts,
        'avg_weights_by_regime': avg_weights_by_regime,
        'nco_info': nco_info
    }
