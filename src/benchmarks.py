import logging
import warnings

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.covariance import LedoitWolf
from sklearn.metrics import silhouette_score

from src.metrics import compute_all_metrics
from src.nco import (
    apply_ledoit_wolf,
    compute_regime_covariance,
    cov_to_distance,
    min_variance_long_only,
    nco_weights,
    select_clusters,
)


def _drift_weights(weights: np.ndarray, returns: np.ndarray) -> np.ndarray:
    """Drift portfolio weights by one period of asset returns."""
    gross = 1.0 + float(weights @ returns)
    if gross <= 0:
        n = len(weights)
        return np.ones(n) / n
    return weights * (1.0 + returns) / gross


def equal_weight_backtest(
    monthly_returns: pd.DataFrame,
    oos_start: str = '2021-01-01',
    cost_bps: float = 10.0
) -> dict:
    """
    Equal Weight benchmark: constant 1/N weights, rebalance monthly.
    Returns dict with keys: 'returns' (pd.Series of monthly net returns),
    'nav' (pd.Series of NAV), 'turnover' (pd.Series), 'weights' (pd.DataFrame),
    'metrics' (dict from compute_all_metrics), 'n_rebalances' (int).
    """
    oos_mask = monthly_returns.index >= oos_start
    oos_returns = monthly_returns[oos_mask]
    n_assets = monthly_returns.shape[1]
    n_oos = len(oos_returns)
    cost_factor = cost_bps / 10000.0

    target_w = np.ones(n_assets) / n_assets

    nav = 1.0
    executed_w = np.zeros(n_assets)  # start fully in cash

    net_returns, navs, turnovers, weights_list = [], [], [], []
    n_rebalances = 0

    for t in range(n_oos):
        r_t = oos_returns.iloc[t].values.astype(float)

        # Pretrade weights: drift from previous period's executed weights
        if t == 0:
            pretrade_w = np.zeros(n_assets)
        else:
            pretrade_w = _drift_weights(executed_w, oos_returns.iloc[t - 1].values.astype(float))

        # Rebalance to EW
        turnover = float(np.sum(np.abs(target_w - pretrade_w)))
        cost = cost_factor * turnover

        gross_ret = float(target_w @ r_t)
        net_ret = gross_ret - cost

        executed_w = target_w.copy()
        nav *= (1.0 + net_ret)

        net_returns.append(net_ret)
        navs.append(nav)
        turnovers.append(turnover)
        weights_list.append(target_w.copy())
        n_rebalances += 1

    dates = oos_returns.index
    returns_s = pd.Series(net_returns, index=dates)
    nav_s = pd.Series(navs, index=dates)
    turnover_s = pd.Series(turnovers, index=dates)
    weights_df = pd.DataFrame(weights_list, index=dates, columns=monthly_returns.columns)
    metrics = compute_all_metrics(returns_s, turnover_s, n_rebalances)

    return {
        'returns': returns_s,
        'nav': nav_s,
        'turnover': turnover_s,
        'weights': weights_df,
        'metrics': metrics,
        'n_rebalances': n_rebalances
    }


def buy_and_hold_backtest(
    monthly_returns: pd.DataFrame,
    oos_start: str = '2021-01-01',
    cost_bps: float = 10.0
) -> dict:
    """
    Buy-and-Hold: start with equal weights, never rebalance after.
    Initial trade incurs transaction cost.
    Weights drift with market.
    Returns same dict structure as equal_weight_backtest.
    """
    oos_mask = monthly_returns.index >= oos_start
    oos_returns = monthly_returns[oos_mask]
    n_assets = monthly_returns.shape[1]
    n_oos = len(oos_returns)
    cost_factor = cost_bps / 10000.0

    initial_w = np.ones(n_assets) / n_assets

    nav = 1.0
    current_w = initial_w.copy()

    net_returns, navs, turnovers, weights_list = [], [], [], []
    n_rebalances = 0

    for t in range(n_oos):
        r_t = oos_returns.iloc[t].values.astype(float)

        if t == 0:
            # Initial purchase from cash: turnover = 1
            turnover = float(np.sum(np.abs(initial_w - np.zeros(n_assets))))
            cost = cost_factor * turnover
            n_rebalances += 1
        else:
            turnover = 0.0
            cost = 0.0

        gross_ret = float(current_w @ r_t)
        net_ret = gross_ret - cost

        nav *= (1.0 + net_ret)

        net_returns.append(net_ret)
        navs.append(nav)
        turnovers.append(turnover)
        weights_list.append(current_w.copy())

        # Drift weights for next period
        current_w = _drift_weights(current_w, r_t)

    dates = oos_returns.index
    returns_s = pd.Series(net_returns, index=dates)
    nav_s = pd.Series(navs, index=dates)
    turnover_s = pd.Series(turnovers, index=dates)
    weights_df = pd.DataFrame(weights_list, index=dates, columns=monthly_returns.columns)
    metrics = compute_all_metrics(returns_s, turnover_s, n_rebalances)

    return {
        'returns': returns_s,
        'nav': nav_s,
        'turnover': turnover_s,
        'weights': weights_df,
        'metrics': metrics,
        'n_rebalances': n_rebalances
    }


def min_variance_backtest(
    monthly_returns: pd.DataFrame,
    oos_start: str = '2021-01-01',
    window: int = 24,
    cost_bps: float = 10.0
) -> dict:
    """
    Rolling Minimum Variance (unconditional, no regime).
    At each month-end, fit LW covariance on trailing 24-month window,
    solve long-only min-var optimization.
    Rebalance monthly. Apply transaction costs.
    Returns same dict structure.
    """
    all_dates = monthly_returns.index
    oos_mask = all_dates >= oos_start
    oos_indices = np.where(oos_mask)[0]

    if len(oos_indices) == 0:
        raise ValueError('No OOS dates found')

    first_oos_pos = oos_indices[0]
    if first_oos_pos < window:
        raise ValueError(
            f'Not enough history before OOS start: need {window}, have {first_oos_pos}'
        )

    n_assets = monthly_returns.shape[1]
    cost_factor = cost_bps / 10000.0

    nav = 1.0
    executed_w = np.zeros(n_assets)

    net_returns, navs, turnovers, weights_list = [], [], [], []
    n_rebalances = 0

    for loop_t, idx in enumerate(oos_indices):
        r_t = monthly_returns.iloc[idx].values.astype(float)

        # Training window: [idx-window, idx)
        train = monthly_returns.iloc[idx - window:idx].values.astype(float)

        # Ledoit-Wolf covariance
        try:
            lw = LedoitWolf()
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                lw.fit(train)
            cov = lw.covariance_
        except Exception as e:
            logging.warning(f'LW failed at {all_dates[idx]}: {e}; using sample cov')
            cov = np.cov(train.T)

        target_w = min_variance_long_only(cov)

        # Pretrade weights: drift from previous executed weights
        if loop_t == 0:
            pretrade_w = np.zeros(n_assets)
        else:
            prev_r = monthly_returns.iloc[oos_indices[loop_t - 1]].values.astype(float)
            pretrade_w = _drift_weights(executed_w, prev_r)

        turnover = float(np.sum(np.abs(target_w - pretrade_w)))
        cost = cost_factor * turnover
        gross_ret = float(target_w @ r_t)
        net_ret = gross_ret - cost

        executed_w = target_w.copy()
        nav *= (1.0 + net_ret)

        net_returns.append(net_ret)
        navs.append(nav)
        turnovers.append(turnover)
        weights_list.append(target_w.copy())
        n_rebalances += 1

    dates = all_dates[oos_indices]
    returns_s = pd.Series(net_returns, index=dates)
    nav_s = pd.Series(navs, index=dates)
    turnover_s = pd.Series(turnovers, index=dates)
    weights_df = pd.DataFrame(weights_list, index=dates, columns=monthly_returns.columns)
    metrics = compute_all_metrics(returns_s, turnover_s, n_rebalances)

    return {
        'returns': returns_s,
        'nav': nav_s,
        'turnover': turnover_s,
        'weights': weights_df,
        'metrics': metrics,
        'n_rebalances': n_rebalances
    }


def standard_nco_backtest(
    monthly_returns: pd.DataFrame,
    oos_start: str = '2021-01-01',
    window: int = 24,
    cost_bps: float = 10.0,
    linkage_method: str = 'average'
) -> dict:
    """
    Standard NCO without regime conditioning.
    Same NCO architecture as regime-aware but uses full rolling window,
    no HMM, no regime filtering.
    Regime mask = all True (use all data).
    Rebalance monthly. Apply transaction costs.
    Returns same dict structure.
    """
    all_dates = monthly_returns.index
    oos_mask = all_dates >= oos_start
    oos_indices = np.where(oos_mask)[0]

    if len(oos_indices) == 0:
        raise ValueError('No OOS dates found')

    first_oos_pos = oos_indices[0]
    if first_oos_pos < window:
        raise ValueError(
            f'Not enough history before OOS start: need {window}, have {first_oos_pos}'
        )

    n_assets = monthly_returns.shape[1]
    cost_factor = cost_bps / 10000.0

    nav = 1.0
    executed_w = np.zeros(n_assets)

    net_returns, navs, turnovers, weights_list = [], [], [], []
    n_rebalances = 0

    for loop_t, idx in enumerate(oos_indices):
        r_t = monthly_returns.iloc[idx].values.astype(float)

        # Training window: [idx-window, idx)
        train = monthly_returns.iloc[idx - window:idx].values.astype(float)
        regime_mask = np.ones(len(train), dtype=bool)  # use all data

        # Step 1: regime-blended cov (alpha=0, pure unconditional)
        Sigma_tilde, alpha_t, T_k = compute_regime_covariance(
            train, regime_mask, n_assets, W_min=window
        )

        # Step 2: LW shrinkage
        cov_denoised = apply_ledoit_wolf(Sigma_tilde, train, regime_mask)

        # Step 3: distance matrix
        dist_matrix = cov_to_distance(cov_denoised)

        # Step 4: clustering
        optimal_M, cluster_labels = select_clusters(
            dist_matrix, n_assets, linkage_method=linkage_method
        )

        # Step 5: NCO weights
        target_w = nco_weights(cov_denoised, cluster_labels, n_assets)

        # Pretrade weights: drift from previous executed weights
        if loop_t == 0:
            pretrade_w = np.zeros(n_assets)
        else:
            prev_r = monthly_returns.iloc[oos_indices[loop_t - 1]].values.astype(float)
            pretrade_w = _drift_weights(executed_w, prev_r)

        turnover = float(np.sum(np.abs(target_w - pretrade_w)))
        cost = cost_factor * turnover
        gross_ret = float(target_w @ r_t)
        net_ret = gross_ret - cost

        executed_w = target_w.copy()
        nav *= (1.0 + net_ret)

        net_returns.append(net_ret)
        navs.append(nav)
        turnovers.append(turnover)
        weights_list.append(target_w.copy())
        n_rebalances += 1

    dates = all_dates[oos_indices]
    returns_s = pd.Series(net_returns, index=dates)
    nav_s = pd.Series(navs, index=dates)
    turnover_s = pd.Series(turnovers, index=dates)
    weights_df = pd.DataFrame(weights_list, index=dates, columns=monthly_returns.columns)
    metrics = compute_all_metrics(returns_s, turnover_s, n_rebalances)

    return {
        'returns': returns_s,
        'nav': nav_s,
        'turnover': turnover_s,
        'weights': weights_df,
        'metrics': metrics,
        'n_rebalances': n_rebalances
    }
