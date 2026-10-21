import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.benchmarks import (
    equal_weight_backtest, buy_and_hold_backtest,
    min_variance_backtest, standard_nco_backtest
)


def make_monthly_returns(T=80, N=5, seed=42, start='2019-01-31'):
    rng = np.random.RandomState(seed)
    dates = pd.date_range(start, periods=T, freq='ME')
    returns = pd.DataFrame(
        rng.normal(0.01, 0.05, (T, N)),
        index=dates,
        columns=['SBN', 'LQ45', 'Gold_IDR', 'CNY_IDR', 'BTC_IDR']
    )
    return returns


def test_equal_weight_nav_starts_near_one():
    """NAV at end of first OOS month should be close to 1.0 (initial + one period return)."""
    returns = make_monthly_returns()
    result = equal_weight_backtest(returns, oos_start='2021-01-01')
    # NAV after first period: 1.0 * (1 + first_month_net_return); expect in reasonable range
    assert 0.5 < result['nav'].iloc[0] < 1.5, "First NAV should be near 1.0"
    assert all(result['nav'] > 0), "All NAV values must be positive"


def test_equal_weight_weights_sum_to_one():
    returns = make_monthly_returns()
    result = equal_weight_backtest(returns, oos_start='2021-01-01')
    if result.get('weights') is not None:
        for _, w in result['weights'].iterrows():
            assert abs(w.sum() - 1.0) < 1e-6


def test_buy_and_hold_no_rebalance():
    returns = make_monthly_returns()
    result = buy_and_hold_backtest(returns, oos_start='2021-01-01')
    # After initial, no more rebalancing -> turnover should be 0
    assert result['n_rebalances'] <= 1  # only initial trade


def test_min_variance_weights_long_only():
    returns = make_monthly_returns()
    result = min_variance_backtest(returns, oos_start='2021-01-01', window=24)
    if result.get('weights') is not None:
        for _, w in result['weights'].iterrows():
            assert all(wi >= -1e-8 for wi in w)
            assert abs(w.sum() - 1.0) < 1e-6


def test_standard_nco_returns_correct_structure():
    returns = make_monthly_returns()
    result = standard_nco_backtest(returns, oos_start='2021-01-01', window=24)
    assert 'returns' in result
    assert 'nav' in result
    assert 'metrics' in result
    assert len(result['returns']) > 0


def test_all_navs_positive():
    returns = make_monthly_returns()
    for fn in [equal_weight_backtest, buy_and_hold_backtest,
               min_variance_backtest, standard_nco_backtest]:
        result = fn(returns, oos_start='2021-01-01')
        assert all(result['nav'] > 0), f'{fn.__name__} NAV should always be positive'
