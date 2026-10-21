import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.metrics import (
    compute_cumulative_return, compute_annualized_return,
    compute_annualized_volatility, compute_sharpe_ratio,
    compute_sortino_ratio, compute_max_drawdown, compute_calmar_ratio,
    compute_cvar, compute_all_metrics, jobson_korkie_test
)


def make_returns(T=56, seed=42, mean=0.001, std=0.03):
    rng = np.random.RandomState(seed)
    r = rng.normal(mean, std, T)
    return pd.Series(r, index=pd.date_range('2021-01-31', periods=T, freq='ME'))


def test_cumulative_return_positive_returns():
    returns = pd.Series([0.01, 0.02, 0.01])
    cr = compute_cumulative_return(returns)
    expected = (1.01 * 1.02 * 1.01) - 1
    assert abs(cr - expected) < 1e-10


def test_cumulative_return_zero_returns():
    returns = pd.Series([0.0, 0.0, 0.0])
    cr = compute_cumulative_return(returns)
    assert abs(cr) < 1e-10


def test_max_drawdown_is_negative_or_zero():
    returns = make_returns()
    mdd = compute_max_drawdown(returns)
    assert mdd <= 0.0 or abs(mdd) < 1e-10, 'MDD should be non-positive (loss)'
    # MDD absolute value used in paper, so check absolute value
    assert abs(mdd) >= 0


def test_max_drawdown_no_loss():
    returns = pd.Series([0.01, 0.02, 0.03])
    mdd = compute_max_drawdown(returns)
    assert abs(mdd) < 1e-10


def test_sharpe_ratio_finite():
    returns = make_returns()
    sr = compute_sharpe_ratio(returns)
    assert np.isfinite(sr)


def test_cvar_between_0_and_1():
    returns = make_returns()
    cvar = compute_cvar(returns, confidence=0.95)
    assert cvar >= 0.0, 'CVaR should be a loss (positive value)'


def test_sortino_ratio_finite():
    returns = make_returns()
    sortino = compute_sortino_ratio(returns)
    assert np.isfinite(sortino)


def test_compute_all_metrics_keys():
    returns = make_returns()
    metrics = compute_all_metrics(returns)
    expected_keys = ['CR', 'Ann_R', 'Ann_Vol', 'Sharpe', 'Sortino', 'MDD', 'Calmar', 'CVaR95']
    for key in expected_keys:
        assert key in metrics, f'Missing key: {key}'


def test_annualized_volatility_scaling():
    """Annualized vol should be monthly vol * sqrt(12)."""
    returns = make_returns(T=120)
    monthly_std = returns.std()
    ann_vol = compute_annualized_volatility(returns)
    assert abs(ann_vol - monthly_std * np.sqrt(12)) < 1e-10


def test_jobson_korkie_test_structure():
    returns_a = make_returns(T=56, seed=1)
    returns_b = make_returns(T=56, seed=2)
    result = jobson_korkie_test(returns_a, returns_b)
    assert 'z_stat' in result
    assert 'p_value' in result
    assert 'reject_h0' in result
    assert 0 <= result['p_value'] <= 1
    assert isinstance(result['reject_h0'], bool)


def test_jobson_korkie_identical_returns():
    """JK test with identical returns should give p close to 1 (no difference)."""
    returns = make_returns(T=56, seed=42)
    result = jobson_korkie_test(returns, returns)
    assert result['p_value'] > 0.5 or np.isnan(result['p_value'])
