import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.hmm_regime import (
    fit_hmm, decode_states, label_states, get_regime_labels,
    compute_bic_for_k, rolling_hmm_decode
)


def make_synthetic_features(T=60, n_features=4, seed=42):
    """Create synthetic HMM feature data."""
    rng = np.random.RandomState(seed)
    return rng.randn(T, n_features)


def test_fit_hmm_returns_model():
    """Test that fit_hmm returns a fitted GaussianHMM model."""
    X = make_synthetic_features(T=50)
    model = fit_hmm(X, n_components=3, n_iter=100, n_restarts=2)
    assert model is not None
    assert hasattr(model, 'means_')
    assert model.means_.shape == (3, 4)


def test_decode_states_length():
    """Test decode_states returns correct length."""
    X = make_synthetic_features(T=50)
    model = fit_hmm(X, n_components=3, n_iter=100, n_restarts=2)
    states = decode_states(model, X)
    assert len(states) == 50
    assert states.min() >= 0
    assert states.max() <= 2


def test_label_states_maps_correctly():
    """Test state labeling by equity return feature."""
    X = make_synthetic_features(T=50)
    model = fit_hmm(X, n_components=3, n_iter=100, n_restarts=2)
    raw_states = decode_states(model, X)
    labeled, state_map = label_states(model, raw_states, equity_feature_idx=0)
    # Verify all states in {0, 1, 2}
    assert set(np.unique(labeled)).issubset({0, 1, 2})
    # Verify state_map maps all raw states
    assert len(state_map) == 3


def test_get_regime_labels_k3():
    """Test regime label generation for K=3."""
    labeled_states = np.array([0, 1, 2, 0, 1, 2])
    labels = get_regime_labels(labeled_states, K=3)
    assert 'Bullish' in labels
    assert 'Sideways' in labels
    assert 'Bearish' in labels


def test_bic_model_selection():
    """Test BIC computation for different K values."""
    X = make_synthetic_features(T=60)
    bic_dict = compute_bic_for_k(X, k_values=[2, 3, 4], n_restarts=2)
    assert set(bic_dict.keys()) == {2, 3, 4}
    for k, bic in bic_dict.items():
        assert np.isfinite(bic), f'BIC for K={k} should be finite'


def test_hmm_states_are_integers():
    """Test that decoded states are integers."""
    X = make_synthetic_features(T=40)
    model = fit_hmm(X, n_components=2, n_iter=50, n_restarts=2)
    states = decode_states(model, X)
    assert states.dtype in [np.int32, np.int64, int]


def test_rolling_hmm_decode_output_shape():
    """Test rolling HMM decode returns correct shape."""
    T = 36
    rng = np.random.RandomState(42)
    dates = pd.date_range('2019-01-31', periods=T, freq='ME')
    features = pd.DataFrame(rng.randn(T, 4), index=dates,
                            columns=['mean_eq', 'vol_eq', 'mean_bond', 'delta_fx'])
    monthly_returns = pd.DataFrame(rng.randn(T, 5) * 0.05, index=dates,
                                   columns=['SBN', 'LQ45', 'Gold_IDR', 'CNY_IDR', 'BTC_IDR'])
    oos_dates = dates[24:]  # last 12 months
    result = rolling_hmm_decode(
        hmm_features=features,
        monthly_returns=monthly_returns,
        window=24, n_components=3, n_iter=100, n_restarts=2
    )
    assert isinstance(result, pd.DataFrame)
    assert 'regime_label' in result.columns
    assert len(result) > 0
