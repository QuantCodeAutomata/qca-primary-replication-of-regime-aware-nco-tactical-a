import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.nco import (
    compute_regime_covariance, apply_ledoit_wolf,
    cov_to_distance, select_clusters,
    min_variance_long_only, nco_weights, compute_nco_portfolio
)


def make_returns(T=24, N=5, seed=42):
    rng = np.random.RandomState(seed)
    return rng.randn(T, N) * 0.05


def test_regime_covariance_shape():
    returns = make_returns(T=24, N=5)
    mask = np.array([True]*10 + [False]*14)
    sigma, alpha, T_k = compute_regime_covariance(returns, mask, n_assets=5)
    assert sigma.shape == (5, 5)
    assert 0.0 <= alpha <= 1.0
    assert T_k == 10


def test_regime_covariance_fallback_few_obs():
    """When T_k <= 1, alpha should be 0 and fallback to unconditional."""
    returns = make_returns(T=24, N=5)
    mask = np.array([True] + [False]*23)  # only 1 regime observation
    sigma, alpha, T_k = compute_regime_covariance(returns, mask, n_assets=5)
    assert alpha == 0.0
    assert sigma.shape == (5, 5)


def test_ledoit_wolf_positive_definite():
    returns = make_returns(T=24, N=5)
    mask = np.ones(24, dtype=bool)
    cov = compute_regime_covariance(returns, mask, n_assets=5)[0]
    shrunk = apply_ledoit_wolf(cov, returns, mask)
    eigenvalues = np.linalg.eigvalsh(shrunk)
    assert np.all(eigenvalues > -1e-10), 'Covariance should be PSD after LW shrinkage'


def test_cov_to_distance_range():
    returns = make_returns(T=24, N=5)
    cov = np.cov(returns.T)
    dist = cov_to_distance(cov)
    assert dist.shape == (5, 5)
    assert np.all(dist >= -1e-10), 'Distances should be non-negative'
    assert np.all(dist <= 1.0 + 1e-10), 'Angular distances should be <= 1'
    assert np.allclose(np.diag(dist), 0.0, atol=1e-10), 'Diagonal should be 0'


def test_select_clusters_range():
    returns = make_returns(T=24, N=5)
    cov = np.cov(returns.T)
    dist = cov_to_distance(cov)
    M, labels = select_clusters(dist, n_assets=5, linkage_method='average')
    assert 2 <= M <= 4
    assert len(labels) == 5


def test_min_variance_weights_sum_to_one():
    returns = make_returns(T=24, N=5)
    cov = np.cov(returns.T)
    w = min_variance_long_only(cov)
    assert abs(sum(w) - 1.0) < 1e-6
    assert all(wi >= -1e-8 for wi in w)
    assert len(w) == 5


def test_min_variance_single_asset():
    cov = np.array([[0.01]])
    w = min_variance_long_only(cov)
    assert abs(w[0] - 1.0) < 1e-10


def test_nco_weights_sum_to_one():
    returns = make_returns(T=24, N=5)
    cov = np.cov(returns.T)
    dist = cov_to_distance(cov)
    M, labels = select_clusters(dist, n_assets=5)
    w = nco_weights(cov, labels, n_assets=5)
    assert abs(sum(w) - 1.0) < 1e-6
    assert all(wi >= -1e-8 for wi in w)
    assert len(w) == 5


def test_compute_nco_portfolio_output():
    returns = make_returns(T=24, N=5)
    mask = np.ones(24, dtype=bool)
    result = compute_nco_portfolio(returns, mask, n_assets=5)
    assert 'weights' in result
    assert abs(sum(result['weights']) - 1.0) < 1e-5
    assert all(w >= -1e-8 for w in result['weights'])
