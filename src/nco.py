import logging
import warnings

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.optimize import minimize
from scipy.spatial.distance import squareform
from sklearn.covariance import LedoitWolf
from sklearn.metrics import silhouette_score


def compute_regime_covariance(
    returns_window: np.ndarray,
    regime_mask: np.ndarray,
    n_assets: int,
    W_min: int = 10
) -> tuple[np.ndarray, float, int]:
    """
    Compute blended regime-conditional covariance.

    Steps:
    1. Compute unconditional sample covariance Sigma_hat on full returns_window
    2. Compute T_k = sum(regime_mask) regime observations
    3. If T_k >= 2: compute regime-conditional sample cov Sigma_hat_k = np.cov(regime_returns.T)
       If T_k <= 1: Sigma_hat_k = None, alpha = 0
    4. alpha_t = min(1.0, T_k / W_min) if T_k >= 2 else 0.0
    5. Blended: Sigma_tilde = alpha_t * Sigma_hat_k + (1 - alpha_t) * Sigma_hat
    6. Return (Sigma_tilde, alpha_t, T_k)

    returns_window: shape (T, N) array of returns
    regime_mask: boolean array of length T, True for regime observations
    """
    Sigma_hat = np.cov(returns_window.T)
    if n_assets == 1:
        Sigma_hat = Sigma_hat.reshape(1, 1)

    T_k = int(np.sum(regime_mask))

    if T_k >= 2:
        regime_returns = returns_window[regime_mask]
        Sigma_hat_k = np.cov(regime_returns.T)
        if n_assets == 1:
            Sigma_hat_k = np.array([[Sigma_hat_k]])
        alpha_t = min(1.0, T_k / W_min)
    else:
        Sigma_hat_k = None
        alpha_t = 0.0

    if Sigma_hat_k is not None:
        Sigma_tilde = alpha_t * Sigma_hat_k + (1.0 - alpha_t) * Sigma_hat
    else:
        Sigma_tilde = Sigma_hat

    return Sigma_tilde, alpha_t, T_k


def apply_ledoit_wolf(
    cov_matrix: np.ndarray,
    returns_window: np.ndarray,
    regime_mask: np.ndarray,
    ridge: float = 1e-8
) -> np.ndarray:
    """
    Apply Ledoit-Wolf shrinkage to the covariance matrix.
    # Using sklearn.covariance.LedoitWolf — Context7 confirmed

    Strategy:
    - If regime returns (regime_mask True) have >= n_assets+1 observations: fit LW on regime returns
    - Otherwise: fit LW on full window returns
    - If LW fails: add ridge diagonal and return
    Returns shrunk covariance matrix of shape (N, N).
    """
    n_assets = cov_matrix.shape[0]
    regime_returns = returns_window[regime_mask]
    T_regime = len(regime_returns)

    try:
        lw = LedoitWolf()
        if T_regime >= n_assets + 1:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                lw.fit(regime_returns)
        else:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                lw.fit(returns_window)
        return lw.covariance_
    except Exception as e:
        logging.warning(f'LedoitWolf failed: {e}. Falling back to ridge regularization.')
        return cov_matrix + ridge * np.eye(n_assets)


def cov_to_distance(
    cov_matrix: np.ndarray
) -> np.ndarray:
    """
    Convert covariance matrix to angular distance matrix.
    1. Compute correlation: C_ij = Sigma_ij / sqrt(Sigma_ii * Sigma_jj)
    2. Clip correlations to [-1, 1]
    3. Distance: d_ij = sqrt((1 - C_ij) / 2)
    Returns distance matrix of same shape as cov_matrix.
    """
    diag = np.sqrt(np.diag(cov_matrix))
    # Avoid division by zero
    diag = np.where(diag == 0, 1e-10, diag)
    outer = np.outer(diag, diag)
    corr = cov_matrix / outer
    corr = np.clip(corr, -1.0, 1.0)
    dist = np.sqrt((1.0 - corr) / 2.0)
    # Ensure diagonal is zero
    np.fill_diagonal(dist, 0.0)
    return dist


def select_clusters(
    dist_matrix: np.ndarray,
    n_assets: int,
    linkage_method: str = 'average'
) -> tuple[int, np.ndarray]:
    """
    Perform hierarchical clustering and select optimal number of clusters M
    by maximizing silhouette score over M in {2, 3, 4} (excluding 1 and N).

    Steps:
    1. Compute condensed distance matrix via squareform(dist_matrix)
    2. linkage(condensed, method=linkage_method)
    3. For M in admissible range [2, min(4, n_assets-1)]:
       a. cluster_labels = fcluster(Z, t=M, criterion='maxclust')
       b. score = silhouette_score(dist_matrix, cluster_labels, metric='precomputed')
    4. Select M* = argmax silhouette score
    5. If n_assets <= 2: return M=1 (single cluster), all same label

    Returns (optimal_M, cluster_labels_array) where cluster_labels are 1-indexed.
    """
    if n_assets <= 2:
        return 1, np.ones(n_assets, dtype=int)

    condensed = squareform(dist_matrix)
    Z = linkage(condensed, method=linkage_method)

    max_M = min(4, n_assets - 1)
    best_M = 2
    best_score = -np.inf
    best_labels = None

    for M in range(2, max_M + 1):
        labels = fcluster(Z, t=M, criterion='maxclust')
        # silhouette_score requires at least 2 distinct labels
        if len(np.unique(labels)) < 2:
            continue
        try:
            score = silhouette_score(dist_matrix, labels, metric='precomputed')
        except Exception as e:
            logging.warning(f'Silhouette score failed for M={M}: {e}')
            continue
        if score > best_score:
            best_score = score
            best_M = M
            best_labels = labels

    if best_labels is None:
        # Fallback: use M=2
        best_labels = fcluster(Z, t=2, criterion='maxclust')
        best_M = 2

    return best_M, best_labels


def min_variance_long_only(
    cov_matrix: np.ndarray,
    ridge: float = 1e-8
) -> np.ndarray:
    """
    Solve long-only fully-invested minimum variance optimization.
    # Custom — using scipy.optimize.minimize with SLSQP

    min w' Sigma w subject to: sum(w) = 1, w >= 0

    If n_assets == 1: return array([1.0])
    Try scipy.optimize.minimize first.
    Fallback: equal weights if optimization fails.
    Returns weight array of length n_assets summing to 1, all >= 0.
    """
    n = cov_matrix.shape[0]
    if n == 1:
        return np.array([1.0])

    # Regularize slightly for numerical stability
    Sigma = cov_matrix + ridge * np.eye(n)

    w0 = np.ones(n) / n
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
    bounds = [(0.0, 1.0)] * n

    try:
        result = minimize(
            fun=lambda w: w @ Sigma @ w,
            x0=w0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'ftol': 1e-9, 'maxiter': 1000}
        )
        if result.success or result.status == 0:
            w = np.clip(result.x, 0.0, 1.0)
            total = w.sum()
            if total > 0:
                return w / total
    except Exception as e:
        logging.warning(f'min_variance_long_only optimization failed: {e}')

    # Fallback: equal weights
    return np.ones(n) / n


def nco_weights(
    cov_denoised: np.ndarray,
    cluster_labels: np.ndarray,
    n_assets: int
) -> np.ndarray:
    """
    Compute NCO portfolio weights.

    1. For each unique cluster m in cluster_labels:
       a. Get asset indices I_m for cluster m
       b. Extract intra-cluster covariance Sigma_m = cov_denoised[I_m][:, I_m]
       c. Solve u_m = min_variance_long_only(Sigma_m)
       d. Store intra-cluster weights embedded in N-dim space

    2. Build mapping matrix U (N x M): column m has u_m weights at positions I_m, 0 elsewhere

    3. Compute cluster-level covariance: Sigma_cl = U.T @ cov_denoised @ U (M x M)

    4. Solve inter-cluster min-var: v = min_variance_long_only(Sigma_cl)

    5. Final weights: w_i = v_{c(i)} * u_{c(i), i}  for each asset i

    6. Normalize to sum to 1 (handle numerical precision)

    Returns weight array of length n_assets.
    """
    unique_clusters = np.unique(cluster_labels)
    M = len(unique_clusters)

    # Map cluster label -> column index in U
    cluster_col = {label: idx for idx, label in enumerate(unique_clusters)}

    # U: (N x M) mapping matrix
    U = np.zeros((n_assets, M))

    # Step 1 & 2: intra-cluster weights
    for label in unique_clusters:
        m = cluster_col[label]
        I_m = np.where(cluster_labels == label)[0]
        Sigma_m = cov_denoised[np.ix_(I_m, I_m)]
        u_m = min_variance_long_only(Sigma_m)
        U[I_m, m] = u_m

    # Step 3: cluster-level covariance
    Sigma_cl = U.T @ cov_denoised @ U  # (M x M)

    # Step 4: inter-cluster min-var weights
    v = min_variance_long_only(Sigma_cl)

    # Step 5: final weights
    w = np.zeros(n_assets)
    for label in unique_clusters:
        m = cluster_col[label]
        I_m = np.where(cluster_labels == label)[0]
        for i in I_m:
            w[i] = v[m] * U[i, m]

    # Step 6: normalize
    total = w.sum()
    if total > 0:
        w = w / total
    else:
        w = np.ones(n_assets) / n_assets

    return w


def compute_nco_portfolio(
    returns_window: np.ndarray,
    regime_mask: np.ndarray,
    n_assets: int,
    W_min: int = 10,
    linkage_method: str = 'average',
    ridge: float = 1e-8
) -> dict:
    """
    Full NCO pipeline for one decision date.

    Returns dict with keys:
    - 'weights': np.ndarray of shape (n_assets,)
    - 'alpha': blending alpha used
    - 'T_k': number of regime observations
    - 'n_clusters': optimal cluster count M
    - 'cluster_labels': cluster assignments
    - 'cov_source': str describing which covariance was used
    - 'cov_matrix': final denoised covariance used
    """
    # Step 1: regime-blended covariance
    Sigma_tilde, alpha_t, T_k = compute_regime_covariance(
        returns_window, regime_mask, n_assets, W_min=W_min
    )

    # Step 2: Ledoit-Wolf shrinkage
    cov_denoised = apply_ledoit_wolf(
        Sigma_tilde, returns_window, regime_mask, ridge=ridge
    )

    # Determine covariance source description
    if alpha_t >= 1.0:
        cov_source = 'regime_only'
    elif alpha_t > 0.0:
        cov_source = f'blended_alpha={alpha_t:.2f}'
    else:
        cov_source = 'unconditional'

    # Step 3: convert to distance matrix
    dist_matrix = cov_to_distance(cov_denoised)

    # Step 4: hierarchical clustering
    optimal_M, cluster_labels = select_clusters(
        dist_matrix, n_assets, linkage_method=linkage_method
    )

    # Step 5: NCO weights
    weights = nco_weights(cov_denoised, cluster_labels, n_assets)

    return {
        'weights': weights,
        'alpha': alpha_t,
        'T_k': T_k,
        'n_clusters': optimal_M,
        'cluster_labels': cluster_labels,
        'cov_source': cov_source,
        'cov_matrix': cov_denoised
    }
