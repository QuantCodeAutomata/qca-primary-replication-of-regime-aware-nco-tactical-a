import logging
import warnings

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.covariance import LedoitWolf


def fit_hmm(
    features: np.ndarray,
    n_components: int = 3,
    n_iter: int = 500,
    n_restarts: int = 5,
    random_state: int = 42,
    tol: float = 1e-4
) -> GaussianHMM:
    """
    Fit GaussianHMM with multiple random restarts, selecting the highest log-likelihood model.
    Uses hmmlearn.hmm.GaussianHMM with covariance_type='full', algorithm='viterbi'.
    """
    # Sanitise: replace NaN/Inf with 0 to prevent hmmlearn failures
    features = np.where(np.isfinite(features), features, 0.0)

    best_model = None
    best_score = -np.inf
    for seed in range(random_state, random_state + n_restarts):
        try:
            model = GaussianHMM(
                n_components=n_components,
                covariance_type='full',
                n_iter=n_iter,
                tol=tol,
                random_state=seed,
                algorithm='viterbi'
            )
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                model.fit(features)
            score = model.score(features)
            if score > best_score:
                best_score = score
                best_model = model
        except Exception as e:
            logging.warning(f'HMM fit failed seed={seed}: {e}')
    return best_model


def decode_states(
    model: GaussianHMM,
    features: np.ndarray
) -> np.ndarray:
    """
    Decode the most likely state sequence using Viterbi algorithm.
    Returns array of integer states of length len(features).
    """
    _, states = model.decode(features, algorithm='viterbi')
    return states


def label_states(
    model: GaussianHMM,
    raw_states: np.ndarray,
    equity_feature_idx: int = 0
) -> tuple[np.ndarray, dict]:
    """
    Re-label raw integer states by ranking the mean equity-return feature.
    Highest mean equity return -> 0 (Bullish)
    Middle -> 1 (Sideways)
    Lowest mean equity return -> 2 (Bearish)
    Returns (labeled_states, state_map) where state_map maps raw->label int.
    For K=2: highest=0 (Bullish), lowest=2 (Bearish), no sideways.
    For K=4: rank descending, labels 0,1,2,3 (Bullish,SideUp,SideDown,Bearish).
    Always: 0=Bullish, K-1=Bearish.
    """
    K = model.n_components
    means = model.means_[:, equity_feature_idx]  # shape (K,)

    # Rank states by descending mean equity return
    sorted_raw = np.argsort(means)[::-1]  # raw state indices sorted high->low mean

    # Build mapping: raw_state -> label (0=Bullish, K-1=Bearish)
    state_map = {}
    for label, raw in enumerate(sorted_raw):
        state_map[int(raw)] = label

    labeled_states = np.array([state_map[s] for s in raw_states], dtype=int)
    return labeled_states, state_map


def get_regime_labels(labeled_states: np.ndarray, K: int = 3) -> list[str]:
    """
    Convert integer labeled states to string labels.
    0 -> 'Bullish', K-1 -> 'Bearish', others -> 'Sideways' (or 'Sideways_1', 'Sideways_2' for K=4).
    """
    labels = []
    for s in labeled_states:
        if s == 0:
            labels.append('Bullish')
        elif s == K - 1:
            labels.append('Bearish')
        else:
            # Middle states: 'Sideways' for K=3, 'Sideways_1'/'Sideways_2' for K=4
            if K == 4:
                labels.append(f'Sideways_{s}')
            else:
                labels.append('Sideways')
    return labels


def compute_bic_for_k(
    features: np.ndarray,
    k_values: list[int],
    n_restarts: int = 5,
    random_state: int = 42
) -> dict[int, float]:
    """
    Fit HMMs for each K in k_values and return BIC scores.
    Uses hmmlearn model.bic(features) which implements -2*loglik + m_K * ln(T).
    Returns dict mapping K -> BIC value (lower is better).
    """
    bic_scores = {}
    for k in k_values:
        model = fit_hmm(
            features,
            n_components=k,
            n_restarts=n_restarts,
            random_state=random_state
        )
        if model is None:
            logging.warning(f'HMM fit returned None for K={k}; skipping BIC.')
            continue
        try:
            bic = model.bic(features)
            bic_scores[k] = bic
        except Exception as e:
            logging.warning(f'BIC computation failed for K={k}: {e}')
    return bic_scores


def rolling_hmm_decode(
    hmm_features: pd.DataFrame,
    monthly_returns: pd.DataFrame,
    window: int = 24,
    n_components: int = 3,
    n_iter: int = 500,
    n_restarts: int = 5,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Perform causal rolling HMM decoding for the full out-of-sample period.

    For each month t in the out-of-sample period:
    1. Take feature window of length 'window' ending at t
    2. Fit GaussianHMM(n_components) on this window
    3. Decode Viterbi states for the full window
    4. Label states by equity return feature ranking
    5. Current regime = last decoded state after re-labeling

    Returns DataFrame with columns:
    ['raw_state', 'labeled_state', 'regime_label', 'hmm_converged']
    indexed by the out-of-sample dates.

    hmm_features: DataFrame of feature vectors, must be long enough to contain training + OOS
    monthly_returns: full monthly returns DataFrame (same index as hmm_features)
    window: rolling estimation window in months (default 24)
    """
    all_dates = hmm_features.index
    oos_dates = all_dates[window:]  # first window months used for warm-up

    records = []
    for t_idx, date in enumerate(oos_dates):
        # Window indices [start, end) in the full feature array
        end_pos = window + t_idx  # exclusive upper bound (position of current date +1)
        start_pos = end_pos - window

        feat_window = hmm_features.iloc[start_pos:end_pos].values.astype(float)

        try:
            model = fit_hmm(
                feat_window,
                n_components=n_components,
                n_iter=n_iter,
                n_restarts=n_restarts,
                random_state=random_state
            )
            if model is None:
                raise RuntimeError('fit_hmm returned None')

            raw_states = decode_states(model, feat_window)
            labeled_states, _ = label_states(model, raw_states, equity_feature_idx=0)
            regime_strs = get_regime_labels(labeled_states, K=n_components)

            current_raw = int(raw_states[-1])
            current_labeled = int(labeled_states[-1])
            current_label = regime_strs[-1]
            converged = model.monitor_.converged

        except Exception as e:
            logging.warning(f'Rolling HMM decode failed at {date}: {e}')
            current_raw = -1
            current_labeled = -1
            current_label = 'Unknown'
            converged = False

        records.append({
            'raw_state': current_raw,
            'labeled_state': current_labeled,
            'regime_label': current_label,
            'hmm_converged': converged
        })

    result = pd.DataFrame(records, index=oos_dates)
    return result
