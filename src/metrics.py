import numpy as np
import pandas as pd
from scipy import stats

MONTHLY_RF = 0.00479       # 0.479% monthly risk-free rate
ANNUALIZATION_FACTOR = 12  # monthly to annual


def compute_cumulative_return(returns: pd.Series) -> float:
    """Cumulative return from monthly simple returns. Product of (1 + r_t) - 1."""
    return float(np.prod(1.0 + returns) - 1.0)


def compute_annualized_return(returns: pd.Series) -> float:
    """Annualized return: (1 + cumulative_return) ^ (12/T) - 1."""
    T = len(returns)
    cum_ret = compute_cumulative_return(returns)
    return float((1.0 + cum_ret) ** (12.0 / T) - 1.0)


def compute_annualized_volatility(returns: pd.Series) -> float:
    """Annualized volatility: std(returns) * sqrt(12)."""
    return float(returns.std() * np.sqrt(ANNUALIZATION_FACTOR))


def compute_sharpe_ratio(
    returns: pd.Series,
    rf_monthly: float = MONTHLY_RF
) -> float:
    """Sharpe ratio: (mean(r - rf)) / std(r - rf) * sqrt(12)."""
    excess = returns - rf_monthly
    std_exc = excess.std()
    if std_exc == 0:
        return 0.0
    return float(excess.mean() / std_exc * np.sqrt(ANNUALIZATION_FACTOR))


def compute_sortino_ratio(
    returns: pd.Series,
    rf_monthly: float = MONTHLY_RF
) -> float:
    """Sortino ratio: mean(r - rf) / downside_std(r) * sqrt(12). Downside std uses only negative excess returns."""
    excess = returns - rf_monthly
    downside = excess[excess < 0]
    if len(downside) == 0:
        return np.inf
    downside_std = downside.std()
    if downside_std == 0:
        return 0.0
    return float(excess.mean() / downside_std * np.sqrt(ANNUALIZATION_FACTOR))


def compute_max_drawdown(returns: pd.Series) -> float:
    """Maximum drawdown from peak. Uses cumulative wealth index."""
    wealth = (1.0 + returns).cumprod()
    rolling_peak = wealth.cummax()
    drawdown = (wealth - rolling_peak) / rolling_peak
    return float(drawdown.min())


def compute_calmar_ratio(returns: pd.Series) -> float:
    """Calmar ratio: annualized_return / abs(max_drawdown). Return 0 if MDD = 0."""
    ann_ret = compute_annualized_return(returns)
    mdd = compute_max_drawdown(returns)
    if mdd == 0:
        return 0.0
    return float(ann_ret / abs(mdd))


def compute_cvar(
    returns: pd.Series,
    confidence: float = 0.95
) -> float:
    """CVaR (Conditional Value at Risk) at given confidence level. Returns positive value (loss)."""
    var_threshold = returns.quantile(1.0 - confidence)
    tail_losses = returns[returns <= var_threshold]
    if len(tail_losses) == 0:
        return 0.0
    return float(-tail_losses.mean())


def compute_all_metrics(
    returns: pd.Series,
    turnover_series: pd.Series = None,
    n_rebalances: int = None,
    rf_monthly: float = MONTHLY_RF
) -> dict:
    """
    Compute all performance metrics for a return series.
    Returns dict with keys:
    CR, Ann_R, Ann_Vol, Sharpe, Sortino, MDD, Calmar, CVaR95,
    Avg_Turnover, N_Rebalances
    All values as floats (proportional, not percentages).
    """
    avg_turnover = float(turnover_series.mean()) if turnover_series is not None and len(turnover_series) > 0 else 0.0
    n_reb = n_rebalances if n_rebalances is not None else 0

    return {
        'CR': compute_cumulative_return(returns),
        'Ann_R': compute_annualized_return(returns),
        'Ann_Vol': compute_annualized_volatility(returns),
        'Sharpe': compute_sharpe_ratio(returns, rf_monthly=rf_monthly),
        'Sortino': compute_sortino_ratio(returns, rf_monthly=rf_monthly),
        'MDD': compute_max_drawdown(returns),
        'Calmar': compute_calmar_ratio(returns),
        'CVaR95': compute_cvar(returns, confidence=0.95),
        'Avg_Turnover': avg_turnover,
        'N_Rebalances': n_reb
    }


def jobson_korkie_test(
    returns_a: pd.Series,
    returns_b: pd.Series,
    rf_monthly: float = MONTHLY_RF
) -> dict:
    """
    Jobson-Korkie test for equality of Sharpe ratios.
    Standard implementation without Memmel correction.

    H0: SR_a = SR_b
    H1: SR_a != SR_b (two-sided)

    Test statistic: z = (mu_a * sigma_b - mu_b * sigma_a) / sqrt(theta)
    where theta = (1/T) * (2*sigma_a^2*sigma_b^2 - 2*sigma_a*sigma_b*sigma_ab +
                           0.5*mu_a^2*sigma_b^2 + 0.5*mu_b^2*sigma_a^2 -
                           mu_a*mu_b*sigma_ab)
    mu_a, mu_b = mean excess returns
    sigma_a, sigma_b = std of excess returns
    sigma_ab = cov(excess_a, excess_b)

    Returns dict: {'z_stat': float, 'p_value': float, 'reject_h0': bool, 'sr_a': float, 'sr_b': float}
    """
    excess_a = returns_a - rf_monthly
    excess_b = returns_b - rf_monthly

    T = len(excess_a)
    mu_a = float(excess_a.mean())
    mu_b = float(excess_b.mean())
    sigma_a = float(excess_a.std())
    sigma_b = float(excess_b.std())
    sigma_ab = float(np.cov(excess_a, excess_b)[0, 1])

    sr_a = mu_a / sigma_a * np.sqrt(ANNUALIZATION_FACTOR) if sigma_a > 0 else 0.0
    sr_b = mu_b / sigma_b * np.sqrt(ANNUALIZATION_FACTOR) if sigma_b > 0 else 0.0

    theta = (1.0 / T) * (
        2.0 * sigma_a ** 2 * sigma_b ** 2
        - 2.0 * sigma_a * sigma_b * sigma_ab
        + 0.5 * mu_a ** 2 * sigma_b ** 2
        + 0.5 * mu_b ** 2 * sigma_a ** 2
        - mu_a * mu_b * sigma_ab
    )

    if theta <= 0:
        z_stat = 0.0
        p_value = 1.0
    else:
        z_stat = float((mu_a * sigma_b - mu_b * sigma_a) / np.sqrt(theta))
        p_value = float(2.0 * (1.0 - stats.norm.cdf(abs(z_stat))))

    return {
        'z_stat': z_stat,
        'p_value': p_value,
        'reject_h0': p_value < 0.05,
        'sr_a': sr_a,
        'sr_b': sr_b
    }


def format_metrics_table(metrics_dict: dict[str, dict]) -> pd.DataFrame:
    """
    Format metrics from multiple strategies into a comparison table.
    strategies_metrics: dict mapping strategy name -> metrics dict
    Returns DataFrame with strategies as columns and metrics as rows.
    Values formatted as percentages where appropriate.
    """
    # Metrics that should be displayed as percentages
    pct_metrics = {'CR', 'Ann_R', 'Ann_Vol', 'MDD', 'CVaR95', 'Avg_Turnover'}

    rows = {}
    for strategy, mdict in metrics_dict.items():
        col = {}
        for key, val in mdict.items():
            if key in pct_metrics:
                col[key] = f'{val * 100:.2f}%'
            elif key == 'N_Rebalances':
                col[key] = str(int(val))
            else:
                col[key] = f'{val:.4f}'
        rows[strategy] = col

    df = pd.DataFrame(rows)
    return df
