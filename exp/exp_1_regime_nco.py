import os, sys, logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.fetch import load_data
from src.hmm_regime import compute_bic_for_k, fit_hmm, decode_states, label_states, get_regime_labels
from src.nco import compute_nco_portfolio
from src.backtest import regime_nco_backtest
from src.metrics import compute_all_metrics, format_metrics_table

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
RESULTS_DIR = Path(__file__).parent.parent / 'results'
RESULTS_DIR.mkdir(exist_ok=True)


def run_bic_selection(monthly_returns, train_end='2020-12-31'):
    """Run BIC model selection on training data.

    Feature vectors are computed from the training-period monthly returns
    using ``_build_window_features`` (same method used inside the backtest),
    so no pre-computed ``hmm_features`` alignment is required.
    """
    from src.backtest import _build_window_features
    from src.hmm_regime import compute_bic_for_k

    train_mask = monthly_returns.index <= train_end
    train_returns = monthly_returns[train_mask].values.astype(float)
    if len(train_returns) < 4:
        raise ValueError('Training period too short for BIC selection.')

    train_features = _build_window_features(train_returns, monthly_returns.columns.tolist())
    bic_dict = compute_bic_for_k(train_features, k_values=[2, 3, 4])
    print('\nBIC Model Selection (Training Period 2019-2020):')
    for k, bic in bic_dict.items():
        print(f'  K={k}: BIC={bic:.4f}')
    best_k = min(bic_dict, key=bic_dict.get)
    print(f'  Best K by BIC: {best_k} (paper uses K=3 for main backtest)')
    return bic_dict


def run_main_backtest(monthly_returns, hmm_features):
    """Run regime-aware NCO backtest."""
    # Convert log returns to simple returns for NAV compounding
    simple_returns = np.exp(monthly_returns) - 1
    result = regime_nco_backtest(
        monthly_returns=simple_returns,
        hmm_features=hmm_features,
        oos_start='2021-01-01',
        window=24,
        n_components=3,
        cost_bps=10.0,
        policy='adaptive',
        delta=0.05,
        linkage_method='average',
        n_restarts=5,
        random_state=42
    )
    return result


def print_results(result):
    """Print summary of backtest results."""
    metrics = result['metrics']
    print('\n=== EXP1: Regime-Aware NCO Results ===')
    print(f'Cumulative Return:     {metrics["CR"]*100:.2f}%')
    print(f'Annualized Return:     {metrics["Ann_R"]*100:.2f}%')
    print(f'Annualized Volatility: {metrics["Ann_Vol"]*100:.2f}%')
    print(f'Sharpe Ratio:          {metrics["Sharpe"]:.4f}')
    print(f'Sortino Ratio:         {metrics["Sortino"]:.4f}')
    print(f'Max Drawdown:          {metrics["MDD"]*100:.2f}%')
    print(f'Calmar Ratio:          {metrics["Calmar"]:.4f}')
    print(f'CVaR 95%:              {metrics["CVaR95"]*100:.2f}%')
    print(f'Num Rebalances:        {result["n_rebalances"]}')
    print(f'Total Turnover:        {result["turnover"].sum()*100:.2f}%')

    print('\nRegime Counts:')
    for regime, count in result['regime_counts'].items():
        print(f'  {regime}: {count}')

    if result.get('avg_weights_by_regime') is not None:
        print('\nAverage Weights by Regime:')
        print(result['avg_weights_by_regime'].to_string())


def save_plots(result, bic_dict):
    """Save all plots for exp_1."""
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    fig.suptitle('Exp 1: Regime-Aware NCO TAA', fontsize=14, fontweight='bold')

    # Plot 1: NAV series
    ax = axes[0, 0]
    result['nav'].plot(ax=ax, color='navy', linewidth=1.5)
    ax.set_title('Portfolio NAV')
    ax.set_ylabel('NAV')
    ax.grid(True, alpha=0.3)

    # Plot 2: Regime labels over time
    ax = axes[0, 1]
    regime_numeric = result['regimes'].map({'Bullish': 1, 'Sideways': 0, 'Bearish': -1}).fillna(0)
    colors = {'Bullish': 'green', 'Sideways': 'orange', 'Bearish': 'red'}
    regime_colors = result['regimes'].map(colors).fillna('gray')
    ax.bar(result['regimes'].index, regime_numeric, color=regime_colors.values, width=20)
    ax.set_title('Regime Labels')
    ax.set_yticks([-1, 0, 1])
    ax.set_yticklabels(['Bearish', 'Sideways', 'Bullish'])
    ax.grid(True, alpha=0.3)

    # Plot 3: Target weights
    ax = axes[1, 0]
    if result.get('target_weights') is not None:
        result['target_weights'].plot(ax=ax, linewidth=1)
        ax.set_title('Target Portfolio Weights')
        ax.set_ylabel('Weight')
        ax.legend(loc='upper right', fontsize=7)
        ax.grid(True, alpha=0.3)

    # Plot 4: Monthly returns
    ax = axes[1, 1]
    result['returns'].plot.bar(ax=ax, color=['red' if r < 0 else 'green' for r in result['returns']])
    ax.set_title('Monthly Returns')
    ax.set_ylabel('Return')
    ax.grid(True, alpha=0.3)

    # Plot 5: BIC values
    ax = axes[2, 0]
    k_vals = list(bic_dict.keys())
    bic_vals = list(bic_dict.values())
    ax.bar(k_vals, bic_vals, color='steelblue')
    ax.set_title('BIC Model Selection')
    ax.set_xlabel('K (Number of Regimes)')
    ax.set_ylabel('BIC')
    ax.grid(True, alpha=0.3)

    # Plot 6: Average weights by regime
    ax = axes[2, 1]
    if result.get('avg_weights_by_regime') is not None:
        result['avg_weights_by_regime'].plot(kind='bar', ax=ax)
        ax.set_title('Average Weights by Regime')
        ax.set_ylabel('Weight')
        ax.set_xlabel('Regime')
        ax.legend(loc='upper right', fontsize=7)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'exp1_regime_nco.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Plot saved to {RESULTS_DIR}/exp1_regime_nco.png')


def save_results_csv(result, bic_dict):
    """Save detailed results to CSV."""
    # Monthly diagnostics table
    diag = pd.DataFrame({
        'nav': result['nav'],
        'returns': result['returns'],
        'regime': result['regimes'],
        'rebalance': result['rebalance_flag'],
        'trigger': result['trigger'],
        'turnover': result['turnover'],
        'costs': result['costs'],
    })
    if result.get('target_weights') is not None:
        for col in result['target_weights'].columns:
            diag[f'target_{col}'] = result['target_weights'][col]
    diag.to_csv(RESULTS_DIR / 'exp1_monthly_diagnostics.csv')

    # Metrics
    metrics_df = pd.DataFrame([result['metrics']], index=['Regime-NCO'])
    metrics_df.to_csv(RESULTS_DIR / 'exp1_metrics.csv')

    # BIC
    bic_df = pd.DataFrame({'K': list(bic_dict.keys()), 'BIC': list(bic_dict.values())})
    bic_df.to_csv(RESULTS_DIR / 'exp1_bic.csv', index=False)
    print('Results saved to results/ directory.')


def main():
    logging.info('Loading data...')
    daily_prices, monthly_returns, hmm_features = load_data()

    logging.info('Running BIC model selection...')
    bic_dict = run_bic_selection(monthly_returns)

    logging.info('Running main backtest (Regime-Aware NCO, K=3)...')
    result = run_main_backtest(monthly_returns, hmm_features)

    print_results(result)
    save_plots(result, bic_dict)
    save_results_csv(result, bic_dict)

    return result, bic_dict


if __name__ == '__main__':
    main()
