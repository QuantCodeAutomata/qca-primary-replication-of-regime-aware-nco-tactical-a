import os, sys, logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.fetch import load_data
from src.backtest import regime_nco_backtest
from src.metrics import compute_all_metrics

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
RESULTS_DIR = Path(__file__).parent.parent / 'results'
RESULTS_DIR.mkdir(exist_ok=True)

# Rebalancing policy configurations to test
POLICIES = {
    'No-Rebalance':   {'policy': 'none',     'delta': None},
    'Monthly':        {'policy': 'monthly',   'delta': None},
    'Adaptive-0.02':  {'policy': 'adaptive',  'delta': 0.02},
    'Adaptive-0.05':  {'policy': 'adaptive',  'delta': 0.05},
    'Adaptive-0.10':  {'policy': 'adaptive',  'delta': 0.10},
}

BASE_KWARGS = dict(
    oos_start='2021-01-01',
    window=24,
    n_components=3,
    cost_bps=10.0,
    linkage_method='average',
    n_restarts=5,
    random_state=42,
)


def run_all_policies(monthly_returns, hmm_features):
    """Run regime-NCO backtest under each rebalancing policy."""
    simple_returns = np.exp(monthly_returns) - 1
    results = {}
    for label, cfg in POLICIES.items():
        logging.info(f'Running policy: {label}')
        kwargs = {**BASE_KWARGS, 'policy': cfg['policy']}
        if cfg['delta'] is not None:
            kwargs['delta'] = cfg['delta']
        results[label] = regime_nco_backtest(
            monthly_returns=simple_returns,
            hmm_features=hmm_features,
            **kwargs
        )
    return results


def build_summary_table(results):
    """Build a summary DataFrame with Sharpe, MDD, TO, N_Reb, Net_Ann_R."""
    rows = []
    for label, res in results.items():
        m = res['metrics']
        rows.append({
            'Policy': label,
            'Net_Ann_R%': round(m['Ann_R'] * 100, 4),
            'Sharpe': round(m['Sharpe'], 4),
            'MDD%': round(m['MDD'] * 100, 4),
            'CVaR95%': round(m['CVaR95'] * 100, 4),
            'N_Reb': res['n_rebalances'],
            'Total_TO%': round(res['turnover'].sum() * 100, 4),
            'Avg_TO%': round(res['turnover'].mean() * 100, 4),
        })
    return pd.DataFrame(rows).set_index('Policy')


def print_summary_table(table):
    """Print formatted summary table."""
    print('\n=== EXP3: Rebalancing Policy Comparison ===')
    print(table.to_string())


def save_plots(results, table):
    """Save comparison plots for rebalancing experiment."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Exp 3: Rebalancing Policy Analysis', fontsize=14, fontweight='bold')

    colors = ['navy', 'darkorange', 'green', 'firebrick', 'purple']

    # Plot 1: NAV curves by policy
    ax = axes[0, 0]
    for (label, res), color in zip(results.items(), colors):
        res['nav'].plot(ax=ax, label=label, color=color, linewidth=1.5)
    ax.set_title('Cumulative NAV by Rebalancing Policy')
    ax.set_ylabel('NAV')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Plot 2: Sharpe ratio by policy
    ax = axes[0, 1]
    table['Sharpe'].plot.bar(ax=ax, color='steelblue')
    ax.set_title('Sharpe Ratio by Policy')
    ax.set_ylabel('Sharpe')
    ax.grid(True, alpha=0.3)
    plt.setp(ax.get_xticklabels(), rotation=30, ha='right')

    # Plot 3: Total turnover by policy
    ax = axes[1, 0]
    table['Total_TO%'].plot.bar(ax=ax, color='darkorange')
    ax.set_title('Total Turnover (%) by Policy')
    ax.set_ylabel('Turnover %')
    ax.grid(True, alpha=0.3)
    plt.setp(ax.get_xticklabels(), rotation=30, ha='right')

    # Plot 4: Number of rebalances by policy
    ax = axes[1, 1]
    table['N_Reb'].plot.bar(ax=ax, color='seagreen')
    ax.set_title('Number of Rebalances by Policy')
    ax.set_ylabel('N_Rebalances')
    ax.grid(True, alpha=0.3)
    plt.setp(ax.get_xticklabels(), rotation=30, ha='right')

    plt.tight_layout()
    out_path = RESULTS_DIR / 'exp3_rebalancing.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Plot saved to {out_path}')


def save_results(results, table):
    """Save summary table and per-month diagnostics to CSV."""
    table.to_csv(RESULTS_DIR / 'exp3_rebalancing_table.csv')

    # Per-policy monthly diagnostics
    nav_df = pd.DataFrame({label: res['nav'] for label, res in results.items()})
    nav_df.to_csv(RESULTS_DIR / 'exp3_navs.csv')

    turnover_df = pd.DataFrame({label: res['turnover'] for label, res in results.items()})
    turnover_df.to_csv(RESULTS_DIR / 'exp3_turnover.csv')

    print('Results saved to results/ directory.')


def main():
    logging.info('Loading data...')
    daily_prices, monthly_returns, hmm_features = load_data()

    logging.info('Running rebalancing policy comparison...')
    results = run_all_policies(monthly_returns, hmm_features)

    table = build_summary_table(results)
    print_summary_table(table)
    save_plots(results, table)
    save_results(results, table)

    return results, table


if __name__ == '__main__':
    main()
