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

# Baseline configuration (matches exp_1 primary backtest)
BASELINE = dict(
    oos_start='2021-01-01',
    n_components=3,
    cost_bps=10.0,
    policy='adaptive',
    delta=0.05,
    linkage_method='average',
    window=24,
    n_restarts=5,
    random_state=42,
)

# Subtest definitions: (subtest_label, sweep_param, sweep_values)
SUBTESTS = [
    ('A_Window',   'window',         [12, 18, 24, 36]),
    ('B_Linkage',  'linkage_method', ['average', 'complete', 'single']),
    ('C_Delta',    'delta',          [0.02, 0.05, 0.10]),
    ('D_Cost',     'cost_bps',       [0, 5, 10, 20]),
]


def run_subtest(simple_returns, hmm_features, subtest_label, sweep_param, sweep_values):
    """Run a single subtest sweeping one parameter over its values."""
    rows = []
    nav_series = {}
    for val in sweep_values:
        cfg = {**BASELINE, sweep_param: val}
        label = f'{sweep_param}={val}'
        logging.info(f'  [{subtest_label}] {label}')
        res = regime_nco_backtest(
            monthly_returns=simple_returns,
            hmm_features=hmm_features,
            **cfg
        )
        m = res['metrics']
        rows.append({
            'Subtest': subtest_label,
            'Parameter': sweep_param,
            'Value': str(val),
            'Ann_R%': round(m['Ann_R'] * 100, 4),
            'Ann_Vol%': round(m['Ann_Vol'] * 100, 4),
            'Sharpe': round(m['Sharpe'], 4),
            'Sortino': round(m['Sortino'], 4),
            'MDD%': round(m['MDD'] * 100, 4),
            'Calmar': round(m['Calmar'], 4),
            'CVaR95%': round(m['CVaR95'] * 100, 4),
            'N_Reb': res['n_rebalances'],
            'Total_TO%': round(res['turnover'].sum() * 100, 4),
        })
        nav_series[label] = res['nav']
    return pd.DataFrame(rows), nav_series


def run_all_subtests(monthly_returns, hmm_features):
    """Run all four robustness subtests and collect results."""
    simple_returns = np.exp(monthly_returns) - 1
    all_rows = []
    all_navs = {}  # keyed by subtest_label -> {label -> nav series}

    for subtest_label, sweep_param, sweep_values in SUBTESTS:
        logging.info(f'Running subtest {subtest_label}: sweep {sweep_param}')
        df, navs = run_subtest(
            simple_returns, hmm_features, subtest_label, sweep_param, sweep_values
        )
        all_rows.append(df)
        all_navs[subtest_label] = navs

    full_table = pd.concat(all_rows, ignore_index=True)
    return full_table, all_navs


def print_summary_table(table):
    """Print the full robustness results table."""
    print('\n=== EXP4: Robustness Analysis ===')
    for subtest in table['Subtest'].unique():
        sub = table[table['Subtest'] == subtest][
            ['Parameter', 'Value', 'Ann_R%', 'Sharpe', 'MDD%', 'Calmar', 'N_Reb', 'Total_TO%']
        ].copy()
        sub = sub.set_index('Value')
        print(f'\n-- Subtest {subtest} (sweep: {sub["Parameter"].iloc[0]}) --')
        print(sub.drop(columns='Parameter').to_string())


def save_plots(all_navs, table):
    """Save 4-panel plot: one panel per subtest showing NAV curves."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Exp 4: Robustness Analysis', fontsize=14, fontweight='bold')
    axes_flat = axes.flatten()

    colors_cycle = ['navy', 'darkorange', 'green', 'firebrick', 'purple', 'brown']

    for idx, (subtest_label, sweep_param, sweep_values) in enumerate(SUBTESTS):
        ax = axes_flat[idx]
        navs = all_navs[subtest_label]
        for (label, nav), color in zip(navs.items(), colors_cycle):
            nav.plot(ax=ax, label=label, color=color, linewidth=1.5)
        ax.set_title(f'Subtest {subtest_label[-1]}: sweep {sweep_param}')
        ax.set_ylabel('NAV')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = RESULTS_DIR / 'exp4_robustness.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Plot saved to {out_path}')

    # Additional Sharpe sensitivity bar charts (one figure per subtest)
    fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10))
    fig2.suptitle('Exp 4: Sharpe Sensitivity', fontsize=14, fontweight='bold')
    axes2_flat = axes2.flatten()

    for idx, (subtest_label, sweep_param, _) in enumerate(SUBTESTS):
        ax = axes2_flat[idx]
        sub = table[table['Subtest'] == subtest_label].copy()
        sub = sub.set_index('Value')
        sub['Sharpe'].plot.bar(ax=ax, color='steelblue')
        ax.set_title(f'Sharpe vs {sweep_param}')
        ax.set_ylabel('Sharpe')
        ax.grid(True, alpha=0.3)
        plt.setp(ax.get_xticklabels(), rotation=30, ha='right')

    plt.tight_layout()
    out_path2 = RESULTS_DIR / 'exp4_robustness_sharpe.png'
    plt.savefig(out_path2, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Sharpe sensitivity plot saved to {out_path2}')


def save_results(table, all_navs):
    """Save results to CSV."""
    table.to_csv(RESULTS_DIR / 'exp4_robustness_table.csv', index=False)

    # Save NAV series for each subtest
    for subtest_label, navs in all_navs.items():
        nav_df = pd.DataFrame(navs)
        nav_df.to_csv(RESULTS_DIR / f'exp4_navs_{subtest_label}.csv')

    print('Results saved to results/ directory.')


def main():
    logging.info('Loading data...')
    daily_prices, monthly_returns, hmm_features = load_data()

    logging.info('Running robustness analysis (4 subtests)...')
    table, all_navs = run_all_subtests(monthly_returns, hmm_features)

    print_summary_table(table)
    save_plots(all_navs, table)
    save_results(table, all_navs)

    return table, all_navs


if __name__ == '__main__':
    main()
