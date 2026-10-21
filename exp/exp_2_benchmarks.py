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
from src.benchmarks import (equal_weight_backtest, buy_and_hold_backtest,
                              min_variance_backtest, standard_nco_backtest)
from src.metrics import compute_all_metrics, jobson_korkie_test, format_metrics_table

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
RESULTS_DIR = Path(__file__).parent.parent / 'results'
RESULTS_DIR.mkdir(exist_ok=True)


def run_all_strategies(monthly_returns, hmm_features):
    """Run all strategies and return results dict."""
    simple_returns = np.exp(monthly_returns) - 1
    results = {}

    logging.info('Running Regime-NCO...')
    results['Regime-NCO'] = regime_nco_backtest(
        monthly_returns=simple_returns,
        hmm_features=hmm_features,
        oos_start='2021-01-01',
        window=24, n_components=3, cost_bps=10.0, policy='adaptive',
        delta=0.05, linkage_method='average', n_restarts=5, random_state=42
    )

    logging.info('Running Equal Weight...')
    results['Equal-Weight'] = equal_weight_backtest(simple_returns, oos_start='2021-01-01', cost_bps=10.0)

    logging.info('Running Buy-and-Hold...')
    results['Buy-and-Hold'] = buy_and_hold_backtest(simple_returns, oos_start='2021-01-01', cost_bps=10.0)

    logging.info('Running Min Variance...')
    results['Min-Variance'] = min_variance_backtest(simple_returns, oos_start='2021-01-01', window=24, cost_bps=10.0)

    logging.info('Running Standard NCO (No Regime)...')
    results['NCO-No-Regime'] = standard_nco_backtest(simple_returns, oos_start='2021-01-01', window=24, cost_bps=10.0)

    # HRP = NCO-No-Regime as per paper (same in 5-asset universe)
    results['HRP'] = results['NCO-No-Regime'].copy()

    return results


def run_jobson_korkie(results):
    """Run Jobson-Korkie tests comparing Regime-NCO vs each benchmark."""
    nco_returns = results['Regime-NCO']['returns']
    jk_results = {}
    for name, res in results.items():
        if name == 'Regime-NCO':
            continue
        jk = jobson_korkie_test(nco_returns, res['returns'])
        jk_results[name] = jk
    return jk_results


def print_comparison_table(results, jk_results):
    """Print comparison metrics table."""
    print('\n=== EXP2: Strategy Comparison ===')
    rows = []
    for name, res in results.items():
        m = res['metrics']
        row = {
            'Strategy': name,
            'CR%': f"{m['CR']*100:.2f}",
            'Ann_R%': f"{m['Ann_R']*100:.2f}",
            'Vol%': f"{m['Ann_Vol']*100:.2f}",
            'Sharpe': f"{m['Sharpe']:.4f}",
            'Sortino': f"{m['Sortino']:.4f}",
            'MDD%': f"{m['MDD']*100:.2f}",
            'Calmar': f"{m['Calmar']:.4f}",
            'CVaR95%': f"{m['CVaR95']*100:.2f}",
            'N_Reb': res['n_rebalances'],
        }
        if name in jk_results:
            jk = jk_results[name]
            row['JK_p'] = f"{jk['p_value']:.4f}"
            row['JK_sig'] = '*' if jk['reject_h0'] else ''
        rows.append(row)
    df = pd.DataFrame(rows).set_index('Strategy')
    print(df.to_string())
    return df


def save_plots(results):
    """Save comparison plots."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Exp 2: Strategy Comparison', fontsize=14, fontweight='bold')

    colors = ['navy', 'orange', 'red', 'green', 'purple', 'brown']

    # NAV comparison
    ax = axes[0, 0]
    for (name, res), color in zip(results.items(), colors):
        res['nav'].plot(ax=ax, label=name, color=color, linewidth=1.5)
    ax.set_title('Cumulative NAV')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Metrics bar chart - Sharpe
    ax = axes[0, 1]
    sharpes = {name: res['metrics']['Sharpe'] for name, res in results.items()}
    pd.Series(sharpes).plot.bar(ax=ax, color='steelblue')
    ax.set_title('Sharpe Ratios')
    ax.set_ylabel('Sharpe')
    ax.grid(True, alpha=0.3)
    plt.setp(ax.get_xticklabels(), rotation=30, ha='right')

    # MDD comparison
    ax = axes[1, 0]
    mdds = {name: res['metrics']['MDD']*100 for name, res in results.items()}
    pd.Series(mdds).plot.bar(ax=ax, color='firebrick')
    ax.set_title('Maximum Drawdown (%)')
    ax.set_ylabel('MDD%')
    ax.grid(True, alpha=0.3)
    plt.setp(ax.get_xticklabels(), rotation=30, ha='right')

    # Ann Return comparison
    ax = axes[1, 1]
    ann_r = {name: res['metrics']['Ann_R']*100 for name, res in results.items()}
    pd.Series(ann_r).plot.bar(ax=ax, color='darkgreen')
    ax.set_title('Annualized Return (%)')
    ax.set_ylabel('Ann_R%')
    ax.grid(True, alpha=0.3)
    plt.setp(ax.get_xticklabels(), rotation=30, ha='right')

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'exp2_benchmarks.png', dpi=150, bbox_inches='tight')
    plt.close()


def main():
    logging.info('Loading data...')
    daily_prices, monthly_returns, hmm_features = load_data()
    simple_returns = np.exp(monthly_returns) - 1

    results = run_all_strategies(monthly_returns, hmm_features)
    jk_results = run_jobson_korkie(results)

    table = print_comparison_table(results, jk_results)
    save_plots(results)

    print('\nJobson-Korkie Test Results (Regime-NCO vs benchmarks):')
    for name, jk in jk_results.items():
        print(f'  vs {name}: z={jk["z_stat"]:.4f}, p={jk["p_value"]:.4f}, sig={jk["reject_h0"]}')

    # Save
    table.to_csv(RESULTS_DIR / 'exp2_comparison_table.csv')
    jk_df = pd.DataFrame(jk_results).T
    jk_df.to_csv(RESULTS_DIR / 'exp2_jobson_korkie.csv')

    # Save NAVs
    navs = pd.DataFrame({name: res['nav'] for name, res in results.items()})
    navs.to_csv(RESULTS_DIR / 'exp2_navs.csv')
    print('Results saved.')
    return results, jk_results


if __name__ == '__main__':
    main()
