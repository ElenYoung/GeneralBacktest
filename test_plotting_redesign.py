
import pandas as pd
import numpy as np
import os
import sys
import inspect
import backtest
from backtest import GeneralBacktest

print(f"Backtest module file: {backtest.__file__}")
print(f"GeneralBacktest methods: {[m for m in dir(GeneralBacktest) if 'plot' in m]}")


def create_mock_data():
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='B')
    assets = ['A', 'B', 'C']
    
    # Mock Prices
    price_data = []
    for asset in assets:
        prices = 100 * (1 + np.random.randn(len(dates)) * 0.02).cumprod()
        df = pd.DataFrame({
            'date': dates,
            'code': asset,
            'open': prices,
            'close': prices,
            'adj_factor': 1.0
        })
        price_data.append(df)
    price_df = pd.concat(price_data)
    
    # Mock Weights
    rebalance_dates = dates[::20] # Rebalance every 20 days
    weights_data = []
    for date in rebalance_dates:
        w = np.random.random(len(assets))
        w = w / w.sum()
        for i, asset in enumerate(assets):
            weights_data.append({
                'date': date,
                'code': asset,
                'weight': w[i]
            })
    weights_df = pd.DataFrame(weights_data)
    
    # Mock Benchmark Weights (Equal Weight)
    bench_weights_data = []
    for date in rebalance_dates:
        for asset in assets:
            bench_weights_data.append({
                'date': date,
                'code': asset,
                'weight': 1.0 / len(assets)
            })
    bench_weights_df = pd.DataFrame(bench_weights_data)
    
    return weights_df, price_df, bench_weights_df

def test_plotting():
    print("Generating mock data...")
    weights_df, price_df, bench_weights_df = create_mock_data()
    
    bt = GeneralBacktest(start_date='2023-01-01', end_date='2023-12-31')
    
    print("Running backtest...")
    bt.run_backtest(
        weights_data=weights_df,
        price_data=price_df,
        buy_price='open',
        sell_price='close',
        adj_factor_col='adj_factor',
        close_price_col='close',
        benchmark_weights=bench_weights_df
    )
    
    output_dir = 'test_output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    print("Testing plot_nav_curve...")
    bt.plot_nav_curve(save_path=f'{output_dir}/nav_curve.png')
    
    print("Testing plot_drawdown...")
    bt.plot_drawdown() # Display only
    
    print("Testing plot_nav_vs_benchmark...")
    bt.plot_nav_vs_benchmark(save_path=f'{output_dir}/nav_vs_benchmark.png')
    
    print("Testing plot_metrics_table...")
    bt.plot_metrics_table(save_path=f'{output_dir}/metrics_table.png')
    
    print("Testing plot_monthly_returns_heatmap...")
    bt.plot_monthly_returns_heatmap(save_path=f'{output_dir}/heatmap.png')
    
    print("Testing plot_turnover_analysis...")
    bt.plot_turnover_analysis(save_path=f'{output_dir}/turnover.png')
    
    print("Testing plot_dashboard...")
    bt.plot_dashboard(save_path=f'{output_dir}/dashboard.png')
    
    print("Testing plot_all (alias)...")
    bt.plot_all(save_path=f'{output_dir}/dashboard_alias.png')
    
    print("All plotting tests finished.")

if __name__ == "__main__":
    test_plotting()
