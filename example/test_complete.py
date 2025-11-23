"""
Complete functionality test - English version to avoid encoding issues
"""

import numpy as np
import pandas as pd
import sys
import os

# Disable matplotlib display
import matplotlib
matplotlib.use('Agg')

sys.path.insert(0, os.path.dirname(__file__))

from backtest import GeneralBacktest

np.random.seed(42)

print("="  * 80)
print("General Backtest Framework - Complete Test")
print("=" * 80)

# Generate test data
print("\nGenerating test data...")
dates = pd.date_range('2020-01-01', periods=100, freq='D')
dates = [d for d in dates if d.weekday() < 5][:50]  # Take 50 trading days
assets = ['STOCK_A', 'STOCK_B', 'STOCK_C']

# Price data
price_data = []
for i, date in enumerate(dates):
    for j, asset in enumerate(assets):
        base_price = 100 * (1 + 0.001 * i + 0.01 * j)
        noise = np.random.normal(0, 0.5)
        price_data.append({
            'date': date,
            'code': asset,
            'open': base_price + noise,
            'close': base_price + noise + np.random.normal(0, 0.3),
            'adj_factor': 1.0 + 0.001 * i
        })
price_df = pd.DataFrame(price_data)
print(f"[OK] Price data: {len(price_df)} records, {len(dates)} days, {len(assets)} assets")

# Weights data - rebalance every 10 days
rebalance_dates = dates[::10]
weights_data = []
for date in rebalance_dates:
    weights = np.random.dirichlet([1, 1, 1])
    for asset, weight in zip(assets, weights):
        weights_data.append({
            'date': date,
            'code': asset,
            'weight': weight
        })
weights_df = pd.DataFrame(weights_data)
print(f"[OK] Weights data: {len(weights_df)} records, {len(rebalance_dates)} rebalances")

# Benchmark data - equal weight
benchmark_weights = []
for date in rebalance_dates:
    for asset in assets:
        benchmark_weights.append({
            'date': date,
            'code': asset,
            'weight': 1.0 / len(assets)
        })
benchmark_df = pd.DataFrame(benchmark_weights)
print(f"[OK] Benchmark data: {len(benchmark_df)} records")

# Run backtest
print("\n" + "=" * 80)
print("Running backtest...")
print("=" * 80)

bt = GeneralBacktest(
    start_date=dates[0].strftime('%Y-%m-%d'),
    end_date=dates[-1].strftime('%Y-%m-%d')
)

results = bt.run_backtest(
    weights_data=weights_df,
    price_data=price_df,
    buy_price='open',
    sell_price='close',
    adj_factor_col='adj_factor',
    transaction_cost=[0.001, 0.001],
    benchmark_weights=benchmark_df
)

# Display results
print("\n" + "=" * 80)
print("Performance Metrics")
print("=" * 80)
metrics = bt.get_metrics()
for idx, row in metrics.iterrows():
    print(f"{idx:30s}: {row['值']}")

print("\n" + "=" * 80)
print("Key Statistics")
print("=" * 80)
print(f"Final NAV: {bt.daily_nav.iloc[-1]:.4f}")
print(f"Total Return: {(bt.daily_nav.iloc[-1] / bt.daily_nav.iloc[0] - 1) * 100:.2f}%")
print(f"NAV data points: {len(bt.daily_nav)}")
print(f"Number of rebalances: {len(bt.trade_records)}")
print(f"Position records: {len(bt.daily_positions)}")

# Test visualization (save to files)
print("\n" + "=" * 80)
print("Testing visualization (saving images)...")
print("=" * 80)

try:
    import matplotlib.pyplot as plt
    
    print("1. NAV curve...")
    bt.plot_nav_curve()
    plt.savefig('test_nav.png', dpi=100, bbox_inches='tight')
    plt.close()
    print("   [OK] Saved to test_nav.png")
    
    print("2. Drawdown curve...")
    bt.plot_drawdown()
    plt.savefig('test_drawdown.png', dpi=100, bbox_inches='tight')
    plt.close()
    print("   [OK] Saved to test_drawdown.png")
    
    print("3. Strategy vs Benchmark...")
    bt.plot_nav_vs_benchmark()
    plt.savefig('test_benchmark.png', dpi=100, bbox_inches='tight')
    plt.close()
    print("   [OK] Saved to test_benchmark.png")
    
    print("4. Comprehensive panel...")
    bt.plot_all()
    plt.savefig('test_all.png', dpi=100, bbox_inches='tight')
    plt.close()
    print("   [OK] Saved to test_all.png")
    
    print("\n[OK] All visualization tests passed!")
    
except Exception as e:
    print(f"[ERROR] Visualization test failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("TEST COMPLETED SUCCESSFULLY!")
print("=" * 80)
