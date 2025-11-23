"""
简单测试脚本 - 验证框架基本功能
"""

import numpy as np
import pandas as pd
import sys
import os

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(__file__))

print("测试 1: 导入模块...")
try:
    from backtest import GeneralBacktest
    print("✓ 成功导入 GeneralBacktest")
except Exception as e:
    print(f"✗ 导入失败: {e}")
    sys.exit(1)

print("\n测试 2: 生成简单测试数据...")
# 生成简单的测试数据
dates = pd.date_range('2020-01-01', periods=10, freq='D')
assets = ['A', 'B']

# 价格数据
price_data = []
for date in dates:
    for asset in assets:
        price_data.append({
            'date': date,
            'code': asset,
            'open': 100.0,
            'close': 100.0,
            'adj_factor': 1.0
        })
price_df = pd.DataFrame(price_data)
print(f"✓ 生成了 {len(price_df)} 条价格记录")

# 权重数据
weights_df = pd.DataFrame([
    {'date': dates[0], 'code': 'A', 'weight': 0.5},
    {'date': dates[0], 'code': 'B', 'weight': 0.5},
    {'date': dates[5], 'code': 'A', 'weight': 0.3},
    {'date': dates[5], 'code': 'B', 'weight': 0.7},
])
print(f"✓ 生成了 {len(weights_df)} 条权重记录")

print("\n测试 3: 初始化回测框架...")
try:
    bt = GeneralBacktest(start_date='2020-01-01', end_date='2020-01-10')
    print("✓ 回测框架初始化成功")
except Exception as e:
    print(f"✗ 初始化失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n测试 4: 运行回测...")
try:
    results = bt.run_backtest(
        weights_data=weights_df,
        price_data=price_df,
        buy_price='open',
        sell_price='close',
        adj_factor_col='adj_factor',
        transaction_cost=[0.001, 0.001]
    )
    print("✓ 回测运行成功")
    print(f"  - 生成净值点数: {len(bt.daily_nav)}")
    print(f"  - 最终净值: {bt.daily_nav.iloc[-1]:.4f}")
except Exception as e:
    print(f"✗ 回测失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n测试 5: 获取性能指标...")
try:
    metrics = bt.get_metrics()
    print("✓ 性能指标计算成功")
    print(f"  - 指标数量: {len(metrics)}")
    print("\n前5个指标:")
    print(metrics.head())
except Exception as e:
    print(f"✗ 获取指标失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("所有测试通过！✓")
print("=" * 60)
