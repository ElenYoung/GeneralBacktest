"""
通用回测框架使用示例
演示如何使用 GeneralBacktest 进行策略回测
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from backtest import GeneralBacktest

# 设置随机种子以保证可复现性
np.random.seed(42)


def generate_sample_data(start_date='2020-01-01', end_date='2023-12-31', n_assets=5):
    """
    生成模拟数据用于测试
    
    Parameters:
    -----------
    start_date : str
        开始日期
    end_date : str
        结束日期
    n_assets : int
        资产数量
        
    Returns:
    --------
    tuple
        (weights_data, price_data)
    """
    # 生成交易日序列（去除周末）
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    trade_dates = [d for d in date_range if d.weekday() < 5]  # 去除周末
    
    # 资产列表
    assets = [f'ASSET_{i:02d}' for i in range(n_assets)]
    
    # ===== 生成价格数据 =====
    print("生成价格数据...")
    price_records = []
    
    # 为每个资产生成价格序列
    for asset in assets:
        # 初始价格
        price = 100.0
        adj_factor = 1.0
        
        for date in trade_dates:
            # 模拟价格波动（几何布朗运动）
            daily_return = np.random.normal(0.0005, 0.015)  # 日均收益0.05%, 波动1.5%
            price = price * (1 + daily_return)
            
            # 模拟开盘价、收盘价、最高价、最低价
            open_price = price * (1 + np.random.uniform(-0.01, 0.01))
            close_price = price * (1 + np.random.uniform(-0.01, 0.01))
            high_price = max(open_price, close_price) * (1 + abs(np.random.uniform(0, 0.01)))
            low_price = min(open_price, close_price) * (1 - abs(np.random.uniform(0, 0.01)))
            
            # 复权因子（假设偶尔有股息）
            if np.random.random() < 0.01:  # 1%的概率分红
                adj_factor *= 1.02  # 分红导致复权因子增加
            
            price_records.append({
                'date': date,
                'code': asset,
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'adj_factor': adj_factor
            })
    
    price_data = pd.DataFrame(price_records)
    
    # ===== 生成权重数据（策略调仓信号）=====
    print("生成权重数据...")
    
    # 调仓频率：每月调仓一次
    rebalance_dates = pd.date_range(start=start_date, end=end_date, freq='MS')
    rebalance_dates = [d for d in rebalance_dates if d in trade_dates]
    
    weights_records = []
    
    for date in rebalance_dates:
        # 生成随机权重（模拟策略输出）
        raw_weights = np.random.dirichlet(np.ones(n_assets), size=1)[0]
        
        # 可选：设置一些权重为0（不是所有资产都持有）
        mask = np.random.random(n_assets) > 0.3  # 70%概率持有
        weights = raw_weights * mask
        
        # 归一化
        if weights.sum() > 0:
            weights = weights / weights.sum()
        
        for i, asset in enumerate(assets):
            if weights[i] > 0:
                weights_records.append({
                    'date': date,
                    'code': asset,
                    'weight': weights[i]
                })
    
    weights_data = pd.DataFrame(weights_records)
    
    print(f"生成了 {len(price_data)} 条价格记录")
    print(f"生成了 {len(weights_data)} 条权重记录")
    print(f"调仓次数: {len(rebalance_dates)}")
    
    return weights_data, price_data


def generate_benchmark_weights(price_data, weight_type='equal'):
    """
    生成基准权重（如等权基准）
    
    Parameters:
    -----------
    price_data : pd.DataFrame
        价格数据
    weight_type : str
        'equal' - 等权, 'market_cap' - 市值权重
        
    Returns:
    --------
    pd.DataFrame
        基准权重
    """
    # 获取所有资产和日期
    assets = price_data['code'].unique()
    dates = sorted(price_data['date'].unique())
    
    # 每月调仓
    rebalance_dates = pd.date_range(start=dates[0], end=dates[-1], freq='MS')
    rebalance_dates = [d for d in rebalance_dates if d in dates]
    
    benchmark_records = []
    
    for date in rebalance_dates:
        if weight_type == 'equal':
            # 等权
            weight = 1.0 / len(assets)
            for asset in assets:
                benchmark_records.append({
                    'date': date,
                    'code': asset,
                    'weight': weight
                })
        elif weight_type == 'market_cap':
            # 简化的市值权重（用价格代替）
            day_prices = price_data[price_data['date'] == date]
            total_price = day_prices['close'].sum()
            
            for _, row in day_prices.iterrows():
                benchmark_records.append({
                    'date': date,
                    'code': row['code'],
                    'weight': row['close'] / total_price
                })
    
    return pd.DataFrame(benchmark_records)


def main():
    """
    主函数：运行回测示例
    """
    print("=" * 80)
    print("通用量化回测框架 - 使用示例")
    print("=" * 80)
    
    # ===== 1. 生成模拟数据 =====
    print("\n【步骤 1】生成模拟数据")
    print("-" * 80)
    
    weights_data, price_data = generate_sample_data(
        start_date='2020-01-01',
        end_date='2023-12-31',
        n_assets=5
    )
    
    # 生成基准
    benchmark_weights = generate_benchmark_weights(price_data, weight_type='equal')
    
    # ===== 2. 初始化回测框架 =====
    print("\n【步骤 2】初始化回测框架")
    print("-" * 80)
    
    backtest = GeneralBacktest(
        start_date='2020-01-01',
        end_date='2023-12-31'
    )
    
    # ===== 3. 运行回测 =====
    print("\n【步骤 3】运行回测")
    print("-" * 80)
    
    results = backtest.run_backtest(
        weights_data=weights_data,
        price_data=price_data,
        buy_price='open',           # 使用开盘价买入
        sell_price='close',         # 使用收盘价卖出/计价
        adj_factor_col='adj_factor',
        date_col='date',
        asset_col='code',
        weight_col='weight',
        rebalance_threshold=0.01,
        transaction_cost=[0.001, 0.001],  # 买入和卖出各0.1%手续费
        initial_capital=1.0,
        benchmark_weights=benchmark_weights
    )
    
    # ===== 4. 查看性能指标 =====
    print("\n【步骤 4】查看性能指标")
    print("-" * 80)

    
    metrics_df = backtest.get_metrics()
    print(metrics_df)
    
    # ===== 5. 查看交易分析 =====
    print("\n【步骤 5】查看交易分析（前10次调仓）")
    print("-" * 80)
    
    trade_analysis = backtest.get_trade_analysis()
    print(trade_analysis.head(10))
    
    # ===== 6. 可视化 =====
    print("\n【步骤 6】生成可视化图表")
    print("-" * 80)
    
    print("绘制图表...")
    
    # 6.1 净值曲线
    backtest.plot_nav_curve()
    
    # 6.2 回撤曲线
    backtest.plot_drawdown()
    
    # 6.3 策略 vs 基准
    backtest.plot_nav_vs_benchmark()
    
    # 6.4 超额收益
    backtest.plot_excess_returns()
    
    # 6.5 交易点位分析
    backtest.plot_trade_points()
    
    # 6.6 持仓热力图
    backtest.plot_position_heatmap()
    
    # 6.7 换手率分析
    backtest.plot_turnover_analysis()
    
    # 6.8 综合展示
    backtest.plot_all()
    
    print("\n" + "=" * 80)
    print("回测示例完成！")
    print("=" * 80)


if __name__ == "__main__":
    main()
