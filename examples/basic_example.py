"""
基础示例：使用随机生成的数据进行回测

这个示例展示了如何使用 GeneralBacktest 框架进行基本的回测。
我们将创建：
- 5 只模拟股票
- 2 年的日度数据
- 季度调仓策略
"""

import numpy as np
import pandas as pd
import sys
import os

# 添加父目录到路径以导入 GeneralBacktest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.GeneralBacktest import GeneralBacktest


def generate_random_stock_data(n_stocks=5, n_days=500, seed=42):
    """
    生成随机股票价格数据
    
    Parameters:
    -----------
    n_stocks : int
        股票数量
    n_days : int
        交易日数量
    seed : int
        随机种子
    
    Returns:
    --------
    pd.DataFrame
        包含价格数据的 DataFrame
    """
    np.random.seed(seed)
    
    # 生成日期序列
    dates = pd.date_range(start='2023-01-01', periods=n_days, freq='B')
    
    # 股票代码
    stock_codes = [f'STOCK_{chr(65+i)}' for i in range(n_stocks)]
    
    data = []
    
    for code in stock_codes:
        # 生成随机价格走势（几何布朗运动）
        initial_price = np.random.uniform(10, 100)
        returns = np.random.normal(0.0005, 0.02, n_days)  # 日均收益率 0.05%，波动率 2%
        
        # 计算累积价格
        prices = initial_price * np.exp(np.cumsum(returns))
        
        # 生成 OHLC 数据
        for i, date in enumerate(dates):
            close_price = prices[i]
            high_price = close_price * np.random.uniform(1.0, 1.02)
            low_price = close_price * np.random.uniform(0.98, 1.0)
            open_price = np.random.uniform(low_price, high_price)
            
            data.append({
                'date': date,
                'code': code,
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'adj_factor': 1.0  # 简化：不考虑除权除息
            })
    
    return pd.DataFrame(data)


def generate_quarterly_weights(dates, stock_codes, seed=42):
    """
    生成季度调仓权重
    
    Parameters:
    -----------
    dates : DatetimeIndex
        日期序列
    stock_codes : list
        股票代码列表
    seed : int
        随机种子
    
    Returns:
    --------
    pd.DataFrame
        权重数据
    """
    np.random.seed(seed)
    
    # 选择季度调仓日（每季度第一个交易日）
    # 使用实际交易日期，每季度选择第一个可用的交易日
    dates_sorted = pd.to_datetime(sorted(dates))
    
    # 按季度分组，选择每个季度的第一个交易日
    dates_df = pd.DataFrame({'date': dates_sorted})
    dates_df['quarter'] = dates_df['date'].dt.to_period('Q')
    rebalance_dates = dates_df.groupby('quarter')['date'].first().tolist()
    
    weights_data = []
    
    for date in rebalance_dates:
        # 生成随机权重并归一化
        raw_weights = np.random.dirichlet(np.ones(len(stock_codes)))
        
        for code, weight in zip(stock_codes, raw_weights):
            weights_data.append({
                'date': date,
                'code': code,
                'weight': weight
            })
    
    return pd.DataFrame(weights_data)


def main():
    print("=" * 60)
    print("GeneralBacktest 基础示例")
    print("=" * 60)
    
    # 1. 生成数据
    print("\n[1/4] 生成随机数据...")
    price_data = generate_random_stock_data(n_stocks=5, n_days=500, seed=42)
    print(f"  - 生成了 {len(price_data['code'].unique())} 只股票")
    print(f"  - 共 {len(price_data['date'].unique())} 个交易日")
    print(f"  - 日期范围：{price_data['date'].min()} 至 {price_data['date'].max()}")
    
    # 2. 生成权重
    print("\n[2/4] 生成季度调仓权重...")
    dates = price_data['date'].unique()
    stock_codes = price_data['code'].unique()
    weights_data = generate_quarterly_weights(dates, stock_codes, seed=42)
    print(f"  - 调仓次数：{len(weights_data['date'].unique())}")
    
    # 3. 运行回测
    print("\n[3/4] 运行回测...")
    bt = GeneralBacktest(
        start_date=price_data['date'].min().strftime('%Y-%m-%d'),
        end_date=price_data['date'].max().strftime('%Y-%m-%d')
    )
    
    results = bt.run_backtest(
        weights_data=weights_data,
        price_data=price_data,
        buy_price='open',
        sell_price='close',
        adj_factor_col='adj_factor',
        close_price_col='close',
        rebalance_threshold=0.005,  # 0.5% 调仓阈值
        transaction_cost=[0.001, 0.001],  # 买卖各 0.1% 手续费
        slippage=0.0005,  # 0.05% 滑点
        initial_capital=1.0
    )
    
    # 4. 展示结果
    print("\n[4/4] 展示回测结果...")
    print("\n" + "=" * 60)
    print("性能指标")
    print("=" * 60)
    bt.print_metrics()
    
    # 5. 生成图表
    print("\n生成可视化图表...")
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'output_demo')
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存图表
    bt.plot_nav_curve(save_path=os.path.join(output_dir, 'basic_nav_curve.png'))
    bt.plot_monthly_returns_heatmap(save_path=os.path.join(output_dir, 'basic_monthly_returns.png'))
    bt.plot_position_heatmap(save_path=os.path.join(output_dir, 'basic_positions.png'))
    
    print(f"\n✓ 图表已保存到 {output_dir}")
    print("\n" + "=" * 60)
    print("示例完成！")
    print("=" * 60)


if __name__ == '__main__':
    main()
