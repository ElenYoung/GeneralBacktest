"""
高级示例：策略对比与基准分析

这个示例展示了：
- 创建主策略和基准策略
- 对比不同策略的表现
- 生成完整的分析报告
"""

import numpy as np
import pandas as pd
import sys
import os

# 添加父目录到路径以导入 GeneralBacktest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.GeneralBacktest import GeneralBacktest


def generate_market_data(n_stocks=10, n_days=750, seed=123):
    """
    生成模拟市场数据（3年）
    """
    np.random.seed(seed)
    
    dates = pd.date_range(start='2021-01-01', periods=n_days, freq='B')
    stock_codes = [f'STOCK_{i:03d}' for i in range(n_stocks)]
    
    data = []
    
    for code in stock_codes:
        initial_price = np.random.uniform(20, 150)
        
        # 不同股票有不同的收益率和波动率
        mu = np.random.uniform(-0.0002, 0.001)  # 日均收益率
        sigma = np.random.uniform(0.015, 0.025)  # 波动率
        
        returns = np.random.normal(mu, sigma, n_days)
        prices = initial_price * np.exp(np.cumsum(returns))
        
        for i, date in enumerate(dates):
            close_price = prices[i]
            high_price = close_price * np.random.uniform(1.0, 1.025)
            low_price = close_price * np.random.uniform(0.975, 1.0)
            open_price = np.random.uniform(low_price, high_price)
            
            data.append({
                'date': date,
                'code': code,
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'adj_factor': 1.0
            })
    
    return pd.DataFrame(data)


def generate_momentum_strategy(price_data, lookback=60, n_select=5):
    """
    生成动量策略权重：选择过去 lookback 天收益率最高的 n_select 只股票
    """
    dates = sorted(price_data['date'].unique())
    
    # 月度调仓
    rebalance_dates = pd.date_range(start=dates[0], end=dates[-1], freq='MS')
    rebalance_dates = [d for d in rebalance_dates if d in dates]
    
    weights_data = []
    
    for rebal_date in rebalance_dates:
        # 计算过去 lookback 天的收益率
        past_date = rebal_date - pd.Timedelta(days=lookback)
        
        returns_dict = {}
        for code in price_data['code'].unique():
            stock_data = price_data[price_data['code'] == code].sort_values('date')
            stock_data = stock_data[stock_data['date'] <= rebal_date]
            
            if len(stock_data) >= lookback:
                past_price = stock_data[stock_data['date'] <= past_date]['close'].iloc[-1] if len(stock_data[stock_data['date'] <= past_date]) > 0 else stock_data['close'].iloc[0]
                current_price = stock_data['close'].iloc[-1]
                returns_dict[code] = (current_price / past_price - 1)
        
        # 选择收益率最高的 n_select 只股票
        if returns_dict:
            top_stocks = sorted(returns_dict.items(), key=lambda x: x[1], reverse=True)[:n_select]
            weight = 1.0 / n_select
            
            for code, _ in top_stocks:
                weights_data.append({
                    'date': rebal_date,
                    'code': code,
                    'weight': weight
                })
    
    return pd.DataFrame(weights_data)


def generate_equal_weight_benchmark(price_data):
    """
    生成等权基准：所有股票等权配置，季度调仓
    """
    dates = sorted(price_data['date'].unique())
    stock_codes = price_data['code'].unique()
    
    # 季度调仓
    rebalance_dates = pd.date_range(start=dates[0], end=dates[-1], freq='QS')
    rebalance_dates = [d for d in rebalance_dates if d in dates]
    
    weights_data = []
    weight = 1.0 / len(stock_codes)
    
    for date in rebalance_dates:
        for code in stock_codes:
            weights_data.append({
                'date': date,
                'code': code,
                'weight': weight
            })
    
    return pd.DataFrame(weights_data)


def main():
    print("=" * 70)
    print("GeneralBacktest 高级示例：策略对比")
    print("=" * 70)
    
    # 1. 生成市场数据
    print("\n[1/5] 生成市场数据...")
    price_data = generate_market_data(n_stocks=10, n_days=750, seed=123)
    print(f"  - 股票数量：{len(price_data['code'].unique())}")
    print(f"  - 交易日数：{len(price_data['date'].unique())}")
    print(f"  - 时间范围：{price_data['date'].min().date()} 至 {price_data['date'].max().date()}")
    
    # 2. 生成策略权重
    print("\n[2/5] 生成策略权重...")
    print("  - 主策略：动量策略（月度调仓，选前5名）")
    strategy_weights = generate_momentum_strategy(price_data, lookback=60, n_select=5)
    print(f"    调仓次数：{len(strategy_weights['date'].unique())}")
    
    print("  - 基准策略：等权组合（季度调仓）")
    benchmark_weights = generate_equal_weight_benchmark(price_data)
    print(f"    调仓次数：{len(benchmark_weights['date'].unique())}")
    
    # 3. 回测主策略
    print("\n[3/5] 回测主策略...")
    bt_strategy = GeneralBacktest(
        start_date=price_data['date'].min().strftime('%Y-%m-%d'),
        end_date=price_data['date'].max().strftime('%Y-%m-%d')
    )
    
    results_strategy = bt_strategy.run_backtest(
        weights_data=strategy_weights,
        price_data=price_data,
        buy_price='open',
        sell_price='close',
        adj_factor_col='adj_factor',
        close_price_col='close',
        rebalance_threshold=0.01,
        transaction_cost=[0.0015, 0.0015],  # 0.15% 手续费
        slippage=0.001,  # 0.1% 滑点
        benchmark_weights=benchmark_weights,
        benchmark_name="等权基准"
    )
    
    # 4. 展示结果
    print("\n[4/5] 展示对比结果...")
    print("\n" + "=" * 70)
    print("策略性能指标")
    print("=" * 70)
    bt_strategy.print_metrics()
    
    # 5. 生成完整的图表集
    print("\n[5/5] 生成可视化报告...")
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'output_demo')
    os.makedirs(output_dir, exist_ok=True)
    
    # 综合报告
    bt_strategy.plot_all(save_path=os.path.join(output_dir, 'advanced_dashboard.png'))
    
    # 策略对比
    bt_strategy.plot_nav_vs_benchmark(save_path=os.path.join(output_dir, 'advanced_comparison.png'))
    
    # 超额收益
    bt_strategy.plot_excess_returns(save_path=os.path.join(output_dir, 'advanced_excess_returns.png'))
    
    # 换手率分析
    bt_strategy.plot_turnover_analysis(save_path=os.path.join(output_dir, 'advanced_turnover.png'))
    
    # 月度收益
    bt_strategy.plot_monthly_returns_heatmap(save_path=os.path.join(output_dir, 'advanced_monthly_returns.png'))
    
    print(f"\n✓ 所有图表已保存到：{output_dir}")
    
    # 6. 导出详细数据
    print("\n导出详细数据...")
    export_file = os.path.join(output_dir, 'strategy_analysis.xlsx')
    
    try:
        with pd.ExcelWriter(export_file) as writer:
            # 性能指标
            metrics_df = pd.DataFrame([bt_strategy.metrics])
            metrics_df.to_excel(writer, sheet_name='性能指标', index=False)
            
            # 净值序列
            nav_df = pd.DataFrame({
                'date': bt_strategy.daily_nav.index,
                'nav': bt_strategy.daily_nav.values
            })
            nav_df.to_excel(writer, sheet_name='净值序列', index=False)
            
            # 调仓记录
            if len(bt_strategy.trade_records) > 0:
                bt_strategy.trade_records.to_excel(writer, sheet_name='调仓记录', index=False)
            
            # 换手率
            if len(bt_strategy.turnover_records) > 0:
                bt_strategy.turnover_records.to_excel(writer, sheet_name='换手率', index=False)
        
        print(f"✓ 数据已导出到：{export_file}")
    except ImportError:
        print("  (需要安装 openpyxl 才能导出 Excel：pip install openpyxl)")
    
    print("\n" + "=" * 70)
    print("高级示例完成！")
    print("=" * 70)
    print("\n主要发现：")
    
    # 简单总结
    strategy_return = bt_strategy.metrics['累计收益率']
    benchmark_return = bt_strategy.metrics.get('基准累计收益率', 0)
    excess_return = bt_strategy.metrics.get('超额收益率', 0)
    
    print(f"  - 动量策略收益：{strategy_return:.2%}")
    print(f"  - 基准收益：{benchmark_return:.2%}")
    print(f"  - 超额收益：{excess_return:.2%}")
    print(f"  - 夏普比率：{bt_strategy.metrics['夏普比率']:.3f}")
    print(f"  - 最大回撤：{bt_strategy.metrics['最大回撤']:.2%}")


if __name__ == '__main__':
    main()
