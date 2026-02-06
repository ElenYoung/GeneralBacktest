"""
GeneralBacktest v1.1.0 完整演示

这个综合示例展示所有核心功能：
1. 标准回测（run_backtest）
2. 现金仓位回测（run_backtest_with_cash）
3. 总仓位控制（position_ratio_col）
4. 所有可视化图表

运行时间：约2-3分钟
"""

import numpy as np
import pandas as pd
import sys
import os
import matplotlib.pyplot as plt

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.GeneralBacktest import GeneralBacktest


def generate_realistic_data():
    """生成真实的模拟数据"""
    print("[1/4] 生成模拟数据...")

    # 设置随机种子（保证可重复）
    np.random.seed(42)

    # 股票代码和价格
    stocks = ['AAPL', 'GOOGL', 'MSFT', 'TSLA']
    start_date = '2023-01-01'
    end_date = '2024-12-31'
    dates = pd.date_range(start=start_date, end=end_date, freq='B')

    # 生成价格数据（不同股票不同走势）
    price_data = []
    for i, stock in enumerate(stocks):
        price = 100 + i * 50  # 不同初始价格
        for date in dates:
            # 不同的收益率特征
            if i == 0:  # AAPL：稳定增长
                ret = np.random.normal(0.001, 0.02)
            elif i == 1:  # GOOGL：高波动
                ret = np.random.normal(0.0008, 0.035)
            elif i == 2:  # MSFT：震荡
                ret = np.random.normal(0, 0.025)
            else:  # TSLA：大起大落
                ret = np.random.normal(0.0015, 0.05)

            price = price * (1 + ret)

            price_data.append({
                'date': date,
                'code': stock,
                'open': price * 0.99,
                'high': price * 1.02,
                'low': price * 0.98,
                'close': price,
                'adj_factor': 1.0
            })

    price_df = pd.DataFrame(price_data)

    # 生成权重数据（季度调仓）
    weights_data = []
    rebalance_dates = pd.date_range(start=start_date, end=end_date, freq='QE')

    for date in rebalance_dates:
        # 模拟动量策略：选择表现好的股票
        date_prices = price_df[price_df['date'] <= date].groupby('code')['close'].last()
        if len(date_prices) >= 4:
            # 选择涨幅最高的2只股票
            returns = date_prices.pct_change(periods=min(60, len(date_prices)))
            top_stocks = returns.sort_values(ascending=False).head(2).index

            weights_data.extend([
                {'date': date, 'code': stock, 'weight': 0.5}
                for stock in top_stocks
            ])

            # 添加仓位控制（牛市高仓位，熊市低仓位）
            avg_price = float(date_prices.mean())
            # 简单的择时：价格高于100说明是牛市
            position_ratio = 0.8 if avg_price > 100 else 0.6

            for w in weights_data:
                if w['date'] == date:
                    w['position_ratio'] = position_ratio

    return price_df, pd.DataFrame(weights_data)


def demo_standard_backtest(price_data, weights_data):
    """演示标准回测"""
    print("\n[2/4] 运行标准回测...")

    bt = GeneralBacktest('2023-01-01', '2024-12-31')

    results = bt.run_backtest(
        weights_data=weights_data,
        price_data=price_data,
        buy_price='open',
        sell_price='close',
        close_price_col='close',
        adj_factor_col='adj_factor',
        rebalance_threshold=0.005,  # 0.5%调仓阈值
        transaction_cost=[0.001, 0.001],  # 买卖各0.1%
        slippage=0.0005,  # 滑点0.05%
        initial_capital=1.0
    )

    print("  → 完成！累计收益：{:.2%}".format(bt.metrics['累计收益率']))
    return results, bt


def demo_cash_backtest(price_data, weights_data):
    """演示现金仓位回测"""
    print("\n[3/4] 运行现金仓位回测...")

    # 使用完整期间数据，展示多个季度调仓（避免水平线）
    start_date = '2023-01-01'
    end_date = '2024-12-31'  # 完整2年期间

    price_subset = price_data[
        (price_data['date'] >= start_date) &
        (price_data['date'] <= end_date)
    ].copy()

    # 包含起始日期之前的最近调仓权重（用于初始持仓）
    weights_subset = weights_data[
        weights_data['date'] <= end_date
    ].copy()

    bt = GeneralBacktest(start_date, end_date)

    results = bt.run_backtest_with_cash(
        weights_data=weights_subset,
        price_data=price_subset,
        initial_capital=1_000_000,  # 100万初始资金
        buy_price='open',
        sell_price='close',
        close_price_col='close',
        lot_size=100,  # 每手100股
        trade_critic='weight_desc',  # 按权重排序
        transaction_cost=[0.001, 0.001],
        slippage=0.0005
    )

    print("  → 完成！最终净值：{:,.2f}".format(results['nav_series'].iloc[-1]))
    print("  → 最终现金：{:,.2f}".format(results['cash_series'].iloc[-1]))

    return results, bt


def demo_position_ratio(price_data, weights_data):
    """演示总仓位控制"""
    print("\n[4/4] 运行总仓位控制回测...")

    bt = GeneralBacktest('2023-01-01', '2023-06-30')

    # 筛选数据
    price_subset = price_data[price_data['date'] <= '2023-06-30'].copy()
    weights_subset = weights_data[weights_data['date'] <= '2023-06-30'].copy()

    results = bt.run_backtest(
        weights_data=weights_subset,
        price_data=price_subset,
        buy_price='open',
        sell_price='close',
        close_price_col='close',
        adj_factor_col='adj_factor',
        position_ratio_col='position_ratio',  # 使用仓位控制
        rebalance_threshold=0.005,
        transaction_cost=[0.001, 0.001],
        slippage=0.0005,
        initial_capital=1.0
    )

    print("  → 完成！累计收益：{:.2%}".format(bt.metrics['累计收益率']))

    # 计算平均现金占比
    positions_df = results['positions_df']
    if len(positions_df) > 0:
        avg_position = positions_df.groupby('date')['weight'].sum().mean()
        avg_cash = 1 - avg_position
        print("  → 平均现金占比：{:.1%}".format(avg_cash))
    else:
        print("  → 平均现金占比：N/A（无持仓数据）")

    return results, bt


def generate_all_plots(standard_results, standard_bt, cash_results, cash_bt,
                       position_results, position_bt):
    """生成所有图表"""
    print("\n" + "="*70)
    print("生成可视化图表...")
    print("="*70)

    output_dir = os.path.join(os.path.dirname(__file__), '..', 'output_demo')
    os.makedirs(output_dir, exist_ok=True)

    # 1. 标准回测图表
    print("\n[1/6] 标准回测图表...")
    standard_bt.plot_nav_curve(
        save_path=os.path.join(output_dir, 'demo_standard_nav.png'),
        title='Standard Backtest NAV Curve'
    )
    print("  → demo_standard_nav.png")

    standard_bt.plot_monthly_returns_heatmap(
        save_path=os.path.join(output_dir, 'demo_standard_monthly.png')
    )
    print("  → demo_standard_monthly.png")

    standard_bt.plot_position_heatmap(
        save_path=os.path.join(output_dir, 'demo_standard_positions.png')
    )
    print("  → demo_standard_positions.png")

    # 2. 现金回测图表
    print("\n[2/6] 现金回测图表...")
    cash_bt.plot_nav_curve(
        save_path=os.path.join(output_dir, 'demo_cash_nav.png'),
        title='Cash Backtest NAV Curve'
    )
    print("  → demo_cash_nav.png")

    cash_bt.plot_turnover_analysis(
        save_path=os.path.join(output_dir, 'demo_cash_turnover.png')
    )
    print("  → demo_cash_turnover.png")

    # 3. 对数坐标图表（从首次调仓开始，避免初始水平线）
    print("\n[3/6] 对数坐标图表...")
    if len(standard_results['nav_series']) > 0:
        first_rebalance = standard_results['nav_series'].index[0]
        log_nav_series = standard_results['nav_series'][first_rebalance:]

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(log_nav_series.index, log_nav_series.values)
        ax.set_yscale('log')
        ax.set_title('Log Scale NAV Curve')
        ax.set_ylabel('NAV')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'demo_log_nav.png'), bbox_inches='tight')
        plt.close()
        print("  → demo_log_nav.png")
    else:
        print("  → Skipped (no data)")

    # 4. 双坐标对比（从首次调仓开始，避免初始水平线）
    print("\n[4/6] 双坐标对比图表...")
    if len(standard_results['nav_series']) > 0:
        first_rebalance = standard_results['nav_series'].index[0]
        dual_nav_series = standard_results['nav_series'][first_rebalance:]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

        # Linear scale
        ax1.plot(dual_nav_series.index, dual_nav_series.values, linewidth=2)
        ax1.set_title('Linear Scale')
        ax1.set_ylabel('NAV')
        ax1.grid(True, alpha=0.3)

        # Log scale
        ax2.plot(dual_nav_series.index, dual_nav_series.values, linewidth=2)
        ax2.set_yscale('log')
        ax2.set_title('Log Scale')
        ax2.set_ylabel('NAV (log)')
        ax2.set_xlabel('Date')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'demo_dual_nav.png'), bbox_inches='tight')
        plt.close()
        print("  → demo_dual_nav.png")
    else:
        print("  → Skipped (no data)")

    # 5. Dashboard
    print("\n[5/6] 综合 Dashboard...")
    standard_bt.plot_all(
        save_path=os.path.join(output_dir, 'demo_standard_dashboard.png')
    )
    print("  → demo_standard_dashboard.png")

    cash_bt.plot_all(
        save_path=os.path.join(output_dir, 'demo_cash_dashboard.png')
    )
    print("  → demo_cash_dashboard.png")

    # 6. 导出数据
    print("\n[6/6] 导出详细数据...")

    # 标准回测数据
    standard_results['nav_series'].to_csv(
        os.path.join(output_dir, 'demo_standard_nav_series.csv')
    )
    print("  → demo_standard_nav_series.csv")

    # 现金回测数据
    cash_results['nav_series'].to_csv(
        os.path.join(output_dir, 'demo_cash_nav_series.csv')
    )
    cash_results['cash_series'].to_csv(
        os.path.join(output_dir, 'demo_cash_series.csv')
    )
    print("  → demo_cash_nav_series.csv")
    print("  → demo_cash_series.csv")

    # 交易记录
    if len(cash_results['trade_records']) > 0:
        cash_results['trade_records'].to_csv(
            os.path.join(output_dir, 'demo_cash_trades.csv'),
            index=False
        )
        print("  → demo_cash_trades.csv")


def print_summary(standard_results, cash_results, position_results, standard_bt, cash_bt, position_bt):
    """打印总结"""
    print("\n" + "="*70)
    print("性能总结")
    print("="*70)

    print("\n[标准回测]")
    print("  累计收益: {:.2%}".format(standard_bt.metrics['累计收益率']))
    print("  年化收益: {:.2%}".format(standard_bt.metrics['年化收益率']))
    print("  年化波动: {:.2%}".format(standard_bt.metrics['年化波动率']))
    print("  最大回撤: {:.2%}".format(standard_bt.metrics['最大回撤']))
    print("  夏普比率: {:.3f}".format(standard_bt.metrics['夏普比率']))

    print("\n[现金回测]")
    final_nav = cash_results['nav_series'].iloc[-1]
    final_cash = cash_results['cash_series'].iloc[-1]
    initial_capital = 1_000_000
    total_return = (final_nav / initial_capital - 1)

    print("  初始资金: {:,.2f}".format(initial_capital))
    print("  最终净值: {:,.2f}".format(final_nav))
    print("  最终现金: {:,.2f}".format(final_cash))
    print("  股票市值: {:,.2f}".format(final_nav - final_cash))
    print("  累计收益: {:.2%}".format(total_return))
    print("  现金占比: {:.1%}".format(final_cash / final_nav))
    print("  夏普比率: {:.3f}".format(cash_bt.metrics['夏普比率']))

    print("\n[总仓位控制]")
    print("  累计收益: {:.2%}".format(position_bt.metrics['累计收益率']))
    print("  年化收益: {:.2%}".format(position_bt.metrics['年化收益率']))
    print("  最大回撤: {:.2%}".format(position_bt.metrics['最大回撤']))

    # 计算平均现金占比
    positions_df = position_results['positions_df']
    if len(positions_df) > 0:
        avg_position = positions_df.groupby('date')['weight'].sum().mean()
        avg_cash = 1 - avg_position
        print("  平均现金占比: {:.1%}".format(avg_cash))
    else:
        print("  平均现金占比: N/A")
    print("  夏普比率: {:.3f}".format(position_bt.metrics['夏普比率']))


def main():
    """主函数"""
    print("="*70)
    print("GeneralBacktest v1.1.0 完整演示")
    print("="*70)
    print("\n本演示将展示：")
    print("  1. 标准回测（理论最优）")
    print("  2. 现金回测（实盘模拟）")
    print("  3. 总仓位控制（动态仓位）")
    print("  4. 所有可视化图表")
    print("\n预计运行时间：2-3分钟")
    print("="*70)

    # 生成数据
    price_data, weights_data = generate_realistic_data()

    # 运行三种回测
    standard_results, standard_bt = demo_standard_backtest(price_data, weights_data)
    cash_results, cash_bt = demo_cash_backtest(price_data, weights_data)
    position_results, position_bt = demo_position_ratio(price_data, weights_data)

    # 打印性能总结
    print_summary(standard_results, cash_results, position_results, standard_bt, cash_bt, position_bt)


    # 生成所有图表
    generate_all_plots(
        standard_results, standard_bt,
        cash_results, cash_bt,
        position_results, position_bt
    )

    # 完成
    print("\n" + "="*70)
    print("演示完成！")
    print("="*70)
    print("\n查看输出结果：")
    print("  图表：output_demo/demo_*.png")
    print("  数据：output_demo/demo_*.csv")
    print("\n推荐查看：")
    print("  1. demo_standard_nav.png - 标准回测净值曲线")
    print("  2. demo_cash_dashboard.png - 现金回测仪表板")
    print("  3. demo_log_nav.png - 对数坐标展示")
    print("  4. demo_dual_nav.png - 双坐标对比")


if __name__ == '__main__':
    main()
