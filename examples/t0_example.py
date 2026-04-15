"""
T+0（日内回转）示例：展示如何做 T 策略回测

本示例展示：
- T+0（日内回转）策略的完整回测流程
- 如何构建包含 phase 列的 weights_data
- A股合规校验（buy_phase 必须在 sell_phase 之后）
- T+0 专用可视化（日内交易点、收益拆分）

运行方式：
    python examples/t0_example.py
"""

import numpy as np
import pandas as pd
import sys
import os
import warnings

warnings.filterwarnings('ignore')

# matplotlib 可选（无 matplotlib 时跳过图表生成）
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("NOTE: matplotlib not installed, skipping chart generation.")

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.GeneralBacktest import TBacktest


def generate_volatile_stock_data(n_days=60, seed=42):
    """
    生成高波动股票数据，更适合展示 T+0 策略效果

    设计思路：日内波动明显（开盘与收盘价差较大），
    使得 sell at open → buy at close 的 T+0 策略有盈利空间。
    """
    np.random.seed(seed)

    dates = pd.date_range(start='2024-01-01', periods=n_days, freq='B')
    stock_codes = ['VOLATILE_A', 'STEADY_B']

    data = []
    for code in stock_codes:
        # 初始参数
        if code == 'VOLATILE_A':
            initial_price = 50.0
            daily_mu = 0.0002    # 微小漂移
            daily_sigma = 0.015  # 较大日内波动
            intraday_drift = 0.0003  # 开盘到收盘的均值漂移（正向有利于做 T）
        else:
            initial_price = 100.0
            daily_mu = 0.0005
            daily_sigma = 0.010
            intraday_drift = 0.0001

        for i, date in enumerate(dates):
            # 生成当日收益率
            daily_ret = np.random.normal(daily_mu, daily_sigma)

            # 分解为开盘缺口 + 日内波动
            open_gap = np.random.normal(0, daily_sigma * 0.3)
            intraday_move = daily_ret - open_gap

            # 确保收盘相对开盘有正向漂移（有利于做 T：开盘卖 → 收盘买）
            if intraday_move < intraday_drift:
                intraday_move = intraday_drift + np.random.normal(0, daily_sigma * 0.5)

            close_price = initial_price * (1 + daily_ret)
            open_price = close_price / (1 + intraday_move)

            # 限制日内波动范围
            intraday_move = close_price / open_price - 1
            high_price = max(open_price, close_price) * (1 + abs(np.random.normal(0, 0.005)))
            low_price = min(open_price, close_price) * (1 - abs(np.random.normal(0, 0.005)))

            data.append({
                'date': date,
                'code': code,
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'adj_factor': 1.0
            })

            initial_price = close_price

    return pd.DataFrame(data)


def generate_t0_weights(price_data, t_stock='VOLATILE_A', t_freq=3):
    """
    生成 T+0 策略权重数据。

    策略说明：
    - 每天收盘买入建仓（phase=None, weight=1.0）
    - 每 t_freq 天做一次 T：
        - sell_phase: 开盘卖出 50%，目标仓位降至 50%
        - buy_phase:  收盘买回 50%，目标仓位恢复 100%
    - 其余日期持有不动（phase=None）

    Parameters:
    -----------
    price_data : pd.DataFrame
        价格数据
    t_stock : str
        做 T 的股票代码
    t_freq : int
        每隔几天做一次 T

    Returns:
    --------
    pd.DataFrame
        包含 phase 列的权重数据
    """
    dates = sorted(price_data['date'].unique())
    weights_data = []

    # 第一天：收盘建仓，100% 持仓
    weights_data.append({
        'date': dates[0],
        'code': t_stock,
        'weight': 1.0,
        'phase': None
    })

    # 后续每天
    for i, date in enumerate(dates[1:], start=1):
        # 判断是否要做 T（跳过第一天，每 t_freq 天做一次）
        is_t_day = (i > 0) and (i % t_freq == 0)

        if is_t_day:
            # T+0 日：先 sell，再 buy
            weights_data.append({
                'date': date,
                'code': t_stock,
                'weight': 0.5,   # sell 后目标持仓 50%
                'phase': 'sell'
            })
            weights_data.append({
                'date': date,
                'code': t_stock,
                'weight': 1.0,   # buy 后目标持仓恢复 100%
                'phase': 'buy'
            })
        else:
            # 普通持有日
            weights_data.append({
                'date': date,
                'code': t_stock,
                'weight': 1.0,
                'phase': None
            })

    return pd.DataFrame(weights_data)


def generate_hold_benchmark(price_data, stock='VOLATILE_A'):
    """
    生成纯持有基准权重（不做 T），用于对比
    """
    dates = sorted(price_data['date'].unique())
    return pd.DataFrame({
        'date': dates,
        'code': [stock] * len(dates),
        'weight': [1.0] * len(dates),
        'phase': [None] * len(dates)
    })


def run_t0_backtest_demo():
    """运行 T+0 回测演示"""
    print("=" * 70)
    print("GeneralBacktest T+0（日内回转）示例")
    print("=" * 70)

    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'output_demo')

    # 1. 生成数据
    print("\n[1/5] 生成高波动股票数据...")
    price_data = generate_volatile_stock_data(n_days=60, seed=42)
    print(f"  - 股票数量：{len(price_data['code'].unique())}")
    print(f"  - 交易日数：{len(price_data['date'].unique())}")
    print(f"  - 日期范围：{price_data['date'].min().date()} 至 {price_data['date'].max().date()}")

    # 2. 生成 T+0 策略权重
    print("\n[2/5] 生成 T+0 策略权重（每隔3天做一次 T）...")
    t0_weights = generate_t0_weights(price_data, t_stock='VOLATILE_A', t_freq=3)
    print(f"  - 总权重记录数：{len(t0_weights)}")

    t_days = t0_weights[t0_weights['phase'].notna()]['date'].nunique()
    t_sell = len(t0_weights[t0_weights['phase'] == 'sell'])
    t_buy = len(t0_weights[t0_weights['phase'] == 'buy'])
    print(f"  - T+0 操作天数：{t_days} 天")
    print(f"  - sell_phase 记录：{t_sell} 条")
    print(f"  - buy_phase 记录：{t_buy} 条")

    # 3. 生成持有基准权重
    print("\n[3/5] 生成纯持有基准（用于对比）...")
    hold_weights = generate_hold_benchmark(price_data, stock='VOLATILE_A')
    print(f"  - 基准记录数：{len(hold_weights)}（每日 100% 持仓，不做 T）")

    # 4. 运行 T+0 回测
    print("\n[4/5] 运行 T+0 回测...")
    start_date = price_data['date'].min().strftime('%Y-%m-%d')
    end_date = price_data['date'].max().strftime('%Y-%m-%d')

    tb = TBacktest(start_date=start_date, end_date=end_date)

    results = tb.run_t0_backtest(
        weights_data=t0_weights,
        price_data=price_data,
        buy_price='close',      # 收盘买入
        sell_price='open',       # 开盘卖出
        adj_factor_col='adj_factor',
        close_price_col='close',
        date_col='date',
        asset_col='code',
        weight_col='weight',
        rebalance_threshold=0.0,  # T+0 需要精确调仓，不设阈值
        transaction_cost=[0.001, 0.001],  # 买卖各 0.1%
        slippage=0.0005,           # 0.05% 滑点
        initial_capital=1.0,
        benchmark_weights=hold_weights,
        benchmark_name="benchmark"
    )

    # 5. 打印结果
    print("\n" + "=" * 70)
    print("T+0 回测性能指标")
    print("=" * 70)
    tb.print_metrics()

    # 6. 生成 T+0 专用图表
    print("\n[5/5] 生成 T+0 专用图表...")
    t0_dir = os.path.join(output_dir, 't0_demo')
    os.makedirs(t0_dir, exist_ok=True)

    if not HAS_MATPLOTLIB:
        print("  (matplotlib not available, skipping chart generation)")
    else:
        # T+0 日内交易点标注
        tb.plot_intraday_trades(
            save_path=os.path.join(t0_dir, 't0_intraday_trades.png'),
            title='T+0 Intraday Trading Points (Open Sell → Close Buy)'
        )
        print(f"  → t0_intraday_trades.png")

        # T+0 收益拆分
        tb.plot_t0_returns_breakdown(
            save_path=os.path.join(t0_dir, 't0_returns_breakdown.png'),
            title='T+0 Returns Breakdown: Sell vs Buy Phase'
        )
        print(f"  → t0_returns_breakdown.png")

        # 综合仪表板（继承自 GeneralBacktest）
        tb.plot_all(save_path=os.path.join(t0_dir, 't0_dashboard.png'))
        print(f"  → t0_dashboard.png")

        # 净值曲线对比
        tb.plot_nav_vs_benchmark(save_path=os.path.join(t0_dir, 't0_vs_benchmark.png'))
        print(f"  → t0_vs_benchmark.png")

    # 导出数据
    results['nav_series'].to_csv(os.path.join(t0_dir, 't0_nav_series.csv'))
    print(f"  → t0_nav_series.csv")

    intraday = tb.get_intraday_records()
    if len(intraday) > 0:
        intraday.to_csv(os.path.join(t0_dir, 't0_intraday_records.csv'), index=False)
        print(f"  → t0_intraday_records.csv")

    # 7. 打印 T+0 专项指标
    print("\n" + "=" * 70)
    print("T+0 专项指标")
    print("=" * 70)
    m = results['metrics']
    t0_keys = ['卖出胜率', '买入胜率', '卖出次数', '买入次数',
               '卖出累计收益', '买入累计收益', '卖出平均收益', '买入平均收益',
               '卖出收益贡献占比', '买入收益贡献占比', '平均佣金率']
    for k in t0_keys:
        if k in m:
            v = m[k]
            if '率' in k or '占比' in k:
                print(f"  {k}: {v:.2%}")
            else:
                print(f"  {k}: {v:.6f}")

    # 8. 手动验证日内记录
    print("\n" + "=" * 70)
    print("日内交易记录（详细）")
    print("=" * 70)
    if len(intraday) > 0:
        display = intraday.copy()
        display['date_str'] = display['date'].dt.strftime('%Y-%m-%d')
        for _, row in display.iterrows():
            tw = row['target_weights']
            tw_str = ', '.join([f"{k}: {v:.0%}" for k, v in tw.items()])
            print(f"  {row['date_str']} | {row['phase'].upper():4s} | "
                  f"收益: {row['return']:+.4%} | 佣金: {row['commission']:.4%} | "
                  f"目标仓位: {tw_str}")

    print("\n" + "=" * 70)
    print("示例完成！")
    print("=" * 70)
    print(f"\n输出目录：{t0_dir}")
    print("\n生成的文件：")
    print(f"  图表：t0_intraday_trades.png, t0_returns_breakdown.png, "
          f"t0_dashboard.png, t0_vs_benchmark.png")
    print(f"  数据：t0_nav_series.csv, t0_intraday_records.csv")

    # 对比分析
    if '年化超额收益' in m:
        print(f"\n策略对比（做 T vs 纯持有）：")
        print(f"  T+0 策略年化收益：{m['年化收益率']:.2%}")
        print(f"  纯持有基准年化收益：{m.get('基准年化收益率', 0):.2%}")
        print(f"  年化超额收益：{m['年化超额收益']:+.2%}")

    return results, tb


def main():
    """主函数"""
    run_t0_backtest_demo()


if __name__ == '__main__':
    main()
