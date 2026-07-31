"""
GeneralBacktest v1.2.0 - 现金仓位 + 可成交量约束 演示

演示 run_backtest_with_cash() 新增的 volume_data 参数：
1. 情形 A：不启用量约束（旧行为）
2. 情形 B：共享池模式（volume_col 单列，买+卖共享同一份当日额度）
3. 情形 C：独立池模式（buy_volume_col / sell_volume_col 分开，
           适合买卖时点差异较大的场景，例如 T 日 10:00 买 / T+1 日 14:50 卖）

对比三种模式下的净值、订单填充率、量约束占比等差异。
"""


import numpy as np
import pandas as pd
import sys
import os
import matplotlib.pyplot as plt

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.GeneralBacktest import GeneralBacktest

# 中文字体（服务器约定）
plt.rcParams.update({
    "font.sans-serif": ["Noto Sans CJK SC", "DejaVu Sans"],
    "axes.unicode_minus": False,
})


# ---------------------- 1. 生成模拟数据 ----------------------
def generate_data():
    np.random.seed(2024)
    stocks = ['A', 'B', 'C', 'D', 'E']
    start, end = '2023-01-01', '2024-06-30'
    dates = pd.date_range(start, end, freq='B')

    # 日线价格
    price_records = []
    for i, s in enumerate(stocks):
        p = 20.0 + i * 5
        for d in dates:
            r = np.random.normal(0.0005, 0.02)
            p = max(1.0, p * (1 + r))
            price_records.append({
                'date': d, 'code': s,
                'open': p * (1 + np.random.normal(0, 0.003)),
                'close': p,
                'adj_factor': 1.0,
                # 日均成交量（用于计算 tradable_shares）
                'daily_volume': np.random.randint(50_000, 500_000),
            })
    price_df = pd.DataFrame(price_records)

    # 权重：月末等权买入表现最好的 3 只
    rebalance_dates = pd.date_range(start, end, freq='ME')

    weight_records = []
    for d in rebalance_dates:
        past = price_df[price_df['date'] <= d]
        if past['date'].nunique() < 20:
            continue
        latest = past.groupby('code')['close'].last()
        earlier = past[past['date'] <= (d - pd.Timedelta(days=20))].groupby('code')['close'].last()
        rets = (latest / earlier - 1).sort_values(ascending=False)
        picked = rets.head(3).index
        for c in picked:
            weight_records.append({'date': d, 'code': c, 'weight': 1.0 / len(picked)})
    weights_df = pd.DataFrame(weight_records)

    # volume_data：假设用户在给定 open 价下，能吃到当日 5% 的成交量作为上限
    # （这里是 mock 的构造方式；真实场景中由用户基于分钟数据自行计算）
    vol_df = price_df[['date', 'code', 'daily_volume']].copy()
    vol_df['tradable_shares'] = (vol_df['daily_volume'] * 0.05).astype(int)
    # 独立池模拟：假设买点位于 T 日 10:00（前半日成交量约占 60%），
    # 卖点位于 T+1 日 14:50（尾盘成交量约占 40%），因此把当日 5% 额度按 60/40 拆分。
    # 真实使用中，两列由外部分钟数据分别计算得到（见 can_trade_amt_build/）。
    vol_df['can_buy_amt'] = (vol_df['tradable_shares'] * 0.6).astype(int)
    vol_df['can_sell_amt'] = (vol_df['tradable_shares'] * 0.4).astype(int)
    vol_df = vol_df[['date', 'code', 'tradable_shares', 'can_buy_amt', 'can_sell_amt']]

    return price_df, weights_df, vol_df



def summarize(name, results):
    m = results['metrics']
    nav = results['nav_series']
    print(f"\n===== {name} =====")
    print(f"Final NAV:            {nav.iloc[-1]:>15,.2f}")
    print(f"Total Return:         {m.get('累计收益率', 0):>15.2%}")
    print(f"Annual Return:        {m.get('年化收益率', 0):>15.2%}")
    print(f"Sharpe:               {m.get('夏普比率', 0):>15.4f}")
    print(f"Max Drawdown:         {m.get('最大回撤', 0):>15.2%}")
    print(f"Final Cash Ratio:     {m.get('最终现金占比', 0):>15.2%}")
    print(f"Avg Cash Ratio:       {m.get('平均现金占比', 0):>15.2%}")
    print(f"Total Orders:         {m.get('订单总数', 0):>15d}")
    print(f"Avg Fill Ratio:       {m.get('平均订单填充率', 1):>15.2%}")
    print(f"Volume-Constrained %: {m.get('量约束订单占比', 0):>15.2%}")


def main():
    print("[1/4] 生成模拟数据 ...")
    price_df, weights_df, vol_df = generate_data()
    print(f"  price rows:  {len(price_df):,}")
    print(f"  weight rows: {len(weights_df):,}")
    print(f"  volume rows: {len(vol_df):,}")

    initial_capital = 1_000_000.0

    # ---- 情形 A：不启用 volume_data ----
    print("\n[2/4] 情形 A - 无量约束回测")
    bt_a = GeneralBacktest('2023-01-01', '2024-06-30')
    res_a = bt_a.run_backtest_with_cash(
        weights_data=weights_df,
        price_data=price_df,
        initial_capital=initial_capital,
        buy_price='open',
        sell_price='close',
        close_price_col='close',
        lot_size=100,
        trade_critic='weight_desc',
        transaction_cost=[0.001, 0.001],
        slippage=0.001,
    )
    summarize("A. 无量约束", res_a)

    # ---- 情形 B：启用 volume_data（共享池模式） ----
    print("\n[3/4] 情形 B - 共享池模式（buy+sell 共用 tradable_shares）")
    bt_b = GeneralBacktest('2023-01-01', '2024-06-30')
    res_b = bt_b.run_backtest_with_cash(
        weights_data=weights_df,
        price_data=price_df,
        initial_capital=initial_capital,
        buy_price='open',
        sell_price='close',
        close_price_col='close',
        lot_size=100,
        trade_critic='weight_desc',
        transaction_cost=[0.001, 0.001],
        slippage=0.001,
        volume_data=vol_df,
        volume_col='tradable_shares',
    )
    summarize("B. 共享池 (shared pool)", res_b)

    # ---- 情形 C：启用 volume_data（独立池模式） ----
    print("\n[4/4] 情形 C - 独立池模式（can_buy_amt / can_sell_amt 分列）")
    bt_c = GeneralBacktest('2023-01-01', '2024-06-30')
    res_c = bt_c.run_backtest_with_cash(
        weights_data=weights_df,
        price_data=price_df,
        initial_capital=initial_capital,
        buy_price='open',
        sell_price='close',
        close_price_col='close',
        lot_size=100,
        trade_critic='weight_desc',
        transaction_cost=[0.001, 0.001],
        slippage=0.001,
        volume_data=vol_df,
        # 关键：同时指定 buy_volume_col / sell_volume_col 即切换到独立池模式，
        # 此时 volume_col 参数被忽略。
        buy_volume_col='can_buy_amt',
        sell_volume_col='can_sell_amt',
    )
    summarize("C. 独立池 (split pool)", res_c)

    # ---- 对比图 ----
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(res_a['nav_series'].index, res_a['nav_series'].values,
            label='A. 无量约束', linewidth=2)
    ax.plot(res_b['nav_series'].index, res_b['nav_series'].values,
            label='B. 共享池（volume_col=tradable_shares）', linewidth=2, linestyle='--')
    ax.plot(res_c['nav_series'].index, res_c['nav_series'].values,
            label='C. 独立池（can_buy_amt / can_sell_amt）', linewidth=2, linestyle=':')
    ax.set_title('run_backtest_with_cash: 三种量约束模式对比')
    ax.set_xlabel('Date')
    ax.set_ylabel('NAV')
    ax.grid(True, alpha=0.3)
    ax.legend()

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'output_demo')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'cash_volume_compare.png')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"\n对比图已保存: {out_path}")
    plt.close(fig)

    # ---- 独立池 vs 共享池 - 订单填充率对比 ----
    print("\n===== 共享池 vs 独立池 关键指标对比 =====")
    print(f"{'Metric':<25}{'B. Shared':>15}{'C. Split':>15}")
    for key in ['平均订单填充率', '量约束订单占比', '最终现金占比', '累计收益率']:
        vb = res_b['metrics'].get(key, 0)
        vc = res_c['metrics'].get(key, 0)
        print(f"{key:<25}{vb:>15.2%}{vc:>15.2%}")



if __name__ == '__main__':
    main()
