"""
dual_track_example.py
====================

演示 v1.3.0 新增的 **双轨（dual-track）现金回测**。

背景：
- 当 buy_price（如 10:00 VWAP）时段早于 sell_price（如 14:50 VWAP）时段时，
  单轨"先卖后买"回测会隐含地用当日尚未发生的卖出所得进行买入，
  违反 A 股 T+1 的真实物理时序。
- v1.3.0 引入 `execution_order='buy_first'` 双轨模式：将 initial_capital
  均分为两个逻辑资金池 A / B，每日 A、B 轮换角色（一条买入 signal_today、
  另一条清仓上一次持仓），每条 track 持有周期为 1 天，天然满足 T+1。

本例：
1. 构造一个 30 个交易日、10 只股票的合成数据集
2. 分别用 `execution_order='sell_first'`（单轨）与 `'buy_first'`（双轨）跑回测
3. 打印两者的关键指标对比
4. 若 matplotlib 可用，画出两条 track 的 NAV 与不平衡度
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

# 允许在源码目录下直接运行
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))

from GeneralBacktest import GeneralBacktest


def build_synthetic_data(n_days: int = 30, n_assets: int = 10, seed: int = 42):
    """构造小规模合成数据（日频价格 + 每日随机 5 只股票的信号）。"""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2024-01-02", periods=n_days)
    assets = [f"s{i:02d}" for i in range(n_assets)]

    # 价格：以 10 为基础，加漂移与噪声
    price_rows = []
    base = {a: 10.0 + rng.random() * 5 for a in assets}
    for d in dates:
        for a in assets:
            # 布朗漂移
            base[a] *= 1.0 + rng.normal(0, 0.015)
            p = max(base[a], 1.0)
            vwap10 = p * (1 + rng.normal(0, 0.003))        # 10:00 VWAP
            vwap1450 = p * (1 + rng.normal(0, 0.003))      # 14:50 VWAP
            close = p * (1 + rng.normal(0, 0.004))
            price_rows.append({
                "date": d, "code": a,
                "vwap_1000": vwap10,
                "vwap_1450": vwap1450,
                "close": close,
            })
    price_data = pd.DataFrame(price_rows)

    # 信号：每日随机 5 只股票等权
    weight_rows = []
    for d in dates:
        chosen = rng.choice(assets, size=5, replace=False)
        for a in chosen:
            weight_rows.append({"date": d, "code": a, "weight": 1.0})
    weights_data = pd.DataFrame(weight_rows)

    return weights_data, price_data


def summarize(results: dict, tag: str) -> dict:
    """打印并返回单次回测的关键指标。"""
    m = results["metrics"]
    print(f"\n===== {tag} =====")
    print(f"  最终 NAV:       {results['nav_series'].iloc[-1]:,.2f}")
    print(f"  累计收益率:     {m.get('累计收益率', float('nan')):.2%}")
    print(f"  年化收益率:     {m.get('年化收益率', float('nan')):.2%}")
    print(f"  年化波动率:     {m.get('年化波动率', float('nan')):.2%}")
    print(f"  夏普比率:       {m.get('夏普比率', float('nan')):.3f}")
    print(f"  最大回撤:       {m.get('最大回撤', float('nan')):.2%}")
    print(f"  平均现金占比:   {m.get('平均现金占比', float('nan')):.2%}")
    if "最大不平衡度" in m:
        print(f"  最大不平衡度:   {m['最大不平衡度']:.2%}")
        print(f"  平均不平衡度:   {m['平均不平衡度']:.2%}")
        print(f"  再平衡次数:     {m['再平衡次数']}")
    return m


def main():
    print("Building synthetic dataset...")
    weights_data, price_data = build_synthetic_data(n_days=40, n_assets=10, seed=2024)

    start = str(price_data["date"].min().date())
    end = str(price_data["date"].max().date())

    # ---------- 1. 单轨（sell_first，legacy） ----------
    bt1 = GeneralBacktest(start_date=start, end_date=end)
    res_sf = bt1.run_backtest_with_cash(
        weights_data=weights_data,
        price_data=price_data,
        initial_capital=1_000_000,
        buy_price="vwap_1000",
        sell_price="vwap_1450",
        close_price_col="close",
        lot_size=100,
        transaction_cost=[0.001, 0.001],
        slippage=0.0,
        execution_order="sell_first",   # legacy 单轨
    )
    m_sf = summarize(res_sf, "sell_first (legacy single-track)")

    # ---------- 2. 双轨（buy_first） ----------
    bt2 = GeneralBacktest(start_date=start, end_date=end)
    res_bf = bt2.run_backtest_with_cash(
        weights_data=weights_data,
        price_data=price_data,
        initial_capital=1_000_000,
        buy_price="vwap_1000",
        sell_price="vwap_1450",
        close_price_col="close",
        lot_size=100,
        transaction_cost=[0.001, 0.001],
        slippage=0.0,
        execution_order="buy_first",
        dual_track_config={
            "imbalance_threshold": 0.10,
            "rebalance_gain": 0.5,
            "initial_split": 0.5,
            "first_buy_track": "A",
        },
    )
    m_bf = summarize(res_bf, "buy_first (dual-track)")

    # ---------- 3. 对比表 ----------
    keys = [
        "累计收益率", "年化收益率", "年化波动率", "夏普比率", "最大回撤",
        "平均现金占比", "订单总数",
    ]
    print("\n===== Comparison (sell_first vs buy_first) =====")
    print(f"{'metric':<16} | {'sell_first':>12} | {'buy_first':>12}")
    print("-" * 48)
    for k in keys:
        v1 = m_sf.get(k, float("nan"))
        v2 = m_bf.get(k, float("nan"))
        if k in ("订单总数",):
            print(f"{k:<16} | {v1:>12} | {v2:>12}")
        elif "率" in k or "回撤" in k:
            print(f"{k:<16} | {v1:>12.2%} | {v2:>12.2%}")
        else:
            print(f"{k:<16} | {v1:>12.3f} | {v2:>12.3f}")

    # 双轨专属
    print(f"{'最大不平衡度':<16} | {'-':>12} | {m_bf['最大不平衡度']:>12.2%}")
    print(f"{'再平衡次数':<16} | {'-':>12} | {int(m_bf['再平衡次数']):>12}")

    # ---------- 4. 可视化（可选） ----------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # 中文字体（服务器上 Noto Sans CJK SC）
        plt.rcParams.update({
            "font.sans-serif": ["Noto Sans CJK SC", "SimHei", "DejaVu Sans"],
            "axes.unicode_minus": False,
        })

        fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

        # 上：两种模式的总 NAV
        axes[0].plot(res_sf["nav_series"].index, res_sf["nav_series"].values,
                     label="sell_first (single-track)", color="#7f7f7f",
                     linestyle="--", linewidth=1.5)
        axes[0].plot(res_bf["nav_series"].index, res_bf["nav_series"].values,
                     label="buy_first total", color="#1f77b4", linewidth=2)
        axes[0].plot(res_bf["track_a"]["nav_series"].index,
                     res_bf["track_a"]["nav_series"].values * 2,
                     label="Track A × 2 (dashed)", color="#2ca02c",
                     linestyle=":", linewidth=1.2, alpha=0.8)
        axes[0].plot(res_bf["track_b"]["nav_series"].index,
                     res_bf["track_b"]["nav_series"].values * 2,
                     label="Track B × 2 (dashed)", color="#d62728",
                     linestyle=":", linewidth=1.2, alpha=0.8)
        axes[0].set_title("NAV: sell_first vs buy_first (dual-track)")
        axes[0].set_ylabel("NAV")
        axes[0].legend(loc="upper left")
        axes[0].grid(alpha=0.3)

        # 下：双轨不平衡度
        imb = res_bf["imbalance_series"].set_index("date")["imbalance"]
        axes[1].plot(imb.index, imb.values, color="#9467bd", linewidth=1.5)
        axes[1].axhline(0.10, color="red", linestyle="--", linewidth=0.8,
                        label="+threshold")
        axes[1].axhline(-0.10, color="red", linestyle="--", linewidth=0.8)
        axes[1].axhline(0.0, color="black", linewidth=0.5)

        # 标记 rebalance 事件
        if len(res_bf["rebalance_events"]) > 0:
            for _, ev in res_bf["rebalance_events"].iterrows():
                axes[1].axvline(ev["date"], color="orange", alpha=0.35,
                                linewidth=0.8)
        axes[1].set_title("Dual-track imbalance (nav_A - nav_B) / total")
        axes[1].set_ylabel("Imbalance")
        axes[1].set_xlabel("Date")
        axes[1].legend(loc="upper left")
        axes[1].grid(alpha=0.3)

        out_path = os.path.join(os.path.dirname(__file__), "dual_track_example.png")
        plt.tight_layout()
        plt.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"\nSaved plot -> {out_path}")

    except Exception as e:  # noqa: BLE001
        print(f"\n(Skip plotting: {e})")


if __name__ == "__main__":
    main()
