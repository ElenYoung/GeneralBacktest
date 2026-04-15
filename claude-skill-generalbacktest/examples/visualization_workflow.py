"""
Visualization workflow template for GeneralBacktest skill.

Covers:
- baseline NAV plot
- log-scale NAV
- dual-scale NAV
- dashboard
- benchmark comparison (when benchmark exists)
"""

import numpy as np
import pandas as pd
from GeneralBacktest import GeneralBacktest


def build_data():
    dates = pd.date_range("2023-01-01", "2023-12-31", freq="B")
    codes = ["AAA", "BBB", "CCC", "DDD"]
    rng = np.random.default_rng(7)

    price_rows = []
    for code in codes:
        px = 50.0 + rng.uniform(0, 20)
        for d in dates:
            ret = rng.normal(0.0004, 0.02)
            open_px = px
            close_px = max(1.0, open_px * (1 + ret))
            price_rows.append(
                {
                    "date": d,
                    "code": code,
                    "open": float(open_px),
                    "close": float(close_px),
                    "adj_factor": 1.0,
                }
            )
            px = close_px

    price_data = pd.DataFrame(price_rows)

    # Quarterly target weights
    rebalance_dates = pd.to_datetime(["2023-01-03", "2023-04-03", "2023-07-03", "2023-10-09"])
    weight_rows = []
    for d in rebalance_dates:
        ws = rng.random(len(codes))
        ws = ws / ws.sum()
        for code, w in zip(codes, ws):
            weight_rows.append({"date": d, "code": code, "weight": float(w)})

    weights_data = pd.DataFrame(weight_rows)
    return weights_data, price_data


def main():
    weights_data, price_data = build_data()

    bt = GeneralBacktest(start_date="2023-01-01", end_date="2023-12-31")
    bt.run_backtest(
        weights_data=weights_data,
        price_data=price_data,
        buy_price="open",
        sell_price="close",
        adj_factor_col="adj_factor",
        close_price_col="close",
        rebalance_threshold=0.005,
        transaction_cost=[0.001, 0.001],
        slippage=0.0005,
        initial_capital=1.0,
    )

    bt.plot_nav_curve(save_path="viz_nav_linear.png")
    bt.plot_nav_curve(log_scale=True, save_path="viz_nav_log.png")
    bt.plot_nav_curve_dual(save_path="viz_nav_dual.png")
    bt.plot_monthly_returns_heatmap(save_path="viz_monthly_heatmap.png")
    bt.plot_turnover_analysis(save_path="viz_turnover.png")
    bt.plot_position_heatmap(save_path="viz_positions.png")
    bt.plot_all(save_path="viz_dashboard.png")


if __name__ == "__main__":
    main()
