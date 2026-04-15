"""
Cash-based backtest template for GeneralBacktest skill.

This example demonstrates realistic execution constraints:
- finite initial capital
- lot-size trading
- cash availability
- trade priority
"""

import numpy as np
import pandas as pd
from GeneralBacktest import GeneralBacktest


def build_demo_data():
    dates = pd.date_range("2023-01-01", "2023-06-30", freq="B")
    codes = ["AAA", "BBB", "CCC"]

    price_rows = []
    rng = np.random.default_rng(42)
    for code in codes:
        px = 100.0
        for d in dates:
            ret = rng.normal(0.0002, 0.015)
            open_px = px
            close_px = max(1.0, open_px * (1.0 + ret))
            price_rows.append(
                {
                    "date": d,
                    "code": code,
                    "open": float(open_px),
                    "close": float(close_px),
                }
            )
            px = close_px

    price_data = pd.DataFrame(price_rows)

    rebalance_dates = pd.to_datetime(["2023-01-03", "2023-03-01", "2023-05-02"])
    weight_rows = []
    target = {
        rebalance_dates[0]: {"AAA": 0.5, "BBB": 0.3, "CCC": 0.2},
        rebalance_dates[1]: {"AAA": 0.2, "BBB": 0.5, "CCC": 0.3},
        rebalance_dates[2]: {"AAA": 0.1, "BBB": 0.4, "CCC": 0.5},
    }
    for d, wmap in target.items():
        for code, w in wmap.items():
            weight_rows.append({"date": d, "code": code, "weight": w})

    weights_data = pd.DataFrame(weight_rows)
    return weights_data, price_data


def main():
    weights_data, price_data = build_demo_data()

    bt = GeneralBacktest(start_date="2023-01-01", end_date="2023-06-30")

    results = bt.run_backtest_with_cash(
        weights_data=weights_data,
        price_data=price_data,
        initial_capital=1_000_000,
        buy_price="open",
        sell_price="close",
        close_price_col="close",
        lot_size=100,
        trade_critic="weight_desc",
        transaction_cost=[0.001, 0.001],
        slippage=0.0005,
    )

    print("Final NAV:", round(results["nav_series"].iloc[-1], 2))
    print("Final Cash:", round(results["cash_series"].iloc[-1], 2))
    print("Cash Ratio:", round(results["metrics"].get("最终现金占比", 0.0), 4))

    bt.plot_all(save_path="cash_dashboard.png")
    bt.plot_nav_curve_dual(save_path="cash_nav_dual.png")


if __name__ == "__main__":
    main()
