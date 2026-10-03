import pandas as pd
import pytest

from GeneralBacktest.backtest import GeneralBacktest


def test_cash_backtest_reinvests_ex_date_share_adjustment_without_nav_gap():
    dates = pd.to_datetime(["2026-01-05", "2026-01-06"])
    prices = pd.DataFrame({
        "date": dates,
        "code": ["A", "A"],
        "buy": [100.0, 50.0],
        "sell": [100.0, 50.0],
        "close": [100.0, 50.0],
        "adjustment": [1.0, 2.0],
        "adjusted_prev_close": [100.0, 50.0],
    })
    weights = pd.DataFrame({
        "date": [dates[0]], "code": ["A"], "weight": [1.0],
    })

    result = GeneralBacktest(dates[0], dates[-1]).run_backtest_with_cash(
        weights_data=weights,
        price_data=prices,
        initial_capital=10_000.0,
        buy_price="buy",
        sell_price="sell",
        close_price_col="close",
        lot_size=100,
        transaction_cost=[0.0, 0.0],
        slippage=0.0,
        sizing_price="prev_close",
        share_adjustment_col="adjustment",
        adjusted_prev_close_col="adjusted_prev_close",
    )

    assert result["nav_series"].iloc[0] == pytest.approx(10_000.0)
    assert result["nav_series"].iloc[1] == pytest.approx(10_000.0)
    assert result["positions_df"].sort_values("date").iloc[-1]["shares"] == pytest.approx(200.0)


def test_cash_backtest_keeps_prior_close_behavior_without_adjustment_columns():
    dates = pd.to_datetime(["2026-01-05", "2026-01-06"])
    prices = pd.DataFrame({"date": dates, "code": ["A", "A"],
                           "buy": [100.0, 100.0], "sell": [100.0, 100.0], "close": [100.0, 100.0]})
    weights = pd.DataFrame({"date": [dates[0]], "code": ["A"], "weight": [1.0]})
    result = GeneralBacktest(dates[0], dates[-1]).run_backtest_with_cash(
        weights_data=weights, price_data=prices, initial_capital=10_000.0,
        buy_price="buy", sell_price="sell", close_price_col="close", lot_size=100,
        transaction_cost=[0.0, 0.0], slippage=0.0,
    )
    assert result["nav_series"].iloc[-1] == pytest.approx(10_000.0)
