import pandas as pd
import pytest

from GeneralBacktest.backtest import GeneralBacktest, _validate_table_name


def _prices():
    dates = pd.to_datetime(['2026-01-05', '2026-01-06'])
    return pd.DataFrame(
        {
            'date': dates,
            'code': ['A', 'A'],
            'open': [10.0, 10.0],
            'close': [10.0, 10.0],
        }
    )


def _weights(position_ratios):
    return pd.DataFrame(
        {
            'date': pd.to_datetime(['2026-01-05', '2026-01-06']),
            'code': ['A', 'A'],
            'weight': [1.0, 1.0],
            'position_ratio': position_ratios,
        }
    )


def test_cash_backtest_applies_daily_position_ratio_and_reports_turnover():
    backtest = GeneralBacktest('2026-01-05', '2026-01-06')

    result = backtest.run_backtest_with_cash(
        weights_data=_weights([0.5, 0.2]),
        price_data=_prices(),
        initial_capital=1000,
        buy_price='open',
        sell_price='open',
        close_price_col='close',
        lot_size=1,
        transaction_cost=[0, 0],
        position_ratio_col='position_ratio',
    )

    assert result['cash_series'].iloc[0] == pytest.approx(500)
    assert result['cash_series'].iloc[1] == pytest.approx(800)
    assert result['positions_df'].groupby('date')['market_value'].sum().iloc[-1] == pytest.approx(200)
    assert result['metrics']['平均换手率'] == pytest.approx(0.2)
    assert not result['turnover_records'].empty


def test_cash_backtest_defaults_to_full_position():
    backtest = GeneralBacktest('2026-01-05', '2026-01-06')

    result = backtest.run_backtest_with_cash(
        weights_data=_weights([0.3, 0.3]).drop(columns='position_ratio'),
        price_data=_prices(),
        initial_capital=1000,
        buy_price='open',
        sell_price='open',
        close_price_col='close',
        lot_size=1,
        transaction_cost=[0, 0],
    )

    assert result['cash_series'].iloc[0] == pytest.approx(0)


def test_dual_track_cash_backtest_applies_position_ratio():
    backtest = GeneralBacktest('2026-01-05', '2026-01-06')

    result = backtest.run_backtest_with_cash(
        weights_data=_weights([0.5, 0.5]),
        price_data=_prices(),
        initial_capital=1000,
        buy_price='open',
        sell_price='open',
        close_price_col='close',
        lot_size=1,
        transaction_cost=[0, 0],
        execution_order='buy_first',
        position_ratio_col='position_ratio',
    )

    first_day = result['positions_df'][
        result['positions_df']['date'] == pd.Timestamp('2026-01-05')
    ]
    assert first_day['market_value'].sum() == pytest.approx(250)
    assert result['cash_series'].iloc[0] == pytest.approx(750)
    assert '平均换手率' in result['metrics']


def test_cash_backtest_rejects_inconsistent_daily_position_ratio():
    weights = pd.DataFrame(
        {
            'date': pd.to_datetime(['2026-01-05', '2026-01-05']),
            'code': ['A', 'B'],
            'weight': [0.5, 0.5],
            'position_ratio': [0.4, 0.6],
        }
    )
    prices = pd.DataFrame(
        {
            'date': pd.to_datetime(['2026-01-05', '2026-01-05']),
            'code': ['A', 'B'],
            'open': [10.0, 10.0],
            'close': [10.0, 10.0],
        }
    )

    with pytest.raises(ValueError, match='one value per date'):
        GeneralBacktest('2026-01-05', '2026-01-05').run_backtest_with_cash(
            weights_data=weights,
            price_data=prices,
            initial_capital=1000,
            buy_price='open',
            sell_price='open',
            close_price_col='close',
            lot_size=1,
            position_ratio_col='position_ratio',
        )


def test_database_table_name_is_caller_supplied_and_validated():
    assert _validate_table_name('market.daily_prices') == 'market.daily_prices'
    with pytest.raises(ValueError, match='required'):
        _validate_table_name(None)
    with pytest.raises(ValueError, match='Invalid'):
        _validate_table_name('market.daily_prices; DROP TABLE x')
