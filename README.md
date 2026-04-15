# GeneralBacktest

[![PyPI version](https://badge.fury.io/py/GeneralBacktest.svg)](https://badge.fury.io/py/GeneralBacktest)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

GeneralBacktest is a flexible and efficient quantitative strategy backtesting framework designed for multi-asset portfolio strategies.
Its core workflow is:

weight -> trades -> NAV

It supports arbitrary rebalancing schedules, realistic trading details, vectorized performance, and rich analytics.

## Documentation Languages

- English (this file): [README.md](README.md)
- Chinese: [README.zh-CN.md](README.zh-CN.md)
- Changelog: [CHANGELOG.md](CHANGELOG.md)

## Key Features

- Flexible rebalancing: rebalance on any dates, not limited to fixed intervals.
- High performance: vectorized implementation for large universes and long histories.
- Realistic trading simulation:
  - Rebalance threshold control (avoid tiny adjustments)
  - Separate buy/sell transaction cost
  - Slippage modeling
  - Rebalance-day PnL split (hold/sell/buy)
- Dynamic total exposure control via `position_ratio_col` (v1.1.0).
- Cash-based backtest mode via `run_backtest_with_cash` (v1.1.0).
- Enhanced visualization (log-scale NAV and dual-scale NAV, v1.1.0).
- 15+ performance metrics and 10+ plotting utilities.

## Installation

### Basic

```bash
pip install GeneralBacktest
```

### With Database Support

If you need `run_backtest_ETF()` or `run_backtest_stock()`:

```bash
pip install GeneralBacktest[database]
```

### All Optional Dependencies

```bash
pip install GeneralBacktest[full]
```

### Dependencies

- Required: `numpy`, `pandas`, `matplotlib`
- Optional:
  - `quantchdb` for database-backed ETF/stock data
  - `openpyxl` for Excel export

## Quick Start

### Basic Backtest with Local Data

```python
from GeneralBacktest import GeneralBacktest
import pandas as pd

weights_data = pd.DataFrame({
    "date": ["2023-01-01", "2023-01-01", "2023-06-01", "2023-06-01"],
    "code": ["stock_A", "stock_B", "stock_A", "stock_B"],
    "weight": [0.6, 0.4, 0.3, 0.7]
})

price_data = pd.DataFrame({
    "date": pd.date_range("2023-01-01", "2023-12-31", freq="D"),
    "code": "stock_A",
    "open": [...],
    "close": [...],
    "adj_factor": [...]
})

bt = GeneralBacktest(start_date="2023-01-01", end_date="2023-12-31")

results = bt.run_backtest(
    weights_data=weights_data,
    price_data=price_data,
    buy_price="open",
    sell_price="close",
    adj_factor_col="adj_factor",
    close_price_col="close",
    rebalance_threshold=0.005,
    transaction_cost=[0.001, 0.001],
    slippage=0.0005,
    initial_capital=1.0
)

bt.print_metrics()
bt.plot_all()
bt.plot_nav_curve()
bt.plot_monthly_returns()
```

## Advanced Features

### Dynamic Total Exposure (`position_ratio_col`)

Set total invested ratio per rebalance date (e.g., 80% invested, 20% cash):

```python
weights_data = pd.DataFrame({
    "date": ["2023-01-01", "2023-01-01", "2023-06-01", "2023-06-01"],
    "code": ["stock_A", "stock_B", "stock_A", "stock_B"],
    "weight": [0.6, 0.4, 0.3, 0.7],
    "position_ratio": [0.8, 0.8, 0.9, 0.9]
})

results = bt.run_backtest(
    weights_data=weights_data,
    price_data=price_data,
    position_ratio_col="position_ratio"
)
```

### Cash Backtest (`run_backtest_with_cash`)

Use actual capital and lot-size constraints for execution-accurate simulation:

```python
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
    slippage=0.001
)

print(f"Final NAV: {results['nav_series'].iloc[-1]:,.2f}")
print(f"Final Cash: {results['cash_series'].iloc[-1]:,.2f}")
print(f"Cash Ratio: {results['metrics']['Cash Ratio']:.2%}")
```

### ETF Database Backtest (Requires DB Config)

`run_backtest_ETF()` and `run_backtest_stock()` need a valid database config. For general usage, `run_backtest()` is recommended.

## Core Concepts

### Total Exposure Control

When `position_ratio_col` is provided:

1. Normalize per-date asset weights so they sum to 1.
2. Multiply normalized weights by `position_ratio` of that date.
3. Final sum of target weights equals `position_ratio`; the remaining part is cash.

Example:

- Raw weights: A=0.6, B=0.4
- `position_ratio` = 0.8
- Final weights: A=0.48, B=0.32
- Cash = 0.2

### Standard vs Cash Backtest

| Item | `run_backtest()` | `run_backtest_with_cash()` |
|------|------------------|----------------------------|
| Capital unit | Relative weight (0-1) | Absolute amount |
| Position tracking | Weights | Share quantity |
| Lot size constraint | No | Yes (`lot_size`) |
| Cash constraint | No | Yes |
| Adj factor | Required for adjusted return | Not required |

## Metrics

The framework includes return, risk, risk-adjusted, tail-risk, relative, and turnover metrics.
Cash-specific metrics in `run_backtest_with_cash()` include:

- `Final Cash`
- `Cash Ratio`
- `Avg Cash Ratio`

## Visualization

```python
bt.plot_all()
bt.plot_nav_curve()                    # linear scale
bt.plot_nav_curve(log_scale=True)      # log scale (v1.1.0)
bt.plot_nav_curve_dual()               # dual linear/log (v1.1.0)
bt.plot_comparison()
bt.plot_excess_returns()
bt.plot_monthly_returns()
bt.plot_turnover()
bt.plot_positions()
bt.plot_return_distribution()
```

## API Overview

### Constructor

```python
GeneralBacktest(start_date: str, end_date: str)
```

### Main Methods

- `run_backtest(...)`
- `run_backtest_ETF(...)`
- `run_backtest_stock(...)`
- `run_backtest_with_cash(...)`

### Plotting and Reporting

- `print_metrics()`
- `plot_all()`
- `plot_nav_curve(log_scale=False)`
- `plot_nav_curve_dual(...)`
- `plot_comparison()`
- `plot_excess_returns()`
- `plot_monthly_returns()`
- `plot_turnover()`
- `plot_positions()`
- `plot_return_distribution()`

## Backward Compatibility

All v1.1.0 features are incremental and backward compatible:

- `position_ratio_col` in `run_backtest()` defaults to `None`.
- `log_scale` in `plot_nav_curve()` defaults to `False`.
- `run_backtest_with_cash()` and `plot_nav_curve_dual()` are new methods.

## Contributing

Issues and pull requests are welcome.

## License

MIT License. See [LICENSE](LICENSE).

## Author

Elen Young - yang13515360252@163.com

## Links

- GitHub: https://github.com/ElenYoung/GeneralBacktest
- Issues: https://github.com/ElenYoung/GeneralBacktest/issues
- Releases: https://github.com/ElenYoung/GeneralBacktest/releases

## Disclaimer

This framework is for research and educational purposes only and does not constitute investment advice.
