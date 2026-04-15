# Changelog

All notable changes to this project will be documented in this file.

## [1.1.0] - 2026-02-06

### Added

- Dynamic total exposure control in `run_backtest()` via `position_ratio_col`.
- Cash-based backtesting with `run_backtest_with_cash()`:
  - actual capital tracking
  - lot-size constraints
  - cash-availability constraints
  - trade priority strategy (`trade_critic`)
- Visualization enhancements:
  - `plot_nav_curve(log_scale=True)` for logarithmic NAV scale
  - `plot_nav_curve_dual()` for linear/log dual-panel comparison
- Additional metrics for cash mode:
  - `Final Cash`
  - `Cash Ratio`
  - `Avg Cash Ratio`
  - turnover tracking

### Changed

- `plot_all()` now adapts display style based on backtest type:
  - normalized NAV for `run_backtest()`
  - absolute capital NAV for `run_backtest_with_cash()`

### Compatibility

- Backward compatible with v1.0.0.
- New parameters keep safe defaults:
  - `position_ratio_col=None`
  - `log_scale=False`

## [1.0.0] - 2026-01-24

### Added

- Initial release.
- Flexible rebalancing schedule support.
- Vectorized high-performance backtesting.
- 15+ performance metrics.
- 8+ visualization charts.
- Benchmark comparison support.
- Realistic trading detail simulation.
