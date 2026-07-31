# Changelog

All notable changes to this project will be documented in this file.

## [1.2.0] - 2026-07-31

### Added

- `run_backtest_with_cash()` 新增可成交量约束（volume_data）：
  - `volume_data` (pd.DataFrame)：用户在给定 `buy_price` / `sell_price` 下自行
    计算的每日最大可成交股数表（date, code, tradable_shares）。
  - `volume_col`：共用可成交量列名，默认 `'tradable_shares'`。
  - `buy_volume_col` / `sell_volume_col`：可选，用于买卖分离的量上限。
  - **严格模式**：`(date, code)` 未出现在 `volume_data` 中的样本 →
    视为当日不可交易。
  - 交易时不做跨日顺延，用户保证 `buy_price` / `sell_price` 与 `tradable_shares`
    对齐于当日。
- `trade_records` 增加字段：
  - `intended_shares`：受约束前的目标成交股数。
  - `constraint_hit`：命中的约束类型（`'none'` / `'cash'` / `'volume'`）。
- 新增指标：
  - `平均订单填充率` (Avg Fill Ratio) = Σ filled / Σ intended
  - `量约束订单占比` (Volume Constrained Ratio)
  - `订单总数` (Total Orders)
- 新增 `utils.validate_volume_data()` 辅助校验函数。
- 新增示例 `examples/cash_volume_example.py`。

### Compatibility

- 完全向后兼容：`volume_data=None`（默认）时行为与 v1.1.0 一致，无量约束。

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
