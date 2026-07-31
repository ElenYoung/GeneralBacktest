# Changelog

All notable changes to this project will be documented in this file.

## [1.2.1] - 2026-07-31

### Added / Improved

- 明确 `run_backtest_with_cash()` 的 volume_data 双池语义：
  - **共享池 (shared pool)**：仅指定 `volume_col`，买+卖当日共享同一份额度。
  - **独立池 (split pool)**：同时指定 `buy_volume_col` 与 `sell_volume_col`,
    买/卖各自独立扣减额度。适用于买卖时点差异较大的场景
    （例如 T 日 10:00 买 / T+1 日 14:50 卖）。
- **严格参数校验**：`buy_volume_col` 与 `sell_volume_col` 必须同时指定或同时
  为 `None`。只填其一会直接抛出 `ValueError`（避免静默回退到共享池导致误用）。
- **买卖对称执行**：新增 `_execute_sell` 与 `_execute_buy` 对称的统一执行函数,
  卖端也遵循 lot_size + volume 约束，并真实扣减 sell 端额度余量。
- **`trade_critic='amount_max'` 仿真修复**：`calculate_total_traded` 现同时考虑
  cash 与 volume 约束（之前仅考虑 cash），保证选中的顺序在真实执行时确实
  可达到期望的成交金额（含 lot_size 取整 + per-order volume 快照扣减）。
- 示例 `examples/cash_volume_example.py` 追加"情形 C：独立池
  （can_buy_amt / can_sell_amt）"演示，末尾输出共享池 vs 独立池的
  关键指标对比表。

### Compatibility

- 完全向后兼容 v1.2.0：`volume_data=None`、单 `volume_col` 用法均无行为改变。


## [1.2.0] - 2026-07-31

### Added

- `run_backtest_with_cash()` 新增可成交量约束（volume_data）：
  - `volume_data` (pd.DataFrame)：用户在给定 `buy_price` / `sell_price` 下自行
    计算的每日最大可成交股数表（date, code, tradable_shares）。
  - `volume_col`：共用可成交量列名，默认 `'tradable_shares'`。
  - `buy_volume_col` / `sell_volume_col`：可选，用于买卖分离的量上限。
  - **严格模式**：`(date, code)` 未出现在 `volume_data` 中的样本 →
    视为当日不可交易。
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
