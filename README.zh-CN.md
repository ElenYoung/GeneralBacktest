# GeneralBacktest

[![PyPI version](https://badge.fury.io/py/GeneralBacktest.svg)](https://badge.fury.io/py/GeneralBacktest)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**GeneralBacktest** 是一个灵活高效的量化策略回测框架，专为多资产组合策略设计。核心理念是“权重 -> 交易 -> 净值”，支持任意调仓频率、真实交易细节模拟、向量化高性能计算和丰富的性能分析。

## 文档导航

- 英文主文档: [README.md](README.md)
- 中文文档: [README.zh-CN.md](README.zh-CN.md)
- 更新日志: [CHANGELOG.md](CHANGELOG.md)

## 主要特性

- 灵活调仓: 支持任意调仓频率和时间点，无需固定周期。
- 高性能计算: 向量化实现，支持大规模标的和长周期数据。
- 真实交易模拟:
  - 调仓阈值控制（避免微小调整）
  - 买卖分开计费
  - 滑点模拟
  - 调仓日盈亏拆分（持仓/卖出/买入）
- 总仓位控制: 支持通过 `position_ratio_col` 动态调整每个调仓日的总仓位比例（v1.1.0）。
- 现金仓位回测: `run_backtest_with_cash()` 支持实际资金、手数和现金约束（v1.1.0）。
- 可视化增强: 支持对数净值与双坐标净值对比（v1.1.0）。
- T+0 日内回转: `TBacktest.run_t0_backtest()` 支持同一天内先卖后买的日内回转策略（v1.2.0）。
- 丰富指标与图表: 15+ 性能指标，10+ 可视化方法。

## 安装

### 基础安装

```bash
pip install GeneralBacktest
```

### 数据库支持

如需使用 `run_backtest_ETF()` 或 `run_backtest_stock()`:

```bash
pip install GeneralBacktest[database]
```

### 安装全部可选依赖

```bash
pip install GeneralBacktest[full]
```

## 快速开始

```python
from GeneralBacktest import GeneralBacktest
import pandas as pd

weights_data = pd.DataFrame({
    'date': ['2023-01-01', '2023-01-01', '2023-06-01', '2023-06-01'],
    'code': ['stock_A', 'stock_B', 'stock_A', 'stock_B'],
    'weight': [0.6, 0.4, 0.3, 0.7]
})

price_data = pd.DataFrame({
    'date': pd.date_range('2023-01-01', '2023-12-31', freq='D'),
    'code': 'stock_A',
    'open': [...],
    'close': [...],
    'adj_factor': [...]
})

bt = GeneralBacktest(start_date='2023-01-01', end_date='2023-12-31')

results = bt.run_backtest(
    weights_data=weights_data,
    price_data=price_data,
    buy_price='open',
    sell_price='close',
    adj_factor_col='adj_factor',
    close_price_col='close',
    rebalance_threshold=0.005,
    transaction_cost=[0.001, 0.001],
    slippage=0.0005,
    initial_capital=1.0
)

bt.print_metrics()
bt.plot_all()
```

## 高级功能

### 总仓位控制

通过 `position_ratio_col` 在每个调仓日设置总仓位比例。例如 `position_ratio = 0.8` 表示股票总仓位 80%，现金 20%。

### 现金仓位回测

`run_backtest_with_cash()` 适合更接近实盘的场景，支持:

- 最小交易单位（如每手 100 股）
- 现金约束（现金不足不能买入）
- 交易优先级策略

### 可视化增强

```python
bt.plot_nav_curve()                    # 普通坐标
bt.plot_nav_curve(log_scale=True)      # 对数坐标
bt.plot_nav_curve_dual()               # 双坐标对比
```

### T+0 日内回转回测

`TBacktest` 支持同一天内先卖出后买入的 T+0 策略。`weight` 为**目标仓位**而非交易量，`phase` 列控制日内执行顺序：

```python
from GeneralBacktest import TBacktest

tb = TBacktest(start_date='2024-01-01', end_date='2024-12-31')

results = tb.run_t0_backtest(
    weights_data=t0_weights,   # 包含 'phase' 列
    price_data=price_data,
    buy_price='close',        # 收盘买入，可配置
    sell_price='open',        # 开盘卖出，可配置
    adj_factor_col='adj_factor',
    close_price_col='close',
    transaction_cost=[0.001, 0.001]
)

# T+0 专用可视化
tb.plot_intraday_trades()          # NAV 曲线 + 日内交易点标注
tb.plot_t0_returns_breakdown()      # 卖出收益 vs 买入收益拆分
tb.plot_nav_vs_benchmark()          # 策略 vs 基准对比
```

权重数据格式（含 `phase` 列）：

| date | code | weight | phase |
|------|------|--------|-------|
| 2024-01-02 | stock_A | 1.0 | NaN | （收盘建仓）|
| 2024-01-03 | stock_A | 0.5 | sell | （开盘卖出，目标仓位 → 50%）|
| 2024-01-03 | stock_A | 1.0 | buy | （收盘买回，目标仓位 → 100%）|

框架内置 A股合规校验：
- `buy_phase` 必须在 `sell_phase` 之后
- 每日净买入约束：总卖出量 ≤ 总买入量（不允许裸做空）
- 目标仓位限制在 [0, 1]

T+0 专用指标：卖出胜率、买入胜率、卖出/买入累计收益、收益贡献占比、平均佣金率。

## 核心接口

- `run_backtest(...)` — 标准权重回测
- `run_backtest_ETF(...)` — 从数据库读取 ETF 数据
- `run_backtest_stock(...)` — 从数据库读取股票数据
- `run_backtest_with_cash(...)` — 现金约束回测
- `TBacktest.run_t0_backtest(...)` — T+0 日内回转回测（v1.2.0）
- `print_metrics()`
- `plot_all()` 及其他图表方法

## 向后兼容性

v1.1.0 和 v1.2.0 所有更新均为增量更新，保持向后兼容:

- `run_backtest()` 新增 `position_ratio_col`，默认 `None`。
- `plot_nav_curve()` 新增 `log_scale`，默认 `False`。
- `run_backtest_with_cash()` 与 `plot_nav_curve_dual()` 为 v1.1.0 新增方法，不影响现有代码。
- `TBacktest` 为 v1.2.0 新增类，不影响现有 `GeneralBacktest` 使用。

## 贡献

欢迎提交 Issue 和 Pull Request。

## 许可证

本项目采用 MIT 许可证，详见 [LICENSE](LICENSE)。

## 作者

Elen Young - yang13515360252@163.com

## 免责声明

本框架仅供学习和研究使用，不构成任何投资建议。
