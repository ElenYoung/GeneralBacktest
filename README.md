# GeneralBacktest

[![PyPI version](https://badge.fury.io/py/GeneralBacktest.svg)](https://badge.fury.io/py/GeneralBacktest)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**GeneralBacktest** 是一个灵活高效的量化策略回测框架，专为多资产组合策略设计。核心理念是"权重→交易→净值"，支持任意调仓频率、真实交易细节模拟、向量化高性能计算和丰富的性能分析。

## ✨ 主要特性

- 🎯 **灵活的调仓策略**：支持任意调仓频率和时间点，无需固定周期
- 🚀 **高性能计算**：向量化实现，支持数千标的、数年数据的快速回测
- 💰 **真实交易模拟**：
  - 调仓阈值控制（避免微小调整）
  - 买卖分开计费
  - 滑点模拟
  - 调仓日盈亏拆分（持仓/卖出/买入）
- 📊 **总仓位控制**：支持动态调整每个调仓日的总仓位比例（v1.1.0 新增）
- 💵 **现金仓位回测**：基于实际资金和交易单位的精确回测（v1.1.0 新增）
- 📈 **增强可视化**：支持对数坐标净值曲线和双坐标对比（v1.1.0 新增）
- 📊 **丰富的性能指标**：15+ 常用指标
  - 收益率指标：累计收益、年化收益
  - 风险指标：波动率、最大回撤
  - 风险调整指标：夏普比率、索提诺比率、卡玛比率
  - 尾部风险：VaR、CVaR
  - 相对指标：信息比率、超额收益
  - 交易指标：换手率
- 📈 **专业的可视化**：10+ 图表类型
  - 净值曲线（支持普通/对数坐标）
  - 策略对比
  - 超额收益分析
  - 月度收益热力图
  - 换手率分析
  - 持仓变化
  - 综合 Dashboard

## 📦 安装

### 基础安装

```bash
pip install GeneralBacktest
```

### 完整安装（包含数据库支持）

如果需要使用 `run_backtest_ETF()` 或 `run_backtest_stock()` 方法连接数据库：

```bash
pip install GeneralBacktest[database]
```

或安装所有可选依赖：

```bash
pip install GeneralBacktest[full]
```

### 依赖说明

- **必需依赖**：`numpy`, `pandas`, `matplotlib`
- **可选依赖**：
  - `quantchdb`：用于数据库连接（ETF/股票数据）
  - `openpyxl`：用于导出 Excel 报告

## 🚀 快速开始

### 基础用法 - 本地数据回测

```python
from GeneralBacktest import GeneralBacktest
import pandas as pd

# 准备权重数据（调仓信号）
weights_data = pd.DataFrame({
    'date': ['2023-01-01', '2023-01-01', '2023-06-01', '2023-06-01'],
    'code': ['stock_A', 'stock_B', 'stock_A', 'stock_B'],
    'weight': [0.6, 0.4, 0.3, 0.7]
})

# 准备价格数据（日线行情）
price_data = pd.DataFrame({
    'date': pd.date_range('2023-01-01', '2023-12-31', freq='D'),
    'code': 'stock_A',
    'open': [...],
    'close': [...],
    'adj_factor': [...]  # 复权因子
})
# ... 更多股票数据

# 创建回测实例
bt = GeneralBacktest(start_date="2023-01-01", end_date="2023-12-31")

# 运行回测
results = bt.run_backtest(
    weights_data=weights_data,
    price_data=price_data,
    buy_price="open",              # 买入价格字段
    sell_price="close",            # 卖出价格字段
    adj_factor_col="adj_factor",   # 复权因子字段
    close_price_col="close",       # 收盘价字段
    rebalance_threshold=0.005,     # 调仓阈值：0.5%
    transaction_cost=[0.001, 0.001],  # 买卖成本各 0.1%
    slippage=0.0005,               # 滑点：0.05%
    initial_capital=1.0
)

# 查看性能指标
bt.print_metrics()

# 生成可视化报告
bt.plot_all()  # 综合 Dashboard
bt.plot_nav_curve()  # 净值曲线
bt.plot_monthly_returns()  # 月度收益热力图
```

### 高级功能 - 总仓位控制与现金仓位回测

#### 总仓位控制

支持动态调整每个调仓日的总仓位比例，例如保持 80% 股票仓位，20% 现金：

```python
# 在 weights_data 中增加 position_ratio 列
weights_data = pd.DataFrame({
    'date': ['2023-01-01', '2023-01-01', '2023-06-01', '2023-06-01'],
    'code': ['stock_A', 'stock_B', 'stock_A', 'stock_B'],
    'weight': [0.6, 0.4, 0.3, 0.7],
    'position_ratio': [0.8, 0.8, 0.9, 0.9]  # 每个调仓日的目标仓位比例
})

results = bt.run_backtest(
    weights_data=weights_data,
    price_data=price_data,
    position_ratio_col='position_ratio',  # 指定仓位比例列名
    ...
)
```

#### 现金仓位回测（精确模拟）

使用 `run_backtest_with_cash()` 方法进行基于实际资金和交易单位的精确回测，适用于需要考虑以下因素的实盘模拟：

- 最小交易单位（每手 100 股）
- 实际资金约束（现金不足时无法买入）
- 交易优先级策略

```python
# 运行现金仓位回测
results = bt.run_backtest_with_cash(
    weights_data=weights_data,
    price_data=price_data,
    initial_capital=1_000_000,  # 初始资金 100 万
    buy_price='open',
    sell_price='close',
    close_price_col='close',
    lot_size=100,  # 每手股数（A 股=100）
    trade_critic='weight_desc',  # 按权重从大到小交易
    transaction_cost=[0.001, 0.001],
    slippage=0.001
)

# 查看回测结果
print(f"最终净值: {results['nav_series'].iloc[-1]:,.2f}")
print(f"最终现金: {results['cash_series'].iloc[-1]:,.2f}")
print(f"最终现金占比: {results['metrics']['Cash Ratio']:.2%}")
```

### 高级用法 - ETF 数据库回测（需要数据库配置）

> **注意**：`run_backtest_ETF()` 和 `run_backtest_stock()` 方法需要特定的数据库配置（`db_config`），这些方法需要有特定数据库支持。一般建议使用 `run_backtest()` 方法。

```python
from GeneralBacktest import GeneralBacktest
import pandas as pd

# 准备权重数据
weights_data = pd.DataFrame({
    'date': ['2023-01-01', '2023-01-01'],
    'code': ['510300', '510500'],  # ETF 代码
    'weight': [0.6, 0.4]
})

# 数据库配置（需要有效的数据库访问权限）
db_config = {
    'host': 'your_host',
    'port': 9000,
    'user': 'your_user',
    'password': 'your_password',
    'database': 'etf'
}

# 创建回测实例
bt = GeneralBacktest(start_date="2023-01-01", end_date="2023-12-31")

# 运行 ETF 回测（自动从数据库获取价格数据）
results = bt.run_backtest_ETF(
    etf_db_config=db_config,
    weights_data=weights_data,
    buy_price='open',
    sell_price='open',
    transaction_cost=[0.001, 0.001],
    rebalance_threshold=0.005
)

bt.print_metrics()
bt.plot_all()
```

## 📚 核心概念

### 总仓位控制（v1.1.0）

支持在 `run_backtest()` 中通过 `position_ratio_col` 参数动态控制每个调仓日的总仓位比例：

**计算逻辑：**
1. 首先将同一天所有标的的权重归一化到 1（各标的权重之和为 1）
2. 然后乘以当天的 `position_ratio` 值，得到最终权重
3. 最终权重之和 = `position_ratio`，剩余部分视为现金（不产生收益）

**示例：**
- 原始权重：A=0.6, B=0.4（和为 1.0）
- `position_ratio` = 0.8
- 最终权重：A=0.48, B=0.32（和为 0.8）
- 现金仓位 = 0.2（20%）

**向后兼容性：**
- `position_ratio_col=None` 时（默认值），行为与之前版本完全一致（满仓）

### 现金仓位回测 vs 标准回测

| 特性 | `run_backtest()` | `run_backtest_with_cash()` |
|------|----------------|---------------------------|
| 资金单位 | 相对权重（0-1） | 绝对金额（如 1,000,000） |
| 持仓追踪 | 权重比例 | 实际股数 |
| 最小交易单位 | 无限制 | 每手 lot_size 股（默认 100） |
| 现金约束 | 无 | 有（现金不足时无法买入） |
| 复权因子 | 需要提供 | 不需要（使用原始价格） |

**`run_backtest_with_cash()` 特有指标：**
- `Final Cash`: 最终现金余额
- `Cash Ratio`: 最终现金占比
- `Avg Cash Ratio`: 平均现金占比

### 权重数据格式

权重数据表示组合在各个调仓日的目标仓位：

**标准回测（`run_backtest`）：**

| date       | code    | weight | position_ratio* |
|------------|---------|--------|-----------------|
| 2023-01-01 | stock_A | 0.6    | 0.8             |
| 2023-01-01 | stock_B | 0.4    | 0.8             |
| 2023-06-01 | stock_A | 0.3    | 1.0             |
| 2023-06-01 | stock_B | 0.7    | 1.0             |

- `weight`：各标的的相对权重，系统会自动归一化
- `position_ratio`（可选）：该调仓日的总仓位比例（0-1），剩余部分为现金
- 若不提供 `position_ratio`，系统默认满仓（position_ratio=1.0）

**现金回测（`run_backtest_with_cash`）：**

| date       | code    | weight |
|------------|---------|--------|
| 2023-01-01 | stock_A | 0.6    |
| 2023-01-01 | stock_B | 0.4    |
| 2023-06-01 | stock_A | 0.3    |
| 2023-06-01 | stock_B | 0.7    |

- 每个调仓日的权重和应为 1.0
- 不同调仓日可以有不同的持仓数量

### 价格数据格式

价格数据包含资产的日线行情：

| date       | code    | open | high | low  | close | adj_factor |
|------------|---------|------|------|------|-------|------------|
| 2023-01-01 | stock_A | 10.0 | 10.5 | 9.8  | 10.2  | 1.0        |
| 2023-01-02 | stock_A | 10.2 | 10.8 | 10.1 | 10.5  | 1.0        |

- `adj_factor`：复权因子，用于计算真实收益
- 买卖价格可以灵活指定（开盘价、收盘价等）

### 调仓机制

1. **调仓阈值**：只有当目标权重与当前权重差异超过阈值时才交易
2. **买卖拆分**：先卖出不需要的仓位，再买入新仓位
3. **成本计算**：买卖分开计费，可设置不同费率
4. **滑点模拟**：买入价格上浮、卖出价格下压

## 📊 性能指标说明

| 指标类别 | 指标名称 | 说明 | 适用方法 |
|---------|---------|------|----------|
| **收益指标** | 累计收益率 | 期末/期初 - 1 | 所有 |
| | 年化收益率 | 按交易日年化 | 所有 |
| **风险指标** | 年化波动率 | 日收益率标准差年化 | 所有 |
| | 最大回撤 | 从峰值到谷底的最大损失 | 所有 |
| | 最大回撤持续期 | 最长回撤时间（天） | 所有 |
| **风险调整** | 夏普比率 | 年化收益/年化波动率 | 所有 |
| | 索提诺比率 | 年化收益/下行波动率 | 所有 |
| | 卡玛比率 | 年化收益/最大回撤 | 所有 |
| **尾部风险** | VaR (95%) | 5%分位数损失 | 所有 |
| | CVaR (95%) | 超过VaR的平均损失 | 所有 |
| **相对指标** | 信息比率 | 超额收益/跟踪误差 | 所有 |
| | 超额年化收益 | 相对基准的年化超额 | 所有 |
| **交易指标** | 换手率 | 调仓时的买卖金额 | 所有 |
| **现金指标** | 最终现金余额 | 回测结束时的现金金额 | `run_backtest_with_cash` |
| | 最终现金占比 | 最终现金 / 最终净值 | `run_backtest_with_cash` |
| | 平均现金占比 | 每日（现金 / 净值）的均值 | `run_backtest_with_cash` |
| | 换手率 | 调仓时的买卖金额 | `run_backtest_with_cash` |

## 🎨 可视化示例

框架提供多种可视化方法：

```python
# 1. 综合 Dashboard（推荐）
bt.plot_all()

# 2. 净值曲线和回撤
bt.plot_nav_curve()  # 普通坐标
bt.plot_nav_curve(log_scale=True)  # 对数坐标（v1.1.0 新增）

# 3. 双坐标净值曲线对比（v1.1.0 新增）
bt.plot_nav_curve_dual(
    figsize=(14, 12),
    title="Strategy Performance",
    save_path="dual_nav.png"
)

# 4. 策略与基准对比
bt.plot_comparison()

# 5. 超额收益分析
bt.plot_excess_returns()

# 6. 月度收益热力图
bt.plot_monthly_returns()

# 7. 换手率分析
bt.plot_turnover()

# 8. 持仓变化热力图
bt.plot_positions()

# 9. 收益分布直方图
bt.plot_return_distribution()
```

**可视化增强（v1.1.0）：**
- `plot_nav_curve()` 新增 `log_scale` 参数，支持对数坐标显示
- 新增 `plot_nav_curve_dual()` 方法，同时显示普通坐标和对数坐标
- `plot_all()` Dashboard 根据回测类型自动调整显示：
  - `run_backtest()`: 显示归一化净值（从 1 开始）
  - `run_backtest_with_cash()`: 显示实际资金净值（带千位分隔符）

## 🔧 高级配置

### 自定义基准

```python
# 提供基准权重数据
benchmark_weights = pd.DataFrame({
    'date': ['2023-01-01'],
    'code': ['index_300'],
    'weight': [1.0]
})

results = bt.run_backtest(
    weights_data=weights_data,
    price_data=price_data,
    benchmark_weights=benchmark_weights,
    benchmark_name="HS300",
    ...
)
```

### 导出结果

```python
# 获取回测结果
nav_series = bt.daily_nav  # 日度净值序列
positions = bt.daily_positions  # 持仓明细
trade_records = bt.trade_records  # 交易记录
metrics = bt.metrics  # 性能指标字典

# 导出为 DataFrame
import pandas as pd
pd.DataFrame(metrics, index=[0]).to_excel('metrics.xlsx')
```

## 📖 API 文档

### GeneralBacktest 类

#### 初始化
```python
GeneralBacktest(start_date: str, end_date: str)
```

#### 主要方法

**标准回测方法：**
- `run_backtest(weights_data, price_data, buy_price, sell_price, close_price_col, adj_factor_col=None, position_ratio_col=None, ...)`: 通用回测方法（推荐）
  - `position_ratio_col` (str, optional): 总仓位比例列名（v1.1.0 新增），用于动态控制现金仓位
- `run_backtest_ETF(...)`: ETF 回测（需数据库配置）
- `run_backtest_stock(...)`: 股票回测（需数据库配置）

**现金仓位回测方法（v1.1.0 新增）：**
- `run_backtest_with_cash(weights_data, price_data, initial_capital, buy_price='open', sell_price='close', close_price_col='close', lot_size=100, trade_critic='weight_desc', ...)`: 现金仓位回测
  - 基于实际资金和交易单位的精确回测
  - 自动计算换手率、现金占比等指标
  - `lot_size`: 每手股数（A股=100）
  - `trade_critic`: 交易优先级策略（'weight_desc', 'weight_asc', 'amount_max'）

**可视化方法：**
- `print_metrics()`: 打印性能指标
- `plot_all()`: 生成综合报告
- `plot_nav_curve(log_scale=False)`: 绘制净值曲线（v1.1.0 新增 log_scale 参数）
- `plot_nav_curve_dual(figsize=(14, 12), title=None, save_path=None)`: 双坐标净值曲线对比（v1.1.0 新增）
- `plot_comparison()`: 策略对比图
- `plot_excess_returns()`: 超额收益图
- `plot_monthly_returns()`: 月度收益热力图
- `plot_turnover()`: 换手率图
- `plot_positions()`: 持仓热力图
- `plot_return_distribution()`: 收益分布图

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

## 👨‍💻 作者

Elen Young - yang13515360252@163.com

## 🔗 相关链接

- [GitHub 仓库](https://github.com/ElenYoung/GeneralBacktest)
- [问题反馈](https://github.com/ElenYoung/GeneralBacktest/issues)
- [更新日志](https://github.com/ElenYoung/GeneralBacktest/releases)

## ⚠️ 免责声明

本框架仅供学习和研究使用，不构成任何投资建议。使用本框架进行的任何投资决策，风险自负。

## 🔙 向后兼容性

所有 v1.1.0 的更新均为**增量更新，100% 向后兼容**：

- `run_backtest()` 新增 `position_ratio_col` 参数，**默认值为 `None`**（满仓），不传此参数时行为与 v1.0.0 完全一致
- `plot_nav_curve()` 新增 `log_scale` 参数，**默认值为 `False`**，不传此参数时行为与 v1.0.0 一致
- `run_backtest_with_cash()` 是**全新方法**，不影响现有代码
- `plot_nav_curve_dual()` 是**全新方法**，不影响现有代码

您可以在不修改现有代码的情况下安全升级到新版本，并根据需要选择使用新功能。

## 📝 更新日志

### v1.1.0 (2026-02-06)
- ✨ **新增总仓位控制** (`position_ratio_col` 参数)：支持动态调整每个调仓日的总仓位比例
- 💵 **新增现金仓位回测** (`run_backtest_with_cash`)：基于实际资金和交易单位的精确回测
- 📈 **增强可视化功能**：
  - `plot_nav_curve()` 新增 `log_scale` 参数，支持对数坐标
  - 新增 `plot_nav_curve_dual()` 方法，同时显示普通/对数坐标
  - Dashboard 根据回测类型自动调整显示（归一化净值 vs 实际资金）
- 📊 **新增性能指标**：换手率、最终现金、现金占比等
- 🔧 **向后兼容**：所有新功能均为增量更新，不影响现有代码

### v1.0.0 (2026-01-24)
- 🎉 首次发布
- ✨ 支持灵活调仓频率
- ✨ 向量化高性能计算
- ✨ 15+ 性能指标
- ✨ 8+ 可视化图表
- ✨ 支持基准对比
- ✨ 真实交易细节模拟
