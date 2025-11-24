# GeneralBacktest - 通用量化策略回测框架

[![Python](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-1.0.0-orange.svg)](README.md)

一个功能强大、灵活易用的通用量化策略回测框架，支持任意调仓频率、向量化高性能计算、丰富的性能指标和专业级可视化。

## ✨ 核心特性

### 🚀 灵活的调仓机制
- ✅ **任意调仓频率** - 无需固定间隔，支持基于信号的动态调仓
- ✅ **调仓阈值控制** - 避免频繁小额交易，降低成本
- ✅ **滑点模拟** - 真实模拟市场冲击成本
- ✅ **精确成本核算** - 买卖双向费率、滑点损耗全面考虑
- ✅ **盘中收益处理** - 调仓日买入价到收盘价的盘中收益准确计算

### ⚡ 高性能向量化计算
- ✅ **零循环设计** - 全程使用pandas/numpy向量化操作
- ✅ **Pivot加速** - 利用透视表实现O(1)时间复杂度的价格查找
- ✅ **权重漂移优化** - 高效计算持仓自然漂移
- ✅ **大规模回测** - 轻松处理数千资产×数年数据

### 📊 丰富的性能指标 (15+)

#### 收益指标
- 累计收益率、年化收益率

#### 风险指标  
- 年化波动率、最大回撤（含时间区间）
- VaR、CVaR

#### 风险调整收益
- 夏普比率 (Sharpe Ratio)
- 索提诺比率 (Sortino Ratio)  
- 卡玛比率 (Calmar Ratio)
- 信息比率 (Information Ratio)

#### 交易指标
- 交易次数、平均换手率、累计换手率、胜率

#### 相对基准指标
- 超额收益、年化超额收益
- 跟踪误差、信息比率

### 📈 专业级可视化 (13个图表)

#### 标准图表 (10个)
- 📈 **净值曲线** (`plot_nav_curve`) - 含回撤子图与最大回撤区间高亮
- 📉 **回撤曲线** (`plot_drawdown`) - 独立回撤分析
- 📊 **策略vs基准** (`plot_nav_vs_benchmark`) - 净值对比 + 超额收益曲线
- 💹 **超额收益** (`plot_excess_returns`) - 累计超额收益分析
- 📍 **交易点位** (`plot_trade_points`) - 净值曲线叠加调仓点
- 🔥 **持仓热力图** (`plot_position_heatmap`) - 智能排序，优先展示核心资产
- 📅 **月度热力图** (`plot_monthly_returns_heatmap`) - 月度/年度收益概览
- 📊 **换手率分析** (`plot_turnover_analysis`) - 时序换手率与均值线
- 📋 **指标表格** (`plot_metrics_table`) - 导出专业指标表格图片
- 📱 **综合看板** (`plot_dashboard`) - 一页展示核心图表与指标

#### 扩展分析
- 📉 **权重时间序列** - 各资产权重演变
- 📊 **持仓集中度** - HHI指数趋势
- 🔍 **单资产详细分析** - 权重+价格+变化三合一
- 📈 **日度收益分布** - 直方图统计

### 🎯 新增功能

- ✅ **`print_metrics()`** - 美观打印所有性能指标
- ✅ **`get_daily_positions()`** - 获取每日持仓明细
- ✅ **`get_position_matrix()`** - 获取权重矩阵（透视表格式）
- ✅ **`get_position_changes()`** - 获取每日权重变化
- ✅ **便捷方法** - `run_backtest_ETF()`、`run_backtest_stock()`

## 📦 安装

### 依赖包
```bash
pip install numpy pandas matplotlib
```

### 可选依赖
```bash
# 如需导出Excel
pip install openpyxl

# 如需数据库支持（ETF/股票回测）
pip install quantchdb
```

## 🚀 快速开始

### 基础用法

```python
import sys
sys.path.append('path/to/GeneralBacktest')

from backtest import GeneralBacktest
import pandas as pd

# 1. 初始化回测框架
bt = GeneralBacktest(
    start_date='2023-01-01',
    end_date='2023-12-31'
)

# 2. 运行回测
results = bt.run_backtest(
    weights_data=your_weights_df,      # 策略权重数据
    price_data=your_price_df,          # 价格数据
    buy_price='open',                  # 买入价格字段
    sell_price='close',                # 卖出价格字段
    adj_factor_col='adj_factor',       # 复权因子字段
    close_price_col='close',           # 收盘价字段
    rebalance_threshold=0.01,          # 1%调仓阈值
    transaction_cost=[0.001, 0.001],   # 买卖各0.1%手续费
    initial_capital=1000000,           # 初始资金100万
    slippage=0.001,                    # 0.1%滑点
    benchmark_weights=benchmark_df     # 基准权重（可选）
)

# 3. 打印性能指标（新功能！）
bt.print_metrics()

# 4. 获取每日权重数据（新功能！）
positions = bt.get_daily_positions()       # 长格式
position_matrix = bt.get_position_matrix() # 矩阵格式
position_changes = bt.get_position_changes() # 权重变化

# 5. 可视化
bt.plot_dashboard(save_path='dashboard.png')  # 综合分析面板
# bt.plot_all() # plot_dashboard 的别名
```

### 数据格式要求

#### 1. 权重数据 (weights_data)

| date       | code      | weight |
|------------|-----------|--------|
| 2023-01-03 | 000001.SZ | 0.25   |
| 2023-01-03 | 600000.SH | 0.30   |
| 2023-01-03 | 600519.SH | 0.45   |
| 2023-02-01 | 000001.SZ | 0.20   |
| ...        | ...       | ...    |

**说明：**
- `date`: 调仓日期（支持任意频率）
- `code`: 资产代码
- `weight`: 目标权重（会自动归一化）

#### 2. 价格数据 (price_data)

| date       | code      | open  | high  | low   | close | adj_factor |
|------------|-----------|-------|-------|-------|-------|------------|
| 2023-01-03 | 000001.SZ | 10.50 | 10.80 | 10.40 | 10.70 | 1.0        |
| 2023-01-04 | 000001.SZ | 10.70 | 10.90 | 10.60 | 10.85 | 1.0        |
| ...        | ...       | ...   | ...   | ...   | ...   | ...        |

**说明：**
- `date`: 交易日期（日频）
- `code`: 资产代码
- `open`, `high`, `low`, `close`: OHLC价格
- `adj_factor`: 累计复权因子

**复权价格计算：** 复权价格 = 原始价格 × adj_factor

## 💡 使用示例

### 示例1：基本回测

```python
from backtest import GeneralBacktest
import pandas as pd
import numpy as np

# 生成模拟数据
dates = pd.date_range('2023-01-01', '2023-12-31', freq='B')
assets = ['000001.SZ', '000002.SZ', '600000.SH']

# 价格数据
price_data = []
for asset in assets:
    prices = 100 * np.exp(np.cumsum(np.random.normal(0.0005, 0.02, len(dates))))
    for i, date in enumerate(dates):
        price_data.append({
            'date': date,
            'code': asset,
            'open': prices[i] * (1 + np.random.uniform(-0.01, 0.01)),
            'close': prices[i],
            'adj_factor': 1.0
        })
price_df = pd.DataFrame(price_data)

# 权重数据（月度调仓）
rebalance_dates = pd.date_range('2023-01-01', '2023-12-31', freq='MS')
weights_data = []
for date in rebalance_dates:
    weights = np.random.dirichlet(np.ones(len(assets)))
    for asset, weight in zip(assets, weights):
        weights_data.append({'date': date, 'code': asset, 'weight': weight})
weights_df = pd.DataFrame(weights_data)

# 运行回测
bt = GeneralBacktest('2023-01-01', '2023-12-31')
results = bt.run_backtest(
    weights_data=weights_df,
    price_data=price_df,
    buy_price='open',
    sell_price='close',
    adj_factor_col='adj_factor',
    close_price_col='close'
)

# 查看结果
bt.print_metrics()
bt.plot_all()
```

### 示例2：权重分析

```python
# 获取每日权重矩阵
position_matrix = bt.get_position_matrix()

# 查看最新持仓
print("最新持仓分布:")
print(position_matrix.iloc[-1].sort_values(ascending=False))

# 分析持仓集中度
hhi = (position_matrix ** 2).sum(axis=1)
print(f"平均HHI: {hhi.mean():.4f}")

# 可视化权重变化
import matplotlib.pyplot as plt
position_matrix.plot(figsize=(14, 6))
plt.title('各资产权重时间序列')
plt.ylabel('权重')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()
```

### 示例3：导出结果

```python
# 导出为Excel
with pd.ExcelWriter('backtest_results.xlsx') as writer:
    bt.get_metrics().to_excel(writer, sheet_name='性能指标')
    bt.get_position_matrix().to_excel(writer, sheet_name='每日权重')
    bt.get_position_changes().to_excel(writer, sheet_name='权重变化')
    bt.trade_records.to_excel(writer, sheet_name='交易记录', index=False)
    bt.turnover_records.to_excel(writer, sheet_name='换手率', index=False)
```

## 📖 API文档

### GeneralBacktest类

#### 初始化
```python
GeneralBacktest(start_date: str, end_date: str)
```
**参数：**
- `start_date`: 回测开始日期 (格式: 'YYYY-MM-DD')
- `end_date`: 回测结束日期 (格式: 'YYYY-MM-DD')

---

#### run_backtest()
```python
run_backtest(
    weights_data: pd.DataFrame,
    price_data: pd.DataFrame,
    buy_price: str,
    sell_price: str,
    adj_factor_col: str,
    close_price_col: str,
    date_col: str = 'date',
    asset_col: str = 'code',
    weight_col: str = 'weight',
    rebalance_threshold: float = 0.005,
    transaction_cost: List[float] = [0.001, 0.001],
    initial_capital: float = 1.0,
    slippage: float = 0.0,
    benchmark_weights: Optional[pd.DataFrame] = None
) -> Dict
```

**参数：**
- `weights_data`: 策略权重数据
- `price_data`: 价格数据
- `buy_price`: 买入价格字段名
- `sell_price`: 卖出价格字段名
- `adj_factor_col`: 复权因子字段名
- `close_price_col`: 收盘价字段名
- `rebalance_threshold`: 调仓阈值（默认0.005）
- `transaction_cost`: [买入成本, 卖出成本]（默认各0.001）
- `initial_capital`: 初始资金（默认1.0）
- `slippage`: 滑点率（默认0.0）
- `benchmark_weights`: 基准权重数据（可选）

---

#### 性能指标方法

##### get_metrics()
```python
get_metrics() -> pd.DataFrame
```
返回性能指标DataFrame。

##### print_metrics() 🆕
```python
print_metrics() -> None
```
美观打印所有性能指标，按类别分组展示。

---

#### 权重分析方法 🆕

##### get_daily_positions()
```python
get_daily_positions() -> pd.DataFrame
```
返回每日持仓明细（长格式），包含date、asset、weight三列。

##### get_position_matrix()
```python
get_position_matrix() -> pd.DataFrame
```
返回权重矩阵（透视表格式），行为日期，列为资产。

##### get_position_changes()
```python
get_position_changes() -> pd.DataFrame
```
返回每日权重变化，正值表示增持，负值表示减持。

---

#### 可视化方法

| 方法 | 说明 |
|------|------|
| `plot_nav_curve()` | 策略净值曲线（含回撤子图与高亮） |
| `plot_drawdown()` | 独立回撤曲线 |
| `plot_nav_vs_benchmark()` | 策略vs基准对比（含超额收益） |
| `plot_excess_returns()` | 累计超额收益曲线 |
| `plot_trade_points()` | 交易点位分析 |
| `plot_position_heatmap()` | 持仓权重热力图（智能排序） |
| `plot_monthly_returns_heatmap()` | 月度/年度收益热力图 |
| `plot_turnover_analysis()` | 换手率分析 |
| `plot_metrics_table()` | 性能指标表格图片 |
| `plot_dashboard()` | 综合分析面板（含NAV、回撤、指标、换手率） |
| `plot_all()` | `plot_dashboard` 的别名 |

**所有可视化方法都支持 `figsize` 参数调整图表大小。**

---

#### 便捷方法

##### run_backtest_ETF()
```python
run_backtest_ETF(
    weights_data: pd.DataFrame,
    buy_price: str = 'OpenPrice',
    sell_price: str = 'ClosePrice',
    transaction_cost: List[float] = [0.001, 0.001],
    rebalance_threshold: float = 0.01,
    benchmark_weights: Optional[pd.DataFrame] = None
)
```
直接从数据库获取ETF价格数据并运行回测。

##### run_backtest_stock()
```python
run_backtest_stock(
    weights_data: pd.DataFrame,
    buy_price: str = 'open',
    sell_price: str = 'close',
    transaction_cost: List[float] = [0.001, 0.001],
    rebalance_threshold: float = 0.01,
    benchmark_weights: Optional[pd.DataFrame] = None
)
```
直接从数据库获取股票价格数据并运行回测。

## 🏗️ 技术架构

### 文件结构
```
GeneralBacktest/
├── __init__.py              # 包初始化（灵活导入支持）
├── backtest.py              # 核心回测类（1000+行）
├── utils.py                 # 辅助函数和指标计算（700+行）
├── db_config.py             # 数据库配置（可选）
└── README.md                # 本文档
```

### 核心算法

#### 净值计算逻辑

**调仓日：**
1. 计算目标权重与当前权重的差异
2. 应用调仓阈值（避免小额交易）
3. 将持仓分解为三部分：
   - **Kept**: 保持不变的部分
   - **Sold**: 需要卖出的部分  
   - **Bought**: 需要买入的部分
4. 分别计算三部分的收益贡献：
   - Sold: [昨收 → 卖出价] + 滑点损耗
   - Kept: [昨收 → 今收]
   - Bought: [买入价 → 今收] + 滑点损耗
5. 扣除交易成本，更新净值
6. 计算收盘时的真实权重（考虑权重漂移）

**非调仓日：**
1. 根据持仓权重和复权价格变化计算收益
2. 更新净值
3. 计算权重自然漂移

**关键公式：**
```
权重漂移 = weight * (1 + return) / (1 + portfolio_return)
换手率 = Σ|target_weight - current_weight| / 2
```

## 🎯 最佳实践

### 1. 数据准备
- ✅ 确保价格数据和权重数据的日期对齐
- ✅ 正确设置复权因子（分红、送股等）
- ✅ 过滤掉停牌/退市的标的
- ✅ 权重归一化（框架会自动处理）

### 2. 参数设置
- ✅ 根据实际市场设置合理的交易成本
- ✅ 使用调仓阈值避免频繁小额交易
- ✅ 滑点设置应考虑资产流动性
- ✅ 初始资金建议设为实际金额（便于理解）

### 3. 性能优化
- ✅ 大规模回测时，减少不必要的可视化
- ✅ 使用向量化操作处理数据
- ✅ 合理设置调仓频率（过高会降低性能）
- ✅ 持仓热力图会自动限制显示资产数量

### 4. 结果分析
- ✅ 使用`print_metrics()`快速查看关键指标
- ✅ 通过`get_position_matrix()`分析持仓变化
- ✅ 用`plot_all()`生成完整分析报告
- ✅ 导出Excel进行深度分析

## ⚠️ 注意事项

1. **数据质量至关重要**
   - 缺失数据会影响回测准确性
   - 建议使用前复权或后复权数据

2. **交易成本不可忽视**
   - 包括手续费、印花税、滑点等
   - 高频策略交易成本影响更大

3. **回测偏差（Survivorship Bias）**
   - 仅使用当前存续标的会高估收益
   - 建议包含退市/摘牌标的

4. **未来函数（Look-Ahead Bias）**
   - 确保权重计算不使用未来数据
   - 框架假设用户已正确处理

5. **市场环境变化**
   - 历史表现不代表未来收益
   - 建议进行样本外测试

## 🔍 常见问题

### Q1: 如何处理停牌股票？
**A:** 停牌期间价格设为前一交易日价格，权重会自然漂移。复权因子保持不变。

### Q2: 支持做空策略吗？
**A:** 理论上支持，权重可以为负。但需确保价格数据支持做空逻辑。

### Q3: 如何实现T+1交易制度？
**A:** 设置`buy_price='close'`和`sell_price='open'`可模拟T+1。

### Q4: 换手率过高怎么办？
**A:** 增大`rebalance_threshold`参数，设置更宽松的调仓阈值。

### Q5: 如何对比多个策略？
**A:** 对每个策略分别运行回测，然后手动合并净值数据进行对比。

### Q6: 支持分钟级回测吗？
**A:** 当前版本为日频设计。分钟级需修改数据频率和时间逻辑。

### Q7: 如何导出所有图表？
**A:** 使用matplotlib的`savefig()`方法保存每个图表。

### Q8: 基准如何设置？
**A:** 通过`benchmark_weights`参数传入，格式与策略权重相同。

## 📝 更新日志

### v1.1.0 (2024-11-25)
- 🎨 **绘图系统重构**
  - 全面升级为专业出版级图表风格
  - 新增 `plot_dashboard` 综合看板
  - 新增 `plot_metrics_table` 指标表格导出
  - 新增 `plot_monthly_returns_heatmap` 月度收益热力图
  - 优化 `plot_position_heatmap` 展示逻辑，优先展示核心资产
  - 优化 `plot_nav_curve` 和 `plot_nav_vs_benchmark` 的展示细节
  - 所有图表标签国际化（英文），解决字体兼容性问题

### v1.0.0 (2024-11-24)
- ✨ **新增功能**
  - 添加`print_metrics()`方法 - 美观打印所有指标
  - 添加`get_daily_positions()`方法 - 获取每日持仓
  - 添加`get_position_matrix()`方法 - 获取权重矩阵
  - 添加`get_position_changes()`方法 - 获取权重变化
  - 添加灵活导入支持（相对+绝对导入）
  
- 🐛 **问题修复**
  - 修复基准计算参数传递问题
  - 修复ETF/股票回测缺少`close_price_col`参数
  - 修复模块导入兼容性问题

- 📚 **文档更新**
  - 完善README文档
  - 添加API文档
  - 添加使用示例和最佳实践

- ⚡ **性能优化**
  - 优化每日权重记录逻辑
  - 改进向量化计算效率

### v0.9.0 (Initial Release)
- ✅ 核心回测功能
- ✅ 15+性能指标
- ✅ 8+可视化图表
- ✅ 向量化计算
- ✅ 调仓阈值支持

## 📄 许可证

MIT License

## 👥 贡献

欢迎提交Issue和Pull Request！



---

**⭐ 如果这个项目对你有帮助，请给个Star支持一下！**
