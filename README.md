# GeneralBacktest - 通用量化策略回测框架

一个功能强大的通用量化策略回测框架，支持灵活的调仓时间、向量化计算、丰富的性能指标和可视化功能。

## 主要特性

### 1. 灵活的调仓机制
- ✅ 支持任意时间点调仓（不需要固定频率）
- ✅ 自动处理调仓日当天的盘中收益
- ✅ 支持调仓阈值设置
- ✅ 准确计算交易成本

### 2. 向量化高性能计算
- ✅ 避免循环操作，使用 pandas/numpy 向量化
- ✅ 高效的净值填充算法
- ✅ 快速的持仓权重计算

### 3. 丰富的性能指标（15+）

**收益类指标：**
- 累计收益率
- 年化收益率
- 月度收益统计

**风险类指标：**
- 年化波动率
- 最大回撤（含时间区间）
- VaR（风险价值）
- CVaR（条件风险价值）

**风险调整收益：**
- 夏普比率（Sharpe Ratio）
- 索提诺比率（Sortino Ratio）
- 卡玛比率（Calmar Ratio）
- 信息比率（Information Ratio）

**交易类指标：**
- 交易次数
- 换手率（平均、累计）
- 胜率

**相对基准：**
- 超额收益
- 跟踪误差
- 信息比率

### 4. 多样化可视化（8+图表）

- 📈 策略累计净值曲线
- 📉 回撤曲线
- 📊 策略 vs 基准对比
- 💹 超额收益曲线
- 📍 交易点位分析
- 🔥 持仓热力图
- 📊 换手率分析
- 📱 综合展示面板

## 安装依赖

```bash
pip install numpy pandas matplotlib
```

## 快速开始

### 基本用法

```python
from backtest import GeneralBacktest

# 1. 初始化回测框架
backtest = GeneralBacktest(
    start_date='2020-01-01',
    end_date='2023-12-31'
)

# 2. 运行回测
results = backtest.run_backtest(
    weights_data=weights_df,      # 策略权重数据
    price_data=price_df,           # 价格数据
    buy_price='open',              # 买入价格字段
    sell_price='close',            # 卖出价格字段
    adj_factor_col='adj_factor',   # 复权因子字段
    transaction_cost=[0.001, 0.001]  # [买入成本, 卖出成本]
)

# 3. 查看性能指标
metrics = backtest.get_metrics()
print(metrics)

# 4. 可视化
backtest.plot_all()  # 综合展示
```

### 数据格式要求

#### weights_data（权重数据）
```
    date        code      weight
0   2020-01-01  ASSET_01  0.25
1   2020-01-01  ASSET_02  0.35
2   2020-01-01  ASSET_03  0.40
...
```

**说明：**
- `date`: 调仓日期（可以是任意频率，不需要固定）
- `code`: 资产代码
- `weight`: 目标权重（会自动归一化）

#### price_data（价格数据）
```
    date        code      open    high    low     close   adj_factor
0   2020-01-01  ASSET_01  100.0   102.0   99.0    101.0   1.0
1   2020-01-02  ASSET_01  101.0   103.0   100.0   102.0   1.0
...
```

**说明：**
- `date`: 交易日期（日频数据）
- `code`: 资产代码
- `open`, `high`, `low`, `close`: OHLC价格
- `adj_factor`: 累计复权因子（用于计算复权价格）

**复权价格计算：** `复权价格 = 原始价格 × adj_factor`

## 核心逻辑说明

### 净值计算逻辑

1. **调仓日当天：**
   - 根据目标权重和买入价格进行调仓
   - 扣除交易成本
   - **重要：计算从买入价到收盘价的盘中收益**
   - 更新净值

2. **非调仓日：**
   - 根据持仓权重和复权价格变化计算收益
   - 更新净值

### 交易成本计算

- 买入成本 = Σ(买入金额) × 买入费率
- 卖出成本 = Σ(卖出金额) × 卖出费率
- 总成本从净值中扣除

### 换手率计算

换手率 = Σ|目标权重 - 当前权重| / 2

## 完整示例

查看 `example.py` 获取完整的使用示例，包括：
- 模拟数据生成
- 回测运行
- 性能分析
- 可视化展示

运行示例：
```bash
python example.py
```

## API 文档

### GeneralBacktest 类

#### 初始化
```python
GeneralBacktest(start_date, end_date)
```

**参数：**
- `start_date` (str): 回测开始日期，格式 'YYYY-MM-DD'
- `end_date` (str): 回测结束日期，格式 'YYYY-MM-DD'

#### run_backtest
```python
run_backtest(
    weights_data,
    price_data,
    buy_price,
    sell_price,
    adj_factor_col,
    date_col='date',
    asset_col='code',
    weight_col='weight',
    rebalance_threshold=0.01,
    transaction_cost=[0.001, 0.001],
    initial_capital=1.0,
    benchmark_weights=None
)
```

**参数：**
- `weights_data` (pd.DataFrame): 权重数据
- `price_data` (pd.DataFrame): 价格数据
- `buy_price` (str): 买入价格字段名
- `sell_price` (str): 卖出价格字段名
- `adj_factor_col` (str): 复权因子字段名
- `date_col` (str): 日期列名，默认 'date'
- `asset_col` (str): 资产列名，默认 'code'
- `weight_col` (str): 权重列名，默认 'weight'
- `rebalance_threshold` (float): 调仓阈值，默认 0.01
- `transaction_cost` (list): [买入成本, 卖出成本]，默认 [0.001, 0.001]
- `initial_capital` (float): 初始资金，默认 1.0
- `benchmark_weights` (pd.DataFrame): 基准权重数据，可选

**返回：**
- dict: 回测结果字典

#### get_metrics
```python
get_metrics() -> pd.DataFrame
```

返回性能指标表。

#### get_trade_analysis
```python
get_trade_analysis() -> pd.DataFrame
```

返回交易记录分析。

#### 可视化方法

- `plot_nav_curve()` - 净值曲线
- `plot_drawdown()` - 回撤曲线
- `plot_nav_vs_benchmark()` - 策略 vs 基准
- `plot_excess_returns()` - 超额收益
- `plot_trade_points()` - 交易点位分析
- `plot_position_heatmap()` - 持仓热力图
- `plot_turnover_analysis()` - 换手率分析
- `plot_all()` - 综合展示面板

## 文件结构

```
GeneralBacktest/
├── __init__.py          # 包初始化文件
├── backtest.py          # 核心回测类
├── utils.py             # 辅助函数和指标计算
├── example.py           # 使用示例
└── README.md            # 本文件
```

## 性能优化建议

1. **向量化优先**：框架已经使用向量化操作，避免手动循环
2. **数据预处理**：提前准备好标准格式的数据
3. **减少调仓频率**：频繁调仓会增加计算量和交易成本
4. **资产数量**：大量资产时，热力图会自动只显示前20个

## 注意事项

1. **数据质量**：确保价格数据和权重数据的日期对齐
2. **复权因子**：正确设置复权因子以准确计算收益
3. **交易成本**：根据实际情况设置合理的交易成本
4. **滑点**：该框架暂未考虑滑点，可通过调整交易成本模拟

## 扩展功能

框架设计为可扩展的，你可以：

1. 在 `utils.py` 中添加自定义指标
2. 在 `GeneralBacktest` 类中添加新的可视化方法
3. 扩展数据预处理逻辑
4. 添加风险控制模块

## 常见问题

### Q1: 如何处理股票停牌？
A: 将停牌期间的价格设为前一日价格，复权因子保持不变。

### Q2: 如何设置只做多策略？
A: 确保权重数据中所有权重都 ≥ 0。

### Q3: 如何实现日内交易？
A: 当前版本支持日频回测。如需日内交易，需要修改为分钟级数据。

### Q4: 基准如何设置？
A: 通过 `benchmark_weights` 参数传入基准的权重数据，格式与策略权重相同。

## 许可证

MIT License

## 作者

量化交易系统开发团队

## 更新日志

### v1.0.0 (2024)
- ✅ 初始版本发布
- ✅ 支持灵活调仓
- ✅ 向量化计算
- ✅ 15+性能指标
- ✅ 8+可视化图表
- ✅ 调仓日盘中收益计算
