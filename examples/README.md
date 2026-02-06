# Examples 示例代码

本目录包含 GeneralBacktest 框架的示例代码。

## 📁 文件说明

### 1. basic_example.py - 基础示例

最简单的使用示例，适合快速入门。

**特点**：
- 使用随机生成的股票数据
- 5 只股票，季度调仓
- 展示基本回测流程

**运行**：
```bash
cd examples
python basic_example.py
```

**生成图表**：
- `basic_nav_curve.png` - 净值曲线
- `basic_monthly_returns.png` - 月度收益
- `basic_positions.png` - 持仓变化

---

### 2. advanced_example.py - 高级示例

策略对比和完整分析。

**特点**：
- 动量策略 vs 等权基准
- 10 只股票，3 年数据
- 导出 Excel 报告

**运行**：
```bash
cd examples
python advanced_example.py
```

**生成输出**：
- `advanced_dashboard.png` - 综合仪表板
- `advanced_comparison.png` - 策略对比
- `advanced_excess_returns.png` - 超额收益
- `advanced_turnover.png` - 换手率
- `strategy_analysis.xlsx` - 详细数据

---

### 3. complete_demo.py - v1.1.0 完整演示（推荐）

综合展示 v1.1.0 所有功能：3种回测方式 + 所有图表。

**演示内容**：
1. **标准回测**：理论最优（无交易限制）
2. **现金回测**：实盘模拟（100万资金，每手100股）
3. **总仓位控制**：动态仓位（择时策略）
4. **完整可视化**：所有图表（支持对数/双坐标）

**运行**：
```bash
cd examples
python complete_demo.py
```

**生成文件**（`output_demo/`）：

**图表**（PNG）：
- `demo_standard_*.png` - 标准回测（净值、月度、持仓、Dashboard）
- `demo_cash_*.png` - 现金回测（净值、换手率、Dashboard）
- `demo_log_nav.png` - 对数坐标
- `demo_dual_nav.png` - 双坐标对比

**数据**（CSV）：
- `demo_standard_nav_series.csv` - 标准净值序列
- `demo_cash_nav_series.csv` - 现金净值序列
- `demo_cash_series.csv` - 现金余额
- `demo_cash_trades.csv` - 交易记录

**v1.1.0 新增功能**：
- `position_ratio_col` - 动态仓位控制
- `run_backtest_with_cash()` - 现金回测
- `log_scale=True` - 对数坐标
- `plot_nav_curve_dual()` - 双坐标

**运行时间**：2-3分钟

---

## 🚀 快速开始

```bash
# 1. 基础示例（入门）
python examples/basic_example.py

# 2. 高级示例（策略对比）
python examples/advanced_example.py

# 3. 完整演示（v1.1.0 所有功能）
python examples/complete_demo.py
```

**推荐顺序**：`basic_example.py` → `advanced_example.py` → `complete_demo.py`

---

## 📊 输出说明

所有输出保存在 `output_demo/` 目录。

---

## 💡 自定义示例

基于示例创建自己的回测：

1. **修改数据**：股票数量、时间范围、波动率
2. **调整策略**：调仓频率、权重分配、交易成本
3. **实验场景**：牛市/熊市、高/低波动

---

## 🔧 使用真实数据

```python
# 价格数据格式
price_data = pd.DataFrame({
    'date': [...],       # 日期
    'code': [...],       # 股票代码
    'open': [...],       # 开盘价
    'close': [...],      # 收盘价
    'adj_factor': [...]  # 复权因子
})

# 权重数据格式
weights_data = pd.DataFrame({
    'date': [...],       # 调仓日期
    'code': [...],       # 股票代码
    'weight': [...]      # 目标权重
})

# 运行回测
bt = GeneralBacktest(start_date='2023-01-01', end_date='2023-12-31')
results = bt.run_backtest(
    weights_data=weights_data,
    price_data=price_data,
    buy_price='open',
    sell_price='close',
    adj_factor_col='adj_factor',
    close_price_col='close',
    rebalance_threshold=0.005,
    transaction_cost=[0.001, 0.001]
)
```

---

## ❓ 常见问题

**Q: v1.1.0 有哪些新功能？**
- 总仓位控制（`position_ratio_col`）
- 现金回测（`run_backtest_with_cash()`）
- 可视化增强（对数坐标、双坐标）
- 100%向后兼容

**Q: v1.0 代码需要修改吗？**
- **不需要**！不传入新参数时行为完全一致

**Q: 什么时候用现金回测？**
- 实盘模拟、资金规划、成本计算时使用
- 快速策略研究用标准回测即可

**Q: 图表不显示？**
- 确保安装 matplotlib
- 或使用 `save_path` 参数保存

---

## 📚 更多资源

- [完整文档](../README.md)
- [PyPI 上传指南](../PYPI_UPLOAD_GUIDE.md)
- [更新日志](../Update_changelog.md) - v1.1.0 详细说明

---

**祝你使用愉快！** 🎉

**最新版本**: v1.1.0
