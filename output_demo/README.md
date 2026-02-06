# Output Demo

本目录包含 GeneralBacktest 框架生成的示例输出图表和数据文件。

## 📊 v1.1.0 完整演示输出

运行 `complete_demo.py` 生成的完整演示文件（推荐查看）：

### 核心图表（9个PNG文件）
- **`demo_standard_nav.png`** - 标准回测净值曲线（归一化）
- **`demo_cash_nav.png`** - 现金回测净值曲线（实际金额）
- **`demo_log_nav.png`** - 对数坐标净值曲线（避免初始平坦段）
- **`demo_dual_nav.png`** - 线性/对数坐标对比图
- **`demo_standard_monthly.png`** - 月度收益热力图
- **`demo_standard_positions.png`** - 持仓权重热力图
- **`demo_cash_turnover.png`** - 换手率分析
- **`demo_standard_dashboard.png`** - 标准回测综合仪表板
- **`demo_cash_dashboard.png`** - 现金回测综合仪表板

### 数据文件（4个CSV文件）
- **`demo_standard_nav_series.csv`** - 标准回测每日NAV
- **`demo_cash_nav_series.csv`** - 现金回测每日NAV（¥1,000,000 → ¥3,030,950）
- **`demo_cash_series.csv`** - 现金余额变化
- **`demo_cash_trades.csv`** - 完整交易记录（6笔交易）

**📖 详细说明**: 查看 [README_v1.1.0.md](./README_v1.1.0.md) 获取完整文档

## 📊 基础示例输出

运行 `basic_example.py` 生成的图表：
- `basic_nav_curve.png` - 净值曲线和回撤分析
- `basic_monthly_returns.png` - 月度收益热力图
- `basic_positions.png` - 持仓变化热力图

## 🚀 生成图表

运行示例代码生成图表：

```bash
# v1.1.0 完整演示（推荐）
python examples/complete_demo.py

# 基础示例
python examples/basic_example.py

# 高级示例
python examples/advanced_example.py
```

## 📝 注意事项

- 所有图表使用随机生成的模拟数据
- 每次运行示例会覆盖现有图表
- 仅作为演示用途，实际策略表现会有所不同
- v1.1.0 解决了之前版本的图表显示问题（水平线、弹窗、中文编码）

## 🎨 自定义输出

在代码中自定义图表保存路径：

```python
bt.plot_nav_curve(save_path='custom_path/my_chart.png')
```
