# Examples 示例代码

本目录包含 GeneralBacktest 框架的示例代码。

## 📁 文件说明

### 1. basic_example.py - 基础示例

最简单的使用示例，适合快速入门。

**特点**：
- 使用随机生成的股票数据
- 5 只模拟股票
- 季度调仓策略
- 展示基本的回测流程

**运行**：
```bash
cd examples
python basic_example.py
```

**生成的图表**：
- `basic_nav_curve.png` - 净值曲线
- `basic_monthly_returns.png` - 月度收益热力图
- `basic_positions.png` - 持仓变化

---

### 2. advanced_example.py - 高级示例

展示更复杂的策略对比和分析。

**特点**：
- 动量策略 vs 等权基准
- 10 只股票，3 年数据
- 完整的性能分析和对比
- 导出 Excel 报告

**运行**：
```bash
cd examples
python advanced_example.py
```

**生成的输出**：
- `advanced_dashboard.png` - 综合仪表板
- `advanced_comparison.png` - 策略对比
- `advanced_excess_returns.png` - 超额收益分析
- `advanced_turnover.png` - 换手率分析
- `advanced_return_dist.png` - 收益分布
- `strategy_analysis.xlsx` - 详细数据（需要 openpyxl）

---

## 🚀 快速开始

### 运行所有示例

```bash
# 基础示例
python examples/basic_example.py

# 高级示例
python examples/advanced_example.py
```

### 安装可选依赖

如果想导出 Excel 报告：

```bash
pip install openpyxl
```

---

## 📊 输出说明

所有示例生成的图表和报告都保存在 `output_demo/` 目录中。

---

## 💡 自定义示例

你可以基于这些示例创建自己的回测：

1. **修改数据生成参数**：
   - 股票数量
   - 时间范围
   - 价格波动率

2. **调整策略参数**：
   - 调仓频率
   - 权重分配方式
   - 交易成本和滑点

3. **实验不同场景**：
   - 牛市 / 熊市
   - 高波动 / 低波动
   - 不同的交易成本

---

## 🔧 使用真实数据

如果你有真实的价格数据，可以参考以下格式：

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

### Q: 为什么使用随机数据？
A: 示例使用随机数据是为了：
- 不依赖外部数据源
- 可以在任何环境运行
- 便于测试和学习

### Q: 如何获取真实数据？
A: 你可以：
- 使用 `tushare`, `akshare` 等免费数据源
- 连接数据库（需要配置 `db_config`）
- 使用 CSV 文件导入

### Q: 图表不显示怎么办？
A: 确保：
- 安装了 matplotlib
- 使用 `save_path` 参数保存图片
- 或在代码末尾添加 `plt.show()`

---

## 📚 更多资源

- [完整文档](../README.md)
- [PyPI 上传指南](../PYPI_UPLOAD_GUIDE.md)
- [项目总结](../PROJECT_SUMMARY.md)

---

**祝你使用愉快！** 🎉
