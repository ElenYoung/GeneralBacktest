# Output Demo

本目录包含 GeneralBacktest 框架生成的示例输出图表。

## 📊 当前包含的图表

### 基础示例输出
- `basic_nav_curve.png` - 净值曲线和回撤分析
- `basic_monthly_returns.png` - 月度收益热力图
- `basic_positions.png` - 持仓变化热力图

## 🚀 生成更多图表

运行示例代码可以生成更多图表：

```bash
# 运行基础示例
python examples/basic_example.py

# 运行高级示例（生成更多图表）
python examples/advanced_example.py
```

高级示例会生成：
- 综合分析仪表板
- 策略与基准对比
- 超额收益分析
- 换手率分析
- Excel 数据报告等

## 📝 注意事项

- 这些图表是使用随机生成的数据创建的
- 每次运行示例都会覆盖现有图表
- 仅作为演示用途，实际策略表现会有所不同

## 🎨 自定义输出

你可以在代码中自定义图表的保存路径：

```python
bt.plot_nav_curve(save_path='custom_path/my_chart.png')
```
