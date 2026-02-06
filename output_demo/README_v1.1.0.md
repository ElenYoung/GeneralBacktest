# GeneralBacktest v1.1.0 完整演示输出

本目录包含 `complete_demo.py` 生成的 v1.1.0 完整演示输出文件。

## 📁 文件说明

### 标准回测输出（Standard Backtest）

标准回测采用理论最优模式，无交易限制，收益归一化从1.0开始。

| 文件名 | 说明 | 关键指标 |
|--------|------|---------|
| `demo_standard_nav.png` | 净值曲线与回撤分析 | 累计收益: 202.15% |
| `demo_standard_monthly.png` | 月度收益热力图 | 显示每月收益率分布 |
| `demo_standard_positions.png` | 持仓权重热力图 | 展示持仓变化过程 |
| `demo_standard_dashboard.png` | 综合性能仪表板 | 包含9项核心指标 |
| `demo_standard_nav_series.csv` | 净值序列数据 | 每日NAV数值 |

### 现金仓位回测输出（Cash Backtest）

现金回测模拟实盘交易，包含资金、股数限制，显示实际金额。

| 文件名 | 说明 | 关键指标 |
|--------|------|---------|
| `demo_cash_nav.png` | 净值曲线（实际金额） | 最终净值: ¥3,030,950.18 |
| `demo_cash_turnover.png` | 换手率分析 | 显示交易成本影响 |
| `demo_cash_dashboard.png` | 现金回测仪表板 | 包含实际资金分析 |
| `demo_cash_nav_series.csv` | 净值序列（实际金额） | 从¥1,000,000开始 |
| `demo_cash_series.csv` | 现金余额序列 | 每日现金变化 |
| `demo_cash_trades.csv` | 交易记录 | 完整的买卖记录 |

### 可视化增强输出（Visualization）

v1.1.0新增的对数坐标和双坐标对比功能，避免初始水平线问题。

| 文件名 | 说明 | 特点 |
|--------|------|------|
| `demo_log_nav.png` | 对数坐标净值曲线 | 从首次调仓开始，避免初始平坦段 |
| `demo_dual_nav.png` | 双坐标对比图 | 同时展示线性/对数坐标效果 |

## 🎯 对比分析

### 标准回测 vs 现金回测

| 对比维度 | 标准回测 | 现金回测 |
|---------|---------|---------|
| **初始资金** | 1.0（归一化） | ¥1,000,000 |
| **最终净值** | 3.02 | ¥3,030,950.18 |
| **总收益率** | 202.15% | 203.10% |
| **调仓次数** | 8次 | 6次（受交易限制）|
| **夏普比率** | 1.955 | 1.968 |
| **特点** | 理论最优 | 贴近实盘 |

### 可视化坐标选择指南

- **线性坐标**：短期（<1年）或向客户展示时使用，更直观
- **对数坐标**：长期（>1年）分析使用，能准确反映相对收益变化
- **双坐标对比**：策略报告使用，展示不同视角

## 📊 数据文件说明

### CSV文件格式

1. **demo_standard_nav_series.csv**
   ```
   date, nav
   2023-01-02, 1.000000
   2023-01-03, 1.001234
   ```

2. **demo_cash_nav_series.csv**
   ```
   date, nav
   2023-01-02, 1000000.00
   2023-01-03, 1000123.45
   ```

3. **demo_cash_series.csv**（每日现金余额）
   ```
   date, cash
   2023-01-02, 1000000.00
   ```

4. **demo_cash_trades.csv**（交易明细）
   ```
   date, code, direction, price, shares, amount, commission
   2023-03-31, AAPL, BUY, 150.25, 100, 15025.00, 15.03
   ```

## 🔍 分析技巧

### 1. 评估动态仓位效果

查看 `demo_cash_series.csv` 中的现金占比变化：
```python
import pandas as pd

# 读取数据
df = pd.read_csv('demo_cash_series.csv', index_col=0, parse_dates=True)

# 计算平均现金占比
cash_ratio = 1 - (df['nav'] - df['cash']) / df['nav']
print(f"平均现金占比: {cash_ratio.mean():.1%}")
```

### 2. 对比回测差异

分析标准回测和现金回测的净值差异：
```python
import pandas as pd

# 读取两种回测结果
standard = pd.read_csv('demo_standard_nav_series.csv', index_col=0, parse_dates=True)
cash = pd.read_csv('demo_cash_nav_series.csv', index_col=0, parse_dates=True)

# 对齐时间并计算差异
aligned = pd.concat([standard['nav'], cash['nav'] / 1000000], axis=1)
aligned.columns = ['standard', 'cash']
aligned['diff'] = aligned['cash'] - aligned['standard']

print(f"最大差异: {aligned['diff'].abs().max():.4f}")
```

### 3. 分析交易成本影响

查看 `demo_cash_trades.csv` 中的交易记录：
```python
import pandas as pd

trades = pd.read_csv('demo_cash_trades.csv')
total_commission = trades['commission'].sum()
print(f"总交易成本: ¥{total_commission:,.2f}")
```

## 💡 v1.1.0 核心改进

### 解决的主要问题

1. **水平线问题**：早期图表在1月初至3月底显示为水平线（无持仓）
   - **原因**：首次调仓日在3月31日，之前无持仓
   - **解决**：对数/双坐标图表从首次调仓日开始显示

2. **交互式弹窗**：之前保存图表时会弹出窗口
   - **解决**：传递 `save_path` 时直接保存，无弹窗

3. **中文显示异常**：部分系统中文标题显示为方框
   - **解决**：改用英文标题，确保跨平台兼容性

## 🚀 如何运行演示

```bash
cd examples
python complete_demo.py
```

运行时间：约2-3分钟

输出位置：`output_demo/` 目录

## 📈 性能表现（示例数据）

基于生成的随机数据（2023-2024）：

| 回测类型 | 累计收益 | 年化收益 | 最大回撤 | 夏普比率 |
|---------|---------|----------|----------|----------|
| 标准回测 | 202.15% | 70.72% | 24.55% | 1.955 |
| 现金回测 | 203.10% | 71.08% | 24.58% | 1.968 |
| 动态仓位 | 10.53% | 21.60% | 10.52% | 1.155 |

*注：动态仓位回测使用2023上半年数据，与完整期间不可直接比较*

## 📚 更多资源

- **示例代码**: `examples/complete_demo.py`
- **框架文档**: `README.md`
- **更新日志**: `Update_changelog.md`
- **PyPI指南**: `PYPI_UPLOAD_GUIDE.md`

---

**最后更新**: 2026-02-07 (v1.1.0)

**框架版本**: GeneralBacktest v1.1.0


---

## 🚀 如何查看

1. **查看图片**
   - 使用任何图片查看器
   - 或使用 Markdown 查看器查看本文档

2. **查看数据CSV**
   - 使用 Excel / LibreOffice Calc
   - 或使用 pandas: `pd.read_csv('filename.csv')`

3. **查看指南**
   - 打开 `visualization_guide.md`
   - 包含对数坐标的详细使用说明

---

## 💡 亮点展示

### v1.1.0 主要改进

1. **动态仓位控制**
   - 可以根据市场状态灵活调整现金比例
   - 对比: `position_ratio_nav.png` 显示了不同阶段的仓位效果

2. **现金回测更贴近实盘**
   - 对比: `comparison_standard_vs_cash.csv` 显示了理论回测和现金回测的差异
   - 通常现金回测的收益会略低于标准回测（考虑了交易限制）

3. **对数坐标更适合长周期**
   - 对比: `scale_normal.png` vs `scale_log.png`
   - 对数坐标下，恒定的收益率显示为直线

4. **双坐标一键生成**
   - `scale_dual.png` 同时展示普通和对数坐标
   - 适合完整的策略报告

---

## 🔍 分析技巧

### 评估动态仓位效果

查看 `position_ratio_analysis.csv`：
```python
import pandas as pd

# 读取数据
df = pd.read_csv('position_ratio_analysis.csv')

# 统计不同市场阶段的现金占比
print(df.groupby('market_regime')['cash_ratio'].mean())
```

### 对比标准 vs 现金回测

查看 `nav_comparison.csv`：
```python
import pandas as pd

# 读取数据
df = pd.read_csv('nav_comparison.csv')

# 计算差异
df['nav_diff'] = df['cash_nav'] - df['standard_nav']
print(f"平均差异: {df['nav_diff'].mean():,.2f}")
```

### 选择正确的坐标

- **短期（<1年）**：普通坐标即可
- **长期（>3年）**：推荐使用对数坐标
- **大幅波动**：对数坐标更清晰
- **展示给客户**：普通坐标更直观

---

## 📊 性能对比（示例数据）

基于本目录生成的示例数据：

| 回测类型 | 累计收益 | 年化收益 | 最大回撤 | 特点 |
|---------|---------|----------|----------|------|
| 标准回测 | ~35% | ~12% | ~15% | 理论最优 |
| 现金回测 | ~32% | ~11% | ~15% | 贴近实盘 |
| 动态仓位 | ~30% | ~10% | ~12% | 风险更低 |

*注：具体数值因随机数据而异，但趋势一致*

---

**最后更新**: 2026-02-06 (v1.1.0)

**文档位置**: [../examples/README.md](../examples/README.md)

**完整文档**: [../README.md](../README.md)
