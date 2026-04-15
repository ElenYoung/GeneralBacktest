# Output Demo

本目录包含 GeneralBacktest 框架各示例脚本生成的输出文件，按功能分类存放。

## 📁 目录结构

```
output_demo/
├── README.md              ← 本文件
├── standard_demo/         ← 标准回测 + 基础示例输出
├── cash_demo/            ← 现金仓位回测输出
├── advanced_demo/         ← 策略对比与基准分析输出
├── t0_demo/              ← T+0 日内回转回测输出
└── .gitkeep
```

## 🚀 快速开始

运行任意示例脚本生成输出：

```bash
cd examples
python t0_example.py          # T+0 回测示例
python complete_demo.py       # 标准 + 现金回测
python advanced_example.py   # 策略对比分析
python basic_example.py       # 基础示例
```

---

## 📂 standard_demo/

**对应脚本**：`complete_demo.py`（标准回测部分）+ `basic_example.py`

运行 `complete_demo.py` 和 `basic_example.py` 后生成：

| 文件 | 类型 | 说明 |
|------|------|------|
| `basic_nav_curve.png` | 图表 | 基础净值曲线 |
| `basic_monthly_returns.png` | 图表 | 基础月度收益热力图 |
| `basic_positions.png` | 图表 | 基础持仓热力图 |
| `demo_standard_nav.png` | 图表 | 标准回测净值曲线 |
| `demo_standard_dashboard.png` | 图表 | 标准回测综合仪表板 |
| `demo_standard_monthly.png` | 图表 | 标准月度收益热力图 |
| `demo_standard_positions.png` | 图表 | 标准持仓权重热力图 |
| `demo_standard_nav_series.csv` | 数据 | 标准回测每日净值 |

**关键指标示例**（随机数据，每次运行不同）：
- 累计收益：~202%
- 年化收益：~70%
- 夏普比率：~1.95

---

## 📂 cash_demo/

**对应脚本**：`complete_demo.py`（现金回测部分）

运行 `complete_demo.py` 后生成：

| 文件 | 类型 | 说明 |
|------|------|------|
| `demo_cash_nav.png` | 图表 | 现金回测净值曲线（实际金额） |
| `demo_cash_dashboard.png` | 图表 | 现金回测综合仪表板 |
| `demo_cash_turnover.png` | 图表 | 换手率分析 |
| `demo_cash_nav_series.csv` | 数据 | 现金回测每日净值（¥1,000,000 → ¥3,030,950）|
| `demo_cash_series.csv` | 数据 | 每日现金余额 |
| `demo_cash_trades.csv` | 数据 | 完整交易明细 |

**特点**：模拟实盘，包含资金限制和交易成本，结果略低于标准回测。

---

## 📂 advanced_demo/

**对应脚本**：`advanced_example.py`

运行 `advanced_example.py` 后生成：

| 文件 | 类型 | 说明 |
|------|------|------|
| `advanced_dashboard.png` | 图表 | 策略综合仪表板 |
| `advanced_comparison.png` | 图表 | 策略 vs 基准净值对比 |
| `advanced_excess_returns.png` | 图表 | 超额收益分析 |
| `advanced_turnover.png` | 图表 | 换手率分析 |
| `advanced_monthly_returns.png` | 图表 | 月度收益热力图 |
| `strategy_analysis.xlsx` | 数据 | 性能指标 + 净值序列 + 调仓记录 + 换手率（需 openpyxl）|

**策略**：动量策略（月度调仓，选过去60天涨幅前5名），基准为等权组合。

---

## 📂 t0_demo/

**对应脚本**：`t0_example.py`（新增）

运行 `t0_example.py` 后生成：

| 文件 | 类型 | 说明 |
|------|------|------|
| `t0_intraday_trades.png` | 图表 | NAV 曲线 + 日内买卖点标注 |
| `t0_returns_breakdown.png` | 图表 | 卖出收益 vs 买入收益拆分（堆叠柱状图）|
| `t0_dashboard.png` | 图表 | T+0 综合仪表板 |
| `t0_vs_benchmark.png` | 图表 | T+0 策略 vs 纯持有基准对比 |
| `t0_nav_series.csv` | 数据 | 每日净值序列 |
| `t0_intraday_records.csv` | 数据 | 日内交易记录（含每阶段收益和佣金）|

**T+0 策略说明**：

```
策略设计：
  - 股票：VOLATILE_A（高波动）
  - 建仓：t 日收盘买入，目标 100% 持仓
  - 做 T：每隔 3 天执行一次 T+0
      - sell_phase：开盘卖出 50%，目标持仓降至 50%
      - buy_phase：收盘买回 50%，目标持仓恢复 100%
  - 买入价：close，收盘价
  - 卖出价：open，开盘价

权重数据格式（phase 列 = 目标仓位模式）：
  date        | code       | weight | phase
  2024-01-02  | VOLATILE_A | 1.0    | NaN    ← 收盘建仓
  2024-01-05  | VOLATILE_A | 0.5    | sell   ← 开盘卖出 50%
  2024-01-05  | VOLATILE_A | 1.0    | buy    ← 收盘买回 50%
  2024-01-08  | VOLATILE_A | 1.0    | NaN    ← 持有不动
```

**A股合规校验**：
- ✅ buy_phase 必须在 sell_phase 之后
- ✅ 同一天总卖出量 ≤ 总买入量（不允许裸做空）
- ✅ 目标仓位在 [0, 1] 范围内
- ✅ 无重复 (date, asset, phase) 记录

**T+0 专用指标**：
- 卖出胜率 / 买入胜率
- 卖出累计收益 / 买入累计收益
- 卖出收益贡献占比 / 买入收益贡献占比
- 平均佣金率

---

## 📊 示例脚本一览

| 脚本 | 回测类型 | 运行时间 | 输出目录 |
|------|---------|---------|---------|
| `t0_example.py` | T+0 日内回转 | ~5s | t0_demo/ |
| `complete_demo.py` | 标准 + 现金 + 动态仓位 | ~2-3min | standard_demo/ + cash_demo/ + advanced_demo/ |
| `basic_example.py` | 标准（基础） | ~10s | standard_demo/ |
| `advanced_example.py` | 策略对比 + 基准 | ~2min | advanced_demo/ |

---

## 🔧 数据文件格式

### NAV 序列（csv）

```csv
date,nav
2024-01-02,1.000000
2024-01-03,1.001234
```

### 日内交易记录（csv）

```csv
date,phase,return,commission,target_weights
2024-01-05,sell,0.00498,0.00050,"{'VOLATILE_A': 0.5}"
2024-01-05,buy,-0.00050,0.00050,"{'VOLATILE_A': 1.0}"
```

### 交易明细（csv）

```csv
date,code,direction,price,shares,amount,commission
2024-01-05,VOLATILE_A,BUY,51.25,980,50225.00,50.23
```

---

**最后更新**：2026-04-15 (v1.2.0, 新增 T+0 日内回转回测支持)

**框架版本**：GeneralBacktest v1.2.0
