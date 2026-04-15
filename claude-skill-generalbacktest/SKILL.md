---
name: generalbacktest-standard-workflow
description: Enforce a consistent, safe, reproducible, and no-lookahead workflow for GeneralBacktest, including cash-mode execution, visualization playbooks, and known limitations.
---

# GeneralBacktest Skill

## Purpose

Use this skill whenever a task requires running, validating, analyzing, or explaining workflows built on `GeneralBacktest`.

Primary goals:

- Enforce reproducible and no-lookahead backtesting behavior.
- Match method choice to strategy realism requirements.
- Standardize outputs for metrics, charts, caveats, and assumptions.
- Make known framework limitations explicit before interpretation.

## Trigger Conditions

Activate this skill for requests containing intents such as:

- "回测", "backtest", "净值", "调仓", "换手率", "现金仓位", "基准对比"
- Running `run_backtest` / `run_backtest_with_cash` / ETF / stock DB APIs
- Backtest plotting, reporting, diagnostics, or result explanation

## Code-Accurate API Facts (Must Follow)

Based on current implementation:

1. `run_backtest(...)` requires `adj_factor_col` and internally computes adjusted buy/sell/close prices.
2. `run_backtest_with_cash(...)` does not use `adj_factor_col`; it simulates holdings in shares and cash.
3. `run_backtest_ETF(...)` / `run_backtest_stock(...)` fetch daily data from DB and call `run_backtest(...)`.
4. `plot_all()` is an alias of `plot_dashboard()`.
5. `plot_nav_curve(log_scale=True)` and `plot_nav_curve_dual()` are supported.

## Hard Rules

1. Never run backtest on unvalidated input data.
2. Never hide timing assumptions. Explicitly declare signal-to-trade timing.
3. Always declare method selection reason (`standard` vs `cash` vs `DB`).
4. Do not change defaults unless user asks or constraints require.
5. Never fabricate output metrics, files, or plots.

## Input Contracts

### A. Weights Data

Required columns:

- `date`
- `code`
- `weight`

Optional columns:

- `position_ratio` (for dynamic total exposure in `run_backtest`)

Validation:

- `date` parseable as datetime.
- no nulls in required columns.
- weights per date must be interpretable (recommended non-negative long-only unless user explicitly specifies otherwise).
- for cash mode: per-date weight sum should be approximately 1.

### B. Price Data

For `run_backtest` required columns:

- `date`, `code`
- buy price column (for `buy_price`)
- sell price column (for `sell_price`)
- close price column (for `close_price_col`)
- `adj_factor_col` (mandatory in current code)

For `run_backtest_with_cash` required columns:

- `date`, `code`
- buy/sell/close columns

Validation:

- no duplicate (`date`, `code`) rows.
- numeric and positive prices.
- date range covers backtest interval.

## Anti-Lookahead Timing Specification (Critical)

This section is mandatory in every serious backtest task.

### Signal and Weights Timing

- `date = t` weights are target holdings to be executed on day `t`.
- Therefore, weights for day `t` must be generated only from data up to `t-1` (or earlier).
- Any use of day-`t` close (or later) to compute day-`t` target weights is lookahead leakage unless strategy explicitly models intraday signal availability.

### Trading Price Timing

- Trading prices must come from day `t` intraday scope to match "rebalance on t" semantics.
- In this framework, rebalance-day return decomposition is tied to same-day buy/sell/close logic.
- Do not use `t+1` prices as day-`t` execution prices when reporting day-`t` holding return.

### Why This Matters

If sell/buy prices are outside rebalance day scope (for example using next-day close as sell price), the decomposition of:

- kept return
- sold return
- bought intraday return

becomes temporally inconsistent and can produce distorted daily PnL.

## Method Selection Decision Tree

1. Need realistic capital constraints, lot trading, or cash tracking:
- choose `run_backtest_with_cash()`.

2. Need DB-native ETF/stock workflow and valid DB config is provided:
- choose `run_backtest_ETF()` or `run_backtest_stock()`.

3. Otherwise:
- choose `run_backtest()`.

## Cash-Mode Guide (`run_backtest_with_cash`)

Use when strategy requires execution realism:

- finite cash
- minimum lot size
- partial fill due to cash shortage
- order priority (`trade_critic`)

Recommended default parameters:

- `lot_size=100`
- `trade_critic='weight_desc'`
- `transaction_cost=[0.001, 0.001]`
- `slippage=0.0005` (or user-defined)

Must report in cash-mode results:

- final NAV
- final cash
- final cash ratio
- average cash ratio
- turnover stats

## Visualization Playbook

### Minimum chart set

- NAV + drawdown: `plot_nav_curve()`
- comprehensive dashboard: `plot_all()`

### When to use log scale

- long horizon or large NAV growth dispersion:
    - `plot_nav_curve(log_scale=True)`

### When to compare trend shape under different scale

- use `plot_nav_curve_dual()`

### Benchmark analysis charts

- if benchmark exists:
    - `plot_nav_vs_benchmark()`
    - `plot_excess_returns()`

### Trading and allocation diagnostics

- turnover: `plot_turnover_analysis()`
- holdings heatmap: `plot_position_heatmap()`
- trade points/cost: `plot_trade_points()`

## Known Limitations (Must Be Disclosed)

The agent must communicate these constraints when relevant:

1. No true intraday bar engine
- Core engine is daily-based; it is not an event-driven intraday simulator.

2. Not suitable for T strategy (same-day round-trip / intraday scalping logic)
- Current design assumes one rebalance decision per day with day-level decomposition.
- It does not model multiple intraday signal-trade cycles.

3. Standard mode is weight-based, not order-book execution
- `run_backtest()` uses weight transitions and adjusted prices, not discrete share-level fill simulation.

4. Cash benchmark is simplified
- In cash mode, benchmark computation is simplified and does not fully mirror cash constraints.

5. Data quality dependency
- Missing/abnormal prices can materially distort results if not cleaned in advance.

## Execution Workflow

### Step 1. Clarify objective

Collect:

- backtest date range
- universe
- rebalance frequency and trigger
- cost/slippage assumptions
- benchmark need
- whether strategy requires cash realism

### Step 2. Validate data and timing

Check:

- schema completeness
- nulls/duplicates
- numeric price sanity
- per-date weight sanity
- no-lookahead timing compliance

If any check fails:

- stop run
- report exact failing rule
- provide minimal correction example

### Step 3. Run and capture outputs

Required capture:

- `nav_series`
- `metrics`
- `trade_records` (if available)
- `turnover_records` (if available)
- `cash_series` (cash mode)

### Step 4. Report in standard format

1. Setup
- method and why
- key params
- timing assumptions

2. Validation summary
- passed checks
- warnings

3. Performance summary
- cumulative return
- annualized return
- annualized vol
- max drawdown
- Sharpe

4. Additional analytics
- turnover
- cash metrics (if cash mode)
- benchmark-relative metrics

5. Reproducibility
- entry script/function
- generated files

6. Limitations
- strategy-framework mismatch notes

## Examples In This Skill

- `examples/cash_backtest_template.py`
- `examples/visualization_workflow.py`
- `examples/data_timing_and_no_lookahead.md`

## Completion Checklist

Before finishing a task:

- [ ] Correct method selected and justified
- [ ] Input schema validated
- [ ] Timing/no-lookahead assumptions explicitly stated
- [ ] Metrics come from actual run outputs
- [ ] Limitations clearly disclosed
- [ ] Reproducible code path and output files provided
