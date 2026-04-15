# Data Timing And No-Lookahead Rules

This note is used by the skill to prevent lookahead bias.

## Timeline Convention

Assume rebalance day is `t`.

1. Signal generation window:
- Use data up to `t-1` (inclusive) to generate target weights for `t`.

2. Execution window:
- Execute target weights on day `t` using day-`t` tradable prices (`open`, `vwap`, or same-day execution convention).

3. Valuation window:
- Day `t` NAV and decomposition should use day-`t` close within the same day scope.

## Correct Example

- Target weights at `2023-06-01` are computed from data ending at `2023-05-31`.
- Rebalance at `2023-06-01` uses `open`/`close` from `2023-06-01`.

## Incorrect Example (Lookahead)

- Target weights at `2023-06-01` computed using `2023-06-01` close.
- Or using `2023-06-02` close as the sell price for `2023-06-01` rebalance-day decomposition.

## Why Incorrect Timing Is Dangerous

In this framework, rebalance-day return is decomposed into sold/kept/bought contributions under a same-day structure.
If trade prices move outside day-`t`, the decomposition can become temporally inconsistent and generate distorted daily returns.

## Practical Agent Checklist

- [ ] Weight at `t` is generated from `t-1` or earlier data.
- [ ] Buy/sell/close prices used for rebalance are from day `t`.
- [ ] No `t+1` fields are mixed into day-`t` execution or PnL decomposition.
