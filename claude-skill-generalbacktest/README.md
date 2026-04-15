# claude-skill-generalbacktest

This directory contains an Anthropic-style skill for standardizing how an agent uses GeneralBacktest.

## Files

- `SKILL.md`: Main skill definition and workflow contract.
- `examples/cash_backtest_template.py`: Cash-mode backtest template.
- `examples/visualization_workflow.py`: Visualization workflow template.
- `examples/data_timing_and_no_lookahead.md`: Timing rules to avoid lookahead bias.

## Suggested Placement

If you want Claude-style skill auto-discovery in a local Claude workspace, place this folder under:

- `.claude/skills/generalbacktest/`

and keep `SKILL.md` as the entry file.

## What This Skill Enforces

- Input schema validation before execution
- Correct backtest method selection
- Standard output structure for metrics and assumptions
- Explicit error handling and reproducibility requirements
- No-lookahead timing constraints for weights and trade prices
- Mandatory disclosure of framework limitations (including T strategy mismatch)
