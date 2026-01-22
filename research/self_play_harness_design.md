# Self-Play Regression Harness Design

## Goals
- Compare current bot vs previous versions and fixed baselines offline.
- Match tournament conditions: engine-driven, process isolation, strict clocks.
- Deterministic, reproducible results with logged seeds.
- Metrics: EV/hand, win rate, variance, action frequencies, discard outcomes.

## Constraints
- Python 3.7 runtime on server.
- No internet access.
- Respect no-pondering rules; bots cannot learn during matches.
- Memory limit ~1 GB; keep logging and aggregation lightweight.

## Proposed Architecture (Approach 1: Engine-Driven Process Orchestrator)
- Use the official engine (`engine-2026/engine.py`) to run matches.
- Spawn bots as separate processes (isolation, prevents shared state).
- Capture stdout/stderr for each bot and engine gamelog.

### Components
1. Runner CLI (new script, e.g. `tools/run_match.py`)
   - Accept bot A/B commands or paths.
   - Pass seed to engine (via config or env if supported; otherwise patch engine to accept seed input).
   - Set number of rounds, stacks, blinds (match competition defaults).
   - Write gamelog to a known output dir with timestamp + seed.
   - Exit nonzero on engine error; surface bot crash/timeouts clearly.

2. Results Aggregator (new script, e.g. `tools/parse_gamelog.py`)
   - Parse engine gamelog to compute:
     - EV/hand (chips or BB/hand)
     - win rate by match
     - variance and standard error
     - action frequencies and discard outcomes
   - Output a compact JSON summary and a human-readable report.

3. Match Suite (new script, e.g. `tools/run_suite.py`)
   - Run N matches with distinct seeds.
   - Optionally run duplicate matches with seat swaps (reduce variance).
   - Aggregate all match summaries into a single report.

## Determinism & Fairness
- Seed control is mandatory; include seed in logs and JSON summaries.
- Use seat swaps or duplicate matches:
  - Example: run seed S twice with bot positions swapped.
  - Combine results to reduce position bias.
- Keep engine settings identical for all comparisons.

## Failure Handling
- If a bot crashes or times out, mark the match as a forfeit.
- Collect stderr/stdout for diagnosis.
- Continue suite execution even when one match fails.

## Output Layout
- `runs/YYYYMMDD_HHMMSS/`
  - `engine_gamelog.txt`
  - `botA.log`, `botB.log`
  - `match_summary.json`
- `runs/summary.json` aggregated across the suite.

## Next Steps
- Confirm how to set seed in `engine-2026/engine.py` or config.
- Implement runner and parser scripts.
- Add a small baseline bot snapshot for regression comparisons.
