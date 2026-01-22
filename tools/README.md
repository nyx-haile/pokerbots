# Harness Tools

This folder contains scripts to run deterministic bot-vs-bot matches and
aggregate results for regression testing.

## Run a Single Match

```bash
python3 tools/run_match.py \
  --engine-dir engine-2026 \
  --bot-a ./neuropoker \
  --bot-b ./engine-2026/python_skeleton \
  --rounds 1000 \
  --seed 123 \
  --output-dir runs
```

Outputs are written to `runs/<timestamp>/` with:
- `gamelog.txt`
- `engine_stdout.txt`
- bot logs emitted by the engine

## Run a Suite (Multiple Seeds)

```bash
python3 tools/run_suite.py \
  --bot-a ./neuropoker \
  --bot-b ./engine-2026/python_skeleton \
  --rounds 1000 \
  --matches 4 \
  --seat-swaps \
  --seed-start 1 \
  --output-dir runs
```

Outputs per match are stored under `runs/suite_<timestamp>/` and include
`suite_summary.json` with aggregate EV/hand and win rate.

## Parse a Gamelog

```bash
python3 tools/parse_gamelog.py runs/<timestamp>/gamelog.txt --json-out summary.json
```

The JSON summary includes per-player EV/hand, win rate, variance, action
frequencies, and discard counts.

## Evaluate Against Baselines

```bash
python3 tools/evaluate_harness.py --manifest tools/evaluation_manifest.json
```

The evaluation runner uses `tools/run_suite.py` to compare the current bot to
baselines and writes `evaluation_summary.json` under `evaluations/`.
