# Roadmap

## Goals
- Build a competitive Toss or Hold'em bot with correct 8-card evaluation (2 hole + 6 board).
- Use discard mechanics and position advantage for measurable EV gains.
- Keep decisions fast, reproducible, and easy to tune.
- Favor exploitative play that adapts to opponent tendencies while staying stable over 1000 hands.

## Status Legend
- DONE: Implemented and working.
- PARTIAL: Implemented but needs verification/tuning.
- TODO: Not implemented.

## Current State
- Python 3.7 compatible code paths with Deuces fallback.
- 8-card best-of evaluation via PokerStove (when available) and LRU caching of hand evals.
- Monte Carlo equity with time/sample caps and caching.
- Preflop heuristic to avoid expensive computation on street 0.
- Discard equity selection plus asymmetric discard adjustments (visible vs hidden).
- Basic betting: pot odds, raise margins, raise caps, and bluff suppression on paired/flushy boards.
- Lightweight logging for runtime diagnostics.
- No self-play regression harness yet; evaluation is ad hoc.

## Phase 0: Self-play regression harness (Priority 0)
- Build an engine-driven match runner that spawns two bots as separate processes (DONE).
- Add deterministic seed control for reproducible runs; log the seed per match (DONE).
- Use seat swaps or duplicate matches to reduce variance (DONE).
- Capture gamelog + bot stdout/stderr; aggregate EV/hand, win rate, and variance (PARTIAL).
- Enforce timeouts and handle crashes as forfeits to keep runs going (DONE).
- Compare current bot vs previous versions and fixed baselines (PARTIAL).

## Phase 1: Baseline fixes (correctness + speed)
- Ensure cp37 PokerStove wheel builds and installs on the server (offline install via scripts/ensure_pokerstove.py). (DONE)
- Verify Deuces fallback is correct and fast enough if PokerStove is missing. (PARTIAL)
- Review discard-equity simulations to cover opponent discard and future board cards; update notes when modeling assumptions change. (PARTIAL)
- Audit all bots for scrimmage server hardware constraints (single CPU core, no GPU) and remove unsupported assumptions. (TODO)
- Cap thread usage and disable GPU-optional code paths where applicable (e.g., set OMP/MKL/BLAS thread caps, skip GPU imports). (TODO)

### Server compliance checklist (per file)
- `neuropoker/player.py`: ensure no multi-process spawns; set conservative runtime caps for single-core CPU. (TODO)
- `neuropoker/strategy.py`: avoid heavy loops per decision; add early exits/low-sample fallbacks for single-core runtime. (TODO)
- `neuropoker/stats.py`: enforce thread caps for BLAS/OpenMP backends; keep CPU-only eval path. (TODO)
- `neuropoker/scripts/ensure_pokerstove.py`: verify wheel install path works offline and is CPU-only. (TODO)
- `engine-2026/config.py`: confirm bot configs do not assume multi-core or GPU resources. (TODO)

## Phase 2: Core decision model
- Strengthen preflop with a tuned 3-card LUT or bucketed heuristic. (DONE)
- Tighten value thresholds for post-discard play (stronger hands more common with 6-board). (TODO)
- Add board-texture-aware value betting and pot control. (PARTIAL)
- Add staged computation with early exits for low-stakes decisions. (PARTIAL)

## Phase 3: Discard intelligence
- Compute self-equity and board-externality for each discard. (PARTIAL)
- Use discard order: as dealer, condition on opponent discard to adjust. (TODO)
- Track opponent discard tendencies and adjust (simple frequency model). (TODO)
- Expand info-penalty to a learned, opponent-conditioned term. (PARTIAL)

## Phase 4: Opponent modeling
- Track fold frequency by bet size and street with decay. (TODO)
- Track raise/call ratios and showdowns to infer range strength. (TODO)
- Track discard patterns to update opponent range after discard. (TODO)
- Use adaptive bet sizing (larger vs overfolds, thinner value vs callers). (TODO)

## Phase 5: Lightweight learning experiments
- Add a random-feature policy (ELM / random kitchen sinks) with a linear readout. (TODO)
- Use online Hebbian updates; test Mimetic updates if activations are invertible. (TODO)
- Keep model size small (dozens of weights) to preserve speed. (TODO)

## Phase 6: Evaluation and tuning
- Use the self-play harness to evaluate every strategy change vs baseline. (TODO)
- Log EV, win rate, action frequencies, and discard outcomes. (TODO)
- Compare against baseline and select stable defaults. (TODO)
- Tune timeouts and sampling budgets to avoid engine timeouts. (TODO)

## Priority Ranking (Highest to Lowest)
P0. Build self-play regression harness (engine-driven process isolation, deterministic seeds, seat swaps/duplicates, metrics/logging). (TODO)
P1. Ensure cp37 PokerStove wheel builds and installs on the server (baseline correctness/speed). (DONE)
P2. Verify Deuces fallback is correct and fast enough if PokerStove is missing. (PARTIAL)
P3. Review discard-equity simulations to cover opponent discard and future board cards. (PARTIAL)
P3.5. Update all bots to comply with server hardware constraints (single CPU core, no GPU) and enforce thread caps. (TODO)
P4. Strengthen preflop with a tuned 3-card LUT or bucketed heuristic. (DONE)
P5. Tighten value thresholds for post-discard play with 6-card boards. (TODO)
P6. Add board-texture-aware value betting and pot control. (PARTIAL)
P7. Add staged computation with early exits for low-stakes decisions. (PARTIAL)
P8. Compute self-equity and board-externality for each discard. (PARTIAL)
P9. Use discard order to condition on opponent discard (dealer advantage). (TODO)
P10. Track opponent discard tendencies and adjust (simple frequency model). (TODO)
P11. Expand info-penalty to a learned, opponent-conditioned term. (PARTIAL)
P12. Track fold frequency by bet size and street with decay. (TODO)
P13. Track raise/call ratios and showdowns to infer range strength. (TODO)
P14. Track discard patterns to update opponent range after discard. (TODO)
P15. Use adaptive bet sizing (larger vs overfolds, thinner value vs callers). (TODO)
P16. Add a random-feature policy (ELM / random kitchen sinks) with a linear readout. (TODO)
P17. Use online Hebbian updates; test Mimetic updates if activations are invertible. (TODO)
P18. Keep model size small (dozens of weights) to preserve speed. (TODO)
P19. Log EV, win rate, action frequencies, and discard outcomes. (TODO)
P20. Compare against baseline and select stable defaults. (TODO)
P21. Tune timeouts and sampling budgets to avoid engine timeouts. (TODO)

## Deliverables
- Updated `stats.py` for 8-card evaluation, caching, and sampling.
- `strategy.py` for baseline decision logic plus discard asymmetry and exploitative hooks.
- `player.py` glue with hero/villain state wiring.
- `notes.md` updates with findings and defaults.
- Self-play regression harness runner + results summaries.
