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
- Python 3.7 compatible code paths with pkrbot as the sole poker evaluation library.
- 8-card best-of evaluation via pkrbot with LRU caching of hand evals.
- Monte Carlo equity with time/sample caps and caching.
- Preflop heuristic to avoid expensive computation on street 0.
- Discard equity selection plus asymmetric discard adjustments (visible vs hidden).
- Basic betting: pot odds, raise margins, raise caps, and bluff suppression on paired/flushy boards.
- Lightweight logging for runtime diagnostics.
- Self-play regression harness and evaluation tooling are in place.

## Research & Design Requirements (Ground Truth)
- Determinism: identical inputs must yield identical outputs; use per-hand seeded randomness only when required.
- No preflop Monte Carlo; preflop must use fast heuristics/lookup. (`design/preflop.txt`)
- Single-core only; no multithreading or GPU assumptions. (`documentation/server_environment.md`)
 - In-match adaptation limited to decayed counters and simple statistics; any major retraining happens offline. (`research/self_play_harness_design.md`)
- Use engine-driven, process-isolated self-play with deterministic seeds and seat swaps for evaluation. (`research/self_play_harness_design.md`)
- Discard mechanics are central: discard order adds information asymmetry; strategies must model public discard impact. (`research/Developing a Fast Winning Strategy for __Toss or Hold’em__ Pokerbots.pdf`)
- Stronger hands are more common with 6-board; thresholds must be tighter post-discard. (`research/Strategy Design for a Top-Performing Toss or Hold'em Pokerbot (Pokerbots 2026).pdf`)
- Prefer exploitative play with opponent modeling and staged computation/caching under tight time limits. (`research/Strategy Design for a Top-Performing Toss or Hold'em Pokerbot (Pokerbots 2026).pdf`)

## Implementation Plan (Extremely Simple Checklist)
1. **Write down the exact strategy variants we will test (low-risk only).**
   - Variant A: lightweight pairwise feature couplings (hand-designed interactions, no copulas).
   - Variant B: fixed graph-style feature transform (no GNN training; deterministic adjacency rules).
   - Variant C: exploitative threshold tuning driven by opponent stats (no online weight updates).
2. **Define inputs, outputs, and constraints for each variant.**
   - Shared inputs (all variants): hole cards (3/2), board cards (0–6), street, pot size, stacks, continue_cost, position/blind, opponent discard stats, opponent bet-response stats.
   - Shared outputs (all variants): deterministic action choice (Fold/Call/Check/Raise/Discard) with a seeded tie-break roll only when scores are equal; raises must respect min/max bounds.
    - Shared constraints (all variants): single-core only; deterministic; no preflop MC; per-decision budget target <= 1–2 ms; in-match adjustments are limited to bounded counters/statistics.
   - Variant A (pairwise couplings): compute fixed pairwise features (rank gaps, suitedness, board-texture pairs); feed into a tiny linear scorer; zero dynamic weights.
   - Variant B (fixed graph transform): build a static adjacency over cards/roles (hole vs board vs discard); aggregate with fixed weights; no training step.
   - Variant C (exploitative thresholds): adjust raise/call/fold thresholds based on decayed opponent stats; no parameter updates beyond counters.
3. **Add feature flags for each variant.**
   - Flags (defaults OFF):
     - `NEUROPOKER_VARIANT_PAIRWISE=1` enables Variant A.
     - `NEUROPOKER_VARIANT_GRAPH=1` enables Variant B.
     - `NEUROPOKER_VARIANT_THRESHOLDS=1` enables Variant C.
   - Precedence order (if multiple set): thresholds → graph → pairwise → baseline.
   - On any error or missing data, log the fallback and use baseline action selection.
4. **Extend evaluation logging for each decision.**
   - Per-decision log line (single line, stable keys):
     - `STRAT` (variant id), `street`, `equity`, `pot_odds`, `action`, `raise_to`, `decision_ms`, `legal_actions`.
   - Per-match summary:
     - EV/hand, variance, Sharpe ratio.
     - Action frequencies by street and position.
     - Latency percentiles (p50/p90/p99).
     - Fallback counts and reasons (illegal raise, missing data, exception).
5. **Update harness to surface required metrics.**
   - Compute EV/hand, variance, Sharpe ratio, win rate, and per-street action counts.
   - Emit both JSON + text summaries with seeds, bot versions, and strategy flags.
   - Record timeouts, illegal-action fallbacks, and exception counts per bot.
   - Enforce deterministic seeds and seat swaps for every suite run.
6. **Run short sanity checks (fast).**
   - 10×1k seat-swapped vs baseline for each variant.
   - Fail fast on timeouts, illegal actions, or latency regressions.
7. **Run full stability checks (slow).**
   - 10×10k seat-swapped vs baseline for top 1–2 variants.
   - Confirm EV gain + acceptable variance + latency within budget.
8. **Select the winner and lock it in.**
   - Choose the variant with highest EV/hand that meets all constraints.
   - Keep a rollback flag to revert to baseline instantly.
9. **Document outcomes.**
   - Record results, selected default, and flags in `documentation/notes.md`.
   - Note any tuning parameters and their rationale.

## New Strategy Enhancements
- Exploit reraises as information: add learning that conditions opponent range and action likelihoods on reraises, tracking reraise frequency and sizing to tighten post-raise call/raise decisions.

## New Requests / Future Work (2026-01-28)
- **TODO:** Add an oracle “god-mode” benchmark bot that sees full runout (flop/turn/river) at deal time; use it as a training adversary and upper-bound performance benchmark.
- **TODO:** Drive action selection directly from the opponent model (turn tracked fold/call/raise + sizing stats into bounded incentives/threshold tweaks).
- **TODO:** Explore a symbolic equity estimator for faster/safer evaluation (lower priority; likely post‑project).
- **TODO:** Add explicit GTO guardrails: cap exploitative bias magnitudes, require minimum samples before adapting, and prefer equilibrium‑safe adjustments.

## Turn + Betting Logic Review (Scrim Analysis)
### Decision Flow (current)
- `strategy.play`: computes equity (quick + full MC), pot odds, raise/call margins, board texture, and opponent bias; then chooses raise/call/fold.
- `pot_odds_to_call`: uses `continue_cost / (pot_total + continue_cost)`.
- `_equity_budget`: turn uses higher MC samples (100) but still noisy for 6-board.
- `_raise_margin_by_street` / `_call_margin_by_street`: same logic for all post-flop streets; turn defaults to raise=0.18, call=0.05.
- `_raise_size` + `_adjust_value_raise`: pot-fraction sizing with small fold-rate adjustments.
- `fold_equity_estimate`: learned from showdown/fold outcomes; does not explicitly model raise frequency.
- `_raise_call_penalty`: triggered by large raise size ratio; currently adds to call margin.

### Observed Scrim Pattern
- Large EV swings on turn (losses in game_log2/4/5), even when flop EV is positive.
- Bot frequently bets and then calls raises; log analysis shows near-zero fold rate after facing a raise.
- Turn decisions often occur with large pots and shallow stacks, so small equity errors produce big EV swings.

### Likely Causes
- **Call-margin semantics are loose:** `equity < pot_odds - call_margin` means *higher* `call_margin` makes calls *easier*. A penalty that increases `call_margin` actually loosens calls vs big raises.
- **Raise-call logic ignores action history:** no explicit penalty for “bet → face raise” or “reraise” states; decisions rely only on pot odds + margin.
- **Turn equity noise:** even 100 MC samples can be noisy; turn bets are high-leverage because stacks are often shallow.
- **Raise frequency not modeled:** opponent raise rates only slightly bias margins (±0.01/0.02), insufficient when raise size is large.

### Action Items (follow-up)
- Add turn-specific call-tightening: reduce `call_margin` or add a positive delta to pot-odds for turn raises (especially after betting into a raise).
- Rework raise-call penalty to *tighten* calling (subtract from `call_margin`, or add to pot-odds threshold).
- Introduce a turn “raise-response” parameter keyed by raise size ratio (e.g., fold unless equity > pot_odds + extra).
- Optionally raise MC samples on turn *only when facing a raise* to reduce variance, while keeping other spots fast.

### Implemented Mitigation (2026-01-23)
- Turn raise-call tightening: apply `_RAISE_CALL_PENALTY` only on turn (street 4) when we are facing a raise after our own bet on the same street.
- Implementation detail: penalty is applied by **reducing** `call_margin` (tightening the call threshold), and only when `continue_cost / pot_total >= _RAISE_CALL_RATIO`.
- Goal: cut the “bet → face raise → call” frequency that drives turn EV swings.
- Added a turn raise-response threshold (`_TURN_RAISE_RATIO` + `_TURN_RAISE_EXTRA`) that further tightens calls when a large raise hits on the turn after our bet.
- Added turn raise MC upshift (`_TURN_RAISE_SAMPLE_MULT`, `_TURN_RAISE_TIME_MULT`) to re-estimate equity with more samples/time when facing a large turn raise after our bet.
  - Applied only when `continue_cost / pot_total >= _TURN_RAISE_RATIO`.
  - Samples capped at 200, time capped at 0.05s to avoid timeouts.

## Top Opponent Analysis (Games 18454 / 18459)
### Observations
- We win preflop + flop EV but consistently lose on **turn**.
  - 18454: A preflop +1450, flop +1725, **turn −1672**.
  - 18459: A preflop +1364, flop +1646, **turn −1542**.
- Opponent folds a lot preflop and on Discard 2:
  - Preflop folds ~289–310 hands; Discard 2 folds ~474–479 hands.
- Opponent bets turn ~71–95 times, and we call almost all of them:
  - 18454: B turn bets 71, A calls 64 (0 folds).
  - 18459: B turn bets 95, A calls 83 (2 folds).
- Opponent rarely raises after our turn bet; the damage is mostly from **calling their turn bets too often**.

### Likely How They Win
- They play tight early and fold often to our Discard 2 bets, but when they continue, their range is strong.
- They then value-bet turn and we overcall with marginal equity; small errors get magnified by large turn pot sizes.

### Plan to Beat Them
1. **Turn bet-call tightening (non-raise path).**
   - Add a turn-specific penalty when facing a **bet** (not raise), using `continue_cost / pot_total`.
   - Require equity > pot_odds + extra margin for large turn bets.
2. **Continuation filter after Discard 2.**
   - Track when opponent **calls** our Discard 2 bet; treat their range as stronger on the next turn.
   - Increase call threshold or reduce thin value bets on turn after such continuation.
3. **Turn pot-control vs tight opponents.**
   - If opponent has high preflop/discard2 fold rates, reduce turn barrels unless equity is clearly strong.
4. **Equity precision for turn bet responses.**
   - Upshift MC budget when facing a turn bet (not just raises), capped to avoid timeouts.
5. **Diagnostic logging.**
   - Log turn decision context: equity, pot odds, bet size ratio, opponent continuation state.
   - Focus on spots where we call turn bets and lose at showdown.

## Phase 0: Self-play regression harness (Priority 0)
- Build an engine-driven match runner that spawns two bots as separate processes (DONE).
- Add deterministic seed control for reproducible runs; log the seed per match (DONE).
- Use seat swaps or duplicate matches to reduce variance (DONE).
- Capture gamelog + bot stdout/stderr; aggregate EV/hand, win rate, and variance (DONE).
- Enforce timeouts and handle crashes as forfeits to keep runs going (DONE).
- Compare current bot vs previous versions and fixed baselines (DONE).

## Phase 1: Baseline fixes (correctness + speed)
- Ensure pkrbot install is present and working on the server (offline install or bundled wheel as needed). (DONE)
- Review discard-equity simulations to cover opponent discard and future board cards; update notes when modeling assumptions change. (DONE)
- Audit all bots for scrimmage server hardware constraints (single CPU core, no GPU) and remove unsupported assumptions. (DONE)
- Remove multi-threaded/parallel code paths; single-core only. (DONE)

### Server compliance checklist (per file)
- `neuropoker/player.py`: ensure no multi-process spawns; set conservative runtime caps for single-core CPU. (DONE)
- `neuropoker/strategy.py`: avoid heavy loops per decision; add early exits/low-sample fallbacks for single-core runtime. (DONE)
- `neuropoker/stats.py`: keep CPU-only eval path and single-core Monte Carlo. (DONE)
- `neuropoker/scripts/ensure_pkrbot.py`: verify wheel install path works offline and is CPU-only (if needed). (DONE)
- `engine-2026/config.py`: confirm bot configs do not assume multi-core or GPU resources. (DONE)

## Phase 2: Core decision model
- Strengthen preflop with a tuned 3-card LUT or bucketed heuristic. (DONE)
- Tighten value thresholds for post-discard play (stronger hands more common with 6-board). (DONE)
- Add board-texture-aware value betting and pot control. (DONE)
- Add staged computation with early exits for low-stakes decisions. (DONE)

## Phase 3: Discard intelligence
- Compute self-equity and board-externality for each discard. (DONE)
- Use discard order: as dealer, condition on opponent discard to adjust. (DONE)
- Track opponent discard tendencies and adjust (simple frequency model). (DONE)
- Expand info-penalty to a learned, opponent-conditioned term. (DONE)

## Phase 4: Opponent modeling
- Track fold frequency by bet size and street with decay. (DONE)
- Track raise/call ratios and showdowns to infer range strength. (DONE)
- Track discard patterns to update opponent range after discard. (DONE)
- Use adaptive bet sizing (larger vs overfolds, thinner value vs callers). (DONE)

## Phase 5: Lightweight learning experiments
- Add a random-feature policy (ELM / random kitchen sinks) with a linear readout that is trained offline. (DONE)
- Track opponent statistics with a simple, decayed frequency counter; no mimetic/hebbian updates are running during a match. (DONE)
- Keep model size small (dozens of weights) to preserve speed. (DONE)

## Phase 6: Evaluation and tuning
- Use the self-play harness to evaluate every strategy change vs baseline. (DONE)
- Log EV, win rate, action frequencies, and discard outcomes. (DONE)
- Compare against baseline and select stable defaults. (DONE)
- Tune timeouts and sampling budgets to avoid engine timeouts. (DONE)

## Priority Ranking (Highest to Lowest)
P0. Build self-play regression harness (engine-driven process isolation, deterministic seeds, seat swaps/duplicates, metrics/logging). (DONE)
P1. Ensure pkrbot is installed and working on the server (baseline correctness/speed). (DONE)
P2. Review discard-equity simulations to cover opponent discard and future board cards. (DONE)
P2.5. Update all bots to comply with server hardware constraints (single CPU core, no GPU) and remove parallelism. (DONE)
P4. Strengthen preflop with a tuned 3-card LUT or bucketed heuristic. (DONE)
P5. Tighten value thresholds for post-discard play with 6-card boards. (DONE)
P6. Add board-texture-aware value betting and pot control. (DONE)
P7. Add staged computation with early exits for low-stakes decisions. (DONE)
P8. Compute self-equity and board-externality for each discard. (DONE)
P9. Use discard order to condition on opponent discard (dealer advantage). (DONE)
P10. Track opponent discard tendencies and adjust (simple frequency model). (DONE)
P11. Expand info-penalty to a learned, opponent-conditioned term. (DONE)
P12. Track fold frequency by bet size and street with decay. (DONE)
P13. Track raise/call ratios and showdowns to infer range strength. (DONE)
P14. Track discard patterns to update opponent range after discard. (DONE)
P15. Use adaptive bet sizing (larger vs overfolds, thinner value vs callers). (DONE)
P16. Add a random-feature policy (ELM / random kitchen sinks) with a linear readout. (DONE)
P17. Use online Hebbian updates; test Mimetic updates if activations are invertible. (DONE)
P18. Keep model size small (dozens of weights) to preserve speed. (DONE)
P19. Log EV, win rate, action frequencies, and discard outcomes. (DONE)
P20. Compare against baseline and select stable defaults. (DONE)
P21. Tune timeouts and sampling budgets to avoid engine timeouts. (DONE)

## Deliverables
- Updated `stats.py` for 8-card evaluation, caching, and sampling.
- `strategy.py` for baseline decision logic plus discard asymmetry and exploitative hooks.
- `player.py` glue with hero/villain state wiring.
- `notes.md` updates with findings and defaults.
- Self-play regression harness runner + results summaries.
