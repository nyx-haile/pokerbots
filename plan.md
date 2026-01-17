# Roadmap

## Goals
- Build a competitive Toss or Hold'em bot with correct 8-card evaluation (2 hole + 6 board).
- Use discard mechanics and position advantage for measurable EV gains.
- Keep decisions fast, reproducible, and easy to tune.

## Phase 0: Baseline fixes (correctness + speed)
- Drop import-time errors and allow PokerStove fallback.
- Fix discard equity call order and selection logic in `player.py`.
- Add a PokerStove-backed 7-card evaluator with 8-card best-of logic.
- Add Monte Carlo samplers for equity with caching and time caps.
- Add low-level evaluator caching and discard-equity caching.
- Review and validate discard-equity simulations so they fill all future board cards (turn + river + opponent discard) before comparison; update config/notes when the modeling assumptions change.

## Phase 1: Core decision model
- Implement pot-odds-aware calling and folding.
- Add value-bet logic tied to equity and board texture.
- Add EV-banded randomness (volatility maximization without donating EV).
- Add preflop heuristic to avoid MC at street 0.
- Add raise caps to reduce preflop raise wars.

## Phase 2: Discard intelligence
- Compute self-equity and board-externality for each discard.
- Use discard order: as dealer, condition on opponent discard to adjust.
- Track opponent discard tendencies and adjust (simple frequency model).
- Add asymmetric discard logic (visible vs hidden).
- Learn and decay info-penalty from round outcomes.

## Phase 3: Opponent modeling
- Track fold frequency by bet size and street.
- Track raise/call ratios and showdowns to infer range strength.
- Use adaptive bet sizing (larger vs overfolds, thinner value vs callers).

## Phase 4: Lightweight learning experiments
- Add a random-feature policy (ELM / random kitchen sinks) with a linear readout.
- Use online Hebbian updates; test Mimetic updates if activations are invertible.
- Keep model size small (dozens of weights) to preserve speed.

## Phase 5: Evaluation and tuning
- Build a small harness for self-play and fixed-opponent matches.
- Log EV, win rate, action frequencies, and discard outcomes.
- Compare against baseline and select stable defaults.
- Tune timeouts and sampling budgets to avoid engine timeouts.

## Deliverables
- Updated `stats.py` for 8-card evaluation, caching, and sampling.
- `strategy.py` for baseline decision logic plus discard asymmetry.
- `player.py` glue with hero/villain state wiring.
- `notes.md` updates with findings and defaults.
