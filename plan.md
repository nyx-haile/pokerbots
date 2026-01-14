# Roadmap

## Goals
- Build a competitive Toss or Hold'em bot with correct 8-card evaluation (2 hole + 6 board).
- Use discard mechanics and position advantage for measurable EV gains.
- Keep decisions fast, reproducible, and easy to tune.

## Phase 0: Baseline fixes (correctness + speed)
- Remove import-time work in `stats3.py` and any slow side effects.
- Fix discard equity call order and selection logic in `player.py`.
- Add an 8-card evaluator (best-of-5 from 8) or integrate a faster backend.
- Add a small Monte Carlo sampler for equity with caching and time caps.
- Review and validate discard-equity simulations so they fill all future board cards (turn + river + opponent discard) before comparison; update config/notes when the modeling assumptions change.

## Phase 1: Core decision model
- Implement pot-odds-aware calling and folding.
- Add value-bet logic tied to equity and board texture.
- Add EV-banded randomness (volatility maximization without donating EV).

## Phase 2: Discard intelligence
- Compute self-equity and board-externality for each discard.
- Use discard order: as dealer, condition on opponent discard to adjust.
- Track opponent discard tendencies and adjust (simple frequency model).

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

## Deliverables
- `equity.py` or updated `stats3.py` for 8-card evaluation and sampling.
- `player.py` strategy logic with configurable flags.
- `notes.md` updates with findings and defaults.
