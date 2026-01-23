Critical path: what runs when
Always per decision (every get_action)

In neuropoker/player.py

Build “view” object: legal actions, stacks, pips, board, button/position, last actions, opponent counters.

Update opponent tracking with the latest action(s).

Call strategy.play(view).

In neuropoker/strategy.py

Route logic (currently ends up at _fallback_action).

Quick feature extraction: pot odds, margins, texture flags, opponent bias knobs.

Preflop: should be O(1) (cached strength + simple mix), no MC.

Postflop: may invoke equity estimation based on thresholds.

This is your hard budget path. Anything added here must be cheap and deterministic.

Sometimes per decision (postflop only, conditional)

In neuropoker/stats.py

estimate_equity(...) called when postflop thresholds are close or texture says “don’t fold too fast”.

“Quick equity” path (few samples) and “deep equity” path (more samples).

This is where most time goes in non-preflop streets.

Only on discard

Discard selection

discard_equity(...) (enumeration or MC) + asymmetry adjustments + opponent discard tendency bias.

This is a major hotspot; good that it’s not every action.

Only on round boundaries

handle_new_round / handle_round_over

Reset per-round state.

Update outcome-driven models (range bias, decay counters, showdown influence).

These can be heavier than per-action code if needed, because frequency is low.

The preflop fold pathology: where it can be coming from (architecture-aligned)

Given your description, the “fold ~100% preflop” is likely one of these, and all live in the player.py -> strategy.py handoff boundary:

The preflop mix is gated by a knob or route

You mentioned: strategy.play currently routes to _fallback_action.
If _fallback_action is “MC baseline”, it may be applying pot-odds logic that is effectively “never call” under your preflop state representation (especially if to_call/pot are computed weirdly preflop).

Also: NEUROPOKER_DISABLE_PREFLOP_MIX exists. In self-play harness, it may be set (explicitly or inherited environment).

Strength proxy scale mismatch

stats.preflop_strength is cached (good) but if it was calibrated for a different game (2-hole Hold’em) it may be too pessimistic for 3-hole + discard.

If typical strengths cluster below your thresholds, your “bucket mix” never triggers.

Legality degradation to fold

You already called this out indirectly: if raise is chosen but min_raise/cap rejects it, and the degrade logic goes to fold instead of call/check, you get “deterministic fold.”

This is especially common in engines where “RaiseAction(amount)” expects raise-to vs raise-by semantics.

Architecturally: this is all fixable cleanly inside strategy.py with better preflop routing + robust legality degrade, while keeping player.py unchanged except for logging/metrics.

Highest-ROI tuning points (in order)
1) Make preflop routing explicit and unskippable

Right now, “strategy.play routes to _fallback_action” is a red flag.

Requirement: preflop should not depend on the MC fallback path at all.

In strategy.play(view): first branch on street == PREFLOP:

call _preflop_open_decision(view) and return if not None

only then fallback (but ideally never on preflop)

This alone prevents postflop MC logic (or margins tuned for later streets) from poisoning preflop.

2) Deterministic mixed policy with a local seed

You already have a knob NEUROPOKER_USE_RANDOM_POLICY. Make sure:

preflop mix uses local random.Random(seed) per hand/decision

never uses module-global RNG

seed includes hand_id + seat + action_index so seat-swapping produces consistent symmetry

This is essential for meaningful self-play comparisons.

3) Harden legality degradation (raise → call/check, never “raise failure → fold”)

This is the most common cause of “fold almost always” in mixed strategies.

Rule: if raise is selected but illegal, degrade to:

Call if to_call > 0 and Call is legal,

else Check if legal (or to_call == 0),

else Fold.

Do not degrade raise failure to fold unless there is literally no other legal non-raise action.

4) Add harness-visible preflop metrics (not just logs)

Logs are good; suites need summary stats to confirm you’ve fixed the failure mode quickly.

Minimal metrics computed from engine output:

fold/call/raise counts by seat

raise legality failure count

histogram of preflop_strength actually observed

histogram of to_call and pot_odds preflop

This immediately tells you whether the issue is thresholds, pot_odds, or legality.

5) Only after 1–4: tune thresholds/probabilities

Tuning without fixing routing + legality is wasted work.

Once preflop actions actually happen, tune:

bucket boundaries (0.30/0.35/0.40/0.58/0.62)

call/raise frequencies

position adjustments (+/- 0.03)

“Critical path” performance notes (where to be strict)

Given your stated hotspots:

Do not add any new MC preflop. Use cached preflop_strength + O(1) math only.

Avoid additional calls to discard_equity or estimate_equity triggered from preflop. (Sounds obvious, but I’ve seen “just check equity quickly” creep in.)

Keep preflop logging behind a verbosity/env flag so suite runs can collect stats without spamming.

Self-play harness integration: what to wire and where

You’ve got the right separation of concerns:

In player.py (best place)

Add counters that update on every get_action:

self.metrics.preflop_actions[seat][action_type] += 1

self.metrics.illegal_raise_degrades += 1 (if strategy reports it)

record strength, pot_odds, to_call summary stats (binning or online mean/var)

Then at handle_round_over or end-of-match:

flush metrics to stdout in a stable single-line JSON blob

easy for harness parser

deterministic

In strategy.py

Return a small “decision metadata” struct (or attach to view) without changing engine API:

bucket name

roll

selected action before legality fix

final action

reason tags: fallback, illegal_raise_degraded, etc.

Even if you don’t want structural changes, you can encode this into the existing log line in a machine-parseable way.

The one thing I’d challenge in your current mental model

You said: “SB/in-position (dealer) can play slightly wider; BB/out-of-position (first discarder) slightly tighter.”

That’s plausible only if your variant’s discard order or action order actually gives BB the informational disadvantage you think. In HU standard Hold’em, SB acts first preflop but is IP postflop. Your game has discard mechanics that can invert or amplify info edges.

Actionable takeaway: position adjustment should be tied to who discards first and who acts first on each street, not the label “SB/BB” by tradition. If your view already knows “acts_first_preflop” and “discards_first_flop”, base the adjustment on that. This is a cheap correctness win.

What I’d do next (zero ambiguity, highest confidence)

In strategy.play, make preflop call _preflop_open_decision first, no MC fallback unless truly needed.

Implement deterministic mixed buckets + legality degradation.

Emit suite-parseable metrics from player.py.

Run your 10×1k seat-swapped suite and validate:

fold rate < 90% per seat

illegal raise degrade rate low (ideally < 10–20% of raises)

strength histogram not collapsed (sanity check)

Only then adjust thresholds.

If you paste (a) the preflop branch in strategy.play, and (b) how legal actions/min_raise/max_raise are represented in the “view”, I’ll pin down—very concretely—whether your fold issue is routing, scale, or legality semantics, and give you the exact patch-level requirements (still no code, but precise enough for the dev team to implement without interpretation errors).
