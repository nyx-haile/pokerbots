# Collaborative Research Notes

## Volatility Maximisation
A strategy objective that exploits a faster hand-solving model by acting to maximize opponent error. The goal is to obfuscate our own hand strength and push the opponent into the widest possible mistake spread (e.g., inducing incorrect calls, folds, or bet sizing). In practice this means choosing lines that increase uncertainty in the opponent's model while preserving favorable expected value when they deviate.

## Copula Activation
A neural activation scheme that uses copulas to model dependencies between features. Instead of treating activations as independent or purely linear combinations, the transition uses a copula-based coupling to capture richer joint structure, enabling more expressive chaining of ideas and feature interactions.

## Graph Isomorphic Activation
An activation scheme that promotes richer feature representation by locating isomorphic subgraphs in the activation graph. Activations are derived by matching structural motifs, enabling the network to reuse and recombine learned patterns in a topology-aware way.

## Continuous Random Draws
A stochastic activation scheme akin to Monte Carlo sampling. Each node draws from a tunable distribution or pool (with bias) to produce its activation, enabling exploration and variability while remaining controllable through distribution parameters.

## Weighted Signature Generation
An activation scheme where each node generates a composable signature of other nodes (e.g., weighted hashes or feature fingerprints). These signatures are combined to form the next activation, enabling structured composition of multiple influences into a single activation value.

## Hebbian Learning
A standard learning rule where connections strengthen when pre- and post-synaptic activations co-occur. This reinforces frequently co-activated pathways.

## Mimetic Learning (Reverse Hebbian)
A learning rule that inverts the Hebbian idea by leveraging the (often) invertible structure of the proposed activations. The intent is to use output behavior to infer and reinforce upstream structure, enabling learning that mirrors or reverses standard co-activation dynamics.

## Strategy Variants (Feature Flags)
To keep experimentation deterministic and single-core safe, strategy variants are toggled via environment flags and default OFF.

### Flags and Precedence
- `NEUROPOKER_VARIANT_THRESHOLDS=1` (highest priority)
- `NEUROPOKER_VARIANT_GRAPH=1`
- `NEUROPOKER_VARIANT_PAIRWISE=1`

If multiple flags are set, precedence is: thresholds → graph → pairwise → baseline.

### Variant A: Pairwise Feature Couplings
Purpose: add small, deterministic equity bias based on simple pairwise interactions.

Inputs:
- Hole ranks/suits vs board ranks/suits
- Simple connectivity of hole ranks

Behavior:
- Adds a capped equity bias (<= 0.02) based on rank matches, suit matches, and connectivity.
- No dynamic weights, no randomness, no learning.

### Variant B: Fixed Graph-Style Transform
Purpose: approximate relational structure without training a GNN.

Inputs:
- Hole-to-board rank/suit matches
- Board texture signals (paired/flushy)

Behavior:
- Uses fixed adjacency-style weights to compute a small equity bias (<= 0.02).
- Deterministic, no learned parameters.

### Variant C: Exploitative Threshold Tuning
Purpose: adjust raise/call margins based on opponent fold tendencies.

Inputs:
- Decayed opponent fold-rate statistics by street

Behavior:
- Loosens raise/call thresholds when opponent folds too often.
- Tightens thresholds when opponent rarely folds.
- No online weight updates beyond existing counters.

### Integration Notes
- Variants are applied inside `neuropoker/strategy.py` and must remain deterministic.
- Preflop still avoids Monte Carlo; variants only add small, bounded adjustments.
- If a variant causes any exception or missing-data condition, the bot falls back to baseline logic.
