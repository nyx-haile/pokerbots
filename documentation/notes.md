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
