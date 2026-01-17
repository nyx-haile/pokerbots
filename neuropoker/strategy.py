from __future__ import annotations
from dataclasses import dataclass, field
import random
from typing import Iterable, Mapping, Protocol, Sequence

import stats

_INFO_PENALTY = 0.05
_INFO_PENALTY_MIN = 0.0
_INFO_PENALTY_MAX = 0.12


@dataclass(frozen=True)
class StrategyConfig:
    """Configurable knobs for strategy behavior and randomness."""

    volatility_weight: float = 0.2
    discard_randomness: float = 0.05
    bluff_rate: float = 0.03
    min_raise_fraction: float = 0.5
    random_seed: int | None = None


@dataclass
class StrategyState:
    """State container for runtime strategy execution."""

    rng: random.Random = field(default_factory=random.Random)
    action_count: int = 0


@dataclass
class OpponentModel:
    """Lightweight counters for opponent tendencies."""

    fold_count: int = 0
    call_count: int = 0
    raise_count: int = 0
    showdown_count: int = 0


class ActorView(Protocol):
    legal_actions: Iterable[object]
    hand: Sequence[str]
    pip: int
    stack: int
    continue_cost: int
    contribution: int
    pot_total: int
    blind: bool
    raise_bounds: tuple[int, int]


class PlayerView(Protocol):
    hero: ActorView
    villain: ActorView
    community: Sequence[str]
    street: int


def init_state(config: StrategyConfig) -> StrategyState:
    """Initialize a state object with deterministic RNG if configured."""
    rng = random.Random(config.random_seed)
    return StrategyState(rng=rng)


def pot_odds_to_call(cost: int, pot: int) -> float:
    """Compute pot-odds threshold; safe for zero-cost checks."""
    if cost <= 0:
        return 0.0
    return cost / max(1, pot + cost)


def volatility_adjustment(base_ev: float, variance_proxy: float, config: StrategyConfig) -> float:
    """Apply a volatility maximization adjustment to a base EV score."""
    return base_ev + config.volatility_weight * variance_proxy


def select_discard(
    equities: Sequence[float],
    config: StrategyConfig,
    state: StrategyState,
) -> int:
    """Pick the discard index, preferring higher equity with mild randomness."""
    if not equities:
        return 0
    best = max(range(len(equities)), key=equities.__getitem__)
    if state.rng.random() < config.discard_randomness:
        return state.rng.randrange(len(equities))
    return best


def decide_action(
    legal_actions: Iterable[str],
    pot_odds: float,
    equity: float,
    config: StrategyConfig,
    state: StrategyState,
) -> str:
    """
    Decide between legal actions using pot odds and a small bluff rate.

    Returns a string action label for now ("fold", "call", "check", "raise").
    """
    state.action_count += 1
    legal = set(legal_actions)
    if "check" in legal and pot_odds == 0:
        return "check"
    if equity < pot_odds and "fold" in legal:
        return "fold"
    if "raise" in legal and state.rng.random() < config.bluff_rate:
        return "raise"
    if "call" in legal:
        return "call"
    if "check" in legal:
        return "check"
    return next(iter(legal))


def update_opponent_model(model: OpponentModel, action: str) -> None:
    """Track opponent action frequencies for later range adjustments."""
    if action == "fold":
        model.fold_count += 1
    elif action == "call":
        model.call_count += 1
    elif action == "raise":
        model.raise_count += 1


def extract_features(values: Mapping[str, float]) -> dict[str, float]:
    """Normalize and sanitize feature inputs for simple learners."""
    return {key: float(val) for key, val in values.items()}


def copula_activation(features: Mapping[str, float]) -> float:
    """Placeholder for copula-style activation using simple aggregation."""
    if not features:
        return 0.0
    return sum(features.values()) / len(features)


def graph_isomorphic_activation(features: Mapping[str, float]) -> float:
    """Placeholder for topology-aware activation; use max feature for now."""
    if not features:
        return 0.0
    return max(features.values())


def continuous_random_draw(features: Mapping[str, float], state: StrategyState) -> float:
    """Sample an activation by mixing features with bounded noise."""
    base = copula_activation(features)
    return base + state.rng.uniform(-0.1, 0.1)


def weighted_signature(features: Mapping[str, float]) -> float:
    """Combine weighted feature signatures into a single activation."""
    if not features:
        return 0.0
    total = 0.0
    for i, value in enumerate(features.values(), start=1):
        total += value / i
    return total / len(features)


def hebbian_update(
    weights: Mapping[str, float],
    features: Mapping[str, float],
    lr: float = 0.1,
) -> dict[str, float]:
    """Apply a simple Hebbian update to a weight map."""
    updated = dict(weights)
    for key, value in features.items():
        updated[key] = updated.get(key, 0.0) + lr * value
    return updated


def mimetic_update(
    weights: Mapping[str, float],
    features: Mapping[str, float],
    lr: float = 0.1,
) -> dict[str, float]:
    """Apply a reverse-Hebbian update to a weight map."""
    updated = dict(weights)
    for key, value in features.items():
        updated[key] = updated.get(key, 0.0) - lr * value
    return updated


def play(bot):
    """
    Strategy entry point called by player.py.

    This function should read the live fields on player and return an action
    instance from skeleton.actions.
    """
    return _fallback_action(bot)


def _fallback_action(player: PlayerView):
    """Baseline equity-driven action selection using Monte Carlo estimates."""
    from skeleton.actions import CallAction, CheckAction, DiscardAction, FoldAction, RaiseAction

    legal_actions = set(player.hero.legal_actions)
    hero_hand = list(player.hero.hand)
    board_cards = list(player.community)

    if DiscardAction in legal_actions:
        discard_samples, discard_seconds = _discard_budget(player)
        equities = stats.discard_equity(
            hero_hand,
            board_cards,
            n_samples=discard_samples,
            max_seconds=discard_seconds,
        )
        best_i = _select_discard_asymmetric(
            hero_hand,
            equities,
            discard_visible=player.hero.blind,
        )
        player.hero.last_discard = hero_hand[best_i]
        player.hero.last_discard_visible = player.hero.blind
        return DiscardAction(best_i)

    if len(legal_actions) == 1:
        action = next(iter(legal_actions))
        return action()

    if player.street <= 0:
        equity = stats.preflop_strength(hero_hand)
    else:
        samples, max_seconds, discard_samples = _equity_budget(player)
        equity = stats.estimate_equity(
            hero_hand,
            board_cards,
            samples=samples,
            max_seconds=max_seconds,
            discard_samples=discard_samples,
        )
    pot_total = max(1, player.hero.pot_total)
    pot_odds = pot_odds_to_call(player.hero.continue_cost, pot_total)
    raise_margin = _raise_margin_by_street(player.street)
    call_margin = _call_margin_by_street(player.street)

    if RaiseAction in legal_actions:
        min_raise, max_raise = getattr(player.hero, "raise_bounds", (0, 0))
        raise_threshold = pot_odds + raise_margin
        raise_cap = max(4, pot_total // 2)
        if min_raise > raise_cap or min_raise > player.hero.stack // 2:
            pass
        elif equity > raise_threshold and min_raise > 0:
            target = _raise_size(pot_total, min_raise, max_raise, equity)
            return RaiseAction(target)
        if equity < pot_odds - 0.1 and _should_bluff(player.street, board_cards):
            target = _raise_size(pot_total, min_raise, max_raise, equity, bluff=True)
            if target > 0:
                return RaiseAction(target)

    if equity < pot_odds - call_margin and FoldAction in legal_actions:
        return FoldAction()

    if CheckAction in legal_actions and player.hero.continue_cost == 0:
        return CheckAction()
    if CallAction in legal_actions:
        return CallAction()
    if CheckAction in legal_actions:
        return CheckAction()
    return FoldAction()


def _raise_margin_by_street(street: int) -> float:
    if street <= 0:
        return 0.2
    if street <= 3:
        return 0.15
    if street <= 4:
        return 0.12
    return 0.1


def _call_margin_by_street(street: int) -> float:
    if street <= 0:
        return 0.05
    if street <= 3:
        return 0.03
    return 0.02


def _raise_size(pot_total: int, min_raise: int, max_raise: int, equity: float, bluff: bool = False) -> int:
    if max_raise <= 0:
        return 0
    if bluff:
        target = max(min_raise, pot_total // 3)
        return min(target, max_raise)
    if equity >= 0.7:
        target = pot_total
    elif equity >= 0.6:
        target = pot_total // 2
    else:
        target = pot_total // 3
    target = max(min_raise, target)
    return min(target, max_raise)


def _should_bluff(street: int, board_cards: Sequence[str]) -> bool:
    if street < 3:
        return False
    board_int = stats._ensure_int_cards(list(board_cards))
    if _board_is_paired(board_int):
        return False
    if _board_is_flushy(board_int):
        return False
    return random.random() < 0.06


def _board_is_paired(board_int: Sequence[int]) -> bool:
    ranks = [stats.Card.get_rank_int(card) for card in board_int]
    return len(set(ranks)) < len(ranks)


def _board_is_flushy(board_int: Sequence[int]) -> bool:
    suits = [stats.Card.get_suit_int(card) for card in board_int]
    for suit in set(suits):
        if suits.count(suit) >= 3:
            return True
    return False


def _equity_budget(player: PlayerView) -> tuple[int, float, int]:
    street = player.street
    if street <= 0:
        samples = 60
        max_seconds = 0.02
        discard_samples = 8
    elif street <= 3:
        samples = 80
        max_seconds = 0.025
        discard_samples = 10
    else:
        samples = 120
        max_seconds = 0.03
        discard_samples = 12

    game_clock = getattr(player, "game_clock", None)
    if game_clock is not None and game_clock < 20:
        samples = max(30, samples // 2)
        max_seconds = max(0.01, max_seconds * 0.5)
        discard_samples = max(5, discard_samples // 2)
    return samples, max_seconds, discard_samples


def _discard_budget(player: PlayerView) -> tuple[int, float]:
    street = player.street
    if street <= 2:
        samples = 50
        max_seconds = 0.0
    else:
        samples = 70
        max_seconds = 0.0
    game_clock = getattr(player, "game_clock", None)
    if game_clock is not None and game_clock < 20:
        samples = max(30, samples // 2)
    return samples, max_seconds


def _select_discard_asymmetric(
    hero_hand: Sequence[str],
    equities: Sequence[float],
    discard_visible: bool,
) -> int:
    if not equities:
        return 0
    if not discard_visible:
        return max(range(len(equities)), key=equities.__getitem__)

    cards_int = stats._ensure_int_cards(list(hero_hand))
    info_penalty = _INFO_PENALTY
    scores = []
    for idx, equity in enumerate(equities):
        rank = stats.Card.get_rank_int(cards_int[idx]) + 2
        scores.append(equity - info_penalty * (rank / 14))
    return max(range(len(scores)), key=scores.__getitem__)


def update_info_penalty_from_round(player: PlayerView) -> None:
    """Adapt the information penalty based on observed outcomes."""
    global _INFO_PENALTY
    discard = getattr(player.hero, "last_discard", None)
    discard_visible = getattr(player.hero, "last_discard_visible", None)
    delta = getattr(player.hero, "delta", 0)
    if discard is None or discard_visible is not True:
        _INFO_PENALTY = _apply_info_penalty_decay(_INFO_PENALTY, player)
        return
    discard_int = stats._ensure_int_cards([discard])[0]
    rank = stats.Card.get_rank_int(discard_int) + 2
    strength = rank / 14
    if delta < 0:
        _INFO_PENALTY = min(_INFO_PENALTY_MAX, _INFO_PENALTY + 0.01 * strength)
    elif delta > 0:
        _INFO_PENALTY = max(_INFO_PENALTY_MIN, _INFO_PENALTY - 0.005 * strength)
    _INFO_PENALTY = _apply_info_penalty_decay(_INFO_PENALTY, player)


def _apply_info_penalty_decay(value: float, player: PlayerView) -> float:
    round_num = getattr(player, "round_num", 0)
    if round_num <= 0:
        return value
    if round_num < 200:
        decay = 0.995
    elif round_num < 600:
        decay = 0.99
    else:
        decay = 0.985
    return max(_INFO_PENALTY_MIN, min(_INFO_PENALTY_MAX, value * decay))


def initial_opponent_range(position):
    """
    Return prior distribution over opponent 3-card hands conditioned on position.
    """
    raise NotImplementedError


def range_after_preflop_action(range3, action, size):
    """
    Update opponent 3-card range after a preflop action using Bayesian weighting.
    """
    raise NotImplementedError


def infer_discard_distribution(player_id, known_hand, community, action_context):
    """
    Predict discard probabilities for a player at the discard step.
    """
    raise NotImplementedError


def update_range_after_discard(range3, discarded_card):
    """
    Condition opponent 3-card range on observed discard and return 2-card range.
    """
    raise NotImplementedError


def bet_size_likelihood(player_id, street, hand_strength_bucket):
    """
    Return likelihood of a player choosing a bet size bucket given inferred strength.
    """
    raise NotImplementedError


def fold_equity_estimate(player_id, bet_size, street):
    """
    Estimate probability opponent folds to a bet of given size on a street.
    """
    raise NotImplementedError


def update_player_model(player_id, outcome):
    """
    Update opponent-facing learned parameters after a hand.
    """
    raise NotImplementedError


def decay_player_model(player_id, factor):
    """
    Apply forgetting to learned opponent statistics.
    """
    raise NotImplementedError


def current_hand_strength_percentile(player_id):
    """
    Return hero strength percentile vs opponent range on current board.
    """
    raise NotImplementedError


def discard_regret(card):
    """
    Return max discard EV minus EV of discarding the given card.
    """
    raise NotImplementedError


def raise_ev(amount):
    """
    Return EV of raising to the given amount.
    """
    raise NotImplementedError


def call_ev(amount):
    """
    Return EV of calling the given amount.
    """
    raise NotImplementedError
