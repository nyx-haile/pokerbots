from dataclasses import dataclass, field
import math
import os
import random
from typing import Any, Iterable, Mapping, Optional, Sequence, Tuple, Dict, List

import stats

_INFO_PENALTY = 0.05
_INFO_PENALTY_MIN = 0.0
_INFO_PENALTY_MAX = 0.12
_DISCARD_MODEL_DECAY = 0.995
_OPPONENT_DISCARD_MODEL = {
    "rank_counts": [0] * 13,
    "suit_counts": [0] * 4,
    "total": 0,
}
_BET_MODEL_DECAY = 0.99
_OPPONENT_BET_MODEL = {
    "by_street": {},
}
_RANGE_MODEL_DECAY = 0.99
_OPPONENT_RANGE_MODEL = {
    "by_street": {},
    "showdowns": {"wins": 0.0, "losses": 0.0},
}
_USE_RANDOM_POLICY = os.environ.get("NEUROPOKER_USE_RANDOM_POLICY", "0") == "1"
_RANDOM_POLICY_LR = 0.02
_RANDOM_POLICY_HIDDEN = 16
_LAST_HIDDEN = None


@dataclass(frozen=True)
class StrategyConfig:
    """Configurable knobs for strategy behavior and randomness."""

    volatility_weight: float = 0.2
    discard_randomness: float = 0.05
    bluff_rate: float = 0.03
    min_raise_fraction: float = 0.5
    random_seed: Optional[int] = None


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


class ActorView:
    legal_actions: Iterable[object]
    hand: Sequence[str]
    pip: int
    stack: int
    continue_cost: int
    contribution: int
    pot_total: int
    blind: bool
    raise_bounds: Tuple[int, int]


class PlayerView:
    hero: ActorView
    villain: ActorView
    community: Sequence[str]
    street: int


class RandomFeaturePolicy:
    """Random-feature policy with a linear readout."""

    def __init__(self, seed: int = 0, hidden_dim: int = _RANDOM_POLICY_HIDDEN):
        rng = random.Random(seed)
        self.hidden_dim = hidden_dim
        self.proj = [[rng.uniform(-1.0, 1.0) for _ in range(8)] for _ in range(hidden_dim)]
        self.weights = [rng.uniform(-0.05, 0.05) for _ in range(hidden_dim)]

    def forward(self, features: List[float]) -> List[float]:
        hidden = []
        for row in self.proj:
            dot = 0.0
            for weight, feat in zip(row, features):
                dot += weight * feat
            hidden.append(math.tanh(dot))
        return hidden

    def score(self, hidden: List[float]) -> float:
        total = 0.0
        for weight, feat in zip(self.weights, hidden):
            total += weight * feat
        return total

    def update(self, hidden: List[float], reward: float, lr: float) -> None:
        for idx, feat in enumerate(hidden):
            self.weights[idx] += lr * reward * feat


_RANDOM_POLICY = RandomFeaturePolicy(seed=7)


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


def extract_features(values: Mapping[str, float]) -> Dict[str, float]:
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
) -> Dict[str, float]:
    """Apply a simple Hebbian update to a weight map."""
    updated = dict(weights)
    for key, value in features.items():
        updated[key] = updated.get(key, 0.0) + lr * value
    return updated


def mimetic_update(
    weights: Mapping[str, float],
    features: Mapping[str, float],
    lr: float = 0.1,
) -> Dict[str, float]:
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
        opponent_discard = None
        if not player.hero.blind and len(board_cards) >= 3:
            opponent_discard = board_cards[-1]
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
            opponent_discard=opponent_discard,
            board_cards=board_cards,
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
        quick_equity = stats.estimate_equity(
            hero_hand,
            board_cards,
            samples=max(20, samples // 4),
            max_seconds=min(0.006, max_seconds * 0.2),
            discard_samples=max(4, discard_samples // 2),
        )
        equity = quick_equity
    pot_total = max(1, player.hero.pot_total)
    pot_odds = pot_odds_to_call(player.hero.continue_cost, pot_total)
    raise_margin = _raise_margin_by_street(player.street)
    call_margin = _call_margin_by_street(player.street)
    discard_bias = _opponent_discard_bias(player.street)
    range_bias = _opponent_range_bias(player.street)
    texture_raise, texture_call, raise_cap_mult = _board_texture_adjustments(board_cards)
    raise_margin += discard_bias
    call_margin += discard_bias
    raise_margin += range_bias
    call_margin += range_bias

    if _USE_RANDOM_POLICY:
        policy_bias = _policy_bias(player, equity, pot_odds)
        raise_margin -= policy_bias
        call_margin -= policy_bias
    raise_margin += texture_raise
    call_margin += texture_call

    preflop_raise_threshold = None
    preflop_call_threshold = None
    if player.street <= 0:
        preflop_raise_threshold = max(pot_odds + 0.05, 0.55)
        preflop_call_threshold = max(pot_odds - 0.02, 0.45)

    if player.street > 0:
        raise_threshold = pot_odds + raise_margin
        if quick_equity > raise_threshold + 0.12:
            equity = quick_equity
        elif quick_equity < pot_odds - call_margin - 0.12:
            equity = quick_equity
        else:
            equity = stats.estimate_equity(
                hero_hand,
                board_cards,
                samples=samples,
                max_seconds=max_seconds,
                discard_samples=discard_samples,
            )

    if RaiseAction in legal_actions:
        min_raise, max_raise = getattr(player.hero, "raise_bounds", (0, 0))
        raise_threshold = pot_odds + raise_margin
        if preflop_raise_threshold is not None:
            raise_threshold = max(raise_threshold, preflop_raise_threshold)
        raise_cap = max(4, int(pot_total // 2 * raise_cap_mult))
        fold_rate = fold_equity_estimate(None, min_raise, player.street, pot_total)
        if min_raise > raise_cap or min_raise > player.hero.stack // 2:
            pass
        elif equity > raise_threshold and min_raise > 0:
            target = _raise_size(pot_total, min_raise, max_raise, equity)
            target = _adjust_value_raise(target, min_raise, max_raise, fold_rate, equity)
            return RaiseAction(target)
        if equity < pot_odds - 0.1 and _should_bluff(player.street, board_cards):
            if fold_rate <= 0.3:
                pass
            else:
                target = _raise_size(pot_total, min_raise, max_raise, equity, bluff=True)
                if target > 0:
                    return RaiseAction(target)

    if preflop_call_threshold is not None and CallAction in legal_actions:
        if equity >= preflop_call_threshold:
            return CallAction()
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
        return 0.18
    return 0.16


def _call_margin_by_street(street: int) -> float:
    if street <= 0:
        return 0.05
    if street <= 3:
        return 0.03
    if street <= 4:
        return 0.05
    return 0.06


def _board_texture_adjustments(board_cards: Sequence[str]) -> Tuple[float, float, float]:
    if not board_cards:
        return 0.0, 0.0, 1.0
    board_int = stats._ensure_int_cards(list(board_cards))
    paired = _board_is_paired(board_int)
    flushy = _board_is_flushy(board_int)
    if paired or flushy:
        return 0.03, 0.02, 0.7
    return 0.0, 0.0, 1.0


def _opponent_discard_bias(street: int) -> float:
    if street < 4:
        return 0.0
    total = _OPPONENT_DISCARD_MODEL["total"]
    if total <= 0:
        return 0.0
    rank_counts = _OPPONENT_DISCARD_MODEL["rank_counts"]
    avg_rank = 0.0
    for idx, count in enumerate(rank_counts):
        avg_rank += (idx + 2) * count
    avg_rank /= float(total)
    if avg_rank <= 6:
        return 0.02
    if avg_rank <= 8:
        return 0.01
    if avg_rank >= 12:
        return -0.01
    return 0.0


def _opponent_range_bias(street: int) -> float:
    bucket = _OPPONENT_RANGE_MODEL["by_street"].get(street)
    if not bucket:
        return 0.0
    total = bucket.get("total", 0.0)
    raises = bucket.get("raises", 0.0)
    if total < 6:
        return 0.0
    raise_rate = raises / total
    if raise_rate >= 0.5:
        return 0.02
    if raise_rate >= 0.3:
        return 0.01
    if raise_rate <= 0.15:
        return -0.01
    return 0.0


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


def _adjust_value_raise(
    target: int,
    min_raise: int,
    max_raise: int,
    fold_rate: float,
    equity: float,
) -> int:
    if target <= 0:
        return target
    if equity < 0.6:
        return target
    if fold_rate >= 0.6:
        target = int(target * 1.2)
    elif fold_rate <= 0.3:
        target = int(target * 0.8)
    target = max(min_raise, target)
    return min(target, max_raise)


def _opponent_fold_rate(street: int, bucket_key: Optional[str]) -> float:
    bucket = _OPPONENT_BET_MODEL["by_street"].get(street)
    if not bucket:
        return 0.0
    if bucket_key is None:
        total = 0.0
        folds = 0.0
        for counts in bucket.values():
            total += counts.get("fold", 0.0) + counts.get("call", 0.0)
            folds += counts.get("fold", 0.0)
    else:
        counts = bucket.get(bucket_key, {})
        total = counts.get("fold", 0.0) + counts.get("call", 0.0)
        folds = counts.get("fold", 0.0)
    if total < 5:
        return 0.0
    return folds / total


def record_opponent_bet_response(street: int, bet_size: int, pot_total: int, folded: bool) -> None:
    street_bucket = _OPPONENT_BET_MODEL["by_street"].setdefault(street, {})
    bucket_key = _bet_size_bucket(bet_size, pot_total)
    counts = street_bucket.setdefault(bucket_key, {"fold": 0.0, "call": 0.0})
    if folded:
        counts["fold"] += 1.0
    else:
        counts["call"] += 1.0


def decay_bet_model() -> None:
    by_street = _OPPONENT_BET_MODEL["by_street"]
    for street, bucket in by_street.items():
        for counts in bucket.values():
            counts["fold"] *= _BET_MODEL_DECAY
            counts["call"] *= _BET_MODEL_DECAY


def _bet_size_bucket(bet_size: int, pot_total: int) -> str:
    pot = max(1, pot_total)
    ratio = bet_size / float(pot)
    if ratio <= 0.5:
        return "small"
    if ratio <= 1.0:
        return "medium"
    return "large"


def fold_equity_estimate(player_id, bet_size: int, street: int, pot_total: int) -> float:
    """Estimate fold probability given bet size and street using a decayed model."""
    bucket_key = _bet_size_bucket(bet_size, pot_total)
    fold_rate = _opponent_fold_rate(street, bucket_key)
    if fold_rate > 0.0:
        return fold_rate
    return _opponent_fold_rate(street, None)


def record_opponent_raise(street: int) -> None:
    bucket = _OPPONENT_RANGE_MODEL["by_street"].setdefault(street, {"raises": 0.0, "calls": 0.0, "total": 0.0})
    bucket["raises"] += 1.0
    bucket["total"] += 1.0


def record_opponent_action(street: int) -> None:
    bucket = _OPPONENT_RANGE_MODEL["by_street"].setdefault(street, {"raises": 0.0, "calls": 0.0, "total": 0.0})
    bucket["total"] += 1.0


def record_opponent_call(street: int) -> None:
    bucket = _OPPONENT_RANGE_MODEL["by_street"].setdefault(street, {"raises": 0.0, "calls": 0.0, "total": 0.0})
    bucket["calls"] += 1.0
    bucket["total"] += 1.0


def record_opponent_showdown(win: bool) -> None:
    key = "wins" if win else "losses"
    _OPPONENT_RANGE_MODEL["showdowns"][key] += 1.0


def decay_range_model() -> None:
    for bucket in _OPPONENT_RANGE_MODEL["by_street"].values():
        bucket["raises"] *= _RANGE_MODEL_DECAY
        bucket["calls"] *= _RANGE_MODEL_DECAY
        bucket["total"] *= _RANGE_MODEL_DECAY
    _OPPONENT_RANGE_MODEL["showdowns"]["wins"] *= _RANGE_MODEL_DECAY
    _OPPONENT_RANGE_MODEL["showdowns"]["losses"] *= _RANGE_MODEL_DECAY


def opponent_range_strength() -> float:
    showdowns = _OPPONENT_RANGE_MODEL["showdowns"]
    total = showdowns["wins"] + showdowns["losses"]
    if total < 3:
        return 0.0
    return (showdowns["wins"] - showdowns["losses"]) / total


def update_range_after_discard(range3, discarded_card):
    record_opponent_discard(discarded_card)
    return range3


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


def _equity_budget(player: PlayerView) -> Tuple[int, float, int]:
    street = player.street
    if street <= 0:
        samples = 60
        max_seconds = 0.018
        discard_samples = 8
    elif street <= 3:
        samples = 70
        max_seconds = 0.02
        discard_samples = 10
    else:
        samples = 100
        max_seconds = 0.025
        discard_samples = 10

    game_clock = getattr(player, "game_clock", None)
    if game_clock is not None and game_clock < 20:
        samples = max(30, samples // 2)
        max_seconds = max(0.01, max_seconds * 0.5)
        discard_samples = max(5, discard_samples // 2)
    if game_clock is not None and game_clock < 10:
        samples = max(20, samples // 2)
        max_seconds = max(0.005, max_seconds * 0.5)
        discard_samples = max(4, discard_samples // 2)
    return samples, max_seconds, discard_samples


def _discard_budget(player: PlayerView) -> Tuple[int, float]:
    street = player.street
    if street <= 2:
        samples = 40
        max_seconds = 0.0
    else:
        samples = 60
        max_seconds = 0.0
    game_clock = getattr(player, "game_clock", None)
    if game_clock is not None and game_clock < 20:
        samples = max(30, samples // 2)
    if game_clock is not None and game_clock < 10:
        samples = max(20, samples // 2)
    return samples, max_seconds


def _select_discard_asymmetric(
    hero_hand: Sequence[str],
    equities: Sequence[float],
    discard_visible: bool,
    opponent_discard: Optional[str] = None,
    board_cards: Optional[Sequence[str]] = None,
) -> int:
    if not equities:
        return 0
    if not discard_visible and opponent_discard is None:
        return max(range(len(equities)), key=equities.__getitem__)

    cards_int = stats._ensure_int_cards(list(hero_hand))
    info_penalty = _INFO_PENALTY if discard_visible else 0.0
    scores = []
    opponent_int = None
    if opponent_discard is not None:
        opponent_int = stats._ensure_int_cards([opponent_discard])[0]
    for idx, equity in enumerate(equities):
        rank = stats.Card.get_rank_int(cards_int[idx]) + 2
        score = equity - info_penalty * (rank / 14)
        if board_cards is not None:
            score -= _discard_externality_penalty(board_cards, cards_int[idx])
        if opponent_int is not None:
            discard_card = cards_int[idx]
            keep_cards = [card for i, card in enumerate(cards_int) if i != idx]
            score -= _discard_order_penalty(keep_cards, discard_card, opponent_int)
        scores.append(score)
    return max(range(len(scores)), key=scores.__getitem__)


def _discard_externality_penalty(board_cards: Sequence[str], discard_card: int) -> float:
    board_int = stats._ensure_int_cards(list(board_cards))
    if not board_int:
        return 0.0
    ranks = [stats.Card.get_rank_int(card) for card in board_int]
    suits = [stats.Card.get_suit_int(card) for card in board_int]
    discard_rank = stats.Card.get_rank_int(discard_card)
    discard_suit = stats.Card.get_suit_int(discard_card)
    penalty = 0.0
    if discard_rank in ranks:
        penalty += 0.012
    if suits.count(discard_suit) >= 2:
        penalty += 0.015
    return penalty


def _discard_order_penalty(
    keep_cards: Sequence[int],
    discard_card: int,
    opponent_discard: int,
) -> float:
    """Penalize discards that strengthen the public board when acting second."""
    penalty = 0.0
    discard_suit = stats.Card.get_suit_int(discard_card)
    discard_rank = stats.Card.get_rank_int(discard_card)
    keep_suits = [stats.Card.get_suit_int(card) for card in keep_cards]
    keep_ranks = [stats.Card.get_rank_int(card) for card in keep_cards]
    opp_suit = stats.Card.get_suit_int(opponent_discard)
    opp_rank = stats.Card.get_rank_int(opponent_discard)
    if discard_suit == opp_suit and opp_suit not in keep_suits:
        penalty += 0.015
    if discard_rank == opp_rank and opp_rank not in keep_ranks:
        penalty += 0.012
    return penalty


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
    opponent_strength = opponent_range_strength()
    if opponent_strength:
        _INFO_PENALTY = min(
            _INFO_PENALTY_MAX,
            max(_INFO_PENALTY_MIN, _INFO_PENALTY + 0.01 * opponent_strength),
        )
    _INFO_PENALTY = _apply_info_penalty_decay(_INFO_PENALTY, player)
    _update_random_policy(player)


def record_opponent_discard(card: str) -> None:
    card_int = stats._ensure_int_cards([card])[0]
    rank = stats.Card.get_rank_int(card_int)
    suit = stats.Card.get_suit_int(card_int)
    _OPPONENT_DISCARD_MODEL["rank_counts"][rank] += 1
    _OPPONENT_DISCARD_MODEL["suit_counts"][suit] += 1
    _OPPONENT_DISCARD_MODEL["total"] += 1


def decay_discard_model() -> None:
    total = _OPPONENT_DISCARD_MODEL["total"]
    if total <= 0:
        return
    rank_counts = []
    suit_counts = []
    for count in _OPPONENT_DISCARD_MODEL["rank_counts"]:
        rank_counts.append(count * _DISCARD_MODEL_DECAY)
    for count in _OPPONENT_DISCARD_MODEL["suit_counts"]:
        suit_counts.append(count * _DISCARD_MODEL_DECAY)
    _OPPONENT_DISCARD_MODEL["rank_counts"] = rank_counts
    _OPPONENT_DISCARD_MODEL["suit_counts"] = suit_counts
    _OPPONENT_DISCARD_MODEL["total"] = total * _DISCARD_MODEL_DECAY


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


def _policy_bias(player: PlayerView, equity: float, pot_odds: float) -> float:
    global _LAST_HIDDEN
    features = _policy_features(player, equity, pot_odds)
    hidden = _RANDOM_POLICY.forward(features)
    _LAST_HIDDEN = hidden
    score = _RANDOM_POLICY.score(hidden)
    return max(-0.05, min(0.05, score))


def _policy_features(player: PlayerView, equity: float, pot_odds: float) -> List[float]:
    pot_total = max(1, player.hero.pot_total)
    return [
        equity,
        pot_odds,
        min(1.0, player.street / 6.0),
        min(1.0, player.hero.stack / float(pot_total)),
        min(1.0, player.hero.continue_cost / float(pot_total)),
        1.0 if player.hero.blind else 0.0,
        1.0,
        0.5,
    ]


def _update_random_policy(player: PlayerView) -> None:
    global _LAST_HIDDEN
    if not _USE_RANDOM_POLICY or _LAST_HIDDEN is None:
        return
    delta = getattr(player.hero, "delta", 0)
    reward = max(-1.0, min(1.0, delta / 100.0))
    _RANDOM_POLICY.update(_LAST_HIDDEN, reward, _RANDOM_POLICY_LR)
    _LAST_HIDDEN = None


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
