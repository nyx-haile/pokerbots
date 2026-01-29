"""Opponent + policy modeling helpers extracted from strategy core."""

import math
import os
import random
from typing import Optional, Sequence, List, Tuple

import stats

from .math import (
    _BLUFF_DISABLE_BEHIND,
    _BLUFF_WEAKNESS_THRESHOLD,
    _PARAM_VALUES,
    _PRESSURE_FOLDRATE_MIN,
    _PRESSURE_EQUITY_THRESHOLD,
    _TIGHT_EQUITY_THRESHOLD,
    _TIGHT_FOLD_LR,
    _load_float_list,
    _board_is_flushy,
    _board_is_paired,
    _board_texture_adjustments,
    _lead_protection_size_mult,
    is_river,
)

_USE_RANDOM_POLICY = os.environ.get("NEUROPOKER_USE_RANDOM_POLICY", "0") == "1"
_RANDOM_POLICY_SEED = int(os.environ.get("NEUROPOKER_RANDOM_POLICY_SEED", "7") or "7")
_RANDOM_POLICY_LR = 0.02
_RANDOM_POLICY_HIDDEN = 16
_LAST_HIDDEN = None

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
_OPPONENT_BET_SIZE_MODEL = {
    "by_street": {},
}
_OVERBET_ACCURACY = {
    "count": 0.0,
    "sum_brier": 0.0,
    "sum_abs": 0.0,
}
_OVERBET_ACCURACY_BY_STREET = {}
_FOLD_EQUITY_ACCURACY = {
    "count": 0.0,
    "sum_brier": 0.0,
    "sum_abs": 0.0,
}
_FOLD_EQUITY_ACCURACY_BY_STREET = {}
_FOLD_EQUITY_OBS = {}
_RANGE_HINT_ACCURACY = {
    "count": 0.0,
    "sum_brier": 0.0,
    "sum_abs": 0.0,
}
_RANGE_HINT_ACCURACY_BY_STREET = {}
_RANGE_HINT_ACCURACY_BY_LINE = {}
_LINE_OUTCOMES = {}

_ACCURACY_MIN_OBS = {
    "overbet": 10,
    "fold_equity": 8,
    "range_hint": 8,
}
_ACCURACY_MIN_CONFIDENCE = 0.25
_RANGE_HINT_MIN_LINE_OBS = 10
_LINE_PRIOR_MIN_OBS = 12
_OVERBET_MIN_POT = 6
_FOLD_EQUITY_MIN_POT = 6
_OVERBET_ADJ_MIN_OBS = 8
_FOLD_EQUITY_MIN_OBS = 8
_RANGE_HINT_MIN_SHOWDOWNS = 5
_OVERBET_PRIOR_ALPHA = 1.0
_OVERBET_PRIOR_BETA = 3.0
_FOLD_PRIOR_ALPHA = 1.0
_FOLD_PRIOR_BETA = 1.5
_RANGE_MODEL_DECAY = 0.99
_OPPONENT_RANGE_MODEL = {
    "by_street": {},
    "showdowns": {"wins": 0.0, "losses": 0.0},
    "inferred": {"sum": 0.0, "total": 0.0},
}

_CONFIDENCE_Z = 1.0

_OPP_PASSIVE_THRESHOLD = _load_float_list(
    "NEUROPOKER_OPP_PASSIVE_THRESHOLD",
    1,
    (0.2,),
    param_key="opp_passive_threshold",
    param_values=_PARAM_VALUES,
)[0]
_OPP_STATION_CALL_RATE = _load_float_list(
    "NEUROPOKER_OPP_STATION_CALL_RATE",
    1,
    (0.6,),
    param_key="opp_station_call_rate",
    param_values=_PARAM_VALUES,
)[0]
_OPP_STATION_FOLD_RATE = _load_float_list(
    "NEUROPOKER_OPP_STATION_FOLD_RATE",
    1,
    (0.15,),
    param_key="opp_station_fold_rate",
    param_values=_PARAM_VALUES,
)[0]
_OPP_PASSIVE_VALUE_MULT = _load_float_list(
    "NEUROPOKER_OPP_PASSIVE_VALUE_MULT",
    1,
    (1.2,),
    param_key="opp_passive_value_mult",
    param_values=_PARAM_VALUES,
)[0]
_OPP_STATION_VALUE_MULT = _load_float_list(
    "NEUROPOKER_OPP_STATION_VALUE_MULT",
    1,
    (1.4,),
    param_key="opp_station_value_mult",
    param_values=_PARAM_VALUES,
)[0]

_POLICY_LR = float(os.environ.get("NEUROPOKER_POLICY_LR", "0.15") or "0.15")
_POLICY_EPSILON = float(os.environ.get("NEUROPOKER_POLICY_EPSILON", "0.15") or "0.15")
_POLICY_STATS = {}


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


_RANDOM_POLICY = RandomFeaturePolicy(seed=_RANDOM_POLICY_SEED)


def update_policy_from_round(player) -> None:
    policy_class = getattr(player.hero, "policy_class", None)
    if not policy_class:
        return
    _ensure_policy_stats(policy_class)
    policy_key = _policy_key(policy_class)
    if policy_key not in _POLICY_STATS:
        return
    delta = getattr(player.hero, "delta", 0)
    reward = max(-1.0, min(1.0, delta / 100.0))
    lr = _POLICY_LR
    bluff_cls, _ = _policy_classes()
    if policy_class is bluff_cls:
        call_rate = _opponent_call_rate(getattr(player, "street", None))
        if call_rate > 0.0:
            lr *= max(0.1, 1.0 - min(0.8, call_rate))
    stats_bucket = _POLICY_STATS[policy_key]
    stats_bucket["avg"] += lr * (reward - stats_bucket["avg"])
    stats_bucket["count"] += 1.0
    stats_bucket["total_delta"] += float(delta)
    stats_bucket["total_reward"] += float(reward)
    street = getattr(player, "_last_hero_bet_street", None)
    if street is None:
        street = getattr(player, "street", None)
    if street is not None:
        street_bucket = stats_bucket["per_street"].setdefault(
            int(street),
            {"count": 0.0, "total_delta": 0.0, "total_reward": 0.0},
        )
        street_bucket["count"] += 1.0
        street_bucket["total_delta"] += float(delta)
        street_bucket["total_reward"] += float(reward)


def _policy_key(policy_class) -> str:
    return getattr(policy_class, "__name__", str(policy_class))


def _policy_classes():
    import policies.bluff as bluff_policy
    import policies.tight as tight_policy

    return (bluff_policy.BluffPolicy, tight_policy.TightPolicy)


def _ensure_policy_stats(policy_class=None) -> None:
    if policy_class is not None:
        key = _policy_key(policy_class)
        _POLICY_STATS.setdefault(
            key,
            {
                "avg": 0.0,
                "count": 0.0,
                "total_delta": 0.0,
                "total_reward": 0.0,
                "per_street": {},
            },
        )
        return
    for cls in _policy_classes():
        key = _policy_key(cls)
        _POLICY_STATS.setdefault(
            key,
            {
                "avg": 0.0,
                "count": 0.0,
                "total_delta": 0.0,
                "total_reward": 0.0,
                "per_street": {},
            },
        )


def _select_round_policy(player):
    policy_class = getattr(player.hero, "policy_class", None)
    policy_round = getattr(player.hero, "policy_round", None)
    round_num = getattr(player, "round_num", 0)
    if policy_class in _policy_classes() and policy_round == round_num:
        return policy_class
    return None


def _opponent_weakness_score(player) -> float:
    fold_rate = _opponent_fold_rate(player.street, None)
    range_bucket = _OPPONENT_RANGE_MODEL["by_street"].get(player.street, {})
    total = range_bucket.get("total", 0.0)
    raises = range_bucket.get("raises", 0.0)
    raise_rate = raises / total if total >= 5 else None
    inferred = _OPPONENT_RANGE_MODEL["inferred"]
    inferred_avg = inferred["sum"] / inferred["total"] if inferred["total"] >= 5 else None
    score = 0.0
    if fold_rate > 0.0:
        score += max(0.0, fold_rate - 0.25)
    if raise_rate is not None:
        score += max(0.0, 0.3 - raise_rate)
    if inferred_avg is not None:
        score += max(0.0, 0.45 - inferred_avg)
    return score


def _opponent_bluff_adjustment(player) -> float:
    fold_rate = _opponent_fold_rate(player.street, None)
    range_bucket = _OPPONENT_RANGE_MODEL["by_street"].get(player.street, {})
    total = range_bucket.get("total", 0.0)
    raises = range_bucket.get("raises", 0.0)
    raise_rate = raises / total if total >= 5 else None
    bias = 0.0
    if fold_rate >= 0.45:
        bias -= 0.04
    elif 0.0 < fold_rate <= 0.2:
        bias += 0.04
    if raise_rate is not None and raise_rate >= 0.45:
        bias += 0.02
    return bias


def _opponent_aggression_rate(street: Optional[int]) -> Tuple[float, float]:
    if street is None:
        return 0.0, 0.0
    bucket = _OPPONENT_RANGE_MODEL["by_street"].get(street, {})
    total = bucket.get("total", 0.0)
    raises = bucket.get("raises", 0.0)
    if total < 6:
        return 0.0, 0.0
    rate = raises / total
    confidence = min(1.0, total / 20.0)
    return rate, confidence


def _anti_exploit_rate(player) -> float:
    street = getattr(player, "street", None)
    if street is None or street <= 0:
        return 0.0
    rate, confidence = _opponent_aggression_rate(street)
    if rate <= 0.3 or confidence <= 0.0:
        return 0.0
    excess = max(0.0, rate - 0.3)
    anti = excess * 0.5 * confidence
    return max(0.0, min(0.12, anti))


def _anti_exploit_discard_index(equities: Sequence[float], best_idx: int, rate: float) -> int:
    if rate <= 0.0 or not equities or best_idx is None:
        return best_idx
    if len(equities) < 2:
        return best_idx
    if random.random() >= rate:
        return best_idx
    ranked = sorted(range(len(equities)), key=equities.__getitem__, reverse=True)
    for idx in ranked:
        if idx != best_idx:
            return idx
    return best_idx


def _bluff_threshold(player) -> float:
    _ensure_policy_stats()
    bluff_cls, tight_cls = _policy_classes()
    bluff_avg = _POLICY_STATS[_policy_key(bluff_cls)]["avg"]
    tight_avg = _POLICY_STATS[_policy_key(tight_cls)]["avg"]
    bias = max(-0.05, min(0.05, (bluff_avg - tight_avg) * 0.2))
    opponent_bias = _opponent_bluff_adjustment(player)
    return max(0.0, _BLUFF_WEAKNESS_THRESHOLD - bias + opponent_bias)


def _tight_equity_threshold(player) -> float:
    base = _TIGHT_EQUITY_THRESHOLD
    fold_rate = _opponent_fold_rate(player.street, None)
    if fold_rate <= 0.0 or _TIGHT_FOLD_LR <= 0.0:
        return base
    adjustment = fold_rate * _TIGHT_FOLD_LR
    return max(0.0, base - adjustment)


def _choose_policy_for_round(player, equity: float):
    from .math import pot_odds_to_call

    bluff_cls, tight_cls = _policy_classes()
    if equity >= _tight_equity_threshold(player):
        return tight_cls
    pot_total = max(1, getattr(player.hero, "pot_total", 1))
    continue_cost = getattr(player.hero, "continue_cost", 0)
    pot_odds = pot_odds_to_call(continue_cost, pot_total)
    if equity >= pot_odds:
        return tight_cls
    weakness = _opponent_weakness_score(player)
    if weakness >= _bluff_threshold(player) and _bluff_allowed(player):
        board_cards = list(getattr(player, "community", []))
        texture_raise, _, _ = _board_texture_adjustments(board_cards)
        if texture_raise > 0.0:
            return None
        fold_rate = _opponent_fold_rate(player.street, None)
        if fold_rate <= 0.0:
            return None
        if fold_rate < _PRESSURE_FOLDRATE_MIN:
            return None
        return bluff_cls
    return None


def _bluff_allowed(player) -> bool:
    if _BLUFF_DISABLE_BEHIND <= 0:
        return True
    bankroll = getattr(player.hero, "bankroll", 0)
    return bankroll >= -_BLUFF_DISABLE_BEHIND


def _opponent_discard_bias(street: int) -> float:
    if street <= 0:
        return 0.0
    total = _OPPONENT_DISCARD_MODEL["total"]
    if total <= 0:
        return 0.0
    rank_counts = _OPPONENT_DISCARD_MODEL["rank_counts"]
    avg_rank = sum(idx * count for idx, count in enumerate(rank_counts)) / total
    strength = min(1.0, max(0.0, avg_rank / 12.0))
    return min(0.04, 0.02 * (strength - 0.5))


def _opponent_range_bias(street: int) -> float:
    if street <= 0:
        return 0.0
    inferred = _OPPONENT_RANGE_MODEL["inferred"]
    if inferred["total"] <= 0:
        return 0.0
    avg = inferred["sum"] / inferred["total"]
    return max(-0.04, min(0.04, (avg - 0.5) * 0.08))


def _opponent_is_passive(street: int) -> bool:
    fold_rate, low, _, conf, width = _opponent_fold_rate_band(street, None)
    if fold_rate <= 0.0:
        return False
    if conf < 0.25 or width > 0.35:
        return False
    return low >= _OPP_PASSIVE_THRESHOLD


def _opponent_is_calling_station(street: int) -> bool:
    fold_rate, _, high, conf, width = _opponent_fold_rate_band(street, None)
    call_rate = _opponent_call_rate(street)
    if fold_rate <= 0.0:
        return False
    if call_rate <= 0.0:
        return False
    if conf < 0.25 or width > 0.35:
        return False
    return high <= _OPP_STATION_FOLD_RATE and call_rate >= _OPP_STATION_CALL_RATE


def _value_extraction_multiplier(player) -> float:
    opp_mult = 1.0
    street = getattr(player, "street", 0)
    if _opponent_is_calling_station(street):
        opp_mult = _OPP_STATION_VALUE_MULT
    elif _opponent_is_passive(street):
        opp_mult = _OPP_PASSIVE_VALUE_MULT
    lead_mult = _lead_protection_size_mult(player)
    variance = _equity_variance_factor(player)
    if variance > 0.0:
        opp_mult *= max(0.85, 1.0 - 0.2 * variance)
    return opp_mult * lead_mult


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
    if total < _FOLD_EQUITY_MIN_OBS:
        return 0.0
    return (folds + _FOLD_PRIOR_ALPHA) / (total + _FOLD_PRIOR_ALPHA + _FOLD_PRIOR_BETA)


def _opponent_fold_rate_counts(street: int, bucket_key: Optional[str]) -> Tuple[float, float]:
    bucket = _OPPONENT_BET_MODEL["by_street"].get(street)
    if not bucket:
        return 0.0, 0.0
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
    return folds, total


def _beta_confidence_band(successes: float, total: float, alpha: float, beta: float) -> Tuple[float, float, float, float]:
    if total <= 0:
        return 0.0, 0.0, 0.0, 0.0
    mean = (successes + alpha) / (total + alpha + beta)
    var = mean * (1.0 - mean) / (total + alpha + beta + 1.0)
    stdev = math.sqrt(max(0.0, var))
    low = max(0.0, mean - _CONFIDENCE_Z * stdev)
    high = min(1.0, mean + _CONFIDENCE_Z * stdev)
    conf = min(1.0, total / 12.0)
    return mean, low, high, conf


def _opponent_fold_rate_band(street: int, bucket_key: Optional[str]) -> Tuple[float, float, float, float, float]:
    folds, total = _opponent_fold_rate_counts(street, bucket_key)
    if total < _FOLD_EQUITY_MIN_OBS:
        return 0.0, 0.0, 0.0, 0.0, 1.0
    mean, low, high, conf = _beta_confidence_band(folds, total, _FOLD_PRIOR_ALPHA, _FOLD_PRIOR_BETA)
    width = max(0.0, high - low)
    return mean, low, high, conf, width


def fold_equity_band(bet_size: int, street: int, pot_total: int) -> Tuple[float, float, float, float, float]:
    if pot_total < _FOLD_EQUITY_MIN_POT:
        return 0.0, 0.0, 0.0, 0.0, 1.0
    bucket_key = _bet_size_bucket(bet_size, pot_total)
    mean, low, high, conf, width = _opponent_fold_rate_band(street, bucket_key)
    if mean > 0.0:
        return mean, low, high, conf, width
    return _opponent_fold_rate_band(street, None)


def _sample_fold_rate(rng: random.Random, folds: float, total: float) -> float:
    if total <= 0.0:
        return 0.0
    alpha = folds + _FOLD_PRIOR_ALPHA
    beta = (total - folds) + _FOLD_PRIOR_BETA
    return rng.betavariate(alpha, beta)


def _bet_escalation_penalty(player, bet_size: int, pot_total: int) -> float:
    if player is None or bet_size <= 0 or pot_total <= 0:
        return 0.0
    history = getattr(player, "_villain_bet_history", None)
    if not history:
        return 0.0
    prev_street, prev_size, prev_pot = history[-1]
    street = getattr(player, "street", 0)
    if prev_street >= street:
        return 0.0
    if prev_size <= 0 or prev_pot <= 0:
        return 0.0
    ratio_now = bet_size / float(max(1, pot_total))
    ratio_prev = prev_size / float(max(1, prev_pot))
    escalation = max(0.0, ratio_now - ratio_prev)
    if bet_size >= prev_size * 1.6:
        escalation += 0.15
    if bet_size >= prev_size * 2.0:
        escalation += 0.1
    penalty = 0.02 + 0.12 * escalation
    if street >= 6:
        penalty *= 1.2
    elif street >= 5:
        penalty *= 1.1
    return min(0.12, max(0.0, penalty))


def _bet_escalation_info(player, bet_size: int, pot_total: int) -> Tuple[float, float, float]:
    if player is None or bet_size <= 0 or pot_total <= 0:
        return 0.0, 0.0, 0.0
    history = getattr(player, "_villain_bet_history", None)
    if not history:
        return 0.0, 0.0, 0.0
    prev_street, prev_size, prev_pot = history[-1]
    street = getattr(player, "street", 0)
    if prev_street >= street or prev_size <= 0 or prev_pot <= 0:
        return 0.0, 0.0, 0.0
    size_ratio = bet_size / float(prev_size)
    ratio_prev = prev_size / float(max(1, prev_pot))
    ratio_now = bet_size / float(max(1, pot_total))
    return size_ratio, ratio_prev, ratio_now


def _opponent_call_rate(street: Optional[int]) -> float:
    if street is None:
        total = 0.0
        calls = 0.0
        for bucket in _OPPONENT_BET_MODEL["by_street"].values():
            for counts in bucket.values():
                total += counts.get("fold", 0.0) + counts.get("call", 0.0)
                calls += counts.get("call", 0.0)
    else:
        bucket = _OPPONENT_BET_MODEL["by_street"].get(street)
        if not bucket:
            return 0.0
        total = 0.0
        calls = 0.0
        for counts in bucket.values():
            total += counts.get("fold", 0.0) + counts.get("call", 0.0)
            calls += counts.get("call", 0.0)
    if total < 5:
        return 0.0
    return calls / total


def record_opponent_bet_response(street: int, bet_size: int, pot_total: int, folded: bool) -> None:
    street_bucket = _OPPONENT_BET_MODEL["by_street"].setdefault(street, {})
    bucket_key = _bet_size_bucket(bet_size, pot_total)
    counts = street_bucket.setdefault(bucket_key, {"fold": 0.0, "call": 0.0})
    if folded:
        counts["fold"] += 1.0
    else:
        counts["call"] += 1.0


def record_opponent_bet_size(street: int, bet_size: int, pot_total: int) -> None:
    if bet_size <= 0:
        return
    if pot_total < _OVERBET_MIN_POT:
        return
    pot = max(1, pot_total)
    ratio = bet_size / float(pot)
    street_bucket = _OPPONENT_BET_SIZE_MODEL["by_street"].setdefault(
        street,
        {"total": 0.0, "overbet": 0.0, "big": 0.0},
    )
    street_bucket["total"] += 1.0
    if ratio > 1.0:
        street_bucket["overbet"] += 1.0
    if ratio > 1.5:
        street_bucket["big"] += 1.0


def record_overbet_observation(street: int, bet_size: int, pot_total: int) -> None:
    if bet_size <= 0:
        return
    if pot_total < _OVERBET_MIN_POT:
        return
    pot = max(1, pot_total)
    ratio = bet_size / float(pot)
    is_overbet = 1.0 if ratio > 1.0 else 0.0
    rate, _, confidence = opponent_overbet_rate(street)
    if confidence <= 0.0:
        return
    if confidence < _ACCURACY_MIN_CONFIDENCE:
        return
    total = _OPPONENT_BET_SIZE_MODEL["by_street"].get(street, {}).get("total", 0.0)
    if total < _ACCURACY_MIN_OBS["overbet"]:
        return
    prob = max(0.0, min(1.0, rate))
    error = prob - is_overbet
    _OVERBET_ACCURACY["count"] += 1.0
    _OVERBET_ACCURACY["sum_brier"] += error * error
    _OVERBET_ACCURACY["sum_abs"] += abs(error)
    street_bucket = _OVERBET_ACCURACY_BY_STREET.setdefault(
        int(street),
        {"count": 0.0, "sum_brier": 0.0, "sum_abs": 0.0},
    )
    street_bucket["count"] += 1.0
    street_bucket["sum_brier"] += error * error
    street_bucket["sum_abs"] += abs(error)


def record_fold_equity_observation(
    street: int,
    bet_size: int,
    pot_total: int,
    folded: bool,
    line_key: Optional[str] = None,
) -> None:
    if bet_size <= 0:
        return
    pred = fold_equity_estimate(None, bet_size, street, pot_total)
    if pred <= 0.0:
        return
    obs_bucket = _FOLD_EQUITY_OBS.setdefault(int(street), {}).setdefault(
        _bet_size_bucket(bet_size, pot_total),
        {"count": 0.0},
    )
    obs_bucket["count"] += 1.0
    if obs_bucket["count"] < _ACCURACY_MIN_OBS["fold_equity"]:
        return
    bucket_key = _bet_size_bucket(bet_size, pot_total)
    street_bucket = _OPPONENT_BET_MODEL["by_street"].get(street, {})
    bucket_counts = street_bucket.get(bucket_key, {})
    bucket_total = bucket_counts.get("fold", 0.0) + bucket_counts.get("call", 0.0)
    confidence = min(1.0, bucket_total / 12.0)
    if confidence < _ACCURACY_MIN_CONFIDENCE:
        return
    actual = 1.0 if folded else 0.0
    error = pred - actual
    weight = _line_reliability_weight(line_key)
    if weight <= 0.5:
        return
    _FOLD_EQUITY_ACCURACY["count"] += weight
    _FOLD_EQUITY_ACCURACY["sum_brier"] += weight * error * error
    _FOLD_EQUITY_ACCURACY["sum_abs"] += weight * abs(error)
    street_accuracy = _FOLD_EQUITY_ACCURACY_BY_STREET.setdefault(
        int(street),
        {"count": 0.0, "sum_brier": 0.0, "sum_abs": 0.0},
    )
    street_accuracy["count"] += weight
    street_accuracy["sum_brier"] += weight * error * error
    street_accuracy["sum_abs"] += weight * abs(error)


def decay_bet_model() -> None:
    by_street = _OPPONENT_BET_MODEL["by_street"]
    for street, bucket in by_street.items():
        for counts in bucket.values():
            counts["fold"] *= _BET_MODEL_DECAY
            counts["call"] *= _BET_MODEL_DECAY
    overbet_by_street = _OPPONENT_BET_SIZE_MODEL["by_street"]
    for bucket in overbet_by_street.values():
        bucket["total"] *= _BET_MODEL_DECAY
        bucket["overbet"] *= _BET_MODEL_DECAY
        bucket["big"] *= _BET_MODEL_DECAY


def _bet_size_bucket(bet_size: int, pot_total: int) -> str:
    pot = max(1, pot_total)
    ratio = bet_size / float(pot)
    if ratio <= 0.5:
        return "small"
    if ratio <= 1.0:
        return "medium"
    return "large"


def fold_equity_estimate(player_id, bet_size: int, street: int, pot_total: int) -> float:
    if pot_total < _FOLD_EQUITY_MIN_POT:
        return 0.0
    bucket_key = _bet_size_bucket(bet_size, pot_total)
    fold_rate = _opponent_fold_rate(street, bucket_key)
    if fold_rate > 0.0:
        return fold_rate
    return _opponent_fold_rate(street, None)


def opponent_overbet_rate(street: Optional[int]) -> Tuple[float, float, float]:
    if street is None:
        total = 0.0
        overbet = 0.0
        big = 0.0
        for bucket in _OPPONENT_BET_SIZE_MODEL["by_street"].values():
            total += bucket.get("total", 0.0)
            overbet += bucket.get("overbet", 0.0)
            big += bucket.get("big", 0.0)
    else:
        bucket = _OPPONENT_BET_SIZE_MODEL["by_street"].get(street, {})
        total = bucket.get("total", 0.0)
        overbet = bucket.get("overbet", 0.0)
        big = bucket.get("big", 0.0)
    if total < 3:
        return 0.0, 0.0, 0.0
    confidence = min(1.0, total / 12.0)
    overbet_rate = (overbet + _OVERBET_PRIOR_ALPHA) / (total + _OVERBET_PRIOR_ALPHA + _OVERBET_PRIOR_BETA)
    big_rate = (big + _OVERBET_PRIOR_ALPHA) / (total + _OVERBET_PRIOR_ALPHA + _OVERBET_PRIOR_BETA)
    return overbet_rate, big_rate, confidence


def opponent_overbet_summary() -> str:
    lines = ["opponent_overbet:"]
    by_street = _OPPONENT_BET_SIZE_MODEL["by_street"]
    if not by_street:
        return "opponent_overbet: none"
    for street in sorted(by_street):
        bucket = by_street[street]
        total = bucket.get("total", 0.0)
        if total <= 0:
            continue
        overbet = bucket.get("overbet", 0.0)
        big = bucket.get("big", 0.0)
        lines.append(
            f"  street={street} total={int(total)} overbet={overbet/total:.3f} big={big/total:.3f}"
        )
    if len(lines) == 1:
        return "opponent_overbet: none"
    return "\n".join(lines)


def overbet_accuracy_summary() -> str:
    count = _OVERBET_ACCURACY.get("count", 0.0)
    if count <= 0:
        return "overbet_accuracy: none"
    brier = _OVERBET_ACCURACY.get("sum_brier", 0.0) / count
    abs_err = _OVERBET_ACCURACY.get("sum_abs", 0.0) / count
    lines = [f"overbet_accuracy: count={int(count)} brier={brier:.4f} abs_err={abs_err:.3f}"]
    for street in sorted(_OVERBET_ACCURACY_BY_STREET):
        bucket = _OVERBET_ACCURACY_BY_STREET[street]
        street_count = bucket.get("count", 0.0)
        if street_count <= 0:
            continue
        street_brier = bucket.get("sum_brier", 0.0) / street_count
        street_abs = bucket.get("sum_abs", 0.0) / street_count
        lines.append(
            f"  street={street} count={int(street_count)} brier={street_brier:.4f} abs_err={street_abs:.3f}"
        )
    return "\n".join(lines)


def fold_equity_accuracy_summary() -> str:
    count = _FOLD_EQUITY_ACCURACY.get("count", 0.0)
    if count <= 0:
        return "fold_equity_accuracy: none"
    brier = _FOLD_EQUITY_ACCURACY.get("sum_brier", 0.0) / count
    abs_err = _FOLD_EQUITY_ACCURACY.get("sum_abs", 0.0) / count
    lines = [f"fold_equity_accuracy: count={int(count)} brier={brier:.4f} abs_err={abs_err:.3f}"]
    for street in sorted(_FOLD_EQUITY_ACCURACY_BY_STREET):
        bucket = _FOLD_EQUITY_ACCURACY_BY_STREET[street]
        street_count = bucket.get("count", 0.0)
        if street_count <= 0:
            continue
        street_brier = bucket.get("sum_brier", 0.0) / street_count
        street_abs = bucket.get("sum_abs", 0.0) / street_count
        lines.append(
            f"  street={street} count={int(street_count)} brier={street_brier:.4f} abs_err={street_abs:.3f}"
        )
    return "\n".join(lines)


def record_range_hint_accuracy(
    strength: float,
    confidence: float,
    hero_delta: int,
    street: int,
    line_key: Optional[str] = None,
) -> None:
    if confidence <= 0.0:
        return
    if confidence < _ACCURACY_MIN_CONFIDENCE:
        return
    actual = 1.0 if hero_delta < 0 else 0.0
    pred = max(0.0, min(1.0, strength))
    weight = max(0.1, confidence) * _line_reliability_weight(line_key)
    if weight <= 0.1:
        return
    error = pred - actual
    _RANGE_HINT_ACCURACY["count"] += weight
    _RANGE_HINT_ACCURACY["sum_brier"] += weight * error * error
    _RANGE_HINT_ACCURACY["sum_abs"] += weight * abs(error)
    street_bucket = _RANGE_HINT_ACCURACY_BY_STREET.setdefault(
        int(street),
        {"count": 0.0, "sum_brier": 0.0, "sum_abs": 0.0},
    )
    street_bucket["count"] += weight
    street_bucket["sum_brier"] += weight * error * error
    street_bucket["sum_abs"] += weight * abs(error)
    if line_key:
        line_bucket = _RANGE_HINT_ACCURACY_BY_LINE.setdefault(
            line_key,
            {"count": 0.0, "sum_brier": 0.0, "sum_abs": 0.0},
        )
        line_bucket["count"] += weight
        line_bucket["sum_brier"] += weight * error * error
        line_bucket["sum_abs"] += weight * abs(error)


def range_hint_accuracy_summary() -> str:
    count = _RANGE_HINT_ACCURACY.get("count", 0.0)
    if count <= 0:
        return "range_hint_accuracy: none"
    brier = _RANGE_HINT_ACCURACY.get("sum_brier", 0.0) / count
    abs_err = _RANGE_HINT_ACCURACY.get("sum_abs", 0.0) / count
    lines = [f"range_hint_accuracy: count={int(count)} brier={brier:.4f} abs_err={abs_err:.3f}"]
    for street in sorted(_RANGE_HINT_ACCURACY_BY_STREET):
        bucket = _RANGE_HINT_ACCURACY_BY_STREET[street]
        street_count = bucket.get("count", 0.0)
        if street_count <= 0:
            continue
        street_brier = bucket.get("sum_brier", 0.0) / street_count
        street_abs = bucket.get("sum_abs", 0.0) / street_count
        lines.append(
            f"  street={street} count={int(street_count)} brier={street_brier:.4f} abs_err={street_abs:.3f}"
        )
    for line_key in sorted(_RANGE_HINT_ACCURACY_BY_LINE):
        bucket = _RANGE_HINT_ACCURACY_BY_LINE[line_key]
        line_count = bucket.get("count", 0.0)
        if line_count < _RANGE_HINT_MIN_LINE_OBS:
            continue
        line_brier = bucket.get("sum_brier", 0.0) / line_count
        line_abs = bucket.get("sum_abs", 0.0) / line_count
        lines.append(
            f"  line={line_key} count={int(line_count)} brier={line_brier:.4f} abs_err={line_abs:.3f}"
        )
    return "\n".join(lines)


def _line_reliability_weight(line_key: Optional[str]) -> float:
    if not line_key:
        return 1.0
    bucket = _LINE_OUTCOMES.get(line_key)
    if not bucket:
        return 1.0
    count = bucket.get("count", 0.0)
    if count < _RANGE_HINT_MIN_LINE_OBS:
        return 1.0
    wins = bucket.get("wins", 0.0)
    losses = bucket.get("losses", 0.0)
    total = max(1.0, wins + losses)
    win_rate = wins / total
    avg_delta = bucket.get("sum_delta", 0.0) / max(1.0, count)
    if win_rate <= 0.25 and avg_delta < 0:
        return 0.25
    if win_rate <= 0.40 and avg_delta < 0:
        return 0.5
    return 1.0


def _opponent_overbet_adjustments(player, bet_size: int, pot_total: int) -> Tuple[float, float]:
    if bet_size <= 0:
        return 0.0, 0.0
    pot = max(1, pot_total)
    ratio = bet_size / float(pot)
    if ratio <= 1.0:
        return 0.0, 0.0
    rate, big_rate, confidence = opponent_overbet_rate(getattr(player, "street", None))
    if confidence <= 0.0:
        return 0.0, 0.0
    street_total = _OPPONENT_BET_SIZE_MODEL["by_street"].get(getattr(player, "street", None), {}).get("total", 0.0)
    if street_total < _OVERBET_ADJ_MIN_OBS:
        return 0.0, 0.0
    call_adj = 0.0
    raise_adj = 0.0
    if rate < 0.2:
        base = min(0.08, 0.04 + 0.04 * (ratio - 1.0))
        raise_adj += base * confidence
        call_adj += min(0.06, 0.03 + 0.03 * (ratio - 1.0)) * confidence
    elif rate > 0.5 and big_rate > 0.25:
        call_adj -= 0.02 * confidence
        raise_adj -= 0.02 * confidence
    call_adj = max(-0.03, min(0.06, call_adj))
    raise_adj = max(-0.04, min(0.08, raise_adj))
    variance = _equity_variance_factor(player)
    if variance > 0.0:
        damp = max(0.5, 1.0 - 0.5 * variance)
        call_adj *= damp
        raise_adj *= damp
    return call_adj, raise_adj


def record_opponent_raise(street: int) -> None:
    bucket = _OPPONENT_RANGE_MODEL["by_street"].setdefault(
        street,
        {"raises": 0.0, "calls": 0.0, "folds": 0.0, "checks": 0.0, "total": 0.0},
    )
    bucket["raises"] += 1.0
    bucket["total"] += 1.0


def record_opponent_action(street: int) -> None:
    record_opponent_check(street)


def record_opponent_call(street: int) -> None:
    bucket = _OPPONENT_RANGE_MODEL["by_street"].setdefault(
        street,
        {"raises": 0.0, "calls": 0.0, "folds": 0.0, "checks": 0.0, "total": 0.0},
    )
    bucket["calls"] += 1.0
    bucket["total"] += 1.0


def record_opponent_fold(street: int) -> None:
    bucket = _OPPONENT_RANGE_MODEL["by_street"].setdefault(
        street,
        {"raises": 0.0, "calls": 0.0, "folds": 0.0, "checks": 0.0, "total": 0.0},
    )
    bucket["folds"] += 1.0
    bucket["total"] += 1.0


def record_opponent_check(street: int) -> None:
    bucket = _OPPONENT_RANGE_MODEL["by_street"].setdefault(
        street,
        {"raises": 0.0, "calls": 0.0, "folds": 0.0, "checks": 0.0, "total": 0.0},
    )
    bucket["checks"] += 1.0
    bucket["total"] += 1.0


def record_inferred_range(strength: float) -> None:
    inferred = _OPPONENT_RANGE_MODEL["inferred"]
    inferred["sum"] += float(strength)
    inferred["total"] += 1.0


def record_opponent_showdown(win: bool) -> None:
    key = "wins" if win else "losses"
    _OPPONENT_RANGE_MODEL["showdowns"][key] += 1.0


def decay_range_model() -> None:
    for bucket in _OPPONENT_RANGE_MODEL["by_street"].values():
        bucket["raises"] *= _RANGE_MODEL_DECAY
        bucket["calls"] *= _RANGE_MODEL_DECAY
        bucket["folds"] *= _RANGE_MODEL_DECAY
        bucket["checks"] *= _RANGE_MODEL_DECAY
        bucket["total"] *= _RANGE_MODEL_DECAY
    _OPPONENT_RANGE_MODEL["showdowns"]["wins"] *= _RANGE_MODEL_DECAY
    _OPPONENT_RANGE_MODEL["showdowns"]["losses"] *= _RANGE_MODEL_DECAY
    _OPPONENT_RANGE_MODEL["inferred"]["sum"] *= _RANGE_MODEL_DECAY
    _OPPONENT_RANGE_MODEL["inferred"]["total"] *= _RANGE_MODEL_DECAY


def opponent_range_strength() -> float:
    showdowns = _OPPONENT_RANGE_MODEL["showdowns"]
    total = showdowns["wins"] + showdowns["losses"]
    if total < 3:
        return 0.0
    return (showdowns["wins"] - showdowns["losses"]) / total


def opponent_range_hint(line_key: Optional[str] = None) -> Tuple[float, float]:
    inferred = _OPPONENT_RANGE_MODEL["inferred"]
    inferred_total = inferred["total"]
    if inferred_total > 0:
        inferred_avg = (inferred["sum"] + 0.5 * 2.0) / (inferred_total + 2.0)
    else:
        inferred_avg = 0.5
    showdowns = _OPPONENT_RANGE_MODEL["showdowns"]
    total_showdowns = showdowns["wins"] + showdowns["losses"]
    if total_showdowns >= _RANGE_HINT_MIN_SHOWDOWNS:
        showdown_bias = (showdowns["wins"] - showdowns["losses"]) / total_showdowns
        showdown_avg = 0.5 + 0.25 * showdown_bias
    else:
        showdown_avg = 0.5
    strength = 0.7 * inferred_avg + 0.3 * showdown_avg
    strength += _line_prior_adjustment(line_key)
    discard_strength_adj, _, discard_conf = opponent_discard_range_adjustment()
    strength += discard_strength_adj
    strength = max(0.25, min(0.92, strength))
    confidence = min(1.0, (inferred_total + total_showdowns) / 10.0)
    if discard_conf > 0:
        confidence = max(confidence, 0.4 * discard_conf)
    return strength, confidence


def record_line_showdown(line_key: Optional[str], hero_delta: int) -> None:
    if not line_key:
        return
    bucket = _LINE_OUTCOMES.setdefault(
        line_key,
        {"count": 0.0, "wins": 0.0, "losses": 0.0, "sum_delta": 0.0},
    )
    bucket["count"] += 1.0
    bucket["sum_delta"] += float(hero_delta)
    if hero_delta > 0:
        bucket["wins"] += 1.0
    elif hero_delta < 0:
        bucket["losses"] += 1.0


def _line_prior_adjustment(line_key: Optional[str]) -> float:
    if not line_key:
        return 0.0
    bucket = _LINE_OUTCOMES.get(line_key)
    if not bucket:
        return 0.0
    count = bucket.get("count", 0.0)
    if count < _LINE_PRIOR_MIN_OBS:
        return 0.0
    wins = bucket.get("wins", 0.0)
    losses = bucket.get("losses", 0.0)
    total = max(1.0, wins + losses)
    win_rate = wins / total
    avg_delta = bucket.get("sum_delta", 0.0) / max(1.0, count)
    bias = (0.5 - win_rate) * 0.06
    if avg_delta < 0:
        bias += min(0.03, -avg_delta / 2000.0)
    elif avg_delta > 0:
        bias -= min(0.02, avg_delta / 3000.0)
    return max(-0.04, min(0.04, bias))


def line_range_calibration(line_key: Optional[str]) -> Tuple[float, float, float, float]:
    if not line_key:
        return 0.0, 0.0, 1.0, 0.0
    bucket = _LINE_OUTCOMES.get(line_key)
    if not bucket:
        return 0.0, 0.0, 1.0, 0.0
    count = bucket.get("count", 0.0)
    if count < max(6.0, _LINE_PRIOR_MIN_OBS):
        return 0.0, 0.0, 1.0, 0.0
    wins = bucket.get("wins", 0.0)
    losses = bucket.get("losses", 0.0)
    total = max(1.0, wins + losses)
    win_rate = wins / total
    avg_delta = bucket.get("sum_delta", 0.0) / max(1.0, count)
    strength_adj = (0.5 - win_rate) * 0.08
    if avg_delta < 0:
        strength_adj += min(0.04, -avg_delta / 2000.0)
    elif avg_delta > 0:
        strength_adj -= min(0.03, avg_delta / 2500.0)
    strength_adj = max(-0.06, min(0.06, strength_adj))
    tight_adj = strength_adj * 0.6
    mix_mult = 1.0 - min(0.2, abs(strength_adj) * 2.5)
    conf = min(1.0, count / 20.0)
    return strength_adj, tight_adj, mix_mult, conf


def opponent_discard_range_adjustment() -> Tuple[float, float, float]:
    total = _OPPONENT_DISCARD_MODEL["total"]
    if total < 3:
        return 0.0, 0.0, 0.0
    rank_counts = _OPPONENT_DISCARD_MODEL["rank_counts"]
    avg_rank = sum(idx * count for idx, count in enumerate(rank_counts)) / total
    discard_strength = max(0.0, min(1.0, avg_rank / 12.0))
    confidence = min(1.0, total / 20.0)
    strength_shift = (0.5 - discard_strength) * 0.12 * confidence
    tightness_shift = strength_shift * 0.5
    return strength_shift, tightness_shift, confidence


def _opponent_threshold_adjustments(player) -> Tuple[float, float, float, float]:
    line_key = getattr(getattr(player, "hero", None), "line_prefix", None)
    strength, confidence = opponent_range_hint(line_key)
    delta = strength - 0.5
    raise_adj = 0.05 * delta * (0.5 + confidence)
    call_adj = 0.03 * delta * (0.5 + confidence)
    discard_strength_adj, _, discard_conf = opponent_discard_range_adjustment()
    if getattr(player, "street", 0) >= 4 and discard_conf > 0:
        raise_adj += discard_strength_adj * 0.6
        call_adj += discard_strength_adj * 0.4
    raise_adj = max(-0.03, min(0.06, raise_adj))
    call_adj = max(-0.02, min(0.04, call_adj))
    river_raise_adj = 0.0
    river_floor_adj = 0.0
    if is_river(getattr(player, "street", 0)):
        strong = max(0.0, strength - 0.55)
        river_raise_adj = 0.03 + 0.03 * confidence + 0.03 * strong
        river_floor_adj = 0.02 + 0.03 * confidence + 0.05 * strong
    variance = _equity_variance_factor(player)
    if variance > 0.0:
        damp = max(0.4, 1.0 - 0.6 * variance)
        raise_adj *= damp
        call_adj *= damp
        river_raise_adj *= damp
        river_floor_adj *= damp
    return raise_adj, call_adj, river_raise_adj, river_floor_adj


def _equity_variance_factor(player, cap: float = 0.12) -> float:
    err = getattr(getattr(player, "hero", None), "last_equity_abs_error", 0.0)
    if err <= 0.0 or cap <= 0.0:
        return 0.0
    return max(0.0, min(1.0, err / cap))


def update_range_after_discard(range3, discarded_card):
    record_opponent_discard(discarded_card)
    return range3


def _should_bluff(street: int, board_cards: Sequence[str], allow_bluff: bool = True) -> bool:
    if not allow_bluff:
        return False
    suppression = _discard_bluff_suppression(None)
    if suppression > 0.0 and random.random() < suppression:
        return False
    if street == 0:
        return random.random() < 0.006
    if street < 3:
        return False
    board_int = stats._ensure_int_cards(list(board_cards))
    board_tuple = tuple(board_int)
    if _board_is_paired(board_tuple):
        return False
    if _board_is_flushy(board_tuple):
        return False
    return random.random() < 0.006


def _discard_bluff_suppression(player) -> float:
    total = _OPPONENT_DISCARD_MODEL["total"]
    if total < 6:
        return 0.0
    rank_counts = _OPPONENT_DISCARD_MODEL["rank_counts"]
    avg_rank = sum(idx * count for idx, count in enumerate(rank_counts)) / total
    discard_strength = max(0.0, min(1.0, avg_rank / 12.0))
    confidence = min(1.0, total / 20.0)
    if discard_strength >= 0.5:
        return 0.0
    bias = (0.5 - discard_strength) / 0.5
    suppression = bias * 0.6 * confidence
    return max(0.0, min(0.6, suppression))


def _adaptive_raise_size(
    player,
    pot_total: int,
    min_raise: int,
    max_raise: int,
    equity: float,
    bluff: bool = False,
    value_mult: float = 1.0,
) -> int:
    if max_raise <= 0:
        return 0
    pot = max(1, pot_total)
    street = getattr(player, "street", 0)
    seed = hash(
        (
            getattr(player, "round_num", 0),
            street,
            tuple(getattr(getattr(player, "hero", None), "hand", ())),
            tuple(getattr(player, "community", ())),
            int(equity * 100),
        )
    ) & 0xFFFFFFFF
    rng = random.Random(seed)
    fold_rate, fold_low, fold_high, fold_conf, fold_width = fold_equity_band(min_raise, street, pot_total)
    if fold_rate <= 0.0:
        fold_rate = _opponent_fold_rate(street, None)
        fold_low = fold_rate
        fold_high = fold_rate
        fold_conf = 0.0
        fold_width = 1.0
    base_ratio = 0.28
    conf_weight = max(0.0, min(1.0, 1.0 - fold_width / 0.45))
    if bluff:
        effective_rate = fold_rate
        if fold_conf > 0.0:
            folds, total = _opponent_fold_rate_counts(street, _bet_size_bucket(min_raise, pot_total))
            effective_rate = _sample_fold_rate(rng, folds, total)
        base_ratio += 0.15 * max(0.0, effective_rate - 0.25) * conf_weight
        base_ratio -= 0.08 * max(0.0, 0.25 - effective_rate) * conf_weight
    else:
        base_ratio += 0.55 * max(0.0, min(1.0, equity))
        base_ratio *= max(0.7, min(1.4, value_mult))
        if fold_conf > 0.0 and fold_width > 0.2:
            base_ratio *= max(0.85, 1.0 - 0.3 * (fold_width - 0.2))
    overbet_rate, big_rate, conf = opponent_overbet_rate(street)
    if conf > 0.0:
        if overbet_rate < 0.2:
            base_ratio *= 1.05 + 0.05 * conf
        elif overbet_rate > 0.5 and big_rate > 0.2:
            base_ratio *= 0.92 - 0.05 * conf
    continue_cost = getattr(getattr(player, "hero", None), "continue_cost", 0)
    if not bluff and continue_cost == 0:
        if street >= 5:
            base_ratio = max(base_ratio, 0.6)
        elif street >= 4:
            base_ratio = max(base_ratio, 0.55)
    base_ratio = max(0.18, min(1.25, base_ratio))

    jitter = rng.uniform(0.88, 1.12)
    target = int(pot * base_ratio * jitter)
    target = max(min_raise, min(max_raise, target))
    target = _ladder_raise_target(player, target, min_raise, max_raise)
    return max(min_raise, min(max_raise, target))


def _ladder_raise_target(player, ideal_target: int, min_raise: int, max_raise: int) -> int:
    # NOTE: Laddering only triggers on re-raises; if calls end the street, this won't realize the plan.
    if player is None or not hasattr(player, "hero"):
        return ideal_target
    street = getattr(player, "street", None)
    continue_cost = getattr(player.hero, "continue_cost", 0)
    plan_target = getattr(player.hero, "raise_plan_target", None)
    plan_street = getattr(player.hero, "raise_plan_street", None)
    if plan_target is not None and plan_street == street and continue_cost > 0:
        player.hero.raise_plan_target = None
        player.hero.raise_plan_street = None
        return max(min_raise, min(max_raise, int(plan_target)))
    if continue_cost == 0:
        fold_rate = fold_equity_estimate(None, min_raise, street, getattr(player.hero, "pot_total", 0))
        _, _, confidence = opponent_overbet_rate(street)
        if fold_rate < 0.35 or confidence < 0.35:
            return ideal_target
        player.hero.raise_plan_target = ideal_target
        player.hero.raise_plan_street = street
        if ideal_target <= int(min_raise * 1.2):
            return ideal_target
        step = min_raise + int((ideal_target - min_raise) * 0.55)
        return max(min_raise, min(max_raise, step))
    return ideal_target


def record_opponent_discard(card: str) -> None:
    card_int = stats._ensure_int_cards([card])[0]
    rank = stats.Card.get_rank_int(card_int)
    suit = stats.Card.get_suit_int(card_int)
    suit_index = {1: 0, 2: 1, 4: 2, 8: 3}.get(suit)
    if suit_index is None:
        return
    _OPPONENT_DISCARD_MODEL["rank_counts"][rank] += 1
    _OPPONENT_DISCARD_MODEL["suit_counts"][suit_index] += 1
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


def _apply_info_penalty_decay(value: float, player) -> float:
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


def update_info_penalty_from_round(player) -> None:
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


def _policy_bias(player, equity: float, pot_odds: float) -> float:
    global _LAST_HIDDEN
    features = _policy_features(player, equity, pot_odds)
    hidden = _RANDOM_POLICY.forward(features)
    _LAST_HIDDEN = hidden
    score = _RANDOM_POLICY.score(hidden)
    return max(-0.05, min(0.05, score))


def _policy_features(player, equity: float, pot_odds: float) -> List[float]:
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


def _update_random_policy(player) -> None:
    global _LAST_HIDDEN
    if not _USE_RANDOM_POLICY or _LAST_HIDDEN is None:
        return
    delta = getattr(player.hero, "delta", 0)
    reward = max(-1.0, min(1.0, delta / 100.0))
    _RANDOM_POLICY.update(_LAST_HIDDEN, reward, _RANDOM_POLICY_LR)
    _LAST_HIDDEN = None


def _should_pressure(player, equity: float) -> bool:
    variance = _equity_variance_factor(player)
    if equity < _PRESSURE_EQUITY_THRESHOLD + 0.02 * variance:
        return False
    fold_rate, low, _, conf, width = _opponent_fold_rate_band(player.street, None)
    if fold_rate <= 0.0:
        return True
    if conf < 0.25 or width > 0.35:
        return False
    return low >= _PRESSURE_FOLDRATE_MIN


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
