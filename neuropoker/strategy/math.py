"""Math + thresholds helpers extracted from strategy core."""

from dataclasses import dataclass, field
from functools import lru_cache
import json
import os
import random
from typing import Iterable, Mapping, Optional, Sequence, Tuple, Dict

import stats
from skeleton.states import BIG_BLIND, NUM_ROUNDS, SMALL_BLIND, STARTING_STACK


def active_variant_id() -> str:
    if _VARIANT_THRESHOLDS:
        return "thresholds"
    if _VARIANT_GRAPH:
        return "graph"
    if _VARIANT_PAIRWISE:
        return "pairwise"
    return "baseline"


def _load_thresholds(
    env_key: str,
    default: Tuple[float, float, float],
    param_key: Optional[str] = None,
    param_values: Optional[Mapping[str, float]] = None,
) -> Tuple[float, float, float]:
    if param_key and param_values:
        raw = param_values.get(param_key)
        if raw is not None:
            return raw
    raw = os.environ.get(env_key)
    if not raw:
        return default
    parts = [part.strip() for part in raw.split(",") if part.strip()]
    if len(parts) != 3:
        return default
    values = tuple(float(part) for part in parts)
    if values[0] < values[1] or values[1] < values[2]:
        return default
    return values


def _load_float_list(
    env_key: str,
    count: int,
    default: Tuple[float, ...],
    param_key: Optional[str] = None,
    param_values: Optional[Mapping[str, float]] = None,
) -> Tuple[float, ...]:
    if param_key and param_values:
        raw = param_values.get(param_key)
        if raw is not None:
            return raw
    raw = os.environ.get(env_key)
    if not raw:
        return default
    parts = [part.strip() for part in raw.split(",") if part.strip()]
    if len(parts) != count:
        return default
    try:
        values = tuple(float(part) for part in parts)
    except ValueError:
        return default
    for value in values:
        if value < 0:
            return default
    return values


def _load_param_file() -> Mapping[str, Tuple[float, ...]]:
    filename = os.environ.get("NEUROPOKER_PARAM_FILE", "best_params.json")
    path = os.path.join(os.path.dirname(__file__), filename)
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r") as handle:
            data = json.load(handle)
    except (ValueError, OSError):
        return {}
    if not isinstance(data, dict):
        return {}
    params = data.get("best_params", data)
    if not isinstance(params, dict):
        return {}
    grouped = {}

    def _maybe_group(key, keys):
        try:
            values = tuple(float(params[name]) for name in keys)
        except (KeyError, TypeError, ValueError):
            return
        grouped[key] = values

    _maybe_group("preflop_raise_thresholds", ("raise_strong", "raise_medium", "raise_light"))
    _maybe_group("preflop_call_thresholds", ("call_strong", "call_medium", "call_light"))
    _maybe_group("raise_size_fractions", ("raise_size_strong", "raise_size_medium", "raise_size_light"))
    _maybe_group("bluff_raise_fraction", ("bluff_raise_frac",))
    _maybe_group(
        "raise_margin_by_street",
        ("raise_margin_pre", "raise_margin_post", "raise_margin_turn", "raise_margin_river"),
    )
    _maybe_group(
        "call_margin_by_street",
        ("call_margin_pre", "call_margin_post", "call_margin_turn", "call_margin_river"),
    )
    _maybe_group("raise_call_ratio", ("raise_call_ratio",))
    _maybe_group("raise_call_penalty", ("raise_call_penalty",))
    _maybe_group("turn_raise_ratio", ("turn_raise_ratio",))
    _maybe_group("turn_raise_extra", ("turn_raise_extra",))
    _maybe_group("turn_raise_sample_mult", ("turn_raise_sample_mult",))
    _maybe_group("turn_raise_time_mult", ("turn_raise_time_mult",))
    _maybe_group("discard_bluff_rate", ("discard_bluff_rate",))
    _maybe_group("discard_bluff_raise_rate", ("discard_bluff_raise_rate",))
    _maybe_group("discard_bluff_raise_fraction", ("discard_bluff_raise_fraction",))
    _maybe_group("nut_raise_equity", ("nut_raise_equity",))
    _maybe_group("aggro_equity", ("aggro_equity",))
    _maybe_group("aggro_raise_bonus", ("aggro_raise_bonus",))
    _maybe_group("hard_fold_equity_by_street", ("hard_fold_post", "hard_fold_turn", "hard_fold_river"))
    _maybe_group("hard_fold_pot_odds_min", ("hard_fold_pot_odds_min",))
    _maybe_group(
        "fold_bias_by_street",
        ("fold_bias_pre", "fold_bias_post", "fold_bias_turn", "fold_bias_river"),
    )
    _maybe_group("tight_equity_threshold", ("tight_equity_threshold",))
    _maybe_group("tight_fold_lr", ("tight_fold_lr",))
    _maybe_group("bluff_weakness_threshold", ("bluff_weakness_threshold",))
    _maybe_group("bluff_disable_behind", ("bluff_disable_behind",))
    _maybe_group("pressure_equity_threshold", ("pressure_equity_threshold",))
    _maybe_group("pressure_raise_bonus", ("pressure_raise_bonus",))
    _maybe_group("pressure_foldrate_min", ("pressure_foldrate_min",))
    _maybe_group("lead_prot_thresholds", ("lead_prot_t1", "lead_prot_t2", "lead_prot_t3", "lead_prot_t4"))
    _maybe_group("lead_prot_adjustments", ("lead_prot_a1", "lead_prot_a2", "lead_prot_a3", "lead_prot_a4"))
    _maybe_group("lead_prot_size_mults", ("lead_prot_s1", "lead_prot_s2", "lead_prot_s3", "lead_prot_s4"))
    _maybe_group("lead_prot_pot_factor", ("lead_prot_pot_factor",))
    _maybe_group("opp_passive_threshold", ("opp_passive_threshold",))
    _maybe_group("opp_station_call_rate", ("opp_station_call_rate",))
    _maybe_group("opp_station_fold_rate", ("opp_station_fold_rate",))
    _maybe_group("opp_passive_value_mult", ("opp_passive_value_mult",))
    _maybe_group("opp_station_value_mult", ("opp_station_value_mult",))
    _maybe_group("early_boost_rounds", ("early_boost_r1", "early_boost_r2"))
    _maybe_group("early_boost_mults", ("early_boost_m1", "early_boost_m2"))
    _maybe_group("desperate_nut_threshold", ("desperate_nut_threshold",))
    _maybe_group("desperate_raise_margin", ("desperate_raise_margin",))
    _maybe_group("desperate_call_penalty", ("desperate_call_penalty",))
    return grouped


_PARAM_VALUES = _load_param_file()

_VARIANT_PAIRWISE = os.environ.get("NEUROPOKER_VARIANT_PAIRWISE", "0") == "1"
_VARIANT_GRAPH = os.environ.get("NEUROPOKER_VARIANT_GRAPH", "0") == "1"
_VARIANT_THRESHOLDS = os.environ.get("NEUROPOKER_VARIANT_THRESHOLDS", "0") == "1"

_PREFLOP_RAISE_THRESHOLDS = _load_thresholds(
    "NEUROPOKER_PREFLOP_RAISE_THRESHOLDS",
    (0.6850, 0.6200, 0.4600),
    param_key="preflop_raise_thresholds",
    param_values=_PARAM_VALUES,
)
_PREFLOP_CALL_THRESHOLDS = _load_thresholds(
    "NEUROPOKER_PREFLOP_CALL_THRESHOLDS",
    (0.6275, 0.5389, 0.3478),
    param_key="preflop_call_thresholds",
    param_values=_PARAM_VALUES,
)
_RAISE_SIZE_FRACTIONS = _load_float_list(
    "NEUROPOKER_RAISE_SIZE_FRACTIONS",
    3,
    (0.65, 0.42, 0.28),
    param_key="raise_size_fractions",
    param_values=_PARAM_VALUES,
)
_BLUFF_RAISE_FRACTION = _load_float_list(
    "NEUROPOKER_BLUFF_RAISE_FRACTION",
    1,
    (0.33,),
    param_key="bluff_raise_fraction",
    param_values=_PARAM_VALUES,
)[0]
_RAISE_MARGIN_BY_STREET = _load_float_list(
    "NEUROPOKER_RAISE_MARGIN_BY_STREET",
    4,
    (0.14, 0.11, 0.13, 0.13),
    param_key="raise_margin_by_street",
    param_values=_PARAM_VALUES,
)
_CALL_MARGIN_BY_STREET = _load_float_list(
    "NEUROPOKER_CALL_MARGIN_BY_STREET",
    4,
    (0.04, 0.03, 0.04, 0.05),
    param_key="call_margin_by_street",
    param_values=_PARAM_VALUES,
)
_RAISE_CALL_RATIO = _load_float_list(
    "NEUROPOKER_RAISE_CALL_RATIO",
    1,
    (0.65,),
    param_key="raise_call_ratio",
    param_values=_PARAM_VALUES,
)[0]
_RAISE_CALL_PENALTY = _load_float_list(
    "NEUROPOKER_RAISE_CALL_PENALTY",
    1,
    (0.10,),
    param_key="raise_call_penalty",
    param_values=_PARAM_VALUES,
)[0]
_TURN_RAISE_RATIO = _load_float_list(
    "NEUROPOKER_TURN_RAISE_RATIO",
    1,
    (0.5,),
    param_key="turn_raise_ratio",
    param_values=_PARAM_VALUES,
)[0]
_TURN_RAISE_EXTRA = _load_float_list(
    "NEUROPOKER_TURN_RAISE_EXTRA",
    1,
    (0.08,),
    param_key="turn_raise_extra",
    param_values=_PARAM_VALUES,
)[0]
_TURN_RAISE_SAMPLE_MULT = _load_float_list(
    "NEUROPOKER_TURN_RAISE_SAMPLE_MULT",
    1,
    (2.0,),
    param_key="turn_raise_sample_mult",
    param_values=_PARAM_VALUES,
)[0]
_TURN_RAISE_TIME_MULT = _load_float_list(
    "NEUROPOKER_TURN_RAISE_TIME_MULT",
    1,
    (1.5,),
    param_key="turn_raise_time_mult",
    param_values=_PARAM_VALUES,
)[0]
_DISCARD_BLUFF_RATE = _load_float_list(
    "NEUROPOKER_DISCARD_BLUFF_RATE",
    1,
    (0.008,),
    param_key="discard_bluff_rate",
    param_values=_PARAM_VALUES,
)[0]
_DISCARD_BLUFF_RAISE_RATE = _load_float_list(
    "NEUROPOKER_DISCARD_BLUFF_RAISE_RATE",
    1,
    (0.7,),
    param_key="discard_bluff_raise_rate",
    param_values=_PARAM_VALUES,
)[0]
_DISCARD_BLUFF_RAISE_FRACTION = _load_float_list(
    "NEUROPOKER_DISCARD_BLUFF_RAISE_FRACTION",
    1,
    (1.0,),
    param_key="discard_bluff_raise_fraction",
    param_values=_PARAM_VALUES,
)[0]
_NUT_RAISE_EQUITY = _load_float_list(
    "NEUROPOKER_NUT_RAISE_EQUITY",
    1,
    (0.84,),
    param_key="nut_raise_equity",
    param_values=_PARAM_VALUES,
)[0]
_AGGRO_EQUITY = _load_float_list(
    "NEUROPOKER_AGGRO_EQUITY",
    1,
    (0.68,),
    param_key="aggro_equity",
    param_values=_PARAM_VALUES,
)[0]
_AGGRO_RAISE_BONUS = _load_float_list(
    "NEUROPOKER_AGGRO_RAISE_BONUS",
    1,
    (0.04,),
    param_key="aggro_raise_bonus",
    param_values=_PARAM_VALUES,
)[0]
_HARD_FOLD_EQUITY_BY_STREET = _load_float_list(
    "NEUROPOKER_HARD_FOLD_EQUITY_BY_STREET",
    3,
    (0.24, 0.28, 0.31),
    param_key="hard_fold_equity_by_street",
    param_values=_PARAM_VALUES,
)
_HARD_FOLD_POT_ODDS_MIN = _load_float_list(
    "NEUROPOKER_HARD_FOLD_POT_ODDS_MIN",
    1,
    (0.05,),
    param_key="hard_fold_pot_odds_min",
    param_values=_PARAM_VALUES,
)[0]
_FOLD_BIAS_BY_STREET = _load_float_list(
    "NEUROPOKER_FOLD_BIAS_BY_STREET",
    4,
    (0.0, 0.015, 0.02, 0.025),
    param_key="fold_bias_by_street",
    param_values=_PARAM_VALUES,
)
_TIGHT_EQUITY_THRESHOLD = _load_float_list(
    "NEUROPOKER_TIGHT_EQUITY_THRESHOLD",
    1,
    (0.6,),
    param_key="tight_equity_threshold",
    param_values=_PARAM_VALUES,
)[0]
_TIGHT_FOLD_LR = _load_float_list(
    "NEUROPOKER_TIGHT_FOLD_LR",
    1,
    (0.12,),
    param_key="tight_fold_lr",
    param_values=_PARAM_VALUES,
)[0]
_BLUFF_WEAKNESS_THRESHOLD = _load_float_list(
    "NEUROPOKER_BLUFF_WEAKNESS_THRESHOLD",
    1,
    (0.08,),
    param_key="bluff_weakness_threshold",
    param_values=_PARAM_VALUES,
)[0]
_BLUFF_DISABLE_BEHIND = _load_float_list(
    "NEUROPOKER_BLUFF_DISABLE_BEHIND",
    1,
    (200.0,),
    param_key="bluff_disable_behind",
    param_values=_PARAM_VALUES,
)[0]
_PRESSURE_EQUITY_THRESHOLD = _load_float_list(
    "NEUROPOKER_PRESSURE_EQUITY_THRESHOLD",
    1,
    (0.60,),
    param_key="pressure_equity_threshold",
    param_values=_PARAM_VALUES,
)[0]
_PRESSURE_RAISE_BONUS = _load_float_list(
    "NEUROPOKER_PRESSURE_RAISE_BONUS",
    1,
    (0.04,),
    param_key="pressure_raise_bonus",
    param_values=_PARAM_VALUES,
)[0]
_PRESSURE_FOLDRATE_MIN = _load_float_list(
    "NEUROPOKER_PRESSURE_FOLDRATE_MIN",
    1,
    (0.10,),
    param_key="pressure_foldrate_min",
    param_values=_PARAM_VALUES,
)[0]

_LEAD_PROT_THRESHOLDS = _load_float_list(
    "NEUROPOKER_LEAD_PROT_THRESHOLDS",
    4,
    (0.2, 0.4, 0.6, 0.8),
    param_key="lead_prot_thresholds",
    param_values=_PARAM_VALUES,
)
_LEAD_PROT_ADJUSTMENTS = _load_float_list(
    "NEUROPOKER_LEAD_PROT_ADJUSTMENTS",
    4,
    (0.02, 0.04, 0.06, 0.10),
    param_key="lead_prot_adjustments",
    param_values=_PARAM_VALUES,
)
_LEAD_PROT_SIZE_MULTS = _load_float_list(
    "NEUROPOKER_LEAD_PROT_SIZE_MULTS",
    4,
    (0.5, 0.7, 0.8, 0.9),
    param_key="lead_prot_size_mults",
    param_values=_PARAM_VALUES,
)
_LEAD_PROT_POT_FACTOR = _load_float_list(
    "NEUROPOKER_LEAD_PROT_POT_FACTOR",
    1,
    (0.4,),
    param_key="lead_prot_pot_factor",
    param_values=_PARAM_VALUES,
)[0]

_EARLY_BOOST_ROUNDS = _load_float_list(
    "NEUROPOKER_EARLY_BOOST_ROUNDS",
    2,
    (100.0, 200.0),
    param_key="early_boost_rounds",
    param_values=_PARAM_VALUES,
)
_EARLY_BOOST_MULTS = _load_float_list(
    "NEUROPOKER_EARLY_BOOST_MULTS",
    2,
    (1.4, 1.25),
    param_key="early_boost_mults",
    param_values=_PARAM_VALUES,
)

_DESPERATE_NUT_THRESHOLD = _load_float_list(
    "NEUROPOKER_DESPERATE_NUT_THRESHOLD",
    1,
    (0.72,),
    param_key="desperate_nut_threshold",
    param_values=_PARAM_VALUES,
)[0]
_DESPERATE_RAISE_MARGIN = _load_float_list(
    "NEUROPOKER_DESPERATE_RAISE_MARGIN",
    1,
    (0.10,),
    param_key="desperate_raise_margin",
    param_values=_PARAM_VALUES,
)[0]
_DESPERATE_CALL_PENALTY = _load_float_list(
    "NEUROPOKER_DESPERATE_CALL_PENALTY",
    1,
    (0.08,),
    param_key="desperate_call_penalty",
    param_values=_PARAM_VALUES,
)[0]

_RIVER_VALUE_FLOOR = _load_float_list(
    "NEUROPOKER_RIVER_VALUE_FLOOR",
    1,
    (0.72,),
)[0]
_RIVER_MAX_RAISE_FRAC = _load_float_list(
    "NEUROPOKER_RIVER_MAX_RAISE_FRAC",
    1,
    (0.45,),
)[0]


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


def init_state(config: StrategyConfig) -> StrategyState:
    rng = random.Random(config.random_seed)
    return StrategyState(rng=rng)


def pot_odds_to_call(cost: int, pot: int) -> float:
    if cost <= 0:
        return 0.0
    return cost / max(1, pot + cost)


def volatility_adjustment(base_ev: float, variance_proxy: float, config: StrategyConfig) -> float:
    return base_ev + config.volatility_weight * variance_proxy


def select_discard(
    equities: Sequence[float],
    hero_hand: Sequence[str],
    config: StrategyConfig,
    state: StrategyState,
) -> int:
    if not equities:
        return _fast_discard_index(hero_hand)
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


def extract_features(values: Mapping[str, float]) -> Dict[str, float]:
    return {key: float(val) for key, val in values.items()}


def hebbian_update(
    weights: Mapping[str, float],
    features: Mapping[str, float],
    lr: float = 0.1,
) -> Dict[str, float]:
    updated = dict(weights)
    for key, value in features.items():
        updated[key] = updated.get(key, 0.0) + lr * value
    return updated


def mimetic_update(
    weights: Mapping[str, float],
    features: Mapping[str, float],
    lr: float = 0.1,
) -> Dict[str, float]:
    updated = dict(weights)
    for key, value in features.items():
        updated[key] = updated.get(key, 0.0) - lr * value
    return updated


def _blind_loss_for_rounds(rounds_left: int, starts_as_bb: bool) -> int:
    if rounds_left <= 0:
        return 0
    pairs = rounds_left // 2
    remainder = rounds_left % 2
    total = pairs * (BIG_BLIND + SMALL_BLIND)
    if remainder:
        total += BIG_BLIND if starts_as_bb else SMALL_BLIND
    return total


def _remaining_fold_loss(player) -> int:
    round_num = getattr(player, "round_num", 0)
    if round_num <= 0:
        return 0
    rounds_left = max(0, NUM_ROUNDS - round_num)
    current_loss = int(max(0, getattr(player.hero, "contribution", 0)))
    if current_loss <= 0:
        current_loss = BIG_BLIND if getattr(player.hero, "blind", False) else SMALL_BLIND
    starts_as_bb = not getattr(player.hero, "blind", False)
    future_loss = _blind_loss_for_rounds(rounds_left, starts_as_bb)
    return current_loss + future_loss


def _max_safe_loss_this_round(player) -> int:
    round_num = getattr(player, "round_num", 0)
    if round_num <= 0:
        return STARTING_STACK
    rounds_left = max(0, NUM_ROUNDS - round_num)
    hero_bankroll = getattr(player.hero, "bankroll", 0)
    opp_starts_as_bb = not getattr(player.hero, "blind", False)
    opp_future_loss = _blind_loss_for_rounds(rounds_left, opp_starts_as_bb)
    max_safe = opp_future_loss + hero_bankroll
    return max(0, max_safe)


def _raise_margin_by_street(street: int) -> float:
    if street <= 0:
        return _RAISE_MARGIN_BY_STREET[0]
    if street <= 3:
        return _RAISE_MARGIN_BY_STREET[1]
    if street <= 4:
        return _RAISE_MARGIN_BY_STREET[1]
    if street <= 5:
        return _RAISE_MARGIN_BY_STREET[2]
    return _RAISE_MARGIN_BY_STREET[3]


def _call_margin_by_street(street: int) -> float:
    if street <= 0:
        return _CALL_MARGIN_BY_STREET[0]
    if street <= 3:
        return _CALL_MARGIN_BY_STREET[1]
    if street <= 4:
        return _CALL_MARGIN_BY_STREET[1]
    if street <= 5:
        return _CALL_MARGIN_BY_STREET[2]
    return _CALL_MARGIN_BY_STREET[3]


def _lead_protection_adjustment(player, pot_total: int = 0) -> float:
    from .core import _ENABLE_LOCK_WIN

    if not _ENABLE_LOCK_WIN:
        return 0.0
    hero_bankroll = getattr(player.hero, "bankroll", 0)
    if hero_bankroll <= 0:
        return 0.0
    remaining = _remaining_fold_loss(player)
    if remaining <= 0:
        return _LEAD_PROT_ADJUSTMENTS[3]
    progress = hero_bankroll / remaining
    base_adjustment = 0.0
    if progress >= 1.0:
        base_adjustment = _LEAD_PROT_ADJUSTMENTS[3]
    elif progress >= _LEAD_PROT_THRESHOLDS[3]:
        base_adjustment = _LEAD_PROT_ADJUSTMENTS[3]
    elif progress >= _LEAD_PROT_THRESHOLDS[2]:
        base_adjustment = _LEAD_PROT_ADJUSTMENTS[2]
    elif progress >= _LEAD_PROT_THRESHOLDS[1]:
        base_adjustment = _LEAD_PROT_ADJUSTMENTS[1]
    elif progress >= _LEAD_PROT_THRESHOLDS[0]:
        base_adjustment = _LEAD_PROT_ADJUSTMENTS[0]
    if base_adjustment > 0 and pot_total > 0 and _LEAD_PROT_POT_FACTOR > 0:
        pot_ratio = min(1.0, pot_total / max(1, remaining))
        base_adjustment *= (1.0 + pot_ratio * _LEAD_PROT_POT_FACTOR)
    return base_adjustment


def _lead_protection_size_mult(player) -> float:
    from .core import _ENABLE_LOCK_WIN

    if not _ENABLE_LOCK_WIN:
        return 1.0
    hero_bankroll = getattr(player.hero, "bankroll", 0)
    if hero_bankroll <= 0:
        return 1.0
    remaining = _remaining_fold_loss(player)
    if remaining <= 0:
        return _LEAD_PROT_SIZE_MULTS[3]
    progress = hero_bankroll / remaining
    if progress >= 1.0:
        return _LEAD_PROT_SIZE_MULTS[3]
    elif progress >= _LEAD_PROT_THRESHOLDS[3]:
        return _LEAD_PROT_SIZE_MULTS[3]
    elif progress >= _LEAD_PROT_THRESHOLDS[2]:
        return _LEAD_PROT_SIZE_MULTS[2]
    elif progress >= _LEAD_PROT_THRESHOLDS[1]:
        return _LEAD_PROT_SIZE_MULTS[1]
    elif progress >= _LEAD_PROT_THRESHOLDS[0]:
        return _LEAD_PROT_SIZE_MULTS[0]
    return 1.0


def _fold_bias_by_street(street: int) -> float:
    if street <= 0:
        return _FOLD_BIAS_BY_STREET[0]
    if street <= 3:
        return _FOLD_BIAS_BY_STREET[1]
    if street <= 4:
        return _FOLD_BIAS_BY_STREET[2]
    return _FOLD_BIAS_BY_STREET[3]


def _hard_fold_equity(street: int) -> float:
    if street <= 3:
        return 0.0
    if street <= 4:
        return _HARD_FOLD_EQUITY_BY_STREET[0]
    if street <= 5:
        return _HARD_FOLD_EQUITY_BY_STREET[1]
    return _HARD_FOLD_EQUITY_BY_STREET[2]


def _trajectory_lock_adjustment(player) -> Tuple[float, float]:
    round_num = getattr(player, "round_num", 0)
    if round_num <= 10:
        return 0.0, 0.0
    hero_bankroll = getattr(player.hero, "bankroll", 0)
    if hero_bankroll >= 0:
        return 0.0, 0.0
    avg_delta = hero_bankroll / max(1.0, float(round_num - 1))
    if avg_delta >= 0:
        return 0.0, 0.0
    remaining = _remaining_fold_loss(player)
    if remaining <= 0:
        return 0.0, 0.0
    deficit_to_lock = remaining + hero_bankroll
    if deficit_to_lock <= 0:
        return 0.0, 0.0
    rounds_to_lock = deficit_to_lock / max(1e-6, -avg_delta)
    window = 200.0
    scale = max(0.0, min(1.0, (window - rounds_to_lock) / window))
    call_adj = 0.04 * scale
    raise_adj = -0.03 * scale
    if rounds_to_lock < 30:
        call_adj += 0.02
        raise_adj -= 0.02
    return raise_adj, call_adj


def is_flop(street: int) -> bool:
    return street == 4


def is_turn(street: int) -> bool:
    return street == 5


def is_river(street: int) -> bool:
    return street == 6


def _large_raise_ratio(player, pot_total: int) -> float:
    continue_cost = player.hero.continue_cost
    if continue_cost <= 0:
        return 0.0
    return continue_cost / float(max(1, pot_total))


def _is_large_raise(player, pot_total: int) -> bool:
    return _large_raise_ratio(player, pot_total) >= _RAISE_CALL_RATIO


def _should_discard_bluff(player) -> bool:
    if _DISCARD_BLUFF_RATE <= 0.0:
        return False
    return random.random() < _DISCARD_BLUFF_RATE


def _discard_bluff_index(hero_hand: Sequence[str]) -> int:
    cards_int = stats._ensure_int_cards(list(hero_hand))
    ranks = [stats.Card.get_rank_int(card) for card in cards_int]
    return max(range(len(ranks)), key=ranks.__getitem__)


def _discard_bluff_active(player) -> bool:
    return bool(getattr(player.hero, "discard_bluff", False))


def _consume_discard_bluff(player) -> None:
    if hasattr(player.hero, "discard_bluff"):
        player.hero.discard_bluff = False


def _should_discard_bluff_raise(player) -> bool:
    if not _discard_bluff_active(player):
        return False
    if player.street <= 3:
        return False
    if _DISCARD_BLUFF_RAISE_RATE <= 0.0:
        return False
    return random.random() < _DISCARD_BLUFF_RAISE_RATE


def _discard_bluff_raise_target(min_raise: int, max_raise: int) -> int:
    if max_raise <= 0:
        return min_raise
    target = int(max_raise * _DISCARD_BLUFF_RAISE_FRACTION)
    target = max(min_raise, target)
    return min(target, max_raise)


def _nut_raise_target(min_raise: int, max_raise: int) -> int:
    if max_raise <= 0:
        return min_raise
    return max(min_raise, max_raise)


def _raise_call_penalty(player, pot_total: int) -> float:
    if not is_turn(player.street):
        return 0.0
    continue_cost = player.hero.continue_cost
    if continue_cost <= 0:
        return 0.0
    last_bet_street = getattr(player, "_last_bet_street", None)
    if last_bet_street != player.street:
        return 0.0
    ratio = continue_cost / float(max(1, pot_total))
    if ratio < _RAISE_CALL_RATIO:
        return 0.0
    return _RAISE_CALL_PENALTY


def _turn_raise_extra(player, pot_total: int) -> float:
    if not is_turn(player.street):
        return 0.0
    continue_cost = player.hero.continue_cost
    if continue_cost <= 0:
        return 0.0
    last_bet_street = getattr(player, "_last_bet_street", None)
    if last_bet_street != player.street:
        return 0.0
    ratio = continue_cost / float(max(1, pot_total))
    if ratio < _TURN_RAISE_RATIO:
        return 0.0
    return _TURN_RAISE_EXTRA


def _raise_size(pot_total: int, min_raise: int, max_raise: int, equity: float, bluff: bool = False, value_mult: float = 1.0) -> int:
    if max_raise <= 0:
        return 0
    if bluff:
        target = int(pot_total * _BLUFF_RAISE_FRACTION)
        target = max(min_raise, target)
        return min(target, max_raise)
    strong_frac, medium_frac, light_frac = _RAISE_SIZE_FRACTIONS
    if equity >= 0.7:
        target = int(pot_total * strong_frac * value_mult)
    elif equity >= 0.6:
        target = int(pot_total * medium_frac * value_mult)
    else:
        target = int(pot_total * light_frac)
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


def _line_call_penalty(line_key: Optional[str], street: int) -> float:
    if street < 5 or not line_key:
        return 0.0
    if line_key in ("C-C-K-K", "C-K-C-K", "K-K-K-K"):
        return 0.06 if street >= 6 else 0.04
    return 0.0


def _confidence_call_penalty(confidence: float, street: int) -> float:
    if street < 5:
        return 0.0
    if confidence >= 0.35:
        return 0.0
    if confidence < 0.15:
        return 0.05 if street >= 6 else 0.035
    return 0.03 if street >= 6 else 0.02


def _suppress_medium_raise_target(
    target: int,
    min_raise: int,
    pot_total: int,
    equity: float,
) -> int:
    if pot_total <= 0 or target <= 0:
        return target
    ratio = target / float(max(1, pot_total))
    if ratio <= 0.5 or ratio > 1.0:
        return target
    if equity >= 0.74:
        return target
    if equity >= 0.62:
        return max(min_raise, int(pot_total * 0.45))
    return max(min_raise, int(pot_total * 0.32))


@lru_cache(maxsize=10_000)
def _fast_discard_index_cached(hero_hand_tuple: Tuple[str, ...]) -> int:
    if not hero_hand_tuple:
        return 0
    cards_int = stats._ensure_int_cards(list(hero_hand_tuple))
    ranks = [stats.Card.get_rank_int(card) for card in cards_int]
    suits = [stats.Card.get_suit_int(card) for card in cards_int]
    scores = []
    for idx, rank in enumerate(ranks):
        score = rank
        if ranks.count(rank) > 1:
            score += 6
        if suits.count(suits[idx]) > 1:
            score += 2
        for jdx, other in enumerate(ranks):
            if idx == jdx:
                continue
            if abs(rank - other) <= 3:
                score += 1
                break
        scores.append(score)
    best_keep = max(range(len(scores)), key=scores.__getitem__)
    discard = min(
        (i for i in range(len(scores)) if i != best_keep),
        key=lambda i: (scores[i], ranks[i]),
    )
    return discard


def _fast_discard_index(hero_hand: Sequence[str]) -> int:
    return _fast_discard_index_cached(tuple(hero_hand))


@lru_cache(maxsize=50_000)
def _board_is_paired(board_tuple: Tuple[int, ...]) -> bool:
    ranks = [stats.Card.get_rank_int(card) for card in board_tuple]
    return len(set(ranks)) < len(ranks)


@lru_cache(maxsize=50_000)
def _board_is_flushy(board_tuple: Tuple[int, ...]) -> bool:
    suits = [stats.Card.get_suit_int(card) for card in board_tuple]
    for suit in set(suits):
        if suits.count(suit) >= 3:
            return True
    return False


def _board_texture_adjustments_cached(board_tuple: Tuple[str, ...]) -> Tuple[float, float, float]:
    if not board_tuple:
        return 0.0, 0.0, 1.0
    board_int = stats._ensure_int_cards(list(board_tuple))
    board_int_tuple = tuple(board_int)
    paired = _board_is_paired(board_int_tuple)
    flushy = _board_is_flushy(board_int_tuple)
    if paired or flushy:
        return 0.03, 0.02, 0.7
    return 0.0, 0.0, 1.0


def _board_texture_adjustments(board_cards: Sequence[str]) -> Tuple[float, float, float]:
    return _board_texture_adjustments_cached(tuple(board_cards))


def _pairwise_adjustments(hero_hand: Sequence[str], board_cards: Sequence[str]) -> Tuple[float, float, float]:
    if not hero_hand:
        return 0.0, 0.0, 0.0
    cards_int = stats._ensure_int_cards(list(hero_hand))
    board_int = stats._ensure_int_cards(list(board_cards))
    ranks = [stats.Card.get_rank_int(card) for card in cards_int]
    suits = [stats.Card.get_suit_int(card) for card in cards_int]
    board_ranks = [stats.Card.get_rank_int(card) for card in board_int]
    board_suits = [stats.Card.get_suit_int(card) for card in board_int]
    matches = sum(1 for rank in ranks if rank in board_ranks)
    suit_matches = sum(1 for suit in suits if suit in board_suits)
    connectivity = 0.0
    ranks_sorted = sorted(ranks)
    for i in range(len(ranks_sorted) - 1):
        if abs(ranks_sorted[i + 1] - ranks_sorted[i]) <= 2:
            connectivity += 1.0
    bias = min(0.02, 0.005 * matches + 0.004 * suit_matches + 0.004 * connectivity)
    return bias, bias, 0.0


def _graph_adjustments(hero_hand: Sequence[str], board_cards: Sequence[str]) -> Tuple[float, float, float]:
    if not hero_hand:
        return 0.0, 0.0, 0.0
    cards_int = stats._ensure_int_cards(list(hero_hand))
    board_int = stats._ensure_int_cards(list(board_cards))
    board_ranks = [stats.Card.get_rank_int(card) for card in board_int]
    board_suits = [stats.Card.get_suit_int(card) for card in board_int]
    ranks = [stats.Card.get_rank_int(card) for card in cards_int]
    suits = [stats.Card.get_suit_int(card) for card in cards_int]
    paired = len(set(board_ranks)) < len(board_ranks)
    flushy = False
    for suit in set(board_suits):
        if board_suits.count(suit) >= 3:
            flushy = True
            break
    match_score = sum(1 for rank in ranks if rank in board_ranks)
    suit_score = sum(1 for suit in suits if suit in board_suits)
    texture = 0.5 if paired or flushy else 0.0
    bias = min(0.02, 0.004 * match_score + 0.003 * suit_score + 0.003 * texture)
    return bias, bias, 0.0


def _threshold_adjustments(player) -> Tuple[float, float, float]:
    if player.street <= 0:
        return 0.0, 0.0, 0.0
    fold_rate = 0.0
    if hasattr(player, "opponent_fold_rate"):
        fold_rate = player.opponent_fold_rate
    if fold_rate <= 0.0:
        return 0.0, 0.0, 0.0
    raise_adj = max(-0.05, min(0.05, (0.3 - fold_rate) * 0.2))
    call_adj = max(-0.05, min(0.05, (0.3 - fold_rate) * 0.2))
    return 0.0, raise_adj, call_adj


def _variant_adjustments(
    variant_id: str,
    player,
    hero_hand: Sequence[str],
    board_cards: Sequence[str],
) -> Tuple[float, float, float]:
    if variant_id == "pairwise":
        return _pairwise_adjustments(hero_hand, board_cards)
    if variant_id == "graph":
        return _graph_adjustments(hero_hand, board_cards)
    if variant_id == "thresholds":
        return _threshold_adjustments(player)
    return 0.0, 0.0, 0.0


def _equity_budget(player) -> Tuple[int, float, int]:
    street = player.street
    round_num = getattr(player, "round_num", 0)
    hero_bankroll = getattr(player.hero, "bankroll", 0)
    game_clock = getattr(player, "game_clock", None)
    if street <= 0:
        samples = 60
        max_seconds = 0.012
        discard_samples = 8
    elif street <= 3:
        samples = 80
        max_seconds = 0.015
        discard_samples = 10
    else:
        samples = 120
        max_seconds = 0.020
        discard_samples = 12
    early_r1, early_r2 = _EARLY_BOOST_ROUNDS
    early_m1, early_m2 = _EARLY_BOOST_MULTS
    if round_num <= early_r1 and (game_clock is None or game_clock > 40):
        samples = int(samples * early_m1)
        max_seconds *= (early_m1 * 0.9)
        discard_samples = int(discard_samples * (1.0 + (early_m1 - 1.0) * 0.6))
    elif round_num <= early_r2 and (game_clock is None or game_clock > 35):
        samples = int(samples * early_m2)
        max_seconds *= (early_m2 * 0.95)
        discard_samples = int(discard_samples * (1.0 + (early_m2 - 1.0) * 0.5))
    if game_clock is None or game_clock > 30:
        scale = 1.0
        if hero_bankroll < -50:
            scale = max(scale, 1.3)
        elif hero_bankroll < -20:
            scale = max(scale, 1.15)
        if round_num > 750:
            scale = max(scale, 1.25)
        elif round_num > 500:
            scale = max(scale, 1.1)
        samples = int(samples * scale)
        discard_samples = int(discard_samples * min(1.15, scale))
    if game_clock is not None and game_clock < 20:
        samples = max(40, samples // 2)
        max_seconds = max(0.008, max_seconds * 0.5)
        discard_samples = max(6, discard_samples // 2)
    if game_clock is not None and game_clock < 10:
        samples = max(20, samples // 2)
        max_seconds = max(0.004, max_seconds * 0.5)
        discard_samples = max(4, discard_samples // 2)
    max_seconds *= 1.3
    return samples, max_seconds, discard_samples


def _turn_raise_ratio(player, pot_total: int) -> Optional[float]:
    if not is_turn(player.street):
        return None
    continue_cost = player.hero.continue_cost
    if continue_cost <= 0:
        return None
    last_bet_street = getattr(player, "_last_bet_street", None)
    if last_bet_street != player.street:
        return None
    return continue_cost / float(max(1, pot_total))


def _adjust_budget_for_turn_raise(player, pot_total: int, samples: int, max_seconds: float, discard_samples: int) -> Tuple[int, float, int]:
    ratio = _turn_raise_ratio(player, pot_total)
    if ratio is None or ratio < _TURN_RAISE_RATIO:
        return samples, max_seconds, discard_samples
    sample_mult = max(1.0, _TURN_RAISE_SAMPLE_MULT)
    time_mult = max(1.0, _TURN_RAISE_TIME_MULT)
    samples = min(200, int(samples * sample_mult))
    max_seconds = min(0.05, max_seconds * time_mult)
    discard_samples = min(30, max(4, int(discard_samples * 1.2)))
    return samples, max_seconds, discard_samples


def _river_raise_call_penalty(player, pot_total: int) -> float:
    if not is_river(player.street):
        return 0.0
    continue_cost = player.hero.continue_cost
    if continue_cost <= 0:
        return 0.0
    last_bet_street = getattr(player, "_last_bet_street", None)
    if last_bet_street != player.street:
        return 0.0
    ratio = continue_cost / float(max(1, pot_total))
    if ratio < _RAISE_CALL_RATIO:
        return 0.0
    return _RAISE_CALL_PENALTY


def _river_raise_extra(player, pot_total: int) -> float:
    if not is_river(player.street):
        return 0.0
    continue_cost = player.hero.continue_cost
    if continue_cost <= 0:
        return 0.0
    last_bet_street = getattr(player, "_last_bet_street", None)
    if last_bet_street != player.street:
        return 0.0
    ratio = continue_cost / float(max(1, pot_total))
    if ratio < _TURN_RAISE_RATIO:
        return 0.0
    return _TURN_RAISE_EXTRA


def _river_raise_ratio(player, pot_total: int) -> Optional[float]:
    if not is_river(player.street):
        return None
    continue_cost = player.hero.continue_cost
    if continue_cost <= 0:
        return None
    last_bet_street = getattr(player, "_last_bet_street", None)
    if last_bet_street != player.street:
        return None
    return continue_cost / float(max(1, pot_total))


def _adjust_budget_for_river_raise(player, pot_total: int, samples: int, max_seconds: float, discard_samples: int) -> Tuple[int, float, int]:
    ratio = _river_raise_ratio(player, pot_total)
    if ratio is None or ratio < _TURN_RAISE_RATIO:
        return samples, max_seconds, discard_samples
    sample_mult = max(1.0, _TURN_RAISE_SAMPLE_MULT)
    time_mult = max(1.0, _TURN_RAISE_TIME_MULT)
    samples = min(200, int(samples * sample_mult))
    max_seconds = min(0.05, max_seconds * time_mult)
    discard_samples = min(30, max(4, int(discard_samples * 1.2)))
    return samples, max_seconds, discard_samples


def _discard_budget(player) -> Tuple[int, float]:
    street = player.street
    if street <= 2:
        samples = 60
        max_seconds = 0.015
    else:
        samples = 80
        max_seconds = 0.015
    game_clock = getattr(player, "game_clock", None)
    if game_clock is not None and game_clock < 20:
        samples = max(30, samples // 2)
    if game_clock is not None and game_clock < 10:
        samples = max(15, samples // 2)
    max_seconds *= 1.3
    return samples, max_seconds
