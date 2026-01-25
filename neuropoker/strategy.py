from dataclasses import dataclass, field
from functools import lru_cache
import json
import math
import os
import random
import time
from typing import Any, Iterable, Mapping, Optional, Sequence, Tuple, Dict, List
from skeleton.actions import CallAction, CheckAction, DiscardAction, FoldAction, RaiseAction

import stats
from skeleton.states import BIG_BLIND, NUM_ROUNDS, SMALL_BLIND, STARTING_STACK

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
    "inferred": {"sum": 0.0, "total": 0.0},
}

_USE_RANDOM_POLICY = os.environ.get("NEUROPOKER_USE_RANDOM_POLICY", "0") == "1"
_RANDOM_POLICY_SEED = int(os.environ.get("NEUROPOKER_RANDOM_POLICY_SEED", "7") or "7")
_DISABLE_PREFLOP_MIX = os.environ.get("NEUROPOKER_DISABLE_PREFLOP_MIX", "0") == "1"
_VARIANT_PAIRWISE = os.environ.get("NEUROPOKER_VARIANT_PAIRWISE", "0") == "1"
_VARIANT_GRAPH = os.environ.get("NEUROPOKER_VARIANT_GRAPH", "0") == "1"
_VARIANT_THRESHOLDS = os.environ.get("NEUROPOKER_VARIANT_THRESHOLDS", "0") == "1"
_RANDOM_POLICY_LR = 0.02
_RANDOM_POLICY_HIDDEN = 16
_LAST_HIDDEN = None
_POLICY_EPSILON = float(os.environ.get("NEUROPOKER_POLICY_EPSILON", "0.15") or "0.15")
_POLICY_LR = float(os.environ.get("NEUROPOKER_POLICY_LR", "0.15") or "0.15")
_POLICY_STATS = {}
_ENABLE_LOCK_WIN = os.environ.get("NEUROPOKER_ENABLE_LOCK_WIN", "1") == "1"


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
    try:
        values = tuple(float(part) for part in parts)
    except ValueError:
        return default
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
    _maybe_group(
        "preflop_raise_thresholds",
        ("raise_strong", "raise_medium", "raise_light"),
    )
    _maybe_group(
        "preflop_call_thresholds",
        ("call_strong", "call_medium", "call_light"),
    )
    _maybe_group(
        "raise_size_fractions",
        ("raise_size_strong", "raise_size_medium", "raise_size_light"),
    )
    _maybe_group(
        "bluff_raise_fraction",
        ("bluff_raise_frac",),
    )
    _maybe_group(
        "raise_margin_by_street",
        ("raise_margin_pre", "raise_margin_post", "raise_margin_turn", "raise_margin_river"),
    )
    _maybe_group(
        "call_margin_by_street",
        ("call_margin_pre", "call_margin_post", "call_margin_turn", "call_margin_river"),
    )
    _maybe_group(
        "raise_call_ratio",
        ("raise_call_ratio",),
    )
    _maybe_group(
        "raise_call_penalty",
        ("raise_call_penalty",),
    )
    _maybe_group(
        "turn_raise_ratio",
        ("turn_raise_ratio",),
    )
    _maybe_group(
        "turn_raise_extra",
        ("turn_raise_extra",),
    )
    _maybe_group(
        "turn_raise_sample_mult",
        ("turn_raise_sample_mult",),
    )
    _maybe_group(
        "turn_raise_time_mult",
        ("turn_raise_time_mult",),
    )
    _maybe_group(
        "discard_bluff_rate",
        ("discard_bluff_rate",),
    )
    _maybe_group(
        "discard_bluff_raise_rate",
        ("discard_bluff_raise_rate",),
    )
    _maybe_group(
        "discard_bluff_raise_fraction",
        ("discard_bluff_raise_fraction",),
    )
    _maybe_group(
        "nut_raise_equity",
        ("nut_raise_equity",),
    )
    _maybe_group(
        "aggro_equity",
        ("aggro_equity",),
    )
    _maybe_group(
        "aggro_raise_bonus",
        ("aggro_raise_bonus",),
    )
    _maybe_group(
        "hard_fold_equity_by_street",
        ("hard_fold_post", "hard_fold_turn", "hard_fold_river"),
    )
    _maybe_group(
        "hard_fold_pot_odds_min",
        ("hard_fold_pot_odds_min",),
    )
    _maybe_group(
        "fold_bias_by_street",
        ("fold_bias_pre", "fold_bias_post", "fold_bias_turn", "fold_bias_river"),
    )
    _maybe_group(
        "tight_equity_threshold",
        ("tight_equity_threshold",),
    )
    _maybe_group(
        "tight_fold_lr",
        ("tight_fold_lr",),
    )
    _maybe_group(
        "bluff_weakness_threshold",
        ("bluff_weakness_threshold",),
    )
    _maybe_group(
        "bluff_disable_behind",
        ("bluff_disable_behind",),
    )
    _maybe_group(
        "pressure_equity_threshold",
        ("pressure_equity_threshold",),
    )
    _maybe_group(
        "pressure_raise_bonus",
        ("pressure_raise_bonus",),
    )
    _maybe_group(
        "pressure_foldrate_min",
        ("pressure_foldrate_min",),
    )
    # Lead protection parameters
    _maybe_group(
        "lead_prot_thresholds",
        ("lead_prot_t1", "lead_prot_t2", "lead_prot_t3", "lead_prot_t4"),
    )
    _maybe_group(
        "lead_prot_adjustments",
        ("lead_prot_a1", "lead_prot_a2", "lead_prot_a3", "lead_prot_a4"),
    )
    _maybe_group(
        "lead_prot_size_mults",
        ("lead_prot_s1", "lead_prot_s2", "lead_prot_s3", "lead_prot_s4"),
    )
    _maybe_group(
        "lead_prot_pot_factor",
        ("lead_prot_pot_factor",),
    )
    # Opponent modeling
    _maybe_group(
        "opp_passive_threshold",
        ("opp_passive_threshold",),
    )
    _maybe_group(
        "opp_station_call_rate",
        ("opp_station_call_rate",),
    )
    _maybe_group(
        "opp_station_fold_rate",
        ("opp_station_fold_rate",),
    )
    _maybe_group(
        "opp_passive_value_mult",
        ("opp_passive_value_mult",),
    )
    _maybe_group(
        "opp_station_value_mult",
        ("opp_station_value_mult",),
    )
    # Early-round boost
    _maybe_group(
        "early_boost_rounds",
        ("early_boost_r1", "early_boost_r2"),
    )
    _maybe_group(
        "early_boost_mults",
        ("early_boost_m1", "early_boost_m2"),
    )
    # Desperate policy
    _maybe_group(
        "desperate_nut_threshold",
        ("desperate_nut_threshold",),
    )
    _maybe_group(
        "desperate_raise_margin",
        ("desperate_raise_margin",),
    )
    _maybe_group(
        "desperate_call_penalty",
        ("desperate_call_penalty",),
    )
    return grouped


_PARAM_VALUES = _load_param_file()

_PREFLOP_RAISE_THRESHOLDS = _load_thresholds(
    "NEUROPOKER_PREFLOP_RAISE_THRESHOLDS",
    (0.7213, 0.6632, 0.5000),
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
    (0.75, 0.5, 0.33),
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
    (0.2, 0.15, 0.18, 0.16),
    param_key="raise_margin_by_street",
    param_values=_PARAM_VALUES,
)
_CALL_MARGIN_BY_STREET = _load_float_list(
    "NEUROPOKER_CALL_MARGIN_BY_STREET",
    4,
    (0.05, 0.03, 0.05, 0.06),
    param_key="call_margin_by_street",
    param_values=_PARAM_VALUES,
)
_RAISE_CALL_RATIO = _load_float_list(
    "NEUROPOKER_RAISE_CALL_RATIO",
    1,
    (0.6,),
    param_key="raise_call_ratio",
    param_values=_PARAM_VALUES,
)[0]
_RAISE_CALL_PENALTY = _load_float_list(
    "NEUROPOKER_RAISE_CALL_PENALTY",
    1,
    (0.08,),
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
    (0.06,),
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
    (0.86,),
    param_key="nut_raise_equity",
    param_values=_PARAM_VALUES,
)[0]
_AGGRO_EQUITY = _load_float_list(
    "NEUROPOKER_AGGRO_EQUITY",
    1,
    (0.7,),
    param_key="aggro_equity",
    param_values=_PARAM_VALUES,
)[0]
_AGGRO_RAISE_BONUS = _load_float_list(
    "NEUROPOKER_AGGRO_RAISE_BONUS",
    1,
    (0.05,),
    param_key="aggro_raise_bonus",
    param_values=_PARAM_VALUES,
)[0]
_HARD_FOLD_EQUITY_BY_STREET = _load_float_list(
    "NEUROPOKER_HARD_FOLD_EQUITY_BY_STREET",
    3,
    (0.26, 0.30, 0.33),
    param_key="hard_fold_equity_by_street",
    param_values=_PARAM_VALUES,
)
_HARD_FOLD_POT_ODDS_MIN = _load_float_list(
    "NEUROPOKER_HARD_FOLD_POT_ODDS_MIN",
    1,
    (0.06,),
    param_key="hard_fold_pot_odds_min",
    param_values=_PARAM_VALUES,
)[0]
_FOLD_BIAS_BY_STREET = _load_float_list(
    "NEUROPOKER_FOLD_BIAS_BY_STREET",
    4,
    (0.0, 0.02, 0.03, 0.04),
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
    (0.62,),
    param_key="pressure_equity_threshold",
    param_values=_PARAM_VALUES,
)[0]
_PRESSURE_RAISE_BONUS = _load_float_list(
    "NEUROPOKER_PRESSURE_RAISE_BONUS",
    1,
    (0.05,),
    param_key="pressure_raise_bonus",
    param_values=_PARAM_VALUES,
)[0]
_PRESSURE_FOLDRATE_MIN = _load_float_list(
    "NEUROPOKER_PRESSURE_FOLDRATE_MIN",
    1,
    (0.12,),
    param_key="pressure_foldrate_min",
    param_values=_PARAM_VALUES,
)[0]

# Lead protection: stepwise function for nonlinear tuning
# Each tier: (progress_threshold, margin_adjustment, size_multiplier)
# Using separate arrays for Optuna compatibility
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
    (0.95, 0.9, 0.85, 0.8),
    param_key="lead_prot_size_mults",
    param_values=_PARAM_VALUES,
)
# Pot-relative modifier: extra tightening when pot is large relative to our lead
_LEAD_PROT_POT_FACTOR = _load_float_list(
    "NEUROPOKER_LEAD_PROT_POT_FACTOR",
    1,
    (0.3,),  # multiply adjustment by (1 + pot/remaining * factor)
    param_key="lead_prot_pot_factor",
    param_values=_PARAM_VALUES,
)[0]

# Opponent modeling thresholds
_OPP_PASSIVE_THRESHOLD = _load_float_list(
    "NEUROPOKER_OPP_PASSIVE_THRESHOLD",
    1,
    (0.20,),
    param_key="opp_passive_threshold",
    param_values=_PARAM_VALUES,
)[0]
_OPP_STATION_CALL_RATE = _load_float_list(
    "NEUROPOKER_OPP_STATION_CALL_RATE",
    1,
    (0.50,),
    param_key="opp_station_call_rate",
    param_values=_PARAM_VALUES,
)[0]
_OPP_STATION_FOLD_RATE = _load_float_list(
    "NEUROPOKER_OPP_STATION_FOLD_RATE",
    1,
    (0.25,),
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

# Early-round sampling boost
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

# Desperate policy thresholds
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
    enable_desperate: True
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


_RANDOM_POLICY = RandomFeaturePolicy(seed=_RANDOM_POLICY_SEED)


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
    hero_hand: Sequence[str],
    config: StrategyConfig,
    state: StrategyState,
) -> int:
    """Pick the discard index, preferring higher equity with mild randomness."""
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


def update_opponent_model(
    model: OpponentModel,
    action: str,
    street: Optional[int] = None,
    inferred_strength: Optional[float] = None,
) -> None:
    """Track opponent action frequencies and lightweight range cues."""
    if action == "fold":
        model.fold_count += 1
        if street is not None:
            record_opponent_fold(street)
    elif action == "call":
        model.call_count += 1
        if street is not None:
            record_opponent_call(street)
    elif action == "raise":
        model.raise_count += 1
        if street is not None:
            record_opponent_raise(street)
    if street is not None:
        record_opponent_action(street)
    if inferred_strength is not None:
        record_inferred_range(inferred_strength)


def extract_features(values: Mapping[str, float]) -> Dict[str, float]:
    """Normalize and sanitize feature inputs for simple learners."""
    return {key: float(val) for key, val in values.items()}


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


def begin_round(player: PlayerView) -> str:
    """Reset round policy selection; play() will choose when needed."""
    player.hero.policy_class = None
    player.hero.policy_round = None
    player.hero.discard_bluff = False
    return "unset"


def update_policy_from_round(player: PlayerView) -> None:
    """Update policy scores based on the round outcome."""
    policy_class = getattr(player.hero, "policy_class", None)
    if not policy_class:
        return
    _ensure_policy_stats()
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


def _policy_key(policy_class) -> str:
    return getattr(policy_class, "__name__", str(policy_class))


def _policy_classes():
    import strategies.bluff as bluff_policy
    import strategies.tight as tight_policy

    return (bluff_policy.BluffPolicy, tight_policy.TightPolicy)


def _ensure_policy_stats() -> None:
    for policy_class in _policy_classes():
        key = _policy_key(policy_class)
        _POLICY_STATS.setdefault(key, {"avg": 0.0, "count": 0.0})


def _select_round_policy(player: PlayerView):
    policy_class = getattr(player.hero, "policy_class", None)
    policy_round = getattr(player.hero, "policy_round", None)
    round_num = getattr(player, "round_num", 0)
    if policy_class in _policy_classes() and policy_round == round_num:
        return policy_class
    return None


def _estimate_policy_equity(player: PlayerView) -> float:
    hero_hand = list(player.hero.hand)
    board_cards = list(player.community)
    if player.street <= 0:
        return stats.preflop_strength(hero_hand)
    samples, max_seconds, discard_samples = _equity_budget(player)
    return stats.estimate_equity(
        hero_hand,
        board_cards,
        samples=max(16, samples // 6),
        max_seconds=min(0.004, max_seconds * 0.2),
        discard_samples=max(4, discard_samples // 2),
    )


def _opponent_weakness_score(player: PlayerView) -> float:
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


def _opponent_bluff_adjustment(player: PlayerView) -> float:
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


def _bluff_threshold(player: PlayerView) -> float:
    _ensure_policy_stats()
    bluff_cls, tight_cls = _policy_classes()
    bluff_avg = _POLICY_STATS[_policy_key(bluff_cls)]["avg"]
    tight_avg = _POLICY_STATS[_policy_key(tight_cls)]["avg"]
    bias = max(-0.05, min(0.05, (bluff_avg - tight_avg) * 0.2))
    opponent_bias = _opponent_bluff_adjustment(player)
    return max(0.0, _BLUFF_WEAKNESS_THRESHOLD - bias + opponent_bias)


def _tight_equity_threshold(player: PlayerView) -> float:
    base = _TIGHT_EQUITY_THRESHOLD
    fold_rate = _opponent_fold_rate(player.street, None)
    if fold_rate <= 0.0 or _TIGHT_FOLD_LR <= 0.0:
        return base
    adjustment = fold_rate * _TIGHT_FOLD_LR
    return max(0.0, base - adjustment)


def _choose_policy_for_round(player: PlayerView, equity: float):
    bluff_cls, tight_cls = _policy_classes()
    # Always use TightPolicy for hands with equity >= threshold
    if equity >= _tight_equity_threshold(player):
        return tight_cls
    # Also use TightPolicy for any +EV hand (equity >= pot_odds)
    pot_total = max(1, getattr(player.hero, "pot_total", 1))
    continue_cost = getattr(player.hero, "continue_cost", 0)
    pot_odds = pot_odds_to_call(continue_cost, pot_total)
    if equity >= pot_odds:
        return tight_cls
    weakness = _opponent_weakness_score(player)
    if weakness >= _bluff_threshold(player) and _bluff_allowed(player):
        return bluff_cls
    return None


def _bluff_allowed(player: PlayerView) -> bool:
    if _BLUFF_DISABLE_BEHIND <= 0:
        return True
    bankroll = getattr(player.hero, "bankroll", 0)
    return bankroll >= -_BLUFF_DISABLE_BEHIND


def _remaining_fold_loss(player: PlayerView) -> int:
    round_num = getattr(player, "round_num", 0)
    if round_num <= 0:
        return 0
    rounds_left = max(0, NUM_ROUNDS - round_num )
    if rounds_left <= 0:
        return 0
    current_loss = int(max(0, getattr(player.hero, "contribution", 0)))
    if current_loss <= 0:
        current_loss = BIG_BLIND if getattr(player.hero, "blind", False) else SMALL_BLIND
    future_loss = 0
    big_blind = not getattr(player.hero, "blind", False)

    future_loss =  3*(rounds_left // 2)
    if big_blind and rounds_left % 2:
        future_loss += 1
    elif rounds_left % 2:
        future_loss += 2
    return future_loss

def _should_lock_win(player: PlayerView) -> bool:
    if not _ENABLE_LOCK_WIN:
        return False
    bankroll = getattr(player.hero, "bankroll", 0)
    if bankroll <= 0:
        return False
    return bankroll > _remaining_fold_loss(player) + player.hero.pot_total


def _lock_win_action(player: PlayerView):
    from skeleton.actions import CallAction, CheckAction, DiscardAction, FoldAction

    legal_actions = set(player.hero.legal_actions)
    if DiscardAction in legal_actions:
        return DiscardAction(0)
    if FoldAction in legal_actions and player.hero.continue_cost > 0:
        return FoldAction()
    if CheckAction in legal_actions and player.hero.continue_cost == 0:
        return CheckAction()
    if CallAction in legal_actions:
        return CallAction()
    if CheckAction in legal_actions:
        return CheckAction()
    return FoldAction()


def _opponent_future_fold_loss(player: PlayerView, rounds_left: int) -> int:
    if rounds_left <= 0:
        return 0
    opponent_big = bool(getattr(player.hero, "blind", False))
    total = 0
    for _ in range(rounds_left):
        total += BIG_BLIND if opponent_big else SMALL_BLIND
        opponent_big = not opponent_big
    return total


def _opponent_can_lock_after_fold(player: PlayerView) -> bool:
    round_num = getattr(player, "round_num", 0)
    if round_num <= 0:
        return False
    rounds_left = max(0, NUM_ROUNDS - round_num)
    if rounds_left <= 0:
        return False
    loss_now = int(max(0, getattr(player.hero, "contribution", 0)))
    if loss_now <= 0:
        loss_now = BIG_BLIND if getattr(player.hero, "blind", False) else SMALL_BLIND
    hero_bankroll = getattr(player.hero, "bankroll", 0)
    opponent_bankroll = -hero_bankroll + loss_now
    return opponent_bankroll > _opponent_future_fold_loss(player, rounds_left)


def _avoid_lock_win_fold(player: PlayerView, action):
    from skeleton.actions import CallAction, CheckAction, FoldAction
    if not player.hero.enable_desperate:
        return action

    if not isinstance(action, FoldAction):
        return action
    if not _opponent_can_lock_after_fold(player):
        return action
    legal_actions = set(player.hero.legal_actions)
    if CheckAction in legal_actions and player.hero.continue_cost == 0:
        return CheckAction()
    if CallAction in legal_actions:
        return CallAction()
    if CheckAction in legal_actions:
        return CheckAction()
    return action


def _blind_loss_for_rounds(rounds_left: int, starts_as_bb: bool) -> int:
    """
    Calculate total blind losses over rounds_left rounds.
    Closed-form: pairs * 3 + remainder (2 if BB, 1 if SB).
    """
    if rounds_left <= 0:
        return 0
    pairs = rounds_left // 2
    remainder = rounds_left % 2
    total = pairs * 3  # Each pair: BB(2) + SB(1) = 3
    if remainder:
        total += BIG_BLIND if starts_as_bb else SMALL_BLIND
    return total


def _max_safe_loss_this_round(player: PlayerView) -> int:
    """
    Calculate the maximum chips we can lose this round without letting
    opponent lock the win. Considers both ahead and behind scenarios.
    """
    if not _ENABLE_LOCK_WIN:
        return STARTING_STACK

    round_num = getattr(player, "round_num", 0)
    if round_num <= 0:
        return STARTING_STACK

    rounds_left = max(0, NUM_ROUNDS - round_num)  # rounds AFTER current
    hero_bankroll = getattr(player.hero, "bankroll", 0)

    # Opponent's blind status next round is opposite of ours now
    opp_starts_as_bb = not getattr(player.hero, "blind", False)
    opp_future_loss = _blind_loss_for_rounds(rounds_left, opp_starts_as_bb)

    # If we lose L chips this round:
    # - Opponent's new bankroll: -hero_bankroll + L
    # - Opponent can lock if: -hero_bankroll + L > opp_future_loss
    # - Safe if: L <= opp_future_loss + hero_bankroll

    max_safe = opp_future_loss + hero_bankroll
    return max(0, max_safe)


def _lock_defense_raise_margin(player: PlayerView) -> float:
    """
    Return an additional raise margin (making raises harder) when
    we're close to the danger zone. Returns 0.0 when very safe.
    """
    if not _ENABLE_LOCK_WIN:
        return 0.0

    max_safe = _max_safe_loss_this_round(player)

    if max_safe >= STARTING_STACK:
        return 0.0  # Very safe, no penalty
    if max_safe <= 0:
        return 0.5  # Extreme danger, large penalty

    # Linear penalty: 0.0 at max_safe=400, 0.3 at max_safe=0
    penalty = 0.3 * (1.0 - max_safe / STARTING_STACK)
    return max(0.0, min(0.3, penalty))


def _cap_raise_for_lock_defense(player: PlayerView, raise_amount: int) -> int:
    """
    Cap a raise amount to prevent giving opponent a win-lock.
    Returns 0 if no raise is safe.
    """
    if not _ENABLE_LOCK_WIN:
        return raise_amount

    max_safe = _max_safe_loss_this_round(player)
    if raise_amount <= max_safe:
        return raise_amount

    # Cap to max_safe, but respect minimum raise
    min_raise, _ = getattr(player.hero, "raise_bounds", (0, 0))
    if max_safe < min_raise:
        return 0  # Can't raise safely - signal "don't raise"
    return max_safe


def _is_desperate(player: PlayerView) -> bool:
    """
    Check if we're in a desperate state where opponent can lock the win
    by folding. In this state, we play tighter to avoid losing chips.
    """
    round_num = getattr(player, "round_num", 0)
    hero_bankroll = getattr(player.hero, "bankroll", 0)
    if hero_bankroll >= 0:
        return False  # We're ahead or even, not desperate

    # Check if opponent can lock by folding all remaining hands
    rounds_left = max(0, NUM_ROUNDS - round_num)
    opp_starts_bb = not getattr(player.hero, "blind", False)
    opp_future_loss = _blind_loss_for_rounds(rounds_left, opp_starts_bb)
    opponent_bankroll = -hero_bankroll + player.hero.pot_total

    # Opponent can lock if their bankroll > their future blind losses
    return opponent_bankroll > opp_future_loss


def _is_near_desperate(player: PlayerView) -> bool:
    """
    Check if we're approaching a desperate state - opponent is close to
    being able to lock. In this state, we should play more cautiously.
    """
    if not _ENABLE_LOCK_WIN:
        return False
    round_num = getattr(player, "round_num", 0)
    if round_num <= 0:
        return False
    hero_bankroll = getattr(player.hero, "bankroll", 0)
    if hero_bankroll >= -5:
        return False  # We're ahead or nearly even

    # Check how close opponent is to lock threshold
    rounds_left = max(0, NUM_ROUNDS - round_num)
    opp_starts_bb = not getattr(player.hero, "blind", False)
    opp_future_loss = _blind_loss_for_rounds(rounds_left, opp_starts_bb)
    opponent_bankroll = -hero_bankroll

    # Opponent is "near lock" if they're within 10 chips of locking
    lock_threshold = opp_future_loss
    return opponent_bankroll > lock_threshold - 10


def play(bot):
    """
    Strategy entry point called by player.py.

    This function should read the live fields on player and return an action
    instance from skeleton.actions.
    """

    from strategies.lockwin import LockWinPolicy, DesperatePolicy
    # Reset the action timer for hard 9-second cap in stats.py
    stats._reset_action_timer()

    # Win-lock: if we can lock the win, do it
    if _should_lock_win(bot) or bot.hero.policy_class == LockWinPolicy:
        print("locking win")
        bot.hero.policy_class = LockWinPolicy
        return LockWinPolicy.play(bot)

    # Defensive mode: if opponent can lock but hasn't, play tight
    if _is_desperate(bot) and bot.hero.enable_desperate:
        print("DEFENSIVE MODE - opponent can lock")
        bot.hero.policy_class = DesperatePolicy
        return DesperatePolicy.play(bot)

    # Near-desperate: flag for tighter play in normal policies
    if _is_near_desperate(bot):
        bot.hero.near_desperate = True
    else:
        bot.hero.near_desperate = False

    policy_class = _select_round_policy(bot)
    if not policy_class and bot.street <= 0:
        policy_class = _maybe_set_policy(bot)
    if not policy_class and _discard_action_required(bot):
        policy_class = _maybe_set_policy(bot)
        if not policy_class:
            policy_class = _policy_classes()[1]
            bot.hero.policy_class = policy_class
            bot.hero.policy_round = getattr(bot, "round_num", 0)
            bot.hero.discard_bluff = False
    if policy_class:
        action = policy_class.play(bot)
        if _discard_action_required(bot):
            action = _force_discard_if_needed(bot, action)
        return _avoid_lock_win_fold(bot, action)
    # Fallback: use pot-odds-aware logic instead of blind folding
    legal_actions = set(bot.hero.legal_actions)
    # Calculate pot odds for the fallback decision
    pot_total = max(1, getattr(bot.hero, "pot_total", 1))
    continue_cost = getattr(bot.hero, "continue_cost", 0)
    pot_odds = pot_odds_to_call(continue_cost, pot_total)
    # Estimate equity for fallback
    hero_hand = list(getattr(bot.hero, "hand", []))
    if bot.street <= 0 and hero_hand:
        equity = stats.preflop_strength(hero_hand)
    else:
        equity = 0.3  # Conservative default for post-flop without policy
    # Only fold if equity is clearly below pot odds
    if FoldAction in legal_actions and continue_cost > 0:
        if equity < pot_odds - 0.05:
            return _avoid_lock_win_fold(bot, FoldAction())
        # Otherwise call if we have reasonable equity
        if CallAction in legal_actions:
            return CallAction()
    if CheckAction in legal_actions and continue_cost == 0:
        return CheckAction()
    if CallAction in legal_actions:
        return CallAction()
    if CheckAction in legal_actions:
        return CheckAction()
    return _avoid_lock_win_fold(bot, FoldAction())


def _force_discard_if_needed(player: PlayerView, action):
    if isinstance(action, DiscardAction):
        return action
    legal_actions = set(player.hero.legal_actions)
    if DiscardAction not in legal_actions:
        return action
    hero_hand = list(player.hero.hand)
    board_cards = list(player.community)
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
    player.hero.discard_bluff = False
    return DiscardAction(best_i)


def _maybe_set_policy(player: PlayerView):
    equity = _estimate_policy_equity(player)
    policy_class = _choose_policy_for_round(player, equity)
    if policy_class:
        player.hero.policy_class = policy_class
        player.hero.policy_round = getattr(player, "round_num", 0)
        player.hero.discard_bluff = policy_class.__name__ == "BluffPolicy"
    return policy_class


def _discard_action_required(player: PlayerView) -> bool:
    legal_actions = set(player.hero.legal_actions)
    return DiscardAction in legal_actions


def _raise_margin_by_street(street: int) -> float:
    if street <= 0:
        return _RAISE_MARGIN_BY_STREET[0]
    if street <= 3:
        return _RAISE_MARGIN_BY_STREET[1]
    if street <= 4:
        return _RAISE_MARGIN_BY_STREET[2]
    return _RAISE_MARGIN_BY_STREET[3]


def _call_margin_by_street(street: int) -> float:
    if street <= 0:
        return _CALL_MARGIN_BY_STREET[0]
    if street <= 3:
        return _CALL_MARGIN_BY_STREET[1]
    if street <= 4:
        return _CALL_MARGIN_BY_STREET[2]
    return _CALL_MARGIN_BY_STREET[3]


def _lead_protection_adjustment(player, pot_total: int = 0) -> float:
    """
    Return an adjustment to tighten play when protecting a lead.
    Positive values make raising/calling harder (more conservative).
    Uses tunable stepwise function with pot-relative modifier.
    """
    if not _ENABLE_LOCK_WIN:
        return 0.0

    hero_bankroll = getattr(player.hero, "bankroll", 0)
    if hero_bankroll <= 0:
        return 0.0  # Not ahead, no protection needed

    # Calculate how close we are to win-lock as a percentage
    remaining = _remaining_fold_loss(player)
    if remaining <= 0:
        return _LEAD_PROT_ADJUSTMENTS[3]  # Already at lock, maximum tightness

    # Progress: 0% = no lead, 100% = can lock
    progress = hero_bankroll / remaining

    # Find the applicable tier using tunable thresholds
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

    # Pot-relative modifier: protect more when pot is large relative to remaining loss
    if base_adjustment > 0 and pot_total > 0 and _LEAD_PROT_POT_FACTOR > 0:
        pot_ratio = min(1.0, pot_total / max(1, remaining))
        base_adjustment *= (1.0 + pot_ratio * _LEAD_PROT_POT_FACTOR)

    return base_adjustment


def _lead_protection_size_mult(player) -> float:
    """
    Return a bet size multiplier when protecting a lead.
    Values < 1.0 reduce bet sizes to risk less chips.
    """
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


def _should_discard_bluff(player: PlayerView) -> bool:
    if _DISCARD_BLUFF_RATE <= 0.0:
        return False
    return random.random() < _DISCARD_BLUFF_RATE


def _should_pressure(player: PlayerView, equity: float) -> bool:
    if equity < _PRESSURE_EQUITY_THRESHOLD:
        return False
    fold_rate = _opponent_fold_rate(player.street, None)
    if fold_rate <= 0.0:
        return True
    return fold_rate >= _PRESSURE_FOLDRATE_MIN


def _discard_bluff_index(hero_hand: Sequence[str]) -> int:
    cards_int = stats._ensure_int_cards(list(hero_hand))
    ranks = [stats.Card.get_rank_int(card) for card in cards_int]
    return max(range(len(ranks)), key=ranks.__getitem__)


def _discard_bluff_active(player: PlayerView) -> bool:
    return bool(getattr(player.hero, "discard_bluff", False))


def _consume_discard_bluff(player: PlayerView) -> None:
    if hasattr(player.hero, "discard_bluff"):
        player.hero.discard_bluff = False


def _should_discard_bluff_raise(player: PlayerView) -> bool:
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


def _raise_call_penalty(player: PlayerView, pot_total: int) -> float:
    if player.street != 5:
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


def _turn_raise_extra(player: PlayerView, pot_total: int) -> float:
    if player.street != 5:
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


def _preflop_open_decision(
    player: PlayerView,
    equity: float,
    pot_odds: float,
    legal_actions: Sequence[object],
    min_raise: int,
    max_raise: int,
    pot_total: int,
):
    roll = _preflop_roll(player)
    raise_strong, raise_medium, raise_light = _PREFLOP_RAISE_THRESHOLDS
    call_strong, call_medium, call_light = _PREFLOP_CALL_THRESHOLDS
    if RaiseAction in legal_actions:
        if equity >= raise_strong:
            raise_prob = 0.7
            bucket = "raise_strong"
        elif equity >= raise_medium:
            raise_prob = 0.45
            bucket = "raise_medium"
        elif equity >= raise_light:
            raise_prob = 0.25
            bucket = "raise_light"
        else:
            raise_prob = 0.0
            bucket = "raise_none"
        if equity < pot_odds:
            raise_prob *= 0.5
        if raise_prob > 0 and roll < raise_prob:
            _set_preflop_debug(player, bucket, roll)
            value_mult = _value_extraction_multiplier(player)
            target = _raise_size(pot_total, min_raise, max_raise, equity, value_mult=value_mult)
            if target > 0:
                return RaiseAction(target)

    if CallAction in legal_actions:
        if equity >= call_strong:
            call_prob = 0.75
            bucket = "call_strong"
        elif equity >= call_medium:
            call_prob = 0.5
            bucket = "call_medium"
        elif equity >= call_light:
            call_prob = 0.25
            bucket = "call_light"
        else:
            call_prob = 0.0
            bucket = "call_none"
        if equity < pot_odds - 0.05:
            call_prob *= 0.5
        if call_prob > 0 and roll < call_prob:
            _set_preflop_debug(player, bucket, roll)
            return CallAction()
    if CheckAction in legal_actions and player.hero.continue_cost == 0:
        _set_preflop_debug(player, "check_default", roll)
        return CheckAction()
    if CallAction in legal_actions:
        # Do not default call; fall through to odds-based logic in TightPolicy
        # _set_preflop_debug(player, "call_default", roll)
        return None
    if FoldAction in legal_actions:
        # Do not default fold; fall through to odds-based logic
        # _set_preflop_debug(player, "fold_default", roll)
        return None
    _set_preflop_debug(player, "no_action", roll)
    return None


def _preflop_roll(player: PlayerView) -> float:
    round_num = getattr(player, "round_num", 0)
    hand = getattr(player.hero, "hand", [])
    seed = hash((round_num, tuple(hand), "preflop")) & 0xFFFFFFFF
    return random.Random(seed).random()


def _set_preflop_debug(player: PlayerView, bucket: str, roll: float) -> None:
    setattr(player.hero, "preflop_bucket", bucket)
    setattr(player.hero, "preflop_roll", roll)


@lru_cache(maxsize=50_000)
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


def _variant_adjustments(
    variant_id: str,
    player: PlayerView,
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


def _pairwise_adjustments(
    hero_hand: Sequence[str],
    board_cards: Sequence[str],
) -> Tuple[float, float, float]:
    if not board_cards:
        return 0.0, 0.0, 0.0
    hole_int = stats._ensure_int_cards(list(hero_hand))
    if not hole_int:
        return 0.0, 0.0, 0.0
    board_int = stats._ensure_int_cards(list(board_cards))
    board_ranks = {stats.Card.get_rank_int(card) for card in board_int}
    board_suits = {stats.Card.get_suit_int(card) for card in board_int}
    hole_ranks = [stats.Card.get_rank_int(card) for card in hole_int]
    hole_suits = [stats.Card.get_suit_int(card) for card in hole_int]
    pair_match = sum(1 for rank in hole_ranks if rank in board_ranks)
    suit_match = sum(1 for suit in hole_suits if suit in board_suits)
    connected = 1 if max(hole_ranks) - min(hole_ranks) <= 4 else 0
    equity_bias = min(0.02, 0.008 * pair_match + 0.004 * suit_match + 0.004 * connected)
    return equity_bias, 0.0, 0.0


def _graph_adjustments(
    hero_hand: Sequence[str],
    board_cards: Sequence[str],
) -> Tuple[float, float, float]:
    if not board_cards:
        return 0.0, 0.0, 0.0
    hole_int = stats._ensure_int_cards(list(hero_hand))
    if not hole_int:
        return 0.0, 0.0, 0.0
    board_int = stats._ensure_int_cards(list(board_cards))
    equity_bias = 0.0
    for hcard in hole_int:
        hrank = stats.Card.get_rank_int(hcard)
        hsuit = stats.Card.get_suit_int(hcard)
        for bcard in board_int:
            if hrank == stats.Card.get_rank_int(bcard):
                equity_bias += 0.003
            if hsuit == stats.Card.get_suit_int(bcard):
                equity_bias += 0.002
    board_tuple = tuple(board_int)
    if _board_is_paired(board_tuple):
        equity_bias += 0.004
    if _board_is_flushy(board_tuple):
        equity_bias += 0.004
    return min(0.02, equity_bias), 0.0, 0.0


def _threshold_adjustments(player: PlayerView) -> Tuple[float, float, float]:
    fold_rate = _opponent_fold_rate(player.street, None)
    if fold_rate >= 0.45:
        return 0.0, -0.02, -0.005
    if fold_rate <= 0.2:
        return 0.0, 0.02, 0.01
    return 0.0, 0.0, 0.0


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


def _opponent_is_passive(street: int) -> bool:
    """Check if opponent has been passive (low raise rate) on this street."""
    bucket = _OPPONENT_RANGE_MODEL["by_street"].get(street)
    if not bucket:
        return False
    total = bucket.get("total", 0.0)
    raises = bucket.get("raises", 0.0)
    if total < 8:
        return False
    return (raises / total) < _OPP_PASSIVE_THRESHOLD


def _opponent_is_calling_station(street: int) -> bool:
    """Check if opponent calls frequently but rarely raises."""
    bucket = _OPPONENT_RANGE_MODEL["by_street"].get(street)
    if not bucket:
        return False
    total = bucket.get("total", 0.0)
    calls = bucket.get("calls", 0.0)
    folds = bucket.get("folds", 0.0)
    if total < 10:
        return False
    call_rate = calls / total
    fold_rate = folds / total
    return call_rate > _OPP_STATION_CALL_RATE and fold_rate < _OPP_STATION_FOLD_RATE


def _value_extraction_multiplier(player) -> float:
    """
    Return a multiplier for raise sizing when we have a strong hand.
    Combines lead protection (reduces bet size) with opponent exploitation (increases against passive).
    """
    street = getattr(player, "street", 0)

    # Start with opponent exploitation using tunable multipliers
    opp_mult = 1.0
    if _opponent_is_calling_station(street):
        opp_mult = _OPP_STATION_VALUE_MULT
    elif _opponent_is_passive(street):
        opp_mult = _OPP_PASSIVE_VALUE_MULT

    # Lead protection multiplier using the stepwise function
    lead_mult = _lead_protection_size_mult(player)

    # Combine: exploit weak opponents but still protect lead
    return opp_mult * lead_mult


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
    bucket = _OPPONENT_RANGE_MODEL["by_street"].setdefault(
        street,
        {"raises": 0.0, "calls": 0.0, "folds": 0.0, "total": 0.0},
    )
    bucket["raises"] += 1.0
    bucket["total"] += 1.0


def record_opponent_action(street: int) -> None:
    bucket = _OPPONENT_RANGE_MODEL["by_street"].setdefault(
        street,
        {"raises": 0.0, "calls": 0.0, "folds": 0.0, "total": 0.0},
    )
    bucket["total"] += 1.0


def record_opponent_call(street: int) -> None:
    bucket = _OPPONENT_RANGE_MODEL["by_street"].setdefault(
        street,
        {"raises": 0.0, "calls": 0.0, "folds": 0.0, "total": 0.0},
    )
    bucket["calls"] += 1.0
    bucket["total"] += 1.0


def record_opponent_fold(street: int) -> None:
    bucket = _OPPONENT_RANGE_MODEL["by_street"].setdefault(
        street,
        {"raises": 0.0, "calls": 0.0, "folds": 0.0, "total": 0.0},
    )
    bucket["folds"] += 1.0
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


def update_range_after_discard(range3, discarded_card):
    record_opponent_discard(discarded_card)
    return range3


def _should_bluff(street: int, board_cards: Sequence[str], allow_bluff: bool = True) -> bool:
    if not allow_bluff:
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


def _equity_budget(player: PlayerView) -> Tuple[int, float, int]:
    street = player.street
    round_num = getattr(player, "round_num", 0)
    hero_bankroll = getattr(player.hero, "bankroll", 0)
    game_clock = getattr(player, "game_clock", None)

    # Base budget by street
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

    # Early-round boost using tunable parameters
    early_r1, early_r2 = _EARLY_BOOST_ROUNDS
    early_m1, early_m2 = _EARLY_BOOST_MULTS
    if round_num <= early_r1 and (game_clock is None or game_clock > 40):
        samples = int(samples * early_m1)
        max_seconds *= (early_m1 * 0.9)  # Slightly less time boost
        discard_samples = int(discard_samples * (1.0 + (early_m1 - 1.0) * 0.6))
    elif round_num <= early_r2 and (game_clock is None or game_clock > 35):
        samples = int(samples * early_m2)
        max_seconds *= (early_m2 * 0.95)
        discard_samples = int(discard_samples * (1.0 + (early_m2 - 1.0) * 0.5))

    # Scale up when behind (need to catch up) - only if time permits
    if game_clock is None or game_clock > 30:
        scale = 1.0
        if hero_bankroll < -50:
            scale = max(scale, 1.3)
        elif hero_bankroll < -20:
            scale = max(scale, 1.15)
        # Late rounds: opponent may be exploitable, invest more
        if round_num > 750:
            scale = max(scale, 1.25)
        elif round_num > 500:
            scale = max(scale, 1.1)
        samples = int(samples * scale)
        discard_samples = int(discard_samples * min(1.15, scale))

    # Time pressure: reduce budget when clock is low
    if game_clock is not None and game_clock < 20:
        samples = max(40, samples // 2)
        max_seconds = max(0.008, max_seconds * 0.5)
        discard_samples = max(6, discard_samples // 2)
    if game_clock is not None and game_clock < 10:
        samples = max(20, samples // 2)
        max_seconds = max(0.004, max_seconds * 0.5)
        discard_samples = max(4, discard_samples // 2)
    return samples, max_seconds, discard_samples


def _turn_raise_ratio(player: PlayerView, pot_total: int) -> Optional[float]:
    if player.street != 5:
        return None
    continue_cost = player.hero.continue_cost
    if continue_cost <= 0:
        return None
    last_bet_street = getattr(player, "_last_bet_street", None)
    if last_bet_street != player.street:
        return None
    return continue_cost / float(max(1, pot_total))


def _adjust_budget_for_turn_raise(
    player: PlayerView,
    pot_total: int,
    samples: int,
    max_seconds: float,
    discard_samples: int,
) -> Tuple[int, float, int]:
    ratio = _turn_raise_ratio(player, pot_total)
    if ratio is None or ratio < _TURN_RAISE_RATIO:
        return samples, max_seconds, discard_samples
    sample_mult = max(1.0, _TURN_RAISE_SAMPLE_MULT)
    time_mult = max(1.0, _TURN_RAISE_TIME_MULT)
    samples = min(200, int(samples * sample_mult))
    max_seconds = min(0.05, max_seconds * time_mult)
    discard_samples = min(30, max(4, int(discard_samples * 1.2)))
    return samples, max_seconds, discard_samples


def _discard_budget(player: PlayerView) -> Tuple[int, float]:
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
