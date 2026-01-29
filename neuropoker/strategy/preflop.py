"""Preflop helpers extracted from strategy core."""

import random
from typing import Sequence

from skeleton.actions import CallAction, CheckAction, FoldAction, RaiseAction

from .core import PlayerView
from .math import _PREFLOP_CALL_THRESHOLDS, _PREFLOP_RAISE_THRESHOLDS
from .models import _value_extraction_multiplier, opponent_range_hint, _opponent_fold_rate, _adaptive_raise_size


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
    effective_equity, stack_ratio = _preflop_position_stack_adjustments(player, equity, pot_total)
    strength_hint, confidence = opponent_range_hint()
    fold_rate = _opponent_fold_rate(0, None)
    if fold_rate > 0.0:
        fold_rate = min(0.35, fold_rate)
    else:
        fold_rate = 0.0
    strength_bias = (strength_hint - 0.5) * 0.06 * confidence
    fold_bias = (fold_rate - 0.2) * 0.04
    effective_equity = max(0.0, min(1.0, effective_equity - strength_bias + fold_bias))
    raise_strong, raise_medium, raise_light = _PREFLOP_RAISE_THRESHOLDS
    call_strong, call_medium, call_light = _PREFLOP_CALL_THRESHOLDS
    pos_raise_scale = 1.1 if not getattr(player.hero, "blind", False) else 0.9
    behind_mult = 1.0
    call_suppress = 1.0
    hero_bankroll = getattr(player.hero, "bankroll", 0)
    if hero_bankroll < 0:
        deficit = min(60, -hero_bankroll)
        effective_equity = min(1.0, effective_equity + 0.01 + 0.00025 * deficit)
        behind_mult = 1.10 + 0.002 * min(30, deficit)
        call_suppress = 0.95
    if RaiseAction in legal_actions:
        if effective_equity >= raise_strong:
            raise_prob = 0.8
            bucket = "raise_strong"
        elif effective_equity >= raise_medium:
            raise_prob = 0.55
            bucket = "raise_medium"
        elif effective_equity >= raise_light:
            raise_prob = 0.35
            bucket = "raise_light"
        else:
            raise_prob = 0.0
            bucket = "raise_none"
        if effective_equity < pot_odds:
            raise_prob *= 0.5
        raise_prob *= _preflop_raise_stack_scale(stack_ratio)
        raise_prob *= pos_raise_scale
        raise_prob *= behind_mult
        if raise_prob > 0 and roll < raise_prob:
            _set_preflop_debug(player, bucket, roll)
            value_mult = _value_extraction_multiplier(player)
            target = _adaptive_raise_size(
                player,
                pot_total,
                min_raise,
                max_raise,
                equity,
                value_mult=value_mult,
            )
            if target > 0:
                return RaiseAction(target)

    if CallAction in legal_actions:
        if effective_equity >= call_strong:
            call_prob = 0.75
            bucket = "call_strong"
        elif effective_equity >= call_medium:
            call_prob = 0.5
            bucket = "call_medium"
        elif effective_equity >= call_light:
            call_prob = 0.25
            bucket = "call_light"
        else:
            call_prob = 0.0
            bucket = "call_none"
        if effective_equity < pot_odds - 0.05:
            call_prob *= 0.5
        call_prob *= _preflop_call_stack_scale(stack_ratio)
        call_prob *= call_suppress
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


def _preflop_position_stack_adjustments(
    player: PlayerView,
    equity: float,
    pot_total: int,
) -> tuple:
    pos_adj = -0.02
    if getattr(player.hero, "blind", False):
        pos_adj = 0.02
    stack_ratio = 0.0
    if pot_total > 0:
        stack_ratio = player.hero.stack / float(max(1, pot_total))
    effective_equity = max(0.0, min(1.0, equity + pos_adj))
    return effective_equity, stack_ratio


def _preflop_raise_stack_scale(stack_ratio: float) -> float:
    if stack_ratio <= 0:
        return 1.0
    if stack_ratio < 2.0:
        return 0.6
    if stack_ratio < 4.0:
        return 0.85
    if stack_ratio > 8.0:
        return 1.1
    return 1.0


def _preflop_call_stack_scale(stack_ratio: float) -> float:
    if stack_ratio <= 0:
        return 1.0
    if stack_ratio < 2.0:
        return 0.8
    if stack_ratio < 4.0:
        return 0.95
    if stack_ratio > 8.0:
        return 1.05
    return 1.0


def _set_preflop_debug(player: PlayerView, bucket: str, roll: float) -> None:
    setattr(player.hero, "preflop_bucket", bucket)
    setattr(player.hero, "preflop_roll", roll)


__all__ = ["_preflop_open_decision", "_preflop_roll", "_set_preflop_debug"]
