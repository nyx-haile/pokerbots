"""Core strategy routing and lock-win logic."""

from dataclasses import dataclass
import os
from typing import Iterable, Sequence, Tuple

from skeleton.actions import CallAction, CheckAction, DiscardAction, FoldAction, RaiseAction
from skeleton.states import BIG_BLIND, NUM_ROUNDS, SMALL_BLIND, STARTING_STACK

import stats
import lumberjack
from .math import (
    _DESPERATE_CALL_PENALTY,
    _DESPERATE_NUT_THRESHOLD,
    _DESPERATE_RAISE_MARGIN,
    _discard_budget,
    _equity_budget,
    _max_safe_loss_this_round,
    _remaining_fold_loss,
    pot_odds_to_call,
    is_river,
    _board_texture_adjustments,
)
from .models import (
    _policy_classes,
    _select_round_policy,
    _choose_policy_for_round,
    _select_discard_asymmetric,
    opponent_range_hint,
    opponent_discard_range_adjustment,
    line_range_calibration,
    _anti_exploit_rate,
    _anti_exploit_discard_index,
)

_DISABLE_PREFLOP_MIX = os.environ.get("NEUROPOKER_DISABLE_PREFLOP_MIX", "0") == "1"
_ENABLE_LOCK_WIN = os.environ.get("NEUROPOKER_ENABLE_LOCK_WIN", "1") == "1"
_LOCK_MARGIN = 2


@dataclass
class ActorView:
    legal_actions: Iterable[object] = ()
    hand: Sequence[str] = ()
    pip: int = 0
    enable_desperate: bool = True
    stack: int = 0
    continue_cost: int = 0
    contribution: int = 0
    pot_total: int = 0
    blind: bool = False
    raise_bounds: Tuple[int, int] = (0, 0)
    desperate_state: str = "ready"


@dataclass
class PlayerView:
    hero: ActorView
    villain: ActorView
    community: Sequence[str]
    street: int


def begin_round(player: PlayerView) -> str:
    player.hero.policy_class = None
    player.hero.policy_round = None
    player.hero.discard_bluff = False
    return "unset"


def _estimate_policy_equity(player: PlayerView) -> float:
    hero_hand = list(player.hero.hand)
    board_cards = list(player.community)
    if player.street <= 0:
        return stats.preflop_strength(hero_hand)
    samples, max_seconds, discard_samples = _equity_budget(player)
    if len(hero_hand) == 2:
        return _range_conditioned_equity(
            player,
            hero_hand,
            board_cards,
            samples=max(24, samples // 5),
            max_seconds=min(0.006, max_seconds * 0.25),
        )
    return stats.estimate_equity(
        hero_hand,
        board_cards,
        samples=max(16, samples // 6),
        max_seconds=min(0.004, max_seconds * 0.2),
        discard_samples=max(4, discard_samples // 2),
    )




def _range_conditioned_equity(
    player: PlayerView,
    hero_hand,
    board_cards,
    samples: int,
    max_seconds: float,
) -> float:
    street = getattr(player, "street", 0)
    cache = getattr(getattr(player, "hero", None), "_equity_cache", None)
    cache_key = None
    if cache is not None:
        cache_key = (street, tuple(hero_hand), tuple(board_cards))
        cached = cache.get(cache_key)
        if cached:
            if cached.get("samples", 0) >= samples and cached.get("max_seconds", 0.0) >= max_seconds:
                if hasattr(player, "hero"):
                    player.hero.last_equity_abs_error = cached.get("abs_error", 0.0)
                    player.hero.last_equity_samples = cached.get("samples_used", cached.get("samples", 0))
                    player.hero.last_equity_extended = cached.get("extended", False)
                return cached.get("equity", 0.0)
    if len(hero_hand) != 2:
        equity = stats.estimate_equity(
            hero_hand,
            board_cards,
            samples=samples,
            max_seconds=max_seconds,
        )
        if cache is not None and cache_key is not None:
            cache[cache_key] = {
                "equity": equity,
                "samples": samples,
                "max_seconds": max_seconds,
                "samples_used": samples,
                "abs_error": 0.0,
                "extended": False,
            }
        return equity
    line_key = getattr(getattr(player, "hero", None), "line_prefix", None)
    strength, confidence = opponent_range_hint(line_key)
    discard_strength_adj, discard_tightness_adj, discard_conf = opponent_discard_range_adjustment()
    strength += discard_strength_adj
    if discard_conf > 0:
        confidence = max(confidence, 0.4 * discard_conf)
    texture_strength, texture_tight, texture_mix = _board_texture_adjustments(board_cards)
    line_strength_adj, line_tight_adj, line_mix_mult, line_conf = line_range_calibration(line_key)
    strong = max(0.0, strength - 0.5)
    river_tighten = 0.0
    if is_river(getattr(player, "street", 0)):
        river_tighten = 0.04 + 0.04 * confidence + 0.04 * strong
    strength = min(0.97, strength + 0.06 * strong + 0.04 * confidence * strong + river_tighten)
    tightness = 0.35 + 0.5 * confidence + 0.25 * strong + discard_tightness_adj + (0.1 if river_tighten > 0 else 0.0)
    tightness = min(0.9, max(0.25, tightness))
    if confidence < 0.25:
        strength = min(0.98, strength + 0.02)
        tightness = min(0.95, tightness + 0.04)
    if texture_strength or texture_tight:
        strength = min(0.99, strength + texture_strength)
        tightness = min(0.95, tightness + texture_tight)
    mix_uniform = 0.45 - 0.3 * confidence - 0.15 * strong - (0.08 if river_tighten > 0 else 0.0)
    mix_uniform = max(0.15, mix_uniform)
    if texture_mix < 1.0:
        mix_uniform *= texture_mix
    if line_conf > 0.0:
        strength = max(0.2, min(0.99, strength + line_strength_adj * line_conf))
        tightness = max(0.2, min(0.97, tightness + line_tight_adj * line_conf))
        mix_uniform *= 1.0 - (1.0 - line_mix_mult) * line_conf
    mix_uniform = max(0.1, min(0.7, mix_uniform))
    opponent_range = stats.build_opponent_range(
        hero_hand + board_cards,
        target_strength=strength,
        tightness=tightness,
        mix_uniform=mix_uniform,
    )
    quick_samples = max(12, samples // 3)
    quick_seconds = max(0.004, max_seconds * 0.4)
    quick_equity = stats.estimate_showdown_equity(
        hero_hand,
        opponent_range,
        board_cards,
        samples=quick_samples,
        max_seconds=quick_seconds,
    )
    equity = stats.estimate_showdown_equity(
        hero_hand,
        opponent_range,
        board_cards,
        samples=samples,
        max_seconds=max_seconds,
    )
    abs_error = abs(equity - quick_equity)
    extended = False
    game_clock = getattr(player, "game_clock", None)
    seconds_used = max_seconds
    if abs_error > 0.05 and (game_clock is None or game_clock > 18):
        boosted_samples = int(samples * 1.6)
        boosted_seconds = max_seconds * 1.6
        equity = stats.estimate_showdown_equity(
            hero_hand,
            opponent_range,
            board_cards,
            samples=boosted_samples,
            max_seconds=boosted_seconds,
        )
        abs_error = abs(equity - quick_equity)
        extended = True
        samples_used = boosted_samples
        seconds_used = boosted_seconds
    else:
        samples_used = samples
    if hasattr(player, "hero"):
        player.hero.last_equity_abs_error = abs_error
        player.hero.last_equity_samples = samples_used
        player.hero.last_equity_extended = extended
    lumberjack.record_equity_error(street, abs_error, samples_used, extended)
    if cache is not None and cache_key is not None:
        cache[cache_key] = {
            "equity": equity,
            "samples": samples_used,
            "max_seconds": seconds_used,
            "samples_used": samples_used,
            "abs_error": abs_error,
            "extended": extended,
        }
    return equity




def _should_lock_win(player: PlayerView) -> bool:
    if not _ENABLE_LOCK_WIN:
        return False
    bankroll = getattr(player.hero, "bankroll", 0)
    if bankroll <= 0:
        return False
    return bankroll > _remaining_fold_loss(player)


def _lock_win_action(player: PlayerView):
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
    starts_as_bb = bool(getattr(player.hero, "blind", False))
    return _remaining_fold_loss(player) - (BIG_BLIND if starts_as_bb else SMALL_BLIND)


def _opponent_can_lock_after_loss(player: PlayerView, extra_loss: int) -> bool:
    round_num = getattr(player, "round_num", 0)
    if round_num <= 0:
        return False
    rounds_left = max(0, NUM_ROUNDS - round_num)
    if rounds_left <= 0:
        return False
    loss_now = int(max(0, getattr(player.hero, "contribution", 0)))
    if loss_now <= 0:
        loss_now = BIG_BLIND if getattr(player.hero, "blind", False) else SMALL_BLIND
    loss_now += max(0, int(extra_loss))
    hero_bankroll = getattr(player.hero, "bankroll", 0)
    opponent_bankroll = -hero_bankroll + loss_now
    return opponent_bankroll > _opponent_future_fold_loss(player, rounds_left) + _LOCK_MARGIN


def _opponent_can_lock_after_fold(player: PlayerView) -> bool:
    return _opponent_can_lock_after_loss(player, 0)


def _avoid_lock_win_fold(player: PlayerView, action):
    if not player.hero.enable_desperate:
        return action
    if not isinstance(action, FoldAction):
        return action
    extra_loss = getattr(getattr(player, "hero", None), "continue_cost", 0)
    if not _opponent_can_lock_after_loss(player, extra_loss):
        return action
    legal_actions = set(player.hero.legal_actions)
    if CheckAction in legal_actions and player.hero.continue_cost == 0:
        player.hero.fold_prevented = True
        player.hero.fold_prevent_street = getattr(player, "street", None)
        player.hero.fold_prevent_action = "CheckAction"
        return CheckAction()
    if CallAction in legal_actions:
        player.hero.fold_prevented = True
        player.hero.fold_prevent_street = getattr(player, "street", None)
        player.hero.fold_prevent_action = "CallAction"
        return CallAction()
    if CheckAction in legal_actions:
        player.hero.fold_prevented = True
        player.hero.fold_prevent_street = getattr(player, "street", None)
        player.hero.fold_prevent_action = "CheckAction"
        return CheckAction()
    return action


def _lock_defense_raise_margin(player: PlayerView) -> float:
    max_safe = _max_safe_loss_this_round(player)
    if max_safe >= STARTING_STACK:
        return 0.0
    if max_safe <= 0:
        return 0.5
    penalty = 0.3 * (1.0 - max_safe / STARTING_STACK)
    return max(0.0, min(0.3, penalty))


def _cap_raise_for_lock_defense(player: PlayerView, raise_amount: int) -> int:
    max_safe = _max_safe_loss_this_round(player)
    if raise_amount <= max_safe:
        return raise_amount
    min_raise, _ = getattr(player.hero, "raise_bounds", (0, 0))
    if max_safe < min_raise:
        return 0
    return max_safe


def _is_desperate(player: PlayerView) -> bool:
    if getattr(player.hero, "desperate_state", "ready") != "ready":
        return False
    extra_loss = getattr(getattr(player, "hero", None), "continue_cost", 0)
    return _opponent_can_lock_after_loss(player, extra_loss)


def _is_near_desperate(player: PlayerView) -> bool:
    round_num = getattr(player, "round_num", 0)
    if round_num <= 0:
        return False
    hero_bankroll = getattr(player.hero, "bankroll", 0)
    if hero_bankroll >= -5:
        return False
    return _opponent_can_lock_after_loss(player, max(0, int(getattr(player.hero, "continue_cost", 0)) - 4))


def play(bot):
    from policies.lockwin import LockWinPolicy
    from policies.desperate import DesperatePolicy

    stats._reset_action_timer()

    if _should_lock_win(bot) or bot.hero.policy_class == LockWinPolicy:
        bot.hero.policy_class = LockWinPolicy
        return LockWinPolicy.play(bot)

    if _is_desperate(bot) and bot.hero.enable_desperate:
        bot.hero.policy_class = DesperatePolicy
        return DesperatePolicy.play(bot)

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
        if policy_class.__name__ == "BluffPolicy" and isinstance(action, DiscardAction):
            bot.hero.discard_bluff = True
        return _avoid_lock_win_fold(bot, action)

    legal_actions = set(bot.hero.legal_actions)
    pot_total = max(1, getattr(bot.hero, "pot_total", 1))
    continue_cost = getattr(bot.hero, "continue_cost", 0)
    pot_odds = pot_odds_to_call(continue_cost, pot_total)
    hero_hand = list(getattr(bot.hero, "hand", []))
    if bot.street <= 0 and hero_hand:
        equity = stats.preflop_strength(hero_hand)
    else:
        equity = 0.3
    if FoldAction in legal_actions and continue_cost > 0:
        if equity < pot_odds - 0.05:
            return _avoid_lock_win_fold(bot, FoldAction())
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
    anti_rate = _anti_exploit_rate(player)
    best_i = _anti_exploit_discard_index(equities, best_i, anti_rate)
    player.hero.last_discard_ev = lumberjack.record_discard_decision(
        getattr(player.hero, "policy_class", None).__name__ if getattr(player.hero, "policy_class", None) else None,
        player.street,
        equities,
        best_i,
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
