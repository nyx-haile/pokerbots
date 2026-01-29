"""Core strategy routing and lock-win logic."""

from dataclasses import dataclass
import os
from typing import Iterable, Sequence, Tuple

from skeleton.actions import CallAction, CheckAction, DiscardAction, FoldAction
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
)
from .models import (
    _policy_classes,
    _select_round_policy,
    _choose_policy_for_round,
    _select_discard_asymmetric,
    opponent_range_hint,
    opponent_discard_range_adjustment,
    _anti_exploit_rate,
    _anti_exploit_discard_index,
)

_DISABLE_PREFLOP_MIX = os.environ.get("NEUROPOKER_DISABLE_PREFLOP_MIX", "0") == "1"
_ENABLE_LOCK_WIN = os.environ.get("NEUROPOKER_ENABLE_LOCK_WIN", "1") == "0"
#temporarily disabled


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
    if len(hero_hand) != 2:
        return stats.estimate_equity(
            hero_hand,
            board_cards,
            samples=samples,
            max_seconds=max_seconds,
        )
    line_key = getattr(getattr(player, "hero", None), "line_prefix", None)
    strength, confidence = opponent_range_hint(line_key)
    discard_strength_adj, discard_tightness_adj, discard_conf = opponent_discard_range_adjustment()
    strength += discard_strength_adj
    if discard_conf > 0:
        confidence = max(confidence, 0.4 * discard_conf)
    strong = max(0.0, strength - 0.5)
    river_tighten = 0.0
    if is_river(getattr(player, "street", 0)):
        river_tighten = 0.04 + 0.04 * confidence + 0.04 * strong
    strength = min(0.97, strength + 0.06 * strong + 0.04 * confidence * strong + river_tighten)
    tightness = 0.35 + 0.5 * confidence + 0.25 * strong + discard_tightness_adj + (0.1 if river_tighten > 0 else 0.0)
    tightness = min(0.9, max(0.25, tightness))
    mix_uniform = 0.45 - 0.3 * confidence - 0.15 * strong - (0.08 if river_tighten > 0 else 0.0)
    mix_uniform = max(0.15, mix_uniform)
    opponent_range = stats.build_opponent_range(
        hero_hand + board_cards,
        target_strength=strength,
        tightness=tightness,
        mix_uniform=mix_uniform,
    )
    return stats.estimate_showdown_equity(
        hero_hand,
        opponent_range,
        board_cards,
        samples=samples,
        max_seconds=max_seconds,
    )




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


def _lock_defense_raise_margin(player: PlayerView) -> float:
    if not _ENABLE_LOCK_WIN:
        return 0.0
    max_safe = _max_safe_loss_this_round(player)
    if max_safe >= STARTING_STACK:
        return 0.0
    if max_safe <= 0:
        return 0.5
    penalty = 0.3 * (1.0 - max_safe / STARTING_STACK)
    return max(0.0, min(0.3, penalty))


def _cap_raise_for_lock_defense(player: PlayerView, raise_amount: int) -> int:
    if not _ENABLE_LOCK_WIN:
        return raise_amount
    max_safe = _max_safe_loss_this_round(player)
    if raise_amount <= max_safe:
        return raise_amount
    min_raise, _ = getattr(player.hero, "raise_bounds", (0, 0))
    if max_safe < min_raise:
        return 0
    return max_safe


def _is_desperate(player: PlayerView) -> bool:
    round_num = getattr(player, "round_num", 0)
    hero_bankroll = getattr(player.hero, "bankroll", 0)
    if hero_bankroll >= 0:
        return False
    rounds_left = max(0, NUM_ROUNDS - round_num)
    opp_starts_bb = not getattr(player.hero, "blind", False)
    opp_future_loss = _remaining_fold_loss(player) - (BIG_BLIND if opp_starts_bb else SMALL_BLIND)
    opponent_bankroll = -hero_bankroll + player.hero.pot_total
    return opponent_bankroll > opp_future_loss


def _is_near_desperate(player: PlayerView) -> bool:
    if not _ENABLE_LOCK_WIN:
        return False
    round_num = getattr(player, "round_num", 0)
    if round_num <= 0:
        return False
    hero_bankroll = getattr(player.hero, "bankroll", 0)
    if hero_bankroll >= -5:
        return False
    rounds_left = max(0, NUM_ROUNDS - round_num)
    opp_starts_bb = not getattr(player.hero, "blind", False)
    opp_future_loss = _remaining_fold_loss(player) - (BIG_BLIND if opp_starts_bb else SMALL_BLIND)
    opponent_bankroll = -hero_bankroll
    lock_threshold = opp_future_loss
    return opponent_bankroll > lock_threshold - 10


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
