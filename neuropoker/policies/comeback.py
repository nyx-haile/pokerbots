"""
- ComebackPolicy: when opponent is near win-lock, swing back aggressively.
"""
from skeleton.actions import CallAction, CheckAction, DiscardAction, FoldAction, RaiseAction

import lumberjack
import random
import stats
from strategy import policy_api as core


class ComebackPolicy:
    """Aggressive policy to swing back when near lock-loss."""

    @staticmethod
    def play(player):
        legal_actions = set(player.hero.legal_actions)
        hero_hand = list(player.hero.hand)
        board_cards = list(player.community)

        def _record(action, equity_value=None, pot_odds_value=None):
            lumberjack.record_policy_action("ComebackPolicy", player.street, action.__class__.__name__)
            if equity_value is not None and pot_odds_value is not None:
                lumberjack.record_equity_vs_pot_odds(
                    "ComebackPolicy",
                    player.street,
                    equity_value,
                    pot_odds_value,
                )
            return action

        if DiscardAction in legal_actions:
            opponent_discard = None
            if not player.hero.blind and len(board_cards) >= 3:
                opponent_discard = board_cards[-1]
            discard_samples, discard_seconds = core._discard_budget(player)
            equities = stats.discard_equity(
                hero_hand,
                board_cards,
                n_samples=discard_samples,
                max_seconds=discard_seconds,
            )
            best_i = core._select_discard_asymmetric(
                hero_hand,
                equities,
                discard_visible=player.hero.blind,
                opponent_discard=opponent_discard,
                board_cards=board_cards,
            )
            anti_rate = core._anti_exploit_rate(player) * 0.6
            best_i = core._anti_exploit_discard_index(equities, best_i, anti_rate)
            player.hero.last_discard_ev = lumberjack.record_discard_decision(
                "ComebackPolicy",
                player.street,
                equities,
                best_i,
            )
            player.hero.last_discard = hero_hand[best_i]
            player.hero.last_discard_visible = player.hero.blind
            player.hero.discard_bluff = False
            return _record(DiscardAction(best_i))

        pot_total = max(1, player.hero.pot_total)
        continue_cost = player.hero.continue_cost
        pot_odds = core.pot_odds_to_call(continue_cost, pot_total)

        if player.street <= 0:
            equity = stats.preflop_strength(hero_hand)
        else:
            samples, max_seconds, discard_samples = core._equity_budget(player)
            force_full_equity = core._is_large_raise(player, pot_total)
            quick_equity = core._range_conditioned_equity(
                player,
                hero_hand,
                board_cards,
                samples=max(12, samples // 4),
                max_seconds=min(0.008, max_seconds * 0.25),
            )
            if force_full_equity or abs(quick_equity - pot_odds) < 0.16:
                equity = core._range_conditioned_equity(
                    player,
                    hero_hand,
                    board_cards,
                    samples=samples,
                    max_seconds=max_seconds,
                )
            else:
                equity = quick_equity

        raise_margin = core._raise_margin_by_street(player.street) - 0.04
        call_margin = core._call_margin_by_street(player.street) + 0.04
        discard_bias = core._opponent_discard_bias(player.street)
        range_bias = core._opponent_range_bias(player.street)
        texture_raise, texture_call, raise_cap_mult = core._board_texture_adjustments(board_cards)
        raise_margin += discard_bias + range_bias + texture_raise
        call_margin += discard_bias + range_bias + texture_call
        opp_raise_adj, opp_call_adj, opp_river_raise_adj, opp_river_floor_adj = core._opponent_threshold_adjustments(player)
        raise_margin += opp_raise_adj
        call_margin += opp_call_adj
        if continue_cost > 0 and pot_total > 0:
            call_overbet_adj, raise_overbet_adj = core._opponent_overbet_adjustments(
                player,
                continue_cost,
                pot_total,
            )
            call_margin += call_overbet_adj
            raise_margin += raise_overbet_adj
            call_margin += core._bet_escalation_penalty(player, continue_cost, pot_total)

        raise_threshold = pot_odds + raise_margin
        call_threshold = pot_odds + call_margin
        lumberjack.record_thresholds("ComebackPolicy", player.street, call_threshold, raise_threshold, pot_odds)

        hero_raises = getattr(player, "_hero_raise_count", {}).get(int(player.street), 0)
        villain_raises = getattr(player, "_villain_raise_count", {}).get(int(player.street), 0)
        if hero_raises >= 1 and villain_raises >= 1 and equity < 0.72:
            if FoldAction in legal_actions and continue_cost > 0:
                return _record(FoldAction(), equity, pot_odds)
            if CheckAction in legal_actions and continue_cost == 0:
                return _record(CheckAction(), equity, pot_odds)

        if RaiseAction in legal_actions:
            min_raise, max_raise = getattr(player.hero, "raise_bounds", (0, 0))
            raise_cap = max(4, int(pot_total // 2 * raise_cap_mult))
            fold_mean, fold_low, fold_high, fold_conf, fold_width = core.fold_equity_band(
                min_raise, player.street, pot_total
            )
            fold_rate = fold_mean
            if fold_rate <= 0.0:
                fold_rate = core.fold_equity_estimate(None, min_raise, player.street, pot_total)
            if fold_conf > 0.0 and fold_width > 0.25:
                fold_rate = min(fold_rate, fold_low)
            if min_raise <= raise_cap and min_raise <= player.hero.stack // 2:
                if equity >= max(0.42, raise_threshold):
                    value_mult = 1.12 * core._value_extraction_multiplier(player)
                    target = core._adaptive_raise_size(
                        player,
                        pot_total,
                        min_raise,
                        max_raise,
                        equity,
                        value_mult=value_mult,
                    )
                    target = core._suppress_medium_raise_target(target, min_raise, pot_total, equity)
                    target = core._adjust_value_raise(target, min_raise, max_raise, fold_rate, equity)
                    if core.is_river(player.street) and equity < core._RIVER_VALUE_FLOOR + opp_river_floor_adj:
                        pass
                    elif target >= min_raise:
                        return _record(RaiseAction(target), equity, pot_odds)
                if equity < pot_odds - 0.06 and fold_rate >= 0.22 and fold_width <= 0.35:
                    target = core._adaptive_raise_size(
                        player,
                        pot_total,
                        min_raise,
                        max_raise,
                        equity,
                        bluff=True,
                    )
                    if target >= min_raise:
                        return _record(RaiseAction(target), equity, pot_odds)

        if continue_cost > 0 and FoldAction in legal_actions:
            if equity < pot_odds + call_margin + 0.04:
                return _record(FoldAction(), equity, pot_odds)
        if CheckAction in legal_actions and continue_cost == 0:
            return _record(CheckAction(), equity, pot_odds)
        if CallAction in legal_actions and equity >= pot_odds + call_margin + 0.02:
            return _record(CallAction(), equity, pot_odds)
        if CheckAction in legal_actions:
            return _record(CheckAction(), equity, pot_odds)
        return _record(FoldAction(), equity, pot_odds)
