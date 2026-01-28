"""
- DesperatePolicy: When opponent can lock the win (we must gamble)
"""
from skeleton.actions import CallAction, CheckAction, DiscardAction, FoldAction, RaiseAction

import lumberjack
import stats
from strategy import policy_api as core

class DesperatePolicy:
    """
    Policy when opponent can lock the win but hasn't done so yet.
    Play TIGHTER to avoid giving them chips - they're likely playing for value.
    Uses tunable parameters from strategy module.
    """

    @staticmethod
    def play(player):
        legal_actions = set(player.hero.legal_actions)
        hero_hand = list(player.hero.hand)
        board_cards = list(player.community)

        def _record(action, equity_value=None, pot_odds_value=None):
            lumberjack.record_policy_action("DesperatePolicy", player.street, action.__class__.__name__)
            if equity_value is not None and pot_odds_value is not None:
                lumberjack.record_equity_vs_pot_odds(
                    "DesperatePolicy",
                    player.street,
                    equity_value,
                    pot_odds_value,
                )
            return action

        # Handle discards normally
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
            player.hero.last_discard = hero_hand[best_i]
            player.hero.last_discard_visible = player.hero.blind
            player.hero.discard_bluff = False
            return _record(DiscardAction(best_i))

        pot_total = max(1, player.hero.pot_total)
        continue_cost = player.hero.continue_cost
        pot_odds = core.pot_odds_to_call(continue_cost, pot_total)

        # Calculate equity
        if player.street <= 0:
            equity = stats.preflop_strength(hero_hand)
        else:
            samples, max_seconds, discard_samples = core._equity_budget(player)
            samples, max_seconds, discard_samples = core._adjust_budget_for_turn_raise(
                player,
                pot_total,
                samples,
                max_seconds,
                discard_samples,
            )
            force_full_equity = core._is_large_raise(player, pot_total)
            quick_equity = stats.estimate_equity(
                hero_hand,
                board_cards,
                samples=max(10, samples // 4),
                max_seconds=min(0.006, max_seconds * 0.25),
                discard_samples=max(3, discard_samples // 2),
            )
            # Use full equity calculation for close decisions
            if force_full_equity:
                equity = stats.estimate_equity(
                    hero_hand,
                    board_cards,
                    samples=samples,
                    max_seconds=max_seconds,
                    discard_samples=discard_samples,
                )
            elif abs(quick_equity - pot_odds) < 0.12:
                equity = stats.estimate_equity(
                    hero_hand,
                    board_cards,
                    samples=samples,
                    max_seconds=max_seconds,
                    discard_samples=discard_samples,
                )
            else:
                equity = quick_equity

        raise_margin = core._raise_margin_by_street(player.street)
        call_margin = core._call_margin_by_street(player.street)
        discard_bias = core._opponent_discard_bias(player.street)
        range_bias = core._opponent_range_bias(player.street)
        texture_raise, texture_call, raise_cap_mult = core._board_texture_adjustments(board_cards)
        raise_margin += discard_bias + range_bias + texture_raise
        call_margin += discard_bias + range_bias + texture_call
        # Tighten more when desperate
        call_margin -= core._DESPERATE_CALL_PENALTY
        raise_margin += core._DESPERATE_RAISE_MARGIN
        traj_raise_adj, traj_call_adj = core._trajectory_lock_adjustment(player)
        raise_margin += traj_raise_adj
        call_margin += traj_call_adj
        # Lock defense: increase raise threshold when near danger zone
        raise_margin += core._lock_defense_raise_margin(player)

        # Defensive mode: only raise with very strong hands (tunable threshold)
        if RaiseAction in legal_actions:
            min_raise, max_raise = getattr(player.hero, "raise_bounds", (0, 0))
            raise_threshold = max(core._DESPERATE_NUT_THRESHOLD, pot_odds + raise_margin)
            raise_cap = max(4, int(pot_total // 2 * raise_cap_mult))
            if min_raise <= raise_cap and min_raise <= player.hero.stack // 3:
                if equity >= raise_threshold:
                    value_mult = core._value_extraction_multiplier(player)
                    target = int(pot_total * 0.4 * value_mult)
                    target = min(target, int(pot_total * 0.6))
                    target = max(min_raise, min(max_raise, target))
                    target = min(target, raise_cap)
                    target = core._cap_raise_for_lock_defense(player, target)
                    fold_rate = core.fold_equity_estimate(None, target, player.street, pot_total)
                    if target >= min_raise and fold_rate >= 0.25:
                        return _record(RaiseAction(target), equity, pot_odds)

        # Only call if we have good equity relative to pot odds
        if equity >= pot_odds + call_margin:
            if CallAction in legal_actions:
                return _record(CallAction(), equity, pot_odds)
            if CheckAction in legal_actions:
                return _record(CheckAction(), equity, pot_odds)

        # Check if free
        if CheckAction in legal_actions and continue_cost == 0:
            return _record(CheckAction(), equity, pot_odds)

        # Fold marginal hands - don't give opponent chips
        if equity < pot_odds - 0.05:
            # But use lock defense to avoid giving them win-lock
            return _record(core._avoid_lock_win_fold(player, FoldAction()), equity, pot_odds)

        # Default to lock-aware fold
        return _record(core._avoid_lock_win_fold(player, FoldAction()), equity, pot_odds)
