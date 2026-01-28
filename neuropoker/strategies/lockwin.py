"""
Win-lock related policies:
- LockWinPolicy: When we can lock the win by folding
- DesperatePolicy: When opponent can lock the win (we must gamble)
"""
from skeleton.actions import CallAction, CheckAction, DiscardAction, FoldAction, RaiseAction

import stats
import strategy as core


class LockWinPolicy:
    """
    Policy when we can lock the win by folding all remaining hands.
    Just fold/check to secure the victory.
    """

    @staticmethod
    def play(player):
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
            return DiscardAction(best_i)

        pot_total = max(1, player.hero.pot_total)
        continue_cost = player.hero.continue_cost
        pot_odds = core.pot_odds_to_call(continue_cost, pot_total)

        # Calculate equity
        if player.street <= 0:
            equity = stats.preflop_strength(hero_hand)
        else:
            samples, max_seconds, discard_samples = core._equity_budget(player)
            quick_equity = stats.estimate_equity(
                hero_hand,
                board_cards,
                samples=max(5, samples // 4),
                max_seconds=max_seconds * 0.25,
                discard_samples=max(2, discard_samples // 2),
            )
            # Use full equity calculation for important decisions
            if abs(quick_equity - pot_odds) < 0.15:
                equity = stats.estimate_equity(
                    hero_hand,
                    board_cards,
                    samples=stats._HIGH_VALUE_EQUITY_SAMPLES,
                    max_seconds=max_seconds,
                    discard_samples=discard_samples,
                )
            else:
                equity = quick_equity

        # Defensive mode: only raise with very strong hands (tunable threshold)
        if RaiseAction in legal_actions:
            min_raise, max_raise = getattr(player.hero, "raise_bounds", (0, 0))

            # Only raise with nut-level equity
            if equity >= core._DESPERATE_NUT_THRESHOLD and min_raise > 0:
                # Small value raises only - don't bloat the pot
                target = min(max_raise, max(min_raise, int(pot_total * 0.5)))
                return RaiseAction(target)

            # No bluffing when desperate - we can't afford to lose chips

        # Defensive mode: tighter calling using tunable penalty
        base_call_margin = core._call_margin_by_street(player.street)
        defensive_call_margin = base_call_margin - core._DESPERATE_CALL_PENALTY

        # Only call if we have good equity relative to pot odds
        if equity >= pot_odds + defensive_call_margin:
            if CallAction in legal_actions:
                return CallAction()
            if CheckAction in legal_actions:
                return CheckAction()

        # Check if free
        if CheckAction in legal_actions and continue_cost == 0:
            return CheckAction()

        # Fold marginal hands - don't give opponent chips
        if equity < pot_odds - 0.05:
            # But use lock defense to avoid giving them win-lock
            return core._avoid_lock_win_fold(player, FoldAction())

        # Borderline - call small bets, fold large ones
        if continue_cost <= pot_total * 0.20 and equity >= pot_odds - 0.10:
            if CallAction in legal_actions:
                return CallAction()

        # Default to lock-aware fold
        return core._avoid_lock_win_fold(player, FoldAction())
