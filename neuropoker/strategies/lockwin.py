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
    Policy when opponent can lock the win - we need high-variance aggressive play.
    - Lower raise thresholds (raise with weaker hands)
    - Increase bluff frequency
    - Call more liberally (chase draws)
    - Push all-in with marginal +EV hands
    """

    # Desperation adjustments
    RAISE_MARGIN_REDUCTION = 0.15  # Makes raising easier
    CALL_MARGIN_BONUS = 0.12       # Makes calling easier
    BLUFF_RATE_MULT = 3.0          # Triple bluff rate
    ALLIN_EQUITY_THRESHOLD = 0.42  # Push all-in with 42%+ equity

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
                    samples=samples,
                    max_seconds=max_seconds,
                    discard_samples=discard_samples,
                )
            else:
                equity = quick_equity

        # Desperate mode: lower raise threshold, push all-in with marginal hands
        if RaiseAction in legal_actions:
            min_raise, max_raise = getattr(player.hero, "raise_bounds", (0, 0))
            
            # Push all-in with reasonable equity - we need to gamble
            if equity >= DesperatePolicy.ALLIN_EQUITY_THRESHOLD and min_raise > 0:
                return RaiseAction(max_raise)
            
            # Lower raise threshold significantly
            base_raise_margin = core._raise_margin_by_street(player.street)
            desperate_raise_margin = base_raise_margin - DesperatePolicy.RAISE_MARGIN_REDUCTION
            raise_threshold = pot_odds + desperate_raise_margin
            
            if equity > raise_threshold and min_raise > 0:
                # Raise aggressively - pot-sized or larger
                target = min(max_raise, max(min_raise, int(pot_total * 1.5)))
                return RaiseAction(target)
            
            # Increased bluff rate
            base_bluff_rate = 0.006  # From bluff.py
            desperate_bluff_rate = base_bluff_rate * DesperatePolicy.BLUFF_RATE_MULT
            if equity < pot_odds - 0.05:
                import random
                if random.random() < desperate_bluff_rate:
                    target = min(max_raise, max(min_raise, int(pot_total * 0.75)))
                    if target >= min_raise:
                        return RaiseAction(target)

        # Desperate mode: more liberal calling
        base_call_margin = core._call_margin_by_street(player.street)
        desperate_call_margin = base_call_margin + DesperatePolicy.CALL_MARGIN_BONUS

        # Never fold if equity is anywhere close to pot odds
        if equity >= pot_odds - desperate_call_margin:
            if CallAction in legal_actions:
                return CallAction()
            if CheckAction in legal_actions:
                return CheckAction()

        # Even with bad equity, consider calling small bets
        if continue_cost <= pot_total * 0.25 and equity >= 0.20:
            if CallAction in legal_actions:
                return CallAction()

        # Check if possible
        if CheckAction in legal_actions and continue_cost == 0:
            return CheckAction()
        if CallAction in legal_actions:
            return CallAction()
        if CheckAction in legal_actions:
            return CheckAction()
        
        # Avoid folding - use the lock defense
        return core._avoid_lock_win_fold(player, FoldAction())
