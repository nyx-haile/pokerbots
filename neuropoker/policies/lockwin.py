"""
- LockWinPolicy: When we can lock the win by folding
"""
from skeleton.actions import CallAction, CheckAction, DiscardAction, FoldAction, RaiseAction

import lumberjack
import stats
from strategy import policy_api as core


class LockWinPolicy:
    """
    Policy when we can lock the win by folding all remaining hands.
    Just fold/check to secure the victory.
    """

    @staticmethod
    def play(player):
        legal_actions = set(player.hero.legal_actions)

        def _record(action):
            lumberjack.record_policy_action("LockWinPolicy", player.street, action.__class__.__name__)
            return action

        if DiscardAction in legal_actions:
            return _record(DiscardAction(0))
        if FoldAction in legal_actions and player.hero.continue_cost > 0:
            return _record(FoldAction())
        if CheckAction in legal_actions and player.hero.continue_cost == 0:
            return _record(CheckAction())
        if CallAction in legal_actions:
            return _record(CallAction())
        if CheckAction in legal_actions:
            return _record(CheckAction())
        return _record(FoldAction())
