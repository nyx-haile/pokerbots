from skeleton.actions import CallAction, CheckAction, DiscardAction, FoldAction, RaiseAction

import lumberjack
import stats
from strategy import policy_api as core


class BluffPolicy:
    """Bluff policy: act as if holding the nuts for the entire round."""

    @staticmethod
    def play(player):
        legal_actions = set(player.hero.legal_actions)
        hero_hand = list(player.hero.hand)
        board_cards = list(player.community)
        pot_total = max(1, player.hero.pot_total)
        continue_cost = player.hero.continue_cost

        def _record(action):
            lumberjack.record_policy_action("BluffPolicy", player.street, action.__class__.__name__)
            return action

        if DiscardAction in legal_actions:
            best_i = _bluff_signal_discard_index(hero_hand, board_cards)
            player.hero.last_discard = hero_hand[best_i]
            player.hero.last_discard_visible = player.hero.blind
            player.hero.discard_bluff = True
            return _record(DiscardAction(best_i))

        if RaiseAction in legal_actions:
            min_raise, max_raise = getattr(player.hero, "raise_bounds", (0, 0))
            if min_raise > 0 and continue_cost <= 0:
                texture_raise, _, raise_cap_mult = core._board_texture_adjustments(board_cards)
                if texture_raise <= 0.0:
                    raise_cap = max(4, int(pot_total * 0.5 * raise_cap_mult))
                    if min_raise <= raise_cap:
                        target = core._raise_size(pot_total, min_raise, max_raise, equity=0.0, bluff=True)
                        target = min(target, int(pot_total * 0.4))
                        target = min(target, raise_cap)
                        target = core._cap_raise_for_lock_defense(player, target)
                        fold_rate = core.fold_equity_estimate(None, target, player.street, pot_total)
                        if target >= min_raise and fold_rate >= 0.35:
                            core._consume_discard_bluff(player)
                            return _record(RaiseAction(target))

        if CheckAction in legal_actions and player.hero.continue_cost == 0:
            return _record(CheckAction())
        if CallAction in legal_actions:
            return _record(CallAction())
        if CheckAction in legal_actions:
            return _record(CheckAction())
        return _record(FoldAction())


def _bluff_signal_discard_index(hero_hand, board_cards):
    if not hero_hand:
        return 0
    cards_int = stats._ensure_int_cards(list(hero_hand))
    board_int = stats._ensure_int_cards(list(board_cards))
    board_ranks = {stats.Card.get_rank_int(card) for card in board_int}
    board_suits = [stats.Card.get_suit_int(card) for card in board_int]
    scores = []
    for card in cards_int:
        rank = stats.Card.get_rank_int(card)
        suit = stats.Card.get_suit_int(card)
        score = rank
        if rank >= 8:
            score += 1
        if rank in board_ranks:
            score += 5
        if board_suits.count(suit) >= 2:
            score += 3
        scores.append(score)
    return max(range(len(scores)), key=scores.__getitem__)
