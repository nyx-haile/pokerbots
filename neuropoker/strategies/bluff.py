from skeleton.actions import CallAction, CheckAction, DiscardAction, FoldAction, RaiseAction

import stats
import strategy as core


class BluffPolicy:
    """Bluff policy: act as if holding the nuts for the entire round."""

    @staticmethod
    def play(player):
        legal_actions = set(player.hero.legal_actions)
        hero_hand = list(player.hero.hand)
        board_cards = list(player.community)

        if DiscardAction in legal_actions:
            best_i = _bluff_signal_discard_index(hero_hand, board_cards)
            player.hero.last_discard = hero_hand[best_i]
            player.hero.last_discard_visible = player.hero.blind
            player.hero.discard_bluff = True
            return DiscardAction(best_i)

        if RaiseAction in legal_actions:
            min_raise, max_raise = getattr(player.hero, "raise_bounds", (0, 0))
            if min_raise > 0:
                target = core._nut_raise_target(min_raise, max_raise)
                core._consume_discard_bluff(player)
                return RaiseAction(target)

        if CheckAction in legal_actions and player.hero.continue_cost == 0:
            return CheckAction()
        if CallAction in legal_actions:
            return CallAction()
        if CheckAction in legal_actions:
            return CheckAction()
        return FoldAction()


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
