#here we are again
import os

from deuces import Card, Deck, Evaluator
from itertools import combinations
from functools import lru_cache

try:
    from pokerstove import CardSet
except ImportError:
    CardSet = None

_BACKEND = os.environ.get("NEUROPOKER_EVAL_BACKEND", "auto").lower()
_USE_POKERSTOVE = _BACKEND in ("auto", "pokerstove") and CardSet is not None

evaluator = Evaluator()

class DeckEmptyException(Exception):
    pass

class Sdeck(Deck):
    def draw(self, n=1, card=None):
        #update draw to allow monte carlo sim from any point
        if card is not None:
            card_int = Card.new(card)
            if card_int in self.cards:
                self.cards.remove(card_int)
                return card
            raise DeckEmptyException(f'{card} is not in the deck!')

        return super().draw(n)


def convert(card: str) -> int:
    """
    converts cards from string format ('Qh')
    into integer format as used by deuces.
    """
    return Card.new(card)

def _ensure_int_cards(cards):
    if not cards:
        return []
    if isinstance(cards[0], int):
        return list(cards)
    return [convert(card) for card in cards]

def _ensure_str_cards(cards):
    if not cards:
        return []
    if isinstance(cards[0], str):
        return list(cards)
    return [Card.int_to_str(card) for card in cards]

def _deuces_best_eval(cards):
    best = None
    for combo in combinations(cards, 5):
        rank = evaluator.evaluate(list(combo), [])
        if best is None or rank < best:
            best = rank
    return best

def _pokerstove_best_eval(cards):
    best = None
    for combo in combinations(cards, 5):
        score = CardSet("".join(combo)).evaluateHigh().code()
        if best is None or score > best:
            best = score
    return best

def evaluate_best(board, hand):
    cards = list(board) + list(hand)
    if _USE_POKERSTOVE:
        return _pokerstove_best_eval(_ensure_str_cards(cards))
    return _deuces_best_eval(_ensure_int_cards(cards))

def compare_evals(hero_rank, villain_rank):
    if _USE_POKERSTOVE:
        if hero_rank > villain_rank:
            return 1
        if hero_rank < villain_rank:
            return -1
        return 0
    if hero_rank < villain_rank:
        return 1
    if hero_rank > villain_rank:
        return -1
    return 0

def evaluate(board, hand):
    """
    returns ranking of best 5-card hand from the supplied cards
    """
    return evaluate_best(board, hand)


@lru_cache(maxsize=200_000)
def eq_cache(board_tuple, hole_tuple):
    board = list(board_tuple)
    hole = list(hole_tuple)
    return evaluate(board, hole)

def discard_equity(hole: list, board: list, n_samples: int=50) -> tuple:
    """
    monte carlo board approximation

    H = [c1, c2, c3]
    B = [b1, b2, b3?]
    D = deck-H-B

    hand = H- {c}
    board = B + [c]


    """
    results = {}
    hole = _ensure_int_cards(hole)
    board = _ensure_int_cards(board)
    known = hole + board
    full_deck = Deck().cards
    deck = set(full_deck).symmetric_difference(known)

    remaining_cards = 6-len(board)-1

    for discard in hole:
        my_hole = [i for i in hole if i!= discard]
        new_board = board+[discard]
        wins = ties = losses = 0

        for opp_hand in combinations(deck, 2):
            remaining_deck = deck ^ set(opp_hand)

            for board_fill in combinations(remaining_deck, remaining_cards):
                final_board = new_board + list(board_fill)
                #have to fix evaluator.evaluate to work with 8 cards.
                #Might switch backends to the C++ version if speed is
                #that much of an issue.
                hero_rank = evaluate_best(final_board, my_hole)
                villain_rank = evaluate_best(final_board, list(opp_hand))

                result = compare_evals(hero_rank, villain_rank)
                if result > 0:
                    wins += 1
                elif result < 0:
                    losses += 1
                else:
                    ties += 1

        total = wins+losses+ties
        equity = (wins + 0.5*ties) /total
        results[discard]=equity
    return tuple(results[i] for i in results)
