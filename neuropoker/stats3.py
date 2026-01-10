#here we are again
from deuces import Card, Deck, Evaluator
from itertools import combinations
from functools import lru_cache

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

def evaluate(board, hand):
    """
    returns ranking of hands out of 7462
    """
    hand_int = [convert(card) for card in hand]
    board_int = [convert(card) for card in board]
    return evaluator.evaluate(board_int, hand_int)


@lru_cache(maxsize=200_000)
def eq_cache(board_tuple, hole_tuple):
    board = list(board_tuple)
    hole = list(hole_tuple)
    return evaluate(hole, board)

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
    known = hole+board
    full_deck = Deck().cards
    deck = set(full_deck).symmetric_difference(known)

    remaining_cards = 6-len(board)-1

    for discard in hole:
        my_hole = [i for i in hole if i!= discard]
        new_board = board+[discard]
        wins = ties = losses = 0

        for opp_hand in combinations(deck, 2):
            remaining_deck = deck ^ set(opp_hand)

            for board_fill in combinations(remaining_deck, remaining_cards-2):
                final_board = new_board + list(board_fill)
                #have to fix evaluator.evaluate to work with 8 cards.
                #Might switch backends to the C++ version if speed is
                #that much of an issue.
                hero_rank = evaluator.evaluate(final_board, my_hole)
                villain_rank = evaluator.evaluate(final_board, list(opp_hand))

                if hero_rank < villain_rank:
                    wins+=1
                elif hero_rank > villain_rank:
                    losses +=1
                else:
                    ties+=1

        total = wins+losses+ties
        equity = (wins + 0.5*ties) /total
        results[discard]=equity
    return tuple(results[i] for i in results)

d = Deck()
hole = d.draw(3)
board = d.draw(2)
print(discard_equity(hole, board))
