#here we are again
import os
import random
import time

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


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


_DEFAULT_DISCARD_SECONDS = _env_float("NEUROPOKER_DISCARD_MAX_SECONDS", 0.05)
_DEFAULT_DISCARD_SAMPLES = _env_int("NEUROPOKER_DISCARD_SAMPLES", 150)

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
    str_cards = _ensure_str_cards(cards)
    for combo in combinations(str_cards, 5):
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

def _enumerate_discard_equity(hole, board, deck, remaining_cards):
    stats = {discard: {"wins": 0, "losses": 0, "ties": 0} for discard in hole}
    hole_variants = {discard: [card for card in hole if card != discard] for discard in hole}
    deck_set = set(deck)

    for discard, my_hole in hole_variants.items():
        new_board = board + [discard]
        for opp_hand in combinations(deck_set, 2):
            remaining_deck = deck_set - set(opp_hand)
            for board_fill in combinations(remaining_deck, remaining_cards):
                final_board = new_board + list(board_fill)
                hero_rank = evaluate_best(final_board, my_hole)
                villain_rank = evaluate_best(final_board, list(opp_hand))
                result = compare_evals(hero_rank, villain_rank)
                if result > 0:
                    stats[discard]["wins"] += 1
                elif result < 0:
                    stats[discard]["losses"] += 1
                else:
                    stats[discard]["ties"] += 1

    equities = []
    for discard in hole:
        totals = stats[discard]
        total = totals["wins"] + totals["losses"] + totals["ties"]
        if not total:
            equities.append(0.5)
            continue
        equities.append((totals["wins"] + 0.5 * totals["ties"]) / total)
    return tuple(equities)


def discard_equity(hole: list, board: list, n_samples: int | None = None, max_seconds: float | None = None) -> tuple:
    """
    monte carlo board approximation

    H = [c1, c2, c3]
    B = [b1, b2, b3?]
    D = deck-H-B

    hand = H- {c}
    board = B + [c]


    """
    hole = _ensure_int_cards(hole)
    board = _ensure_int_cards(board)
    if not hole:
        return ()

    known = set(hole + board)
    full_deck = Deck().cards
    deck = [card for card in full_deck if card not in known]

    remaining_cards = max(0, 6 - len(board) - 1)
    needed = 2 + remaining_cards
    if needed > len(deck):
        return _enumerate_discard_equity(hole, board, deck, remaining_cards)

    samples_limit = n_samples if n_samples is not None else _DEFAULT_DISCARD_SAMPLES
    if samples_limit <= 0:
        return _enumerate_discard_equity(hole, board, deck, remaining_cards)

    if max_seconds is None:
        max_seconds = _DEFAULT_DISCARD_SECONDS
    rng = random.Random()
    stats = {discard: {"wins": 0, "losses": 0, "ties": 0} for discard in hole}
    hole_variants = {discard: [card for card in hole if card != discard] for discard in hole}

    start_time = time.perf_counter()
    samples = 0
    while samples < samples_limit:
        if max_seconds > 0 and time.perf_counter() - start_time >= max_seconds:
            break
        sample_cards = rng.sample(deck, needed)
        opp_hand = sample_cards[:2]
        board_fill = sample_cards[2:]

        for discard, my_hole in hole_variants.items():
            final_board = board + [discard] + board_fill
            hero_rank = evaluate_best(final_board, my_hole)
            villain_rank = evaluate_best(final_board, list(opp_hand))
            result = compare_evals(hero_rank, villain_rank)
            if result > 0:
                stats[discard]["wins"] += 1
            elif result < 0:
                stats[discard]["losses"] += 1
            else:
                stats[discard]["ties"] += 1

        samples += 1

    if samples == 0:
        return _enumerate_discard_equity(hole, board, deck, remaining_cards)

    equities = []
    for discard in hole:
        totals = stats[discard]
        total = totals["wins"] + totals["losses"] + totals["ties"]
        if not total:
            equities.append(0.5)
            continue
        equities.append((totals["wins"] + 0.5 * totals["ties"]) / total)
    return tuple(equities)
