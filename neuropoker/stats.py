#here we are again
import os
import random
import time
from itertools import combinations
from functools import lru_cache
import pkrbot
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

_BACKEND = os.environ.get("NEUROPOKER_EVAL_BACKEND", "pkrbot").lower()
_USE_PKRBOT = _BACKEND in ("auto", "pkrbot") and pkrbot is not None

_RANKS = "23456789TJQKA"
_SUITS = "cdhs"
_SUIT_TO_INT = {"s": 1, "h": 2, "d": 4, "c": 8}
_FULL_DECK = [rank + suit for rank in _RANKS for suit in _SUITS]
_PKRBOT_CARD_CACHE = {card: pkrbot.Card(card) for card in _FULL_DECK} if pkrbot is not None else {}


def _card_to_str(card) -> str:
    if isinstance(card, str):
        return card
    if pkrbot is not None and isinstance(card, pkrbot.Card):
        return str(card)
    raise ValueError(f"Unsupported card type: {type(card)}")


class Card:
    @staticmethod
    def get_rank_int(card) -> int:
        card_str = _card_to_str(card)
        rank_char = card_str[0].upper()
        return _RANKS.index(rank_char)

    @staticmethod
    def get_suit_int(card) -> int:
        card_str = _card_to_str(card)
        suit_char = card_str[1].lower()
        return _SUIT_TO_INT[suit_char]

    @staticmethod
    def int_to_str(card) -> str:
        return _card_to_str(card)


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))

_HARD_ACTION_TIMEOUT = 9.0  # Hard 9-second cap for any search function
_action_start_time = None
_action_id = 0  # Incremented each reset to detect stale timers

def _reset_action_timer():
    """Reset the action timer at the start of each action."""
    global _action_start_time, _action_id
    _action_start_time = time.perf_counter()
    _action_id += 1

def _time_remaining() -> float:
    """Return seconds remaining before hard timeout, or infinity if not set."""
    if _action_start_time is None:
        return float('inf')
    elapsed = time.perf_counter() - _action_start_time
    # If elapsed is negative, timer is corrupt - return infinity
    if elapsed < 0:
        return float('inf')
    # If elapsed > 30, timer is stale but we should still respect it
    # (action took way too long, abort immediately)
    if elapsed > 30:
        return 0.0
    return max(0.0, _HARD_ACTION_TIMEOUT - elapsed)

def _should_abort() -> bool:
    """Check if we should abort due to approaching timeout."""
    remaining = _time_remaining()
    # Only abort if timer is active and we're out of time
    if remaining == float('inf'):
        return False
    return remaining <= 0.001

_DEFAULT_DISCARD_SECONDS = 0.05
_DEFAULT_DISCARD_SAMPLES = 2500
_DEFAULT_EQUITY_SAMPLES = 2500
_HIGH_VALUE_EQUITY_SAMPLES = 10_000
_DEFAULT_EQUITY_SECONDS = 0.05

class DeckEmptyException(Exception):
    pass

def convert(card: str) -> int:
    """
    normalizes cards to string format ('Qh').
    """
    return _card_to_str(card)

def _ensure_int_cards(cards):
    if not cards:
        return []
    return [_card_to_str(card) for card in cards]

def _ensure_str_cards(cards):
    if not cards:
        return []
    return [_card_to_str(card) for card in cards]

_CHEN_HIGH = {
    14: 10.0,
    13: 8.0,
    12: 7.0,
    11: 6.0,
    10: 5.0,
    9: 4.5,
    8: 4.0,
    7: 3.5,
    6: 3.0,
    5: 2.5,
    4: 2.0,
    3: 1.5,
    2: 1.0,
}


@lru_cache(maxsize=2048)
def _chen_score(rank_high: int, rank_low: int, suited: bool) -> float:
    score = _CHEN_HIGH.get(rank_high, 0.0)
    if rank_high == rank_low:
        score = max(5.0, score * 2.0)
    if suited and rank_high != rank_low:
        score += 2.0
    gap = rank_high - rank_low - 1
    if gap == 1:
        score -= 1.0
    elif gap == 2:
        score -= 2.0
    elif gap == 3:
        score -= 4.0
    elif gap >= 4:
        score -= 5.0
    if gap <= 1 and rank_high <= 12 and rank_high != rank_low:
        score += 1.0
    if rank_high <= 5 and gap <= 1 and rank_high != rank_low:
        score += 0.5
    return max(0.0, score)


@lru_cache(maxsize=300_000)
def _preflop_strength_cached(cards_tuple: Tuple[int, ...]) -> float:
    cards = list(cards_tuple)
    if len(cards) != 3:
        return 0.5

    ranks = [Card.get_rank_int(card) + 2 for card in cards]
    suits = [Card.get_suit_int(card) for card in cards]
    rank_counts = {rank: ranks.count(rank) for rank in ranks}
    suit_counts = {suit: suits.count(suit) for suit in suits}

    scores = []
    pairs = [(0, 1), (0, 2), (1, 2)]
    for i, j in pairs:
        rank_high = max(ranks[i], ranks[j])
        rank_low = min(ranks[i], ranks[j])
        suited = suits[i] == suits[j]
        scores.append(_chen_score(rank_high, rank_low, suited))

    scores.sort(reverse=True)
    base = scores[0] + 0.35 * scores[1] + 0.10 * scores[2]

    if 3 in rank_counts.values():
        base += 3.0
    elif 2 in rank_counts.values():
        kicker = max(rank for rank, count in rank_counts.items() if count == 1)
        base += (kicker / 14.0) * 1.2

    if 3 in suit_counts.values():
        base += 1.0
    elif 2 in suit_counts.values():
        base += 0.5

    span = max(ranks) - min(ranks)
    if span <= 4:
        base += 1.0
    elif span <= 6:
        base += 0.5

    strength = base / 32.0
    return max(0.0, min(1.0, strength))


def preflop_strength(hole_cards: Sequence[Union[str, int]]) -> float:
    """
    Fast, cached heuristic for 3-card preflop strength in Toss or Hold'em.
    Uses a Chen-style two-card score for each pair plus 3-card synergy bonuses.
    Returns a value in [0, 1].
    """
    cards = _ensure_int_cards(list(hole_cards))
    cards.sort()
    return _preflop_strength_cached(tuple(cards))

def _pokerstove_best_eval(cards):
    str_cards = _ensure_str_cards(cards)
    if len(cards) > 7:
        return eval_best_8(str_cards)
    return cs("".join(str_cards)).evaluateHigh().code()

def _pkrbot_best_eval(cards):
    # Cards are already strings from _evaluate_best_cached, skip conversion
    return pkrbot.evaluate([_PKRBOT_CARD_CACHE[card] for card in cards])

def eval_best_8(cards8_str):
    # cards8_str is a list of card str, eg ['Ah', 'Ks', '7h']
    best = None
    for i in range(len(cards8_str)):
        subset = cards8_str[:i] + cards8_str[i+1:] #drop one card
        score = cs("".join(subset)).evaluateHigh().code()
        #get the best hand from this subset's score
        if best is None or score > best:
            best = score
    return best

@lru_cache(maxsize=500_000)
def _evaluate_best_cached(cards_tuple: Tuple[int, ...]) -> int:
    cards = list(cards_tuple)
    if _USE_PKRBOT:
        return _pkrbot_best_eval(cards)
    raise RuntimeError("No supported evaluator available pkrbot")


def evaluate_best(board, hand):
    # Fast path: avoid list concatenation and conversion when cards are strings
    if board.__class__ is list and hand.__class__ is list:
        cards = board + hand
    else:
        cards = list(board) + list(hand)
    return _evaluate_best_cached(tuple(cards))

def compare_evals(hero_rank, villain_rank):
    if _USE_PKRBOT:
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


@lru_cache(maxsize=300_000)
def eq_cache(board_tuple, hole_tuple):
    board = list(board_tuple)
    hole = list(hole_tuple)
    return evaluate(board, hole)

def _enumerate_discard_equity(hole, board, deck, remaining_cards):
    stats_dict = {discard: {"wins": 0, "losses": 0, "ties": 0} for discard in hole}
    hole_variants = {discard: [card for card in hole if card != discard] for discard in hole}
    deck_set = set(deck)
    opp_discard_pending = 1 if len(board) == 2 else 0
    opp_hand_size = 3 if opp_discard_pending else 2
    iteration_count = 0

    for discard, my_hole in hole_variants.items():
        new_board = board + [discard]
        if opp_hand_size == 3:
            for opp_hand in combinations(deck_set, 3):
                # Check timeout every 100 iterations to avoid overhead
                iteration_count += 1
                if iteration_count % 100 == 0 and _should_abort():
                    break
                opp_discard = _choose_opp_discard(opp_hand)
                remaining_hand = _remove_one(list(opp_hand), opp_discard)
                remaining_deck = deck_set - set(opp_hand)
                for board_fill in combinations(remaining_deck, remaining_cards):
                    final_board = new_board + [opp_discard] + list(board_fill)
                    hero_rank = evaluate_best(final_board, my_hole)
                    villain_rank = evaluate_best(final_board, list(remaining_hand))
                    result = compare_evals(hero_rank, villain_rank)
                    if result > 0:
                        stats_dict[discard]["wins"] += 1
                    elif result < 0:
                        stats_dict[discard]["losses"] += 1
                    else:
                        stats_dict[discard]["ties"] += 1
            if _should_abort():
                break
        else:
            for opp_hand in combinations(deck_set, 2):
                # Check timeout every 100 iterations to avoid overhead
                iteration_count += 1
                if iteration_count % 100 == 0 and _should_abort():
                    break
                remaining_deck = deck_set - set(opp_hand)
                for board_fill in combinations(remaining_deck, remaining_cards):
                    final_board = new_board + list(board_fill)
                    hero_rank = evaluate_best(final_board, my_hole)
                    villain_rank = evaluate_best(final_board, list(opp_hand))
                    result = compare_evals(hero_rank, villain_rank)
                    if result > 0:
                        stats_dict[discard]["wins"] += 1
                    elif result < 0:
                        stats_dict[discard]["losses"] += 1
                    else:
                        stats_dict[discard]["ties"] += 1
            if _should_abort():
                break

    equities = []
    for discard in hole:
        totals = stats_dict[discard]
        total = totals["wins"] + totals["losses"] + totals["ties"]
        if not total:
            equities.append(0.5)
            continue
        equities.append((totals["wins"] + 0.5 * totals["ties"]) / total)
    return tuple(equities)


@lru_cache(maxsize=300_000)
def _discard_equity_cached(
    hole_tuple: Tuple[int, ...],
    board_tuple: Tuple[int, ...],
    n_samples: int,
) -> tuple:
    return _discard_equity_impl(
        list(hole_tuple),
        list(board_tuple),
        n_samples=n_samples,
        max_seconds=0.0,
        return_metadata=False,
    )


def discard_equity(
    hole: list,
    board: list,
    n_samples: Optional[int] = None,
    max_seconds: Optional[float] = None,
    return_metadata: bool = False,
) -> tuple:
    """
    monte carlo board approximation

    H = [c1, c2, c3]
    B = [b1, b2, b3?]
    D = deck-H-B

    hand = H- {c}
    board = B + [c]


    """
    hole = list(hole) if hole.__class__ is not list else hole
    board = list(board) if board.__class__ is not list else board
    if not hole:
        return ()

    # Cap max_seconds to remaining time (incremental results returned on timeout)
    remaining = _time_remaining()
    if remaining < float('inf') and max_seconds is not None:
        max_seconds = min(max_seconds, remaining)
    elif remaining < float('inf'):
        max_seconds = remaining

    # Use cached path when no time limit specified and timer isn't critical
    if n_samples is not None and n_samples > 0 and (max_seconds is None or max_seconds == 0):
        if not return_metadata:
            return _discard_equity_cached(tuple(hole), tuple(board), int(n_samples))
    return _discard_equity_impl(hole, board, n_samples=n_samples, max_seconds=max_seconds, return_metadata=return_metadata)


def _discard_equity_impl(
    hole: list,
    board: list,
    n_samples: Optional[int] = None,
    max_seconds: Optional[float] = None,
    return_metadata: bool = False,
) -> tuple:
    known = set(hole + board)

    opp_discard_pending = 1 if len(board) == 2 else 0
    remaining_cards = max(0, 6 - len(board) - 1 - opp_discard_pending)
    opp_hand_size = 3 if opp_discard_pending else 2
    needed = opp_hand_size + remaining_cards

    # Build deck excluding known cards
    full_deck = list(_FULL_DECK)
    deck = [card for card in full_deck if card not in known]

    if needed > len(deck):
        return _enumerate_discard_equity(hole, board, deck, remaining_cards)

    samples_limit = n_samples if n_samples is not None else _DEFAULT_DISCARD_SAMPLES
    if samples_limit <= 0:
        return _enumerate_discard_equity(hole, board, deck, remaining_cards)

    if max_seconds is None:
        max_seconds = _DEFAULT_DISCARD_SECONDS

    # Use pkrbot.Deck for faster sampling
    if pkrbot is not None:
        seed = None
        if max_seconds == 0 and n_samples is not None:
            seed = hash((tuple(hole), tuple(board), int(n_samples))) & 0xFFFFFFFF
        pk_deck = pkrbot.Deck(seed)
        pk_deck.cards = [_PKRBOT_CARD_CACHE[c] for c in deck]
    else:
        pk_deck = None
        seed = None
        if max_seconds == 0 and n_samples is not None:
            seed = hash((tuple(hole), tuple(board), int(n_samples))) & 0xFFFFFFFF
        rng = random.Random(seed)

    stats = {discard: {"wins": 0, "losses": 0, "ties": 0, "samples": 0} for discard in hole}
    hole_variants = {discard: [card for card in hole if card != discard] for discard in hole}

    # Precompute pkrbot Card objects for hero cards
    if pk_deck is not None:
        hole_pk = {discard: [_PKRBOT_CARD_CACHE[c] for c in cards] for discard, cards in hole_variants.items()}
        board_pk = [_PKRBOT_CARD_CACHE[c] for c in board]
        pk_eval = pkrbot.evaluate

    start_time = time.perf_counter()
    samples = 0
    while samples < samples_limit:
        # Check timeout every 10 iterations to reduce overhead
        if samples > 0 and samples % 10 == 0:
            if max_seconds > 0 and time.perf_counter() - start_time >= max_seconds:
                break
        if samples > 0 and samples % 100 == 0 and _should_abort():
            break

        if pk_deck is not None:
            sample_cards = pk_deck.sample(needed)
            if opp_hand_size == 3:
                opp_hand_full = sample_cards[:3]
                # Find lowest rank card to discard (simple heuristic)
                opp_discard_idx = 0
                for i in range(1, 3):
                    if opp_hand_full[i].rank < opp_hand_full[opp_discard_idx].rank:
                        opp_discard_idx = i
                opp_discard_card = opp_hand_full[opp_discard_idx]
                opp_hand = [c for j, c in enumerate(opp_hand_full) if j != opp_discard_idx]
                board_fill = sample_cards[3:]
            else:
                opp_hand = sample_cards[:2]
                opp_discard_card = None
                board_fill = sample_cards[2:]

            for discard, my_hole_pk in hole_pk.items():
                final_board = board_pk + [_PKRBOT_CARD_CACHE[discard]]
                if opp_discard_card is not None:
                    final_board.append(opp_discard_card)
                final_board = final_board + board_fill
                hero_rank = pk_eval(final_board + my_hole_pk)
                villain_rank = pk_eval(final_board + opp_hand)
                if hero_rank > villain_rank:
                    stats[discard]["wins"] += 1
                elif hero_rank < villain_rank:
                    stats[discard]["losses"] += 1
                else:
                    stats[discard]["ties"] += 1
                stats[discard]["samples"] += 1
        else:
            sample_cards = rng.sample(deck, needed)
            if opp_hand_size == 3:
                opp_hand_full = sample_cards[:3]
                opp_discard = _choose_opp_discard(opp_hand_full)
                opp_hand = _remove_one(list(opp_hand_full), opp_discard)
                board_fill = sample_cards[3:]
            else:
                opp_hand = sample_cards[:2]
                opp_discard = None
                board_fill = sample_cards[2:]

            for discard, my_hole in hole_variants.items():
                final_board = board + [discard]
                if opp_hand_size == 3:
                    final_board.append(opp_discard)
                final_board.extend(board_fill)
                hero_rank = evaluate_best(final_board, my_hole)
                villain_rank = evaluate_best(final_board, list(opp_hand))
                result = compare_evals(hero_rank, villain_rank)
                if result > 0:
                    stats[discard]["wins"] += 1
                elif result < 0:
                    stats[discard]["losses"] += 1
                else:
                    stats[discard]["ties"] += 1
                stats[discard]["samples"] += 1

        samples += 1

    if samples == 0:
        return _enumerate_discard_equity(hole, board, deck, remaining_cards)

    equities = []
    metadata = []
    for discard in hole:
        totals = stats[discard]
        total = totals["wins"] + totals["losses"] + totals["ties"]
        if not total:
            equities.append(0.5)
            metadata.append({"discard": discard, "samples": totals["samples"]})
            continue
        equities.append((totals["wins"] + 0.5 * totals["ties"]) / total)
        metadata.append({"discard": discard, "samples": totals["samples"]})

    if return_metadata:
        return tuple(equities), metadata
    return tuple(equities)


def _choose_opp_discard(cards: Iterable[str]) -> str:
    best = None
    best_rank = None
    for card in cards:
        rank = Card.get_rank_int(card)
        if best is None or rank < best_rank:
            best = card
            best_rank = rank
    return best


def _remove_one(cards: List[str], card: str) -> List[str]:
    removed = False
    remaining = []
    for item in cards:
        if not removed and item == card:
            removed = True
            continue
        remaining.append(item)
    return remaining


def estimate_equity(
    hero_hand: Sequence[Union[str, int]],
    board_cards: Sequence[Union[str, int]],
    samples: Optional[int] = None,
    max_seconds: Optional[float] = None,
    discard_samples: Optional[int] = None,
) -> float:
    if samples is None:
        samples = _DEFAULT_EQUITY_SAMPLES
    if max_seconds is None:
        max_seconds = _DEFAULT_EQUITY_SECONDS
    if discard_samples is None:
        discard_samples = _DEFAULT_DISCARD_SAMPLES // 5

    # Cap max_seconds to remaining time (incremental results returned on timeout)
    remaining = _time_remaining()
    if remaining < float('inf'):
        max_seconds = min(max_seconds, remaining)

    hero_int = list(hero_hand) if hero_hand.__class__ is not list else hero_hand
    board_int = list(board_cards) if board_cards.__class__ is not list else board_cards
    return _estimate_equity_cached(
        tuple(hero_int),
        tuple(board_int),
        int(samples),
        float(max_seconds),
        int(discard_samples),
    )


def _equity_seed(hero_tuple, board_tuple, samples, max_seconds, discard_samples) -> int:
    return hash((hero_tuple, board_tuple, samples, max_seconds, discard_samples)) & 0xFFFFFFFF


@lru_cache(maxsize=200_000)
def _estimate_equity_cached(
    hero_tuple: Tuple[int, ...],
    board_tuple: Tuple[int, ...],
    samples: int,
    max_seconds: float,
    discard_samples: int,
) -> float:
    seed = _equity_seed(hero_tuple, board_tuple, samples, max_seconds, discard_samples)
    hero_int = list(hero_tuple)
    board_int = list(board_tuple)
    start_time = time.perf_counter()
    known = set(hero_int + list(board_int))
    deck = [card for card in _FULL_DECK if card not in known]

    # Use pkrbot.Deck for faster flop sampling
    if pkrbot is not None:
        pk_deck = pkrbot.Deck(seed)
        pk_deck.cards = [_PKRBOT_CARD_CACHE[c] for c in deck]
        rng = None
    else:
        rng = random.Random(seed)
        pk_deck = None

    if len(hero_int) == 3 and len(board_int) < 2:
        total = 0.0
        runs = 0
        for i in range(samples):
            # Check both local max_seconds and global action timeout
            if max_seconds > 0 and time.perf_counter() - start_time >= max_seconds:
                break
            if i % 50 == 0 and _should_abort():
                break
            if pk_deck is not None:
                flop = [str(c) for c in pk_deck.sample(2)]
            else:
                flop = rng.sample(deck, 2)
            # Pass remaining time to inner call
            remaining = _time_remaining()
            inner_max = min(0.002, remaining) if remaining < float('inf') else 0.002
            equities = discard_equity(
                hero_int,
                flop,
                n_samples=discard_samples,
                max_seconds=inner_max,
            )
            total += max(equities) if equities else 0.5
            runs += 1
        return total / max(1, runs)

    if len(hero_int) == 3 and len(board_int) == 2:
        remaining = _time_remaining()
        inner_max = min(0.01, remaining) if remaining < float('inf') else 0.01
        equities = discard_equity(hero_int, board_int, n_samples=discard_samples, max_seconds=inner_max)
        return max(equities) if equities else 0.5

    remaining_board = max(0, 6 - len(board_int))
    if remaining_board + 2 > len(deck):
        return 0.5

    wins, losses, ties, runs = _mc_equity_serial(
        hero_int,
        board_int,
        deck,
        remaining_board,
        samples,
        max_seconds,
    )
    total = wins + losses + ties
    if total == 0:
        return 0.5
    return (wins + 0.5 * ties) / total


def _mc_equity_serial(
    hero_int: List[int],
    board_int: List[int],
    deck: List[int],
    remaining_board: int,
    samples: int,
    max_seconds: float,
) -> Tuple[int, int, int, int]:
    board_base = list(board_int)
    wins = losses = ties = 0
    runs = 0
    start_time = time.perf_counter()

    # Fast path using pkrbot.Deck and direct evaluation
    if pkrbot is not None:
        pk_deck = pkrbot.Deck()
        pk_deck.cards = [_PKRBOT_CARD_CACHE[c] for c in deck]
        hero_cards = [_PKRBOT_CARD_CACHE[c] for c in hero_int]
        board_cards = [_PKRBOT_CARD_CACHE[c] for c in board_base]
        pk_eval = pkrbot.evaluate
        draw_count = 2 + remaining_board

        for i in range(samples):
            # Check timeout every 20 iterations to reduce overhead
            if i > 0 and i % 20 == 0:
                if max_seconds > 0 and time.perf_counter() - start_time >= max_seconds:
                    break
            if i % 100 == 0 and _should_abort():
                break
            drawn = pk_deck.sample(draw_count)
            opp_hand = drawn[:2]
            board_fill = drawn[2:]
            final_board = board_cards + board_fill
            hero_rank = pk_eval(final_board + hero_cards)
            villain_rank = pk_eval(final_board + opp_hand)
            if hero_rank > villain_rank:
                wins += 1
            elif hero_rank < villain_rank:
                losses += 1
            else:
                ties += 1
            runs += 1
        return wins, losses, ties, runs

    # Fallback for non-pkrbot
    rng = random.Random()
    sample = rng.sample
    eval_best = evaluate_best
    compare = compare_evals
    for _ in range(samples):
        # Check both local max_seconds and global action timeout
        if max_seconds > 0 and time.perf_counter() - start_time >= max_seconds:
            break
        if _should_abort():
            break
        drawn = sample(deck, 2 + remaining_board)
        opp_hand = drawn[:2]
        board_fill = drawn[2:]
        final_board = board_base + board_fill
        hero_rank = eval_best(final_board, hero_int)
        villain_rank = eval_best(final_board, opp_hand)
        result = compare(hero_rank, villain_rank)
        if result > 0:
            wins += 1
        elif result < 0:
            losses += 1
        else:
            ties += 1
        runs += 1
    return wins, losses, ties, runs




def estimate_showdown_equity(my_hand, opponent_range, community_cards):
    """
    Estimate hero equity at showdown vs opponent range.

    Inputs:
    - my_hand: hero 2-card hand
    - opponent_range: distribution over opponent 2-card hands
    - community_cards: current board (may be partial)
    """
    raise NotImplementedError


def equity_given_future_discard(my_hand, opponent_range, board, discard_policy):
    """
    Evaluate pre-discard equity integrating over future discard behavior.
    """
    raise NotImplementedError


def expected_discarded_card_value(hero_hand, opponent_range, board, action_context):
    """
    Return EV by candidate discard, accounting for retention, externality, and info effects.
    """
    raise NotImplementedError
