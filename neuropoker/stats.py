#here we are again
import os
import random
import time
import multiprocessing

from deuces import Card, Deck, Evaluator
from itertools import combinations
from functools import lru_cache
try:
    from pokerstove import CardSet as cs
except ImportError:
    cs = None
from typing import Sequence

_BACKEND = os.environ.get("NEUROPOKER_EVAL_BACKEND", "auto").lower()
_USE_POKERSTOVE = _BACKEND in ("auto", "pokerstove") and cs is not None

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


_DEFAULT_DISCARD_SECONDS = _env_float("NEUROPOKER_DISCARD_MAX_SECONDS", 0.02)
_DEFAULT_DISCARD_SAMPLES = _env_int("NEUROPOKER_DISCARD_SAMPLES", 100)
_DEFAULT_EQUITY_SAMPLES = _env_int("NEUROPOKER_EQUITY_SAMPLES", 120)
_DEFAULT_EQUITY_SECONDS = _env_float("NEUROPOKER_EQUITY_MAX_SECONDS", 0.03)
_DEFAULT_MC_WORKERS = _env_int("NEUROPOKER_MC_WORKERS", 2)
_DEFAULT_MC_PARALLEL_MIN = _env_int("NEUROPOKER_MC_PARALLEL_MIN", 200)

class DeckEmptyException(Exception):
    pass

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
    str_cards = _ensure_str_cards(cards)
    if len(cards) > 7:
        return eval_best_8(str_cards)
    return cs("".join(str_cards)).evaluateHigh().code()

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

@lru_cache(maxsize=400_000)
def _evaluate_best_cached(cards_tuple: tuple[int, ...]) -> int:
    cards = list(cards_tuple)
    if _USE_POKERSTOVE:
        return _pokerstove_best_eval(_ensure_str_cards(cards))
    return _deuces_best_eval(cards)


def evaluate_best(board, hand):
    cards = _ensure_int_cards(list(board) + list(hand))
    return _evaluate_best_cached(tuple(cards))

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


def discard_equity(
    hole: list,
    board: list,
    n_samples: int | None = None,
    max_seconds: float | None = None,
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
    stats = {discard: {"wins": 0, "losses": 0, "ties": 0, "samples": 0} for discard in hole}
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


def estimate_equity(
    hero_hand: Sequence[str | int],
    board_cards: Sequence[str | int],
    samples: int | None = None,
    max_seconds: float | None = None,
    discard_samples: int | None = None,
) -> float:
    if samples is None:
        samples = _DEFAULT_EQUITY_SAMPLES
    if max_seconds is None:
        max_seconds = _DEFAULT_EQUITY_SECONDS
    if discard_samples is None:
        discard_samples = max(5, _DEFAULT_DISCARD_SAMPLES // 5)

    hero_int = _ensure_int_cards(hero_hand)
    board_int = _ensure_int_cards(board_cards)
    return _estimate_equity_cached(
        tuple(hero_int),
        tuple(board_int),
        int(samples),
        float(max_seconds),
        int(discard_samples),
    )


def _equity_seed(hero_tuple, board_tuple, samples, max_seconds, discard_samples) -> int:
    return hash((hero_tuple, board_tuple, samples, max_seconds, discard_samples)) & 0xFFFFFFFF


@lru_cache(maxsize=100_000)
def _estimate_equity_cached(
    hero_tuple: tuple[int, ...],
    board_tuple: tuple[int, ...],
    samples: int,
    max_seconds: float,
    discard_samples: int,
) -> float:
    rng = random.Random(_equity_seed(hero_tuple, board_tuple, samples, max_seconds, discard_samples))
    hero_int = list(hero_tuple)
    board_int = list(board_tuple)
    start_time = time.perf_counter()
    known = hero_int + list(board_int)
    deck = [card for card in Deck().cards if card not in known]

    if len(hero_int) == 3 and len(board_int) < 2:
        total = 0.0
        runs = 0
        for _ in range(samples):
            if max_seconds > 0 and time.perf_counter() - start_time >= max_seconds:
                break
            flop = rng.sample(deck, 2)
            equities = discard_equity(
                hero_int,
                flop,
                n_samples=discard_samples,
                max_seconds=0.0,
            )
            total += max(equities) if equities else 0.5
            runs += 1
        return total / max(1, runs)

    if len(hero_int) == 3 and len(board_int) == 2:
        equities = discard_equity(hero_int, board_int)
        return max(equities) if equities else 0.5

    remaining_board = max(0, 6 - len(board_int))
    if remaining_board + 2 > len(deck):
        return 0.5

    use_parallel = _DEFAULT_MC_WORKERS > 1 and samples >= _DEFAULT_MC_PARALLEL_MIN
    if use_parallel:
        wins, losses, ties, runs = _parallel_mc_equity(
            hero_int,
            board_int,
            deck,
            remaining_board,
            samples,
            max_seconds,
        )
    else:
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
    hero_int: list[int],
    board_int: list[int],
    deck: list[int],
    remaining_board: int,
    samples: int,
    max_seconds: float,
) -> tuple[int, int, int, int]:
    rng = random.Random()
    wins = losses = ties = 0
    runs = 0
    start_time = time.perf_counter()
    for _ in range(samples):
        if max_seconds > 0 and time.perf_counter() - start_time >= max_seconds:
            break
        sample = rng.sample(deck, 2 + remaining_board)
        opp_hand = sample[:2]
        board_fill = sample[2:]
        final_board = list(board_int)
        final_board.extend(board_fill)
        hero_rank = evaluate_best(final_board, hero_int)
        villain_rank = evaluate_best(final_board, list(opp_hand))
        result = compare_evals(hero_rank, villain_rank)
        if result > 0:
            wins += 1
        elif result < 0:
            losses += 1
        else:
            ties += 1
        runs += 1
    return wins, losses, ties, runs


def _mc_equity_worker(args) -> tuple[int, int, int, int]:
    hero_int, board_int, deck, remaining_board, samples, seed = args
    rng = random.Random(seed)
    wins = losses = ties = 0
    for _ in range(samples):
        sample = rng.sample(deck, 2 + remaining_board)
        opp_hand = sample[:2]
        board_fill = sample[2:]
        final_board = list(board_int)
        final_board.extend(board_fill)
        hero_rank = evaluate_best(final_board, hero_int)
        villain_rank = evaluate_best(final_board, list(opp_hand))
        result = compare_evals(hero_rank, villain_rank)
        if result > 0:
            wins += 1
        elif result < 0:
            losses += 1
        else:
            ties += 1
    return wins, losses, ties, samples


def _parallel_mc_equity(
    hero_int: list[int],
    board_int: list[int],
    deck: list[int],
    remaining_board: int,
    samples: int,
    max_seconds: float,
) -> tuple[int, int, int, int]:
    workers = max(1, min(_DEFAULT_MC_WORKERS, (multiprocessing.cpu_count() or 2)))
    if workers <= 1:
        return _mc_equity_serial(hero_int, board_int, deck, remaining_board, samples, max_seconds)

    chunk = max(1, samples // workers)
    counts = [chunk] * workers
    counts[0] += samples - sum(counts)
    seeds = [random.randrange(1 << 30) for _ in range(workers)]

    start_time = time.perf_counter()
    wins = losses = ties = runs = 0
    args = [
        (hero_int, board_int, deck, remaining_board, counts[i], seeds[i])
        for i in range(workers)
        if counts[i] > 0
    ]
    ctx = multiprocessing.get_context("spawn")
    pool = ctx.Pool(processes=len(args))
    terminated = False
    try:
        for w, l, t, r in pool.imap_unordered(_mc_equity_worker, args):
            wins += w
            losses += l
            ties += t
            runs += r
            if max_seconds > 0 and time.perf_counter() - start_time >= max_seconds:
                pool.terminate()
                terminated = True
                break
    finally:
        if not terminated:
            pool.close()
        pool.join()
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
