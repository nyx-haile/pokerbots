# stats.py
from itertools import combinations
from functools import lru_cache
from collections import Counter

RANKS = "23456789TJQKA"
SUITS = "♠♥♦♣"
DECK = {r + s for r in RANKS for s in SUITS}

# ------------------------------
# Hand & Board Utilities
# ------------------------------

def all_possible_hands(exclude_cards: set) -> set:
    """Return all legal 3-card hands excluding already dealt cards."""
    remaining = DECK.symmetric_difference(exclude_cards)
    return set(combinations(remaining, 3))

def all_possible_boards(current_board, remaining_cards: set, n: int) -> set:
    """Return all possible n-card board completions from remaining cards."""
    return set(combinations(remaining_cards, n))

@lru_cache(maxsize=None)
def hand_rank(hand, board):
    """
    Deterministically compute best 5-card hand from 3-card hand + board.
    Returns a numeric rank for comparison (higher is better).
    """
    from itertools import combinations
    best = 0
    full_hand = tuple(sorted(hand + board))
    for combo in combinations(full_hand, 5):
        # Placeholder: insert proper hand evaluator here
        rank = _evaluate_5_card_hand(combo)
        if rank > best:
            best = rank
    return best

def compare_hands(hand1, hand2, board):
    """Compare two hands given the board. Return 1 if hand1 wins, 0 tie, -1 if hand2 wins."""
    r1 = hand_rank(hand1, board)
    r2 = hand_rank(hand2, board)
    if r1 > r2:
        return 1
    elif r1 < r2:
        return -1
    else:
        return 0

# ------------------------------
# Equity Computation
# ------------------------------

def compute_showdown_equity(my_hand, board, opponent_hands=None):
    """
    Compute exact showdown equity against all legal opponent hands if opponent_hands=None,
    or provided list of opponent_hands.
    Returns fraction of wins, ties, losses.
    """
    exclude_cards = set(my_hand + board)
    if opponent_hands is None:
        opponent_hands = all_possible_hands(exclude_cards)
    wins = ties = losses = 0
    remaining_deck = [c for c in DECK if c not in my_hand + board]
    # generate all possible turn/river completions (here 2 cards left to complete board)
    for opp_hand in opponent_hands:
        used = set(my_hand + board + list(opp_hand))
        future_cards = [c for c in DECK if c not in used]
        for turn_river in combinations(future_cards, 2):
            final_board = board + turn_river
            result = compare_hands(my_hand, opp_hand, final_board)
            if result == 1:
                wins += 1
            elif result == 0:
                ties += 1
            else:
                losses += 1
    total = wins + ties + losses
    return wins / total, ties / total, losses / total

def equity_after_discard(my_hand, board):
    """
    Returns a dict mapping discard_index -> equity (fractional wins)
    Computes equity for each possible discard from my_hand.
    """
    equities = {}
    for i, discard_card in enumerate(my_hand):
        new_hand = my_hand[:i] + my_hand[i+1:]
        new_board = board + (discard_card,)
        win_frac, tie_frac, _ = compute_showdown_equity(new_hand, new_board)
        equities[i] = win_frac + 0.5 * tie_frac
    return equities

def expected_discard_value(my_hand, discard_index, board):
    """
    Exact expected equity of discarding a specific card.
    """
    new_hand = my_hand[:discard_index] + my_hand[discard_index+1:]
    new_board = board + (my_hand[discard_index],)
    win_frac, tie_frac, _ = compute_showdown_equity(new_hand, new_board)
    return win_frac + 0.5 * tie_frac

def best_discard(my_hand, board):
    """
    Returns the index of the discard with the highest expected equity.
    """
    equities = equity_after_discard(my_hand, board)
    return max(equities, key=equities.get)

# ------------------------------
# Pot & Bet Utilities
# ------------------------------

def pot_size(pips):
    """Return current pot size from contributions."""
    return sum(pips)

def effective_stack(my_stack, opp_stack):
    """Return effective stack for min/max raise calculations."""
    return min(my_stack, opp_stack)

def call_cost(my_pip, opp_pip):
    """Chips required to call."""
    return opp_pip - my_pip

# ------------------------------
# Enumerative Tools
# ------------------------------

def generate_all_completions(my_hand, board, remaining_cards):
    """Yield all possible turn+river completions."""
    for combo in combinations(remaining_cards, 2):
        yield combo

def hand_strength_distribution(my_hand, board):
    """
    Return fraction of opponent hands beaten by my_hand given board.
    """
    exclude_cards = set(my_hand + board)
    opponent_hands = all_possible_hands(exclude_cards)
    wins = 0
    total = 0
    for opp_hand in opponent_hands:
        result = compare_hands(my_hand, opp_hand, board)
        if result == 1:
            wins += 1
        total += 1
    return wins / total

def card_blocker_effect(my_hand, board, candidate_card):
    """
    Return the number of opponent hands blocked by holding or playing candidate_card.
    Deterministic combinatorial count.
    """
    exclude = set(my_hand + board + [candidate_card])
    opponent_hands = all_possible_hands(set(my_hand + board))
    blocked = 0
    for opp_hand in opponent_hands:
        if candidate_card in opp_hand:
            blocked += 1
    return blocked

# ------------------------------
# Placeholder Hand Evaluator
# ------------------------------

def _evaluate_5_card_hand(cards):
    """
    Placeholder: replace with a real poker hand evaluator.
    Returns numeric rank for comparison.
    """
    # Simple stub: rank by sum of rank indices
    rank_indices = {r:i for i,r in enumerate(RANKS)}
    return sum(rank_indices[c[0]] for c in cards)

