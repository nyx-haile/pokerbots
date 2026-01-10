# Computes statistics for gamestates
# Volatility of board, hand win probability, opponent hand range, etc.
# Meant to be used as an oracle for strategy.py

import math
import numpy as np
from itertools import combinations
from functools import lru_cache
from collections import Counter


RANKS = "23456789TJQKA"
SUITS = "♠♥♦♣"
DECK = [r + s for r in RANKS for s in SUITS]


"""
Terminology:
    Player/Hero         Our bot
    Villain             Enemy bot
    Hole                Hero's cards
    Hand                Villain's cards (usually a distribution)
    Board/community     Communal Cards
    Rank                Percentile ranking of the hero's hand + board (if exists)
    Range               Rank distribution of the villain's hand + board
    Equity              Expected payout (for the hero)
    Risk                Expected payout (for the villain)

Architecture

    All functions should have a villain flag that allows
    us to calculate them from the revealed information.

    all functions should cache answers

    all functions should calculate only explicit statistics
    and any learning or models should happen in a separate
    script.
"""


def all_hands(hole: set) -> set:
    """
    All possible opponent hands, excluding own cards.
    """
    other = DECK.symmetric_difference(hole)
    return set(combinations(other, 3))

def all_boards(hole: set, board: set) -> set:
    """
    All possible completions of the current board.
    """
    other = DECK.symmetric_difference(hole+board)

@lru_cache(maxsize=None)
#may have to change this later so
#the 1GB ram limit is not broken.
def hand_rank():
    pass

def compare_hands():
    pass

def rough_rank(hole: set, board: set) -> set:
    """
    Monte Carlo simulate possible poker hands.
    """

class Unused()
"""
    yeah it's an unused class. IT DOESN"T WORK!!"
"""


    def initial_range(self):
        #return all possible opponent hands, given own hand.
        #should throw an error if own_hand is not set.
        pass

    def preflop_range(self):
        #reference action sizes to adjust range estimate
        pass

    def discard_distribution(self, pid, hole, community):
        """
        #reference action context to infer a PD over which card

        #given  villain distribution (or exact, for training/testing?)
        #       public board
        #       positional context
        #       betting history
        #       betting model
        #       discard model

        #outputs: distrbution over cards/types being discarded, to integrate over

        #   derived from
        #   intrinsic utility ~= norm(1/eq(H\c | Board)) for a card c and hand H.
        #   board externality ~= E[eq_opp| board + c ] - E[eq opp | board]
        #   information strategy
        #   basically learn the discard pattern and calculate from range.

        #  present each of those scores as output

        #poisoning matters more for the dealer than OOP.
        #this function must condition on those positions.
        """

        pass

    def discard_range(self):
        #referene action sizes and discard to further adjust range estimate

        """
        assuming you drop a card c:

        MUST ACCOUNT FOR POSITION

        returns:
            self-retention loss: ev_keep(c) - ev_drop(c)
            board externality: ev_risk(board + c)-ev_risk(board)

        """

        pass

    def ev_discard(self, srange, board):
        # calculate game state from player discards
        pass

    def showdown_eq(self):
        #estimate payout post discard
        pass

    def discard_eq(self):
        #estimate payout pre discard
        pass

    def bet_size_likelihood(self, pid):
        #save hand_strength_bucket and reference
        pass

    def fold_eq(pid, bet_size, street):
        """
        this should work for both sides.
        if we have leaked little enough information to the
        opponent that folding is attractive at a certain bet
        size, we can force a win even with a low ev hand.


        fold eq is
            let f be the revealed fold eq of the villain
            let P be the pot size
            let B be the bet size
            let R be the risk
            let Eq be the equity
            (1-f)*[Eq*(P+B)-(R)*B ]
        """
        #inform strategy  from aggression
        pass

    def update_model(pid, outcome):
        pass

    def decay_model(pid, factor):
        pass

    def hand_strength_percentile(self):
        pass

    def discard_regret(self, card):
        pass

    def raise_ev(self, amount):
        pass

    def call_ev(self, amount):
        pass

    def update_hand(self, hand_id, hole, community, street):
        pass

    def update_action(self, player_id, action, amount, street):
        pass

    def finalise_hand(self, results):
        pass #logging for now, optimisation later

    def range(self, player_id, street, drop='optional'):
        #calculate possible outcomes from current game state
        pass

    def equity(self, player_id, street, drop='optional'):
        #calculate EV of current ranges
        pass

    def action_freq(self, pid, street, action_type):
        pass

    def bet_sizing(self, pid, street):
        pass

    def pot_size(self):
        pass

    def effective_stack(self, pid):
        pass

    def pot_odds(self, call_amount):
        pass

    def record_showdown(self, pid, hand, action_history):
        pass

    def decay_old_stats(self):
        pass


