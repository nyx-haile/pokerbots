'''
Simple example pokerbot, written in Python.
'''
from skeleton.actions import FoldAction, CallAction, CheckAction, RaiseAction, DiscardAction
from skeleton.states import GameState, TerminalState, RoundState
from skeleton.states import NUM_ROUNDS, STARTING_STACK, BIG_BLIND, SMALL_BLIND
from skeleton.bot import Bot
from skeleton.runner import parse_args, run_bot

import os
import random
import sys
import traceback

_THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "NEUROPOKER_MC_WORKERS": "1",
}
for key, value in _THREAD_ENV.items():
    os.environ.setdefault(key, value)

import strategy
import stats
from strategy import ActorView

class Player(Bot):
    '''
    A pokerbot.
    '''

    def __init__(self):
        '''
        Called when a new game starts. Called exactly once.

        Arguments:
        Nothing.

        Returns:
        Nothing.
        '''
        self.hero = ActorView()
        self.villain = ActorView()
        self._logged_start = False
        self._last_board_len = 0
        self._pending_hero_discard = None
        self._last_bet_size = None
        self._last_bet_pot = None
        self._last_bet_street = None
        self._last_aggressor = False
        self._last_villain_pip = 0
        self._last_street_seen = 0
        self._last_hero_pip = 0
        self._log(f"init python={sys.version.split()[0]}")
        self._log(f"cwd={os.getcwd()}")
        self._log(f"executable={sys.executable}")
        self._log(f"sys.path[0:3]={sys.path[:3]}")
        self._log(f"env.PYTHONPATH={os.environ.get('PYTHONPATH')}")
        self._log(f"env.NEUROPOKER_EVAL_BACKEND={os.environ.get('NEUROPOKER_EVAL_BACKEND')}")
        try:
            import pokerstove  # noqa: F401
            self._log("pokerstove=ok")
        except Exception as exc:
            self._log(f"pokerstove=missing ({exc})")
        try:
            import stats  # noqa: F401
            self._log("stats=ok")
        except Exception as exc:
            self._log(f"stats=missing ({exc})")

    def _log(self, message: str) -> None:
        print(f"[bot] {message}", flush=True)

    def handle_new_round(self, game_state, round_state, active):
        '''
        Called when a new round starts. Called NUM_ROUNDS times.

        Arguments:
        game_state: the GameState object.
        round_state: the RoundState object.
        active: your player's index.

        Returns:
        Nothing.
        '''
        self.hero.bankroll = game_state.bankroll
        # the total number of chips you've gained
        # or lost from the beginning of the game
        # to the start of this round

        self.game_clock = game_state.game_clock
        # the total number of seconds your
        # bot has left to play this game

        self.round_num = game_state.round_num  # the round number from 1 to NUM_ROUNDS

        self.hero.hand = round_state.hands[active]  # your cards

        self.hero.blind = bool(active)  # True if you are the big blind
        self.villain.blind = not self.hero.blind
        self._last_board_len = len(round_state.board)
        self._pending_hero_discard = None
        self._last_bet_size = None
        self._last_bet_pot = None
        self._last_bet_street = None
        self._last_aggressor = False
        self._last_villain_pip = round_state.pips[1 - active]
        self._last_street_seen = round_state.street
        self._last_hero_pip = round_state.pips[active]
        if self.round_num % 100 == 1:
            self._log(f"round={self.round_num} bankroll={self.hero.bankroll} clock={self.game_clock:.2f}")

    def handle_round_over(self, game_state, terminal_state, active):
        '''
        Called when a round ends. Called NUM_ROUNDS times.

        Arguments:
        game_state: the GameState object.
        terminal_state: the TerminalState object.
        active: your player's index.

        Returns:
        Nothing.
        '''
        self.hero.delta = terminal_state.deltas[active]  # your bankroll change from this round

        self.previous_state = terminal_state.previous_state  # RoundState before payoffs

        self.street = self.previous_state.street  # 0,2,3,4,5,6 representing when this round ended

        self.hero.hand = self.previous_state.hands[active]  # your cards

        self.villain.hand = self.previous_state.hands[1-active] # opponent's cards or [] if not revealed

        strategy.update_info_penalty_from_round(self)
        strategy.decay_discard_model()
        strategy.decay_bet_model()
        strategy.decay_range_model()
        if self._last_aggressor and self._last_bet_size is not None:
            villain_revealed = bool(self.villain.hand)
            if self.hero.delta > 0 and not villain_revealed:
                strategy.record_opponent_bet_response(
                    self._last_bet_street or 0,
                    self._last_bet_size,
                    self._last_bet_pot or 1,
                    True,
                )
            elif villain_revealed:
                strategy.record_opponent_bet_response(
                    self._last_bet_street or 0,
                    self._last_bet_size,
                    self._last_bet_pot or 1,
                    False,
                )
        if self.villain.hand:
            if self.hero.delta > 0:
                strategy.record_opponent_showdown(False)
            elif self.hero.delta < 0:
                strategy.record_opponent_showdown(True)
        if self.round_num % 100 == 1:
            self._log(f"round_over={self.round_num} delta={self.hero.delta}")

    def get_action(self, game_state, round_state, active):
        '''
        Where the magic happens - your code should implement this function.
        Called any time the engine needs an action from your bot.

        Arguments:
        game_state: the GameState object.
        round_state: the RoundState object.
        active: your player's index.

        Returns:
        Your action.
        '''
        self.hero.legal_actions = round_state.legal_actions()
        self.villain.legal_actions = self.hero.legal_actions
        # the actions you are allowed to take

        self.street = round_state.street
        # 0, 3, 4, or 5 representing pre-flop,
        # flop, turn, or river respectively


        hero_index = active
        villain_index = 1 - active
        self.hero.hand = round_state.hands[hero_index]  # your cards
        self.villain.hand = []
        self.community = round_state.board  # the board cards
        if len(self.community) > self._last_board_len:
            new_card = self.community[-1]
            if self._pending_hero_discard == new_card:
                self._pending_hero_discard = None
            else:
                strategy.record_opponent_discard(new_card)
            self._last_board_len = len(self.community)

        # the number of chips you have contributed to the pot this round of betting
        self.hero.pip = round_state.pips[hero_index]

        # the number of chips your opponent has contributed to the pot this round of betting
        self.villain.pip = round_state.pips[villain_index]

        # the number of chips you have remaining
        self.hero.stack = round_state.stacks[hero_index]

        # the number of chips your opponent has remaining
        self.villain.stack = round_state.stacks[villain_index]

        self.hero.continue_cost = self.villain.pip - self.hero.pip
        # the number of chips needed to stay in the pot

        # the number of chips you have contributed to the pot
        self.hero.contribution = STARTING_STACK - self.hero.stack

        # the number of chips your opponent has contributed to the pot
        self.villain.contribution = STARTING_STACK - self.villain.stack

        # opponent-facing public values
        self.villain.continue_cost = self.hero.pip - self.villain.pip
        self.hero.pot_total = self.hero.contribution + self.villain.contribution
        self.villain.pot_total = self.hero.pot_total

        if RaiseAction in self.hero.legal_actions:
            self.hero.raise_bounds = round_state.raise_bounds()
            self.villain.raise_bounds = self.hero.raise_bounds

        if self.hero.continue_cost > 0:
            self._last_aggressor = False

        if self._last_street_seen == self.street:
            if self.villain.pip > self._last_villain_pip:
                strategy.record_opponent_raise(self.street)
            elif self.hero.continue_cost > 0 and self.villain.pip == self._last_villain_pip and self.hero.pip > self._last_hero_pip:
                strategy.record_opponent_call(self.street)
            elif self.villain.pip == self._last_villain_pip and self.hero.continue_cost == 0:
                strategy.record_opponent_action(self.street)
        self._last_villain_pip = self.villain.pip
        self._last_hero_pip = self.hero.pip
        self._last_street_seen = self.street

        # Only use DiscardAction if it's in legal_actions (which already checks street)
        # legal_actions() returns DiscardAction only when street is 2 or 3

        if not self._logged_start:
            self._log(f"first_action street={self.street} legal={self.hero.legal_actions}")
            self._logged_start = True
        try:
            action = strategy.play(self)
        except Exception:
            self._log("strategy error:\n" + traceback.format_exc())
            if CheckAction in self.hero.legal_actions:
                action = CheckAction()
            elif CallAction in self.hero.legal_actions:
                action = CallAction()
            else:
                action = FoldAction()
        if isinstance(action, DiscardAction):
            try:
                self._pending_hero_discard = self.hero.hand[action.card]
            except Exception:
                self._pending_hero_discard = None
        if self.street <= 0:
            self._log(
                "preflop decision action={0} equity={1:.3f} pot_odds={2:.3f} "
                "continue_cost={3} pot_total={4} legal={5}".format(
                    getattr(action, "__class__", type(action)).__name__,
                    stats.preflop_strength(self.hero.hand),
                    strategy.pot_odds_to_call(self.hero.continue_cost, max(1, self.hero.pot_total)),
                    self.hero.continue_cost,
                    self.hero.pot_total,
                    self.hero.legal_actions,
                )
            )
        if isinstance(action, RaiseAction):
            self._last_bet_size = action.amount
            self._last_bet_pot = self.hero.pot_total
            self._last_bet_street = self.street
            self._last_aggressor = True
        return action
        #export computation to strategy engine

        """
        if DiscardAction in legal_actions:
            # use stats discard equity calc
            board_int = [convert(card) for card in board_cards]
            hole_int = [convert(card) for card in my_cards]
            hand_vals = discard_equity(hole_int, board_int)
            best_i = max(range(len(hand_vals)), key=hand_vals.__getitem__)
            return DiscardAction(best_i)

        if RaiseAction in legal_actions:
            # the smallest and largest numbers of chips for a legal bet/raise
            min_raise, max_raise = round_state.raise_bounds()
            min_cost = min_raise - my_pip  # the cost of a minimum bet/raise
            max_cost = max_raise - my_pip  # the cost of a maximum bet/raise

            if random.random() < 0.5:
                return RaiseAction(min_raise)
        if CheckAction in legal_actions:  # check-call
            return CheckAction()

        if random.random() < 0.25:
            return FoldAction()

        return CallAction()
        """

if __name__ == '__main__':
    run_bot(Player(), parse_args())
