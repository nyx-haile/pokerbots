'''
Simple example pokerbot, written in Python.
'''
from skeleton.actions import FoldAction, CallAction, CheckAction, RaiseAction, DiscardAction
from skeleton.states import GameState, TerminalState, RoundState
from skeleton.states import NUM_ROUNDS, STARTING_STACK, BIG_BLIND, SMALL_BLIND
from skeleton.bot import Bot
from skeleton.runner import parse_args, run_bot
from policies.desperate import DesperatePolicy

import os
import random
import sys

_SINGLE_CORE_ENV = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}
for key, value in _SINGLE_CORE_ENV.items():
    os.environ.setdefault(key, value)

_LOG_ROUND_SUMMARY = os.environ.get("NEUROPOKER_LOG_ROUNDS", "0") == "1"
_LOG_ROUND_EVERY = int(os.environ.get("NEUROPOKER_LOG_ROUND_EVERY", "200") or "200")
_LOG_PREFLOP_DECISIONS = os.environ.get("NEUROPOKER_LOG_PREFLOP", "0") == "1"

import strategy
import stats
import lumberjack
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
        self.hero.enable_desperate = True
        self._logged_start = False
        self._last_board_len = 0
        self._pending_hero_discard = None
        self._last_bet_size = None
        self._last_bet_delta = None
        self._last_bet_pot = None
        self._last_bet_street = None
        self._last_aggressor = False
        self._last_hero_call_street = None
        self._hero_bet_buckets = []
        self._hero_line_actions = {}
        self._river_value_bet = False
        self._leak_stats = {
            "end_by_street": {},
            "hero_calls": {},
            "hero_fold_vs_bet": {},
            "villain_bets": {},
            "villain_checks": {},
            "hero_fold_to_raise": {},
            "river_loss_after_bet": {"count": 0, "delta": 0},
        }
        lumberjack.log(f"init python={sys.version.split()[0]}")
        lumberjack.log(f"cwd={os.getcwd()}")
        lumberjack.log(f"executable={sys.executable}")
        lumberjack.log(f"sys.path[0:3]={sys.path[:3]}")
        lumberjack.log(f"env.PYTHONPATH={os.environ.get('PYTHONPATH')}")
        lumberjack.log(f"env.NEUROPOKER_EVAL_BACKEND={os.environ.get('NEUROPOKER_EVAL_BACKEND')}")
        import stats  # noqa: F401
        lumberjack.log("stats=ok")

    def _pot_total_from_state(self, state: RoundState) -> int:
        return (STARTING_STACK - state.stacks[0]) + (STARTING_STACK - state.stacks[1])

    def _bump_bucket(self, stats, street: int, bucket: str, delta: int = 1) -> None:
        street_bucket = stats.setdefault(int(street), {})
        street_bucket[bucket] = street_bucket.get(bucket, 0) + delta

    def _record_villain_action(self, street: int, action: str, size: int, pot_total: int) -> None:
        if action == "raise":
            bucket = strategy._bet_size_bucket(size, pot_total)
            self._bump_bucket(self._leak_stats["villain_bets"], street, bucket)
        elif action == "check":
            self._bump_bucket(self._leak_stats["villain_checks"], street, "check")

    def _record_opponent_action_from_state(self, round_state, hero_index, villain_index) -> None:
        prev = round_state.previous_state
        if prev is None:
            return
        prev_actor = prev.button % 2
        if prev_actor != villain_index:
            return
        # Ignore discard-only streets and discard actions.
        if prev.street in (2, 3):
            return
        prev_continue_cost = prev.pips[1 - prev_actor] - prev.pips[prev_actor]

        # Special-case blind posting (SB CallAction on button 0).
        if (
            prev.street == 0
            and prev.button == 0
            and prev_continue_cost == 0
            and round_state.pips == [BIG_BLIND, BIG_BLIND]
        ):
            return

        # Street advanced: last action must have been a check or call.
        if prev.street != round_state.street:
            if prev_continue_cost > 0:
                strategy.record_opponent_call(prev.street)
                self._record_villain_action(prev.street, "call", prev_continue_cost, self._pot_total_from_state(prev))
            else:
                strategy.record_opponent_check(prev.street)
                self._record_villain_action(prev.street, "check", 0, self._pot_total_from_state(prev))
            return
        # Same street: infer by pip delta.
        delta = round_state.pips[prev_actor] - prev.pips[prev_actor]
        if delta <= 0:
            if prev_continue_cost == 0:
                strategy.record_opponent_check(prev.street)
                self._record_villain_action(prev.street, "check", 0, self._pot_total_from_state(prev))
            return
        if prev_continue_cost > 0 and delta == prev_continue_cost:
            strategy.record_opponent_call(prev.street)
            self._record_villain_action(prev.street, "call", delta, self._pot_total_from_state(prev))
        else:
            strategy.record_opponent_raise(prev.street)
            strategy.record_overbet_observation(prev.street, delta, self._pot_total_from_state(prev))
            strategy.record_opponent_bet_size(prev.street, delta, self._pot_total_from_state(prev))
            self._record_villain_action(prev.street, "raise", delta, self._pot_total_from_state(prev))

    def _hero_line_key(self) -> str:
        codes = []
        for street in (0, 4, 5, 6):
            code = self._hero_line_actions.get(street)
            if code:
                codes.append(code)
        return "-".join(codes) if codes else "none"

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
        self._last_bet_delta = None
        self._last_bet_pot = None
        self._last_bet_street = None
        self._last_aggressor = False
        self._hero_bet_buckets = []
        self._hero_line_actions = {}
        self._river_value_bet = False
        strategy.begin_round(self)
        self.hero.last_discard_ev = None
        self.hero.raise_plan_target = None
        self.hero.raise_plan_street = None
        if _LOG_ROUND_SUMMARY and self.round_num % _LOG_ROUND_EVERY == 1:
            lumberjack.log(f"round={self.round_num} bankroll={self.hero.bankroll} clock={self.game_clock:.2f}")

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
        strategy.update_policy_from_round(self)
        strategy.decay_discard_model()
        strategy.decay_bet_model()
        strategy.decay_range_model()
        self.hero.discard_bluff = False
        if self._last_aggressor and self._last_bet_size is not None:
            villain_revealed = bool(self.villain.hand)
            if self.hero.delta > 0 and not villain_revealed:
                strategy.record_opponent_bet_response(
                    self._last_bet_street or 0,
                    self._last_bet_size,
                    self._last_bet_pot or 1,
                    True,
                )
                strategy.record_fold_equity_observation(
                    self._last_bet_street or 0,
                    self._last_bet_delta if self._last_bet_delta is not None else self._last_bet_size,
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
                strategy.record_fold_equity_observation(
                    self._last_bet_street or 0,
                    self._last_bet_delta if self._last_bet_delta is not None else self._last_bet_size,
                    self._last_bet_pot or 1,
                    False,
                )
        if self.hero.delta > 0 and not self.villain.hand and self._last_bet_street is not None:
            strategy.record_opponent_fold(self._last_bet_street)
        if self.villain.hand:
            if self.hero.delta > 0:
                strategy.record_opponent_showdown(False)
            elif self.hero.delta < 0:
                strategy.record_opponent_showdown(True)
            if len(self.villain.hand) == 2:
                strategy.record_inferred_range(stats.two_card_strength(self.villain.hand))
            strength, confidence = strategy.opponent_range_hint()
            strategy.record_range_hint_accuracy(strength, confidence, int(self.hero.delta))
        if getattr(self.hero, "last_discard_ev", None):
            lumberjack.record_discard_outcome(self.hero.last_discard_ev, int(self.hero.delta))
            self.hero.last_discard_ev = None
        end_street = int(self.previous_state.street)
        end_bucket = self._leak_stats["end_by_street"].setdefault(
            end_street, {"hands": 0, "delta": 0, "loss": 0}
        )
        end_bucket["hands"] += 1
        end_bucket["delta"] += int(self.hero.delta)
        if self.hero.delta < 0:
            end_bucket["loss"] += 1

        if self._last_hero_call_street in (4, 5):
            call_bucket = self._leak_stats["hero_calls"].setdefault(
                int(self._last_hero_call_street),
                {"calls": 0, "wins": 0, "losses": 0},
            )
            call_bucket["calls"] += 1
            if self.villain.hand:
                if self.hero.delta > 0:
                    call_bucket["wins"] += 1
                elif self.hero.delta < 0:
                    call_bucket["losses"] += 1

        hero_folded = bool(not self.villain.hand and self.hero.delta < 0)
        if hero_folded:
            street = int(self.previous_state.street)
            continue_cost = self.previous_state.pips[1 - active] - self.previous_state.pips[active]
            if continue_cost > 0:
                pot_total = self._pot_total_from_state(self.previous_state)
                bucket = strategy._bet_size_bucket(continue_cost, pot_total)
                self._bump_bucket(self._leak_stats["hero_fold_vs_bet"], street, bucket)
            if self._last_bet_street == street and self._last_bet_size is not None:
                self._bump_bucket(self._leak_stats["hero_fold_to_raise"], street, "fold_to_raise")

        if end_street == 6 and self.hero.delta < 0 and self._last_bet_street in (4, 5):
            self._leak_stats["river_loss_after_bet"]["count"] += 1
            self._leak_stats["river_loss_after_bet"]["delta"] += int(self.hero.delta)
        for street, bucket in self._hero_bet_buckets:
            lumberjack.record_bet_size_outcome(street, bucket, int(self.hero.delta))
        if self._river_value_bet:
            lumberjack.record_river_value_bet(int(self.hero.delta))
        if self.villain.hand:
            line_key = self._hero_line_key()
            lumberjack.record_showdown_line(line_key, int(self.hero.delta))
        if _LOG_ROUND_SUMMARY and self.round_num % _LOG_ROUND_EVERY == 1:
            lumberjack.log(f"round_over={self.round_num} delta={self.hero.delta}")

        if game_state.round_num >= NUM_ROUNDS:
            lumberjack.log(lumberjack.policy_summary(strategy._POLICY_STATS))
            lumberjack.log(lumberjack.leak_summary(self._leak_stats))
            lumberjack.log(lumberjack.policy_action_summary())
            lumberjack.log(lumberjack.aggression_ratio_summary())
            lumberjack.log(lumberjack.equity_vs_pot_odds_summary())
            lumberjack.log(lumberjack.discard_ev_summary())
            lumberjack.log(lumberjack.showdown_line_summary())
            lumberjack.log(lumberjack.bet_size_ev_summary())
            lumberjack.log(lumberjack.river_value_bet_summary())
            lumberjack.log(strategy.opponent_overbet_summary())
            lumberjack.log(strategy.overbet_accuracy_summary())
            lumberjack.log(strategy.fold_equity_accuracy_summary())
            lumberjack.log(strategy.range_hint_accuracy_summary())

        if self.hero.policy_class == DesperatePolicy and self.hero.delta < 0:
            self.hero.enable_desperate = False

        self.hero.policy_class = None
        self.hero.policy_round = None
        self._last_hero_call_street = None

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
        if getattr(self.hero, "raise_plan_street", None) not in (None, self.street):
            self.hero.raise_plan_target = None
            self.hero.raise_plan_street = None


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

        self._record_opponent_action_from_state(round_state, hero_index, villain_index)

        # Only use DiscardAction if it's in legal_actions (which already checks street)
        # legal_actions() returns DiscardAction only when street is 2 or 3

        if not self._logged_start:
            lumberjack.log(f"first_action street={self.street} legal={self.hero.legal_actions}")
            self._logged_start = True
        action = strategy.play(self)

        if isinstance(action, DiscardAction):
            self._pending_hero_discard = self.hero.hand[action.card]
        if isinstance(action, CallAction):
            self._last_hero_call_street = self.street
        lumberjack.record_hero_action(self.street, action.__class__.__name__)
        if self.street in (0, 4, 5, 6) and not isinstance(action, DiscardAction):
            if self.street not in self._hero_line_actions:
                if isinstance(action, RaiseAction):
                    code = "R"
                elif isinstance(action, CallAction):
                    code = "C"
                elif isinstance(action, CheckAction):
                    code = "K"
                elif isinstance(action, FoldAction):
                    code = "F"
                else:
                    code = "?"
                self._hero_line_actions[self.street] = code
        if _LOG_PREFLOP_DECISIONS and self.street <= 0:
            bucket = getattr(self.hero, "preflop_bucket", "n/a")
            roll = getattr(self.hero, "preflop_roll", None)
            lumberjack.log(
                "preflop decision action={0} equity={1:.3f} pot_odds={2:.3f} "
                "continue_cost={3} pot_total={4} legal={5} bucket={6} roll={7}".format(
                    getattr(action, "__class__", type(action)).__name__,
                    stats.preflop_strength(self.hero.hand),
                    strategy.pot_odds_to_call(self.hero.continue_cost, max(1, self.hero.pot_total)),
                    self.hero.continue_cost,
                    self.hero.pot_total,
                    self.hero.legal_actions,
                    bucket,
                    "-" if roll is None else "{0:.3f}".format(roll),
                )
            )
        if isinstance(action, RaiseAction):
            bet_delta = max(0, action.amount - self.hero.pip)
            bucket = strategy._bet_size_bucket(bet_delta, max(1, self.hero.pot_total))
            self._hero_bet_buckets.append((int(self.street), bucket))
            if self.street == 6 and self.hero.continue_cost == 0:
                self._river_value_bet = True
            self._last_bet_size = action.amount
            self._last_bet_delta = bet_delta
            self._last_bet_pot = self.hero.pot_total
            self._last_bet_street = self.street
            self._last_aggressor = True
        return action
        #export computation to strategy engine

if __name__ == '__main__':
    run_bot(Player(), parse_args())
