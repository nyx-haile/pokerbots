from skeleton.actions import CallAction, CheckAction, DiscardAction, FoldAction, RaiseAction

import lumberjack
import random
import stats
from strategy import policy_api as core


class TightPolicy:
    """Tight policy: fold early, raise aggressively only when strong."""

    @staticmethod
    def play(player):
        legal_actions = set(player.hero.legal_actions)
        hero_hand = list(player.hero.hand)
        board_cards = list(player.community)

        def _record(action, equity_value=None, pot_odds_value=None):
            lumberjack.record_policy_action("TightPolicy", player.street, action.__class__.__name__)
            if equity_value is not None and pot_odds_value is not None:
                lumberjack.record_equity_vs_pot_odds(
                    "TightPolicy",
                    player.street,
                    equity_value,
                    pot_odds_value,
                )
            return action

        if DiscardAction in legal_actions:
            opponent_discard = None
            if not player.hero.blind and len(board_cards) >= 3:
                opponent_discard = board_cards[-1]
            discard_samples, discard_seconds = core._discard_budget(player)
            equities = stats.discard_equity(
                hero_hand,
                board_cards,
                n_samples=discard_samples,
                max_seconds=discard_seconds,
            )
            best_i = core._select_discard_asymmetric(
                hero_hand,
                equities,
                discard_visible=player.hero.blind,
                opponent_discard=opponent_discard,
                board_cards=board_cards,
            )
            anti_rate = core._anti_exploit_rate(player)
            best_i = core._anti_exploit_discard_index(equities, best_i, anti_rate)
            player.hero.last_discard_ev = lumberjack.record_discard_decision(
                "TightPolicy",
                player.street,
                equities,
                best_i,
            )
            player.hero.last_discard = hero_hand[best_i]
            player.hero.last_discard_visible = player.hero.blind
            player.hero.discard_bluff = False
            return _record(DiscardAction(best_i))

        pot_total = max(1, player.hero.pot_total)
        player.hero.discard_bluff = False
        if player.street <= 0:
            equity = stats.preflop_strength(hero_hand)
        else:
            samples, max_seconds, discard_samples = core._equity_budget(player)
            samples, max_seconds, discard_samples = core._adjust_budget_for_turn_raise(
                player,
                pot_total,
                samples,
                max_seconds,
                discard_samples,
            )
            samples, max_seconds, discard_samples = core._adjust_budget_for_river_raise(
                player,
                pot_total,
                samples,
                max_seconds,
                discard_samples,
            )
            force_full_equity = core._is_large_raise(player, pot_total)
            if core.is_river(player.street) and (player.hero.continue_cost > 0 or RaiseAction in legal_actions):
                force_full_equity = True
            quick_equity = core._range_conditioned_equity(
                player,
                hero_hand,
                board_cards,
                samples=max(20, samples // 4),
                max_seconds=min(0.006, max_seconds * 0.2),
            )
            equity = quick_equity

        pot_odds = core.pot_odds_to_call(player.hero.continue_cost, pot_total)
        base_raise_margin = core._raise_margin_by_street(player.street)
        base_call_margin = core._call_margin_by_street(player.street)
        discard_bias = core._opponent_discard_bias(player.street)
        range_bias = core._opponent_range_bias(player.street)
        texture_raise, texture_call, raise_cap_mult = core._board_texture_adjustments(board_cards)
        variant_id = core.active_variant_id()
        equity_bias, raise_variant, call_variant = core._variant_adjustments(
            variant_id,
            player,
            hero_hand,
            board_cards,
        )
        if player.street <= 0 and equity_bias:
            equity = max(0.0, min(1.0, equity + equity_bias))
        opp_raise_adj, opp_call_adj, opp_river_raise_adj, opp_river_floor_adj = core._opponent_threshold_adjustments(player)
        raise_mult = 1.0
        call_mult = 1.0
        for adj in (discard_bias, range_bias, opp_raise_adj, raise_variant, texture_raise):
            raise_mult += adj
        for adj in (discard_bias, range_bias, opp_call_adj, call_variant, texture_call):
            call_mult += adj

        # Tighten play when approaching win-lock (pot-aware)
        lead_adj = core._lead_protection_adjustment(player, pot_total)
        raise_mult += lead_adj
        call_mult -= lead_adj  # Harder to call when protecting lead
        traj_raise_adj, traj_call_adj = core._trajectory_lock_adjustment(player)
        raise_mult += traj_raise_adj
        call_mult += traj_call_adj

        hero_bankroll = getattr(player.hero, "bankroll", 0)
        if hero_bankroll < -10:
            deficit = min(120, -hero_bankroll)
            raise_mult -= 0.08 + 0.002 * deficit
            call_mult -= 0.02

        if core._USE_RANDOM_POLICY:
            policy_bias = core._policy_bias(player, equity, pot_odds)
            raise_mult -= policy_bias
            call_mult -= policy_bias
        if player.hero.continue_cost > 0 and pot_total > 0:
            ratio = player.hero.continue_cost / float(pot_total)
            if ratio >= 1.0:
                call_mult += 0.10
            elif ratio >= 0.6:
                call_mult += 0.07
            elif ratio >= 0.35:
                call_mult += 0.04
            call_overbet_adj, raise_overbet_adj = core._opponent_overbet_adjustments(
                player,
                player.hero.continue_cost,
                pot_total,
            )
            call_mult += call_overbet_adj
            raise_mult += raise_overbet_adj
            escalation_penalty = core._bet_escalation_penalty(
                player, player.hero.continue_cost, pot_total
            )
            call_mult -= escalation_penalty
        if core.is_flop(player.street):
            call_mult += 0.04
        fold_bias = core._fold_bias_by_street(player.street)
        fold_bias_mult = 2.0
        if core.is_river(player.street) and player.hero.continue_cost > 0:
            fold_bias_mult = 3.0
        call_mult -= fold_bias * fold_bias_mult
        raise_call_penalty = core._raise_call_penalty(player, pot_total)
        turn_raise_extra = core._turn_raise_extra(player, pot_total)
        river_raise_call_penalty = core._river_raise_call_penalty(player, pot_total)
        river_raise_extra = core._river_raise_extra(player, pot_total)
        line_key = getattr(player.hero, "line_prefix", None)
        if player.hero.continue_cost > 0:
            call_mult -= core._line_call_penalty(line_key, player.street)
        strength, confidence = core.opponent_range_hint(line_key)
        if player.hero.continue_cost > 0:
            call_mult -= core._confidence_call_penalty(confidence, player.street)
            if core.is_river(player.street):
                call_mult -= 0.06
            last_call_street = getattr(player, "_last_hero_bet_street", None)
            last_action = getattr(player, "_last_hero_bet_action", None)
            if last_action == "call" and last_call_street is not None and last_call_street != player.street:
                call_mult -= 0.06 if player.street == 5 else 0.08
        variance_factor = core._equity_variance_factor(player)
        if variance_factor > 0.0:
            raise_mult += 0.12 * variance_factor
            call_mult += 0.08 * variance_factor
        if equity >= core._AGGRO_EQUITY:
            raise_mult -= core._AGGRO_RAISE_BONUS
        if core._should_pressure(player, equity):
            raise_mult -= core._PRESSURE_RAISE_BONUS
        # Lock defense: increase raise threshold when near danger zone
        # lock_defense_margin = core._lock_defense_raise_margin(player)
        # raise_mult += lock_defense_margin

        raise_mult = max(0.4, min(1.6, raise_mult))
        call_mult = max(0.4, min(1.6, call_mult))
        raise_margin = base_raise_margin * raise_mult
        call_margin = base_call_margin * call_mult

        call_margin -= raise_call_penalty
        if turn_raise_extra > 0:
            call_margin -= turn_raise_extra
        call_margin -= river_raise_call_penalty
        if river_raise_extra > 0:
            call_margin -= river_raise_extra

        raise_threshold = pot_odds + raise_margin
        call_threshold = pot_odds + call_margin
        lumberjack.record_thresholds("TightPolicy", player.street, call_threshold, raise_threshold, pot_odds)

        if player.street <= 0 and not core._DISABLE_PREFLOP_MIX:
            min_raise, max_raise = getattr(player.hero, "raise_bounds", (0, 0))
            action = core._preflop_open_decision(
                player,
                equity,
                pot_odds,
                legal_actions,
                min_raise,
                max_raise,
                pot_total,
            )
            if action is not None:
                return _record(action, equity, pot_odds)

        if player.street > 0:
            raise_threshold = pot_odds + raise_margin
            if force_full_equity:
                equity = core._range_conditioned_equity(
                    player,
                    hero_hand,
                    board_cards,
                    samples=samples,
                    max_seconds=max_seconds,
                )
            elif quick_equity > raise_threshold + 0.12:
                equity = quick_equity
            elif quick_equity < pot_odds - call_margin - 0.12:
                equity = quick_equity
            else:
                equity = core._range_conditioned_equity(
                    player,
                    hero_hand,
                    board_cards,
                    samples=samples,
                    max_seconds=max_seconds,
                )
            if equity_bias:
                equity = max(0.0, min(1.0, equity + equity_bias))

        if RaiseAction in legal_actions:
            min_raise, max_raise = getattr(player.hero, "raise_bounds", (0, 0))
            raise_threshold = pot_odds + raise_margin
            hero_raises = getattr(player, "_hero_raise_count", {}).get(int(player.street), 0)
            villain_raises = getattr(player, "_villain_raise_count", {}).get(int(player.street), 0)
            raise_war = hero_raises >= 1 and villain_raises >= 1 and equity < 0.70
            if core.is_river(player.street):
                equity = core._range_conditioned_equity(
                    player,
                    hero_hand,
                    board_cards,
                    samples=samples,
                    max_seconds=max_seconds,
                )
                raise_threshold += 0.06 + opp_river_raise_adj
                if player.hero.continue_cost == 0:
                    strength, confidence = core.opponent_range_hint()
                    thin_guard = 0.02 + 0.02 * confidence + 0.03 * max(0.0, strength - 0.55)
                    raise_threshold += thin_guard
            raise_cap = max(4, int(pot_total // 2 * raise_cap_mult))
            fold_mean, fold_low, fold_high, fold_conf, fold_width = core.fold_equity_band(
                min_raise, player.street, pot_total
            )
            fold_rate = fold_mean
            if fold_rate <= 0.0:
                fold_rate = core.fold_equity_estimate(None, min_raise, player.street, pot_total)
            if fold_conf > 0.0 and fold_width > 0.25:
                fold_rate = min(fold_rate, fold_low)
            if player.hero.continue_cost == 0 and pot_total <= 8 and min_raise >= pot_total * 8:
                if fold_conf < 0.5 or fold_width > 0.2:
                    raise_threshold += 0.08
            if min_raise > raise_cap or min_raise > player.hero.stack // 3:
                pass
            elif raise_war:
                pass
            elif equity >= core._NUT_RAISE_EQUITY and min_raise > 0:
                target = core._nut_raise_target(min_raise, max_raise)
                target = core._cap_raise_for_lock_defense(player, target)
                if target >= min_raise:
                    return _record(RaiseAction(target), equity, pot_odds)
            elif core.is_river(player.street) and equity < core._RIVER_VALUE_FLOOR + opp_river_floor_adj:
                pass
            elif equity < core._AGGRO_EQUITY:
                pass
            elif equity > raise_threshold and min_raise > 0:
                value_mult = core._value_extraction_multiplier(player)
                if core.is_river(player.street) and player.hero.continue_cost == 0:
                    if core._is_passive_line(line_key):
                        value_mult *= 1.12
                target = core._adaptive_raise_size(
                    player,
                    pot_total,
                    min_raise,
                    max_raise,
                    equity,
                    value_mult=value_mult,
                )
                target = core._suppress_medium_raise_target(target, min_raise, pot_total, equity)
                if core.is_river(player.street) and equity < core._NUT_RAISE_EQUITY:
                    river_cap = int(pot_total * core._RIVER_MAX_RAISE_FRAC)
                    target = min(target, river_cap)
                target = core._adjust_value_raise(target, min_raise, max_raise, fold_rate, equity)
                target = core._cap_raise_for_lock_defense(player, target)
                if target >= min_raise:
                    return _record(RaiseAction(target), equity, pot_odds)
            if (
                not core.is_river(player.street)
                and equity < pot_odds - 0.1
                and core._should_bluff(player.street, board_cards, player.street == 0)
            ):
                fold_mean, fold_low, fold_high, fold_conf, fold_width = core.fold_equity_band(
                    min_raise, player.street, pot_total
                )
                if fold_mean <= 0.3 or fold_conf < 0.25 or fold_width > 0.35 or fold_low < 0.22:
                    pass
                else:
                    target = core._adaptive_raise_size(
                        player,
                        pot_total,
                        min_raise,
                        max_raise,
                        equity,
                        bluff=True,
                    )
                    if target > 0:
                        target = core._cap_raise_for_lock_defense(player, target)
                        if target >= min_raise:
                            return _record(RaiseAction(target), equity, pot_odds)

        if core.is_river(player.street) and player.hero.continue_cost > 0 and FoldAction in legal_actions:
            passive_line = core._is_passive_line(line_key)
            river_call_gate = 0.03 + 0.04 * max(0.0, 0.35 - confidence) / 0.35
            river_call_gate += 0.02 * max(0.0, strength - 0.55)
            if passive_line:
                river_call_gate += 0.02
            if equity < pot_odds + river_call_gate:
                anti_rate = core._anti_exploit_rate(player)
                if anti_rate > 0 and CallAction in legal_actions and random.random() < anti_rate:
                    return _record(CallAction(), equity, pot_odds)
                return _record(FoldAction(), equity, pot_odds)

        if player.hero.continue_cost > 0 and FoldAction in legal_actions:
            last_call_street = getattr(player, "_last_hero_bet_street", None)
            last_action = getattr(player, "_last_hero_bet_action", None)
            if last_action == "call" and last_call_street is not None and last_call_street != player.street:
                call_down_gate = 0.05 if player.street == 5 else 0.07
                if equity < pot_odds + call_margin + call_down_gate:
                    anti_rate = core._anti_exploit_rate(player)
                    if anti_rate > 0 and CallAction in legal_actions and random.random() < anti_rate:
                        return _record(CallAction(), equity, pot_odds)
                    return _record(FoldAction(), equity, pot_odds)
            size_ratio, ratio_prev, ratio_now = core._bet_escalation_info(
                player, player.hero.continue_cost, pot_total
            )
            if size_ratio >= 2.0 or (ratio_now - ratio_prev) >= 0.4:
                escalation_gate = 0.05 if player.street == 5 else 0.07
                if equity < pot_odds + call_margin + escalation_gate:
                    anti_rate = core._anti_exploit_rate(player)
                    if anti_rate > 0 and CallAction in legal_actions and random.random() < anti_rate:
                        return _record(CallAction(), equity, pot_odds)
                    return _record(FoldAction(), equity, pot_odds)
            hero_raises = getattr(player, "_hero_raise_count", {}).get(int(player.street), 0)
            villain_raises = getattr(player, "_villain_raise_count", {}).get(int(player.street), 0)
            if hero_raises >= 1 and villain_raises >= 1 and equity < 0.70:
                return _record(FoldAction(), equity, pot_odds)

        hard_fold_equity = core._hard_fold_equity(player.street)
        if (
            hard_fold_equity > 0.0
            and player.hero.continue_cost > 0
            and pot_odds >= core._HARD_FOLD_POT_ODDS_MIN
            and equity < hard_fold_equity
            and FoldAction in legal_actions
        ):
            anti_rate = core._anti_exploit_rate(player)
            if anti_rate > 0 and CallAction in legal_actions and random.random() < anti_rate:
                return _record(CallAction(), equity, pot_odds)
            return _record(FoldAction(), equity, pot_odds)

        if equity < pot_odds - call_margin and FoldAction in legal_actions:
            anti_rate = core._anti_exploit_rate(player)
            if anti_rate > 0 and CallAction in legal_actions and random.random() < anti_rate:
                return _record(CallAction(), equity, pot_odds)
            return _record(FoldAction(), equity, pot_odds)

        if CheckAction in legal_actions and player.hero.continue_cost == 0:
            return _record(CheckAction(), equity, pot_odds)
        if CallAction in legal_actions:
            return _record(CallAction(), equity, pot_odds)
        if CheckAction in legal_actions:
            return _record(CheckAction(), equity, pot_odds)
        return _record(FoldAction(), equity, pot_odds)
