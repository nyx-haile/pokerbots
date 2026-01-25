from skeleton.actions import CallAction, CheckAction, DiscardAction, FoldAction, RaiseAction

import stats
import strategy as core


class TightPolicy:
    """Tight policy: fold early, raise aggressively only when strong."""

    @staticmethod
    def play(player):
        legal_actions = set(player.hero.legal_actions)
        hero_hand = list(player.hero.hand)
        board_cards = list(player.community)

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
            player.hero.last_discard = hero_hand[best_i]
            player.hero.last_discard_visible = player.hero.blind
            player.hero.discard_bluff = False
            return DiscardAction(best_i)

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
            quick_equity = stats.estimate_equity(
                hero_hand,
                board_cards,
                samples=max(20, samples // 4),
                max_seconds=min(0.006, max_seconds * 0.2),
                discard_samples=max(4, discard_samples // 2),
            )
            equity = quick_equity

        pot_odds = core.pot_odds_to_call(player.hero.continue_cost, pot_total)
        raise_margin = core._raise_margin_by_street(player.street)
        call_margin = core._call_margin_by_street(player.street)
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
        raise_margin += discard_bias
        call_margin += discard_bias
        raise_margin += range_bias
        call_margin += range_bias
        raise_margin += raise_variant
        call_margin += call_variant
        
        # Tighten play when approaching win-lock (pot-aware)
        lead_adj = core._lead_protection_adjustment(player, pot_total)
        raise_margin += lead_adj
        call_margin -= lead_adj  # Harder to call when protecting lead

        if core._USE_RANDOM_POLICY:
            policy_bias = core._policy_bias(player, equity, pot_odds)
            raise_margin -= policy_bias
            call_margin -= policy_bias
        raise_margin += texture_raise
        call_margin += texture_call
        call_margin -= core._fold_bias_by_street(player.street) * 2.0
        raise_call_penalty = core._raise_call_penalty(player, pot_total)
        call_margin -= raise_call_penalty
        turn_raise_extra = core._turn_raise_extra(player, pot_total)
        if turn_raise_extra > 0:
            call_margin -= turn_raise_extra
        if equity >= core._AGGRO_EQUITY:
            raise_margin -= core._AGGRO_RAISE_BONUS
        if core._should_pressure(player, equity):
            raise_margin -= core._PRESSURE_RAISE_BONUS
        # Lock defense: increase raise threshold when near danger zone
        lock_defense_margin = core._lock_defense_raise_margin(player)
        raise_margin += lock_defense_margin

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
                return action

        if player.street > 0:
            raise_threshold = pot_odds + raise_margin
            if quick_equity > raise_threshold + 0.12:
                equity = quick_equity
            elif quick_equity < pot_odds - call_margin - 0.12:
                equity = quick_equity
            else:
                equity = stats.estimate_equity(
                    hero_hand,
                    board_cards,
                    samples=samples,
                    max_seconds=max_seconds,
                    discard_samples=discard_samples,
                )
            if equity_bias:
                equity = max(0.0, min(1.0, equity + equity_bias))

        if RaiseAction in legal_actions:
            min_raise, max_raise = getattr(player.hero, "raise_bounds", (0, 0))
            raise_threshold = pot_odds + raise_margin
            raise_cap = max(4, int(pot_total // 2 * raise_cap_mult))
            fold_rate = core.fold_equity_estimate(None, min_raise, player.street, pot_total)
            if min_raise > raise_cap or min_raise > player.hero.stack // 2:
                pass
            elif equity >= core._NUT_RAISE_EQUITY and min_raise > 0:
                target = core._nut_raise_target(min_raise, max_raise)
                target = core._cap_raise_for_lock_defense(player, target)
                if target >= min_raise:
                    return RaiseAction(target)
            elif equity < core._AGGRO_EQUITY:
                pass
            elif equity > raise_threshold and min_raise > 0:
                value_mult = core._value_extraction_multiplier(player)
                target = core._raise_size(pot_total, min_raise, max_raise, equity, value_mult=value_mult)
                target = core._adjust_value_raise(target, min_raise, max_raise, fold_rate, equity)
                target = core._cap_raise_for_lock_defense(player, target)
                if target >= min_raise:
                    return RaiseAction(target)
            if equity < pot_odds - 0.1 and core._should_bluff(player.street, board_cards, player.street == 0):
                if fold_rate <= 0.3:
                    pass
                else:
                    target = core._raise_size(pot_total, min_raise, max_raise, equity, bluff=True)
                    if target > 0:
                        target = core._cap_raise_for_lock_defense(player, target)
                        if target >= min_raise:
                            return RaiseAction(target)

        hard_fold_equity = core._hard_fold_equity(player.street)
        if (
            hard_fold_equity > 0.0
            and player.hero.continue_cost > 0
            and pot_odds >= core._HARD_FOLD_POT_ODDS_MIN
            and equity < hard_fold_equity
            and FoldAction in legal_actions
        ):
            return FoldAction()

        if equity < pot_odds - call_margin and FoldAction in legal_actions:
            return FoldAction()

        if CheckAction in legal_actions and player.hero.continue_cost == 0:
            return CheckAction()
        if CallAction in legal_actions:
            return CallAction()
        if CheckAction in legal_actions:
            return CheckAction()
        return FoldAction()
