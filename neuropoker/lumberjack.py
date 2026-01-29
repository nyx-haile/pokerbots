"""Shared logging helpers referenced by the pokerbot."""

from typing import Dict
from contextlib import contextmanager


_POLICY_ACTIONS: Dict[str, Dict[int, Dict[str, int]]] = {}
_DRY_POLICY_ACTIONS: Dict[str, Dict[int, Dict[str, int]]] = {}
_POLICY_EQUITY_ODDS: Dict[str, Dict[int, Dict[str, float]]] = {}
_DISCARD_EV_STATS: Dict[str, Dict[int, Dict[str, float]]] = {}
_HERO_ACTIONS: Dict[int, Dict[str, int]] = {}
_SHOWDOWN_LINES: Dict[str, Dict[str, float]] = {}
_BET_SIZE_EV: Dict[int, Dict[str, Dict[str, float]]] = {}
_RIVER_VALUE_BET: Dict[str, float] = {"count": 0.0, "wins": 0.0, "sum_delta": 0.0}
_THRESHOLD_STATS: Dict[str, Dict[int, Dict[str, float]]] = {}
_EQUITY_ERROR_STATS: Dict[int, Dict[str, float]] = {}
_FOLD_PREVENT_STATS: Dict[int, Dict[str, float]] = {}
_LOG_SUPPRESSED = False
_POLICY_ACTION_SUPPRESSED = False


@contextmanager
def suppress_logs():
    global _LOG_SUPPRESSED
    prev = _LOG_SUPPRESSED
    _LOG_SUPPRESSED = True
    try:
        yield
    finally:
        _LOG_SUPPRESSED = prev


@contextmanager
def suppress_policy_actions():
    global _POLICY_ACTION_SUPPRESSED
    prev = _POLICY_ACTION_SUPPRESSED
    _POLICY_ACTION_SUPPRESSED = True
    try:
        yield
    finally:
        _POLICY_ACTION_SUPPRESSED = prev


def policy_summary(stats: Dict[str, Dict]) -> str:
    if not stats:
        return "policy_stats: none"
    entries = []
    for key in sorted(stats):
        bucket = stats[key]
        count = bucket.get("count", 0.0)
        if count <= 0:
            continue
        avg_delta = bucket.get("total_delta", 0.0) / count
        avg_reward = bucket.get("total_reward", 0.0) / count
        avg = bucket.get("avg", 0.0)
        header = (
            f"{key}: hands={int(count)} avg_delta={avg_delta:.2f} "
            f"avg_reward={avg_reward:.3f} ema_reward={avg:.3f}"
        )
        streets = []
        for street in sorted(bucket.get("per_street", {})):
            street_bucket = bucket["per_street"][street]
            street_count = street_bucket.get("count", 0.0)
            if street_count <= 0:
                continue
            street_avg_delta = street_bucket.get("total_delta", 0.0) / street_count
            street_avg_reward = street_bucket.get("total_reward", 0.0) / street_count
            streets.append(
                f"  street={street} hands={int(street_count)} "
                f"avg_delta={street_avg_delta:.2f} avg_reward={street_avg_reward:.3f}"
            )
        entries.append("\n".join([header] + streets))
    if not entries:
        return "policy_stats: none"
    return "policy_stats:\n" + "\n".join(entries)


def leak_summary(leak_stats: Dict) -> str:
    lines = ["leak_stats:"]
    for key in (
        "end_by_street",
        "hero_calls",
        "hero_fold_vs_bet",
        "villain_bets",
        "villain_checks",
        "hero_fold_to_raise",
    ):
        bucket = leak_stats.get(key, {})
        for street in sorted(bucket):
            entry = bucket[street]
            if key == "end_by_street":
                lines.append(
                    f"  end_street={street} hands={entry.get('hands',0)} "
                    f"delta={entry.get('delta',0)} losses={entry.get('loss',0)}"
                )
            elif key == "hero_calls":
                lines.append(
                    f"  hero_calls street={street} calls={entry.get('calls',0)} "
                    f"wins={entry.get('wins',0)} losses={entry.get('losses',0)}"
                )
            elif key == "hero_fold_vs_bet":
                lines.append(
                    f"  hero_folds_vs_bet street={street} "
                    f"small={entry.get('small',0)} medium={entry.get('medium',0)} "
                    f"large={entry.get('large',0)}"
                )
            elif key == "villain_bets":
                lines.append(
                    f"  villain_bets street={street} "
                    f"small={entry.get('small',0)} medium={entry.get('medium',0)} large={entry.get('large',0)}"
                )
            elif key == "villain_checks":
                lines.append(
                    f"  villain_checks street={street} checks={entry.get('check',0)}"
                )
            elif key == "hero_fold_to_raise":
                lines.append(
                    f"  hero_fold_to_raise street={street} count={entry.get('fold_to_raise',0)}"
                )
    river_loss = leak_stats.get("river_loss_after_bet", {})
    lines.append(
        f"  river_loss_after_bet count={river_loss.get('count',0)} "
        f"delta={river_loss.get('delta',0)}"
    )
    return "\n".join(lines)


def log(message: str) -> None:
    print(f"[bot] {message}", flush=True)


def record_policy_action(policy: str, street: int, action: str) -> None:
    if _LOG_SUPPRESSED or _POLICY_ACTION_SUPPRESSED:
        return
    if not policy:
        policy = "UnknownPolicy"
    street_bucket = _POLICY_ACTIONS.setdefault(policy, {}).setdefault(int(street), {})
    street_bucket[action] = street_bucket.get(action, 0) + 1


def record_dry_policy_action(policy: str, street: int, action: str) -> None:
    if not policy:
        policy = "UnknownPolicy"
    street_bucket = _DRY_POLICY_ACTIONS.setdefault(policy, {}).setdefault(int(street), {})
    street_bucket[action] = street_bucket.get(action, 0) + 1


def record_equity_vs_pot_odds(
    policy: str,
    street: int,
    equity: float,
    pot_odds: float,
) -> None:
    if _LOG_SUPPRESSED:
        return
    if equity is None or pot_odds is None:
        return
    if not policy:
        policy = "UnknownPolicy"
    street_bucket = _POLICY_EQUITY_ODDS.setdefault(policy, {}).setdefault(
        int(street),
        {"count": 0.0, "sum_equity": 0.0, "sum_pot_odds": 0.0, "sum_diff": 0.0, "sum_abs_diff": 0.0},
    )
    diff = equity - pot_odds
    street_bucket["count"] += 1.0
    street_bucket["sum_equity"] += equity
    street_bucket["sum_pot_odds"] += pot_odds
    street_bucket["sum_diff"] += diff
    street_bucket["sum_abs_diff"] += abs(diff)


def record_thresholds(
    policy: str,
    street: int,
    call_threshold: float,
    raise_threshold: float,
    pot_odds: float,
) -> None:
    if _LOG_SUPPRESSED:
        return
    if not policy:
        policy = "UnknownPolicy"
    street_bucket = _THRESHOLD_STATS.setdefault(policy, {}).setdefault(
        int(street),
        {
            "count": 0.0,
            "sum_call": 0.0,
            "sum_raise": 0.0,
            "sum_call_drift": 0.0,
            "sum_raise_drift": 0.0,
        },
    )
    street_bucket["count"] += 1.0
    street_bucket["sum_call"] += float(call_threshold)
    street_bucket["sum_raise"] += float(raise_threshold)
    street_bucket["sum_call_drift"] += float(call_threshold - pot_odds)
    street_bucket["sum_raise_drift"] += float(raise_threshold - pot_odds)


def record_equity_error(street: int, abs_error: float, samples: int, extended: bool = False) -> None:
    if _LOG_SUPPRESSED:
        return
    bucket = _EQUITY_ERROR_STATS.setdefault(
        int(street),
        {"count": 0.0, "sum_abs": 0.0, "sum_samples": 0.0, "extended": 0.0},
    )
    bucket["count"] += 1.0
    bucket["sum_abs"] += float(abs_error)
    bucket["sum_samples"] += float(samples)
    if extended:
        bucket["extended"] += 1.0


def record_fold_prevent_outcome(street: int, action_to: str, delta: int) -> None:
    bucket = _FOLD_PREVENT_STATS.setdefault(
        int(street),
        {"count": 0.0, "sum_delta": 0.0, "wins": 0.0, "losses": 0.0, "calls": 0.0, "checks": 0.0},
    )
    bucket["count"] += 1.0
    bucket["sum_delta"] += float(delta)
    if delta > 0:
        bucket["wins"] += 1.0
    elif delta < 0:
        bucket["losses"] += 1.0
    if action_to == "CallAction":
        bucket["calls"] += 1.0
    elif action_to == "CheckAction":
        bucket["checks"] += 1.0


def record_hero_action(street: int, action: str) -> None:
    street_bucket = _HERO_ACTIONS.setdefault(int(street), {})
    street_bucket[action] = street_bucket.get(action, 0) + 1


def policy_action_summary() -> str:
    if not _POLICY_ACTIONS:
        return "policy_actions: none"
    lines = ["policy_actions:"]
    for policy in sorted(_POLICY_ACTIONS):
        policy_bucket = _POLICY_ACTIONS[policy]
        for street in sorted(policy_bucket):
            actions = policy_bucket[street]
            parts = [f"{name}={actions[name]}" for name in sorted(actions)]
            lines.append(f"  {policy} street={street} " + " ".join(parts))
    return "\n".join(lines)


def dry_policy_action_summary() -> str:
    if not _DRY_POLICY_ACTIONS:
        return "dry_policy_actions: none"
    lines = ["dry_policy_actions:"]
    for policy in sorted(_DRY_POLICY_ACTIONS):
        policy_bucket = _DRY_POLICY_ACTIONS[policy]
        for street in sorted(policy_bucket):
            actions = policy_bucket[street]
            parts = [f"{name}={actions[name]}" for name in sorted(actions)]
            lines.append(f"  {policy} street={street} " + " ".join(parts))
    return "\n".join(lines)


def aggression_ratio_summary() -> str:
    if not _HERO_ACTIONS:
        return "aggression_ratio: none"
    lines = ["aggression_ratio:"]
    for street in sorted(_HERO_ACTIONS):
        bucket = _HERO_ACTIONS[street]
        raises = bucket.get("RaiseAction", 0)
        calls = bucket.get("CallAction", 0)
        checks = bucket.get("CheckAction", 0)
        denom = max(1, calls + checks)
        ratio = raises / denom
        lines.append(
            f"  street={street} raises={raises} calls={calls} checks={checks} ratio={ratio:.3f}"
        )
    return "\n".join(lines)


def equity_vs_pot_odds_summary() -> str:
    if not _POLICY_EQUITY_ODDS:
        return "equity_vs_pot_odds: none"
    lines = ["equity_vs_pot_odds:"]
    for policy in sorted(_POLICY_EQUITY_ODDS):
        policy_bucket = _POLICY_EQUITY_ODDS[policy]
        for street in sorted(policy_bucket):
            bucket = policy_bucket[street]
            count = bucket.get("count", 0.0)
            if count <= 0:
                continue
            avg_equity = bucket.get("sum_equity", 0.0) / count
            avg_pot_odds = bucket.get("sum_pot_odds", 0.0) / count
            avg_diff = bucket.get("sum_diff", 0.0) / count
            avg_abs_diff = bucket.get("sum_abs_diff", 0.0) / count
            lines.append(
                f"  {policy} street={street} count={int(count)} "
                f"equity={avg_equity:.3f} pot_odds={avg_pot_odds:.3f} "
                f"diff={avg_diff:.3f} abs_diff={avg_abs_diff:.3f}"
            )
    return "\n".join(lines)


def equity_error_summary() -> str:
    if not _EQUITY_ERROR_STATS:
        return "equity_error: none"
    lines = ["equity_error:"]
    for street in sorted(_EQUITY_ERROR_STATS):
        bucket = _EQUITY_ERROR_STATS[street]
        count = bucket.get("count", 0.0)
        if count <= 0:
            continue
        avg_error = bucket.get("sum_abs", 0.0) / count
        avg_samples = bucket.get("sum_samples", 0.0) / count
        extended = int(bucket.get("extended", 0.0))
        lines.append(
            f"  street={street} count={int(count)} avg_abs_err={avg_error:.3f} "
            f"avg_samples={avg_samples:.1f} extended={extended}"
        )
    return "\n".join(lines)


def fold_prevent_summary() -> str:
    if not _FOLD_PREVENT_STATS:
        return "fold_prevent: none"
    lines = ["fold_prevent:"]
    for street in sorted(_FOLD_PREVENT_STATS):
        bucket = _FOLD_PREVENT_STATS[street]
        count = bucket.get("count", 0.0)
        if count <= 0:
            continue
        avg_delta = bucket.get("sum_delta", 0.0) / count
        wins = bucket.get("wins", 0.0)
        losses = bucket.get("losses", 0.0)
        calls = bucket.get("calls", 0.0)
        checks = bucket.get("checks", 0.0)
        lines.append(
            f"  street={street} count={int(count)} avg_delta={avg_delta:.2f} "
            f"wins={int(wins)} losses={int(losses)} calls={int(calls)} checks={int(checks)}"
        )
    return "\n".join(lines)


def threshold_summary() -> str:
    if not _THRESHOLD_STATS:
        return "thresholds: none"
    lines = ["thresholds:"]
    for policy in sorted(_THRESHOLD_STATS):
        policy_bucket = _THRESHOLD_STATS[policy]
        for street in sorted(policy_bucket):
            bucket = policy_bucket[street]
            count = bucket.get("count", 0.0)
            if count <= 0:
                continue
            avg_call = bucket.get("sum_call", 0.0) / count
            avg_raise = bucket.get("sum_raise", 0.0) / count
            avg_call_drift = bucket.get("sum_call_drift", 0.0) / count
            avg_raise_drift = bucket.get("sum_raise_drift", 0.0) / count
            lines.append(
                f"  {policy} street={street} count={int(count)} "
                f"call={avg_call:.3f} raise={avg_raise:.3f} "
                f"call_drift={avg_call_drift:.3f} raise_drift={avg_raise_drift:.3f}"
            )
    return "\n".join(lines)


def record_showdown_line(line_key: str, delta: int) -> None:
    if not line_key:
        return
    bucket = _SHOWDOWN_LINES.setdefault(
        line_key,
        {"count": 0.0, "wins": 0.0, "losses": 0.0, "sum_delta": 0.0},
    )
    bucket["count"] += 1.0
    bucket["sum_delta"] += float(delta)
    if delta > 0:
        bucket["wins"] += 1.0
    elif delta < 0:
        bucket["losses"] += 1.0


def showdown_line_summary() -> str:
    if not _SHOWDOWN_LINES:
        return "showdown_lines: none"
    lines = ["showdown_lines:"]
    for line_key in sorted(_SHOWDOWN_LINES):
        bucket = _SHOWDOWN_LINES[line_key]
        count = bucket.get("count", 0.0)
        if count <= 0:
            continue
        wins = bucket.get("wins", 0.0)
        losses = bucket.get("losses", 0.0)
        win_rate = wins / max(1.0, wins + losses)
        avg_delta = bucket.get("sum_delta", 0.0) / count
        lines.append(
            f"  line={line_key} hands={int(count)} win_rate={win_rate:.3f} avg_delta={avg_delta:.2f}"
        )
    return "\n".join(lines)


def record_bet_size_outcome(street: int, bucket: str, delta: int) -> None:
    street_bucket = _BET_SIZE_EV.setdefault(int(street), {})
    bucket_stats = street_bucket.setdefault(
        bucket,
        {"count": 0.0, "wins": 0.0, "losses": 0.0, "sum_delta": 0.0},
    )
    bucket_stats["count"] += 1.0
    bucket_stats["sum_delta"] += float(delta)
    if delta > 0:
        bucket_stats["wins"] += 1.0
    elif delta < 0:
        bucket_stats["losses"] += 1.0


def bet_size_ev_summary() -> str:
    if not _BET_SIZE_EV:
        return "bet_size_ev: none"
    lines = ["bet_size_ev:"]
    for street in sorted(_BET_SIZE_EV):
        bucket = _BET_SIZE_EV[street]
        for size_key in sorted(bucket):
            stats = bucket[size_key]
            count = stats.get("count", 0.0)
            if count <= 0:
                continue
            avg_delta = stats.get("sum_delta", 0.0) / count
            wins = stats.get("wins", 0.0)
            losses = stats.get("losses", 0.0)
            win_rate = wins / max(1.0, wins + losses)
            lines.append(
                f"  street={street} size={size_key} hands={int(count)} "
                f"win_rate={win_rate:.3f} avg_delta={avg_delta:.2f}"
            )
    return "\n".join(lines)


def record_river_value_bet(delta: int) -> None:
    _RIVER_VALUE_BET["count"] += 1.0
    _RIVER_VALUE_BET["sum_delta"] += float(delta)
    if delta > 0:
        _RIVER_VALUE_BET["wins"] += 1.0


def river_value_bet_summary() -> str:
    count = _RIVER_VALUE_BET.get("count", 0.0)
    if count <= 0:
        return "river_value_bet: none"
    wins = _RIVER_VALUE_BET.get("wins", 0.0)
    win_rate = wins / max(1.0, count)
    avg_delta = _RIVER_VALUE_BET.get("sum_delta", 0.0) / count
    return f"river_value_bet: count={int(count)} win_rate={win_rate:.3f} avg_delta={avg_delta:.2f}"


def record_discard_decision(policy: str, street: int, equities, chosen_idx: int):
    if _LOG_SUPPRESSED:
        return None
    if equities is None:
        return None
    equities = list(equities)
    if not equities:
        return None
    if not policy:
        policy = "UnknownPolicy"
    street = int(street)
    if chosen_idx is None or chosen_idx < 0 or chosen_idx >= len(equities):
        return None
    chosen_eq = float(equities[chosen_idx])
    best_idx = max(range(len(equities)), key=equities.__getitem__)
    best_eq = float(equities[best_idx])
    gap = best_eq - chosen_eq
    bucket = _DISCARD_EV_STATS.setdefault(policy, {}).setdefault(
        street,
        {
            "count": 0.0,
            "sum_chosen": 0.0,
            "sum_best": 0.0,
            "sum_gap": 0.0,
            "missed_best": 0.0,
            "wins": 0.0,
            "losses": 0.0,
            "ties": 0.0,
            "loss_gap": 0.0,
        },
    )
    bucket["count"] += 1.0
    bucket["sum_chosen"] += chosen_eq
    bucket["sum_best"] += best_eq
    bucket["sum_gap"] += gap
    if best_idx != chosen_idx:
        bucket["missed_best"] += 1.0
    return {
        "policy": policy,
        "street": street,
        "chosen_idx": chosen_idx,
        "chosen_eq": chosen_eq,
        "best_idx": best_idx,
        "best_eq": best_eq,
        "gap": gap,
    }


def record_discard_outcome(decision, delta: int) -> None:
    if not decision:
        return
    policy = decision.get("policy", "UnknownPolicy")
    street = int(decision.get("street", 0))
    bucket = _DISCARD_EV_STATS.setdefault(policy, {}).setdefault(
        street,
        {
            "count": 0.0,
            "sum_chosen": 0.0,
            "sum_best": 0.0,
            "sum_gap": 0.0,
            "missed_best": 0.0,
            "wins": 0.0,
            "losses": 0.0,
            "ties": 0.0,
            "loss_gap": 0.0,
        },
    )
    if delta > 0:
        bucket["wins"] += 1.0
    elif delta < 0:
        bucket["losses"] += 1.0
        bucket["loss_gap"] += float(decision.get("gap", 0.0))
    else:
        bucket["ties"] += 1.0


def discard_ev_summary() -> str:
    if not _DISCARD_EV_STATS:
        return "discard_ev: none"
    lines = ["discard_ev:"]
    for policy in sorted(_DISCARD_EV_STATS):
        policy_bucket = _DISCARD_EV_STATS[policy]
        for street in sorted(policy_bucket):
            bucket = policy_bucket[street]
            count = bucket.get("count", 0.0)
            if count <= 0:
                continue
            avg_chosen = bucket.get("sum_chosen", 0.0) / count
            avg_best = bucket.get("sum_best", 0.0) / count
            avg_gap = bucket.get("sum_gap", 0.0) / count
            missed = bucket.get("missed_best", 0.0)
            wins = bucket.get("wins", 0.0)
            losses = bucket.get("losses", 0.0)
            ties = bucket.get("ties", 0.0)
            outcome_total = max(1.0, wins + losses + ties)
            win_rate = wins / outcome_total
            loss_gap = bucket.get("loss_gap", 0.0)
            avg_loss_gap = loss_gap / max(1.0, losses)
            lines.append(
                f"  {policy} street={street} count={int(count)} "
                f"chosen={avg_chosen:.3f} best={avg_best:.3f} gap={avg_gap:.3f} "
                f"missed={int(missed)} win_rate={win_rate:.3f} loss_gap={avg_loss_gap:.3f}"
            )
    return "\n".join(lines)
