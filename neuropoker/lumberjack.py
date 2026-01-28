"""Shared logging helpers referenced by the pokerbot."""

from typing import Dict


_POLICY_ACTIONS: Dict[str, Dict[int, Dict[str, int]]] = {}
_POLICY_EQUITY_ODDS: Dict[str, Dict[int, Dict[str, float]]] = {}


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
    if not policy:
        policy = "UnknownPolicy"
    street_bucket = _POLICY_ACTIONS.setdefault(policy, {}).setdefault(int(street), {})
    street_bucket[action] = street_bucket.get(action, 0) + 1


def record_equity_vs_pot_odds(
    policy: str,
    street: int,
    equity: float,
    pot_odds: float,
) -> None:
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
