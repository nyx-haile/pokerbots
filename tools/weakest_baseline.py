#!/usr/bin/env python3
import json
import os
import re
import sys


BOT_B_PATTERN = re.compile(r"^Bot B:\s+(.*)$")


def _iter_match_dirs(root):
    for dirpath, _, filenames in os.walk(root):
        if "summary.json" in filenames and "engine_stdout.txt" in filenames:
            yield dirpath


def _parse_bot_b(engine_path):
    try:
        with open(engine_path, "r") as handle:
            for line in handle:
                match = BOT_B_PATTERN.match(line.strip())
                if match:
                    return match.group(1)
    except OSError:
        return None
    return None


def _parse_ev(summary_path):
    try:
        with open(summary_path, "r") as handle:
            summary = json.load(handle)
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    players = {player.get("name"): player for player in summary.get("players", [])}
    bot_b = players.get("Bot B")
    if not bot_b:
        return None
    ev = bot_b.get("ev_per_hand")
    if ev is None:
        return None
    try:
        return float(ev)
    except (TypeError, ValueError):
        return None


def main():
    roots = [path for path in ("runs", "tuning") if os.path.isdir(path)]
    stats = {}
    for root in roots:
        for match_dir in _iter_match_dirs(root):
            engine_path = os.path.join(match_dir, "engine_stdout.txt")
            summary_path = os.path.join(match_dir, "summary.json")
            bot_b = _parse_bot_b(engine_path)
            if not bot_b or "/baselines/" not in bot_b:
                continue
            ev = _parse_ev(summary_path)
            if ev is None:
                continue
            entry = stats.setdefault(bot_b, {"count": 0, "sum": 0.0})
            entry["count"] += 1
            entry["sum"] += ev

    if not stats:
        print("No baseline matches found under runs/ or tuning/.")
        return 1

    rows = []
    for bot_b, entry in stats.items():
        avg = entry["sum"] / entry["count"] if entry["count"] else 0.0
        rows.append((avg, entry["count"], bot_b))
    rows.sort()
    print("avg_ev_per_hand  matches  bot_b")
    for avg, count, bot_b in rows:
        print("{:>14.4f}  {:>7d}  {}".format(avg, count, bot_b))
    weakest = rows[0]
    print("\nweakest_bot={}".format(weakest[2]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
