#!/usr/bin/env python3
"""
Parse a Pokerbots gamelog and emit summary statistics.

Outputs per-player EV/hand, win rate, variance, action frequencies,
self-discard counts, and seed if present.
"""
import argparse
import json
import os
import re
import sys


ROUND_RE = re.compile(r"^Round #(?P<round>\d+),\s+(?P<p1>.+) \((?P<b1>-?\d+)\),\s+(?P<p2>.+) \((?P<b2>-?\d+)\)")
AWARD_RE = re.compile(r"^(?P<name>.+) awarded (?P<delta>-?\d+)")
ACTION_RE = re.compile(r"^(?P<name>.+?) (?P<action>folds|calls|checks|raises to|bets|discards)\b")
DISCARD_RE = re.compile(r"^(?P<name>.+) discards (?P<card>\S+)")
SEED_RE = re.compile(r"^Seed: (?P<seed>.+)$")


def _init_player_stats(name):
    return {
        "name": name,
        "deltas": [],
        "wins": 0,
        "losses": 0,
        "ties": 0,
        "actions": {
            "folds": 0,
            "calls": 0,
            "checks": 0,
            "raises": 0,
            "bets": 0,
            "discards": 0,
        },
        "discard_cards": {},
    }


def _update_action(stats, action):
    if action == "raises to":
        stats["actions"]["raises"] += 1
    elif action == "bets":
        stats["actions"]["bets"] += 1
    elif action == "folds":
        stats["actions"]["folds"] += 1
    elif action == "calls":
        stats["actions"]["calls"] += 1
    elif action == "checks":
        stats["actions"]["checks"] += 1
    elif action == "discards":
        stats["actions"]["discards"] += 1


def parse_gamelog(path):
    if not os.path.exists(path):
        raise SystemExit("gamelog not found: %s" % path)

    with open(path, "r") as handle:
        lines = [line.rstrip("\n") for line in handle]

    seed = None
    players = []
    stats = {}
    round_awards = {}

    for line in lines:
        if not line:
            continue
        seed_match = SEED_RE.match(line)
        if seed_match:
            seed = seed_match.group("seed")
            continue
        round_match = ROUND_RE.match(line)
        if round_match:
            p1 = round_match.group("p1")
            p2 = round_match.group("p2")
            if not players:
                players = [p1, p2]
                stats[p1] = _init_player_stats(p1)
                stats[p2] = _init_player_stats(p2)
            round_awards = {}
            continue
        award_match = AWARD_RE.match(line)
        if award_match:
            name = award_match.group("name")
            delta = int(award_match.group("delta"))
            if name in stats:
                round_awards[name] = delta
                if len(round_awards) == 2:
                    for player_name, value in round_awards.items():
                        stats[player_name]["deltas"].append(value)
                        if value > 0:
                            stats[player_name]["wins"] += 1
                        elif value < 0:
                            stats[player_name]["losses"] += 1
                        else:
                            stats[player_name]["ties"] += 1
            continue
        action_match = ACTION_RE.match(line)
        if action_match:
            name = action_match.group("name")
            action = action_match.group("action")
            if name in stats:
                _update_action(stats[name], action)
            if action == "discards":
                discard_match = DISCARD_RE.match(line)
                if discard_match:
                    card = discard_match.group("card")
                    counts = stats[name]["discard_cards"]
                    counts[card] = counts.get(card, 0) + 1
            continue

    summary = {
        "gamelog": os.path.abspath(path),
        "seed": seed,
        "players": [],
    }
    for name in players:
        player_stats = stats[name]
        deltas = player_stats["deltas"]
        hands = len(deltas)
        ev_per_hand = (sum(deltas) / float(hands)) if hands else 0.0
        mean = ev_per_hand
        variance = 0.0
        if hands:
            variance = sum((d - mean) ** 2 for d in deltas) / float(hands)
        player_stats["hands"] = hands
        player_stats["ev_per_hand"] = ev_per_hand
        player_stats["variance"] = variance
        win_rate = (player_stats["wins"] / float(hands)) if hands else 0.0
        player_stats["win_rate"] = win_rate
        summary["players"].append(player_stats)

    return summary


def main():
    parser = argparse.ArgumentParser(description="Parse a gamelog and output summary stats.")
    parser.add_argument("gamelog", help="Path to gamelog.txt")
    parser.add_argument("--json-out", default=None, help="Write summary JSON to path")
    args = parser.parse_args()

    summary = parse_gamelog(args.gamelog)
    text = json.dumps(summary, indent=2, sort_keys=True)
    print(text)
    if args.json_out:
        with open(args.json_out, "w") as handle:
            handle.write(text)
            handle.write("\n")


if __name__ == "__main__":
    main()
