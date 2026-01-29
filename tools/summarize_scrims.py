#!/usr/bin/env python3
"""Summarize Pokerbots scrim logs (game_log*)."""
import argparse
import glob
import json
import os
import re
from collections import Counter

ROUND_RE = re.compile(
    r"^Round #(?P<round>\d+),\s+(?P<p1>.+) \((?P<b1>-?\d+)\),\s+(?P<p2>.+) \((?P<b2>-?\d+)\)"
)
FINAL_RE = re.compile(
    r"^Final,\s+(?P<p1>.+) \((?P<b1>-?\d+)\),\s+(?P<p2>.+) \((?P<b2>-?\d+)\)"
)
AWARD_RE = re.compile(r"^(?P<name>.+) awarded (?P<delta>-?\d+)")
ACTION_RE = re.compile(r"^(?P<name>.+?) (?P<action>folds|calls|checks|raises to|bets|discards)\b")
SHOW_RE = re.compile(r"^(?P<name>.+) shows ")
STREET_RE = re.compile(r"^(?P<street>Flop|Discard 1|Discard 2|Turn|River) ")

STREETS = ["Preflop", "Flop", "Discard 1", "Discard 2", "Turn", "River"]
ACTION_KEYS = ["folds", "calls", "checks", "raises", "bets", "discards"]


def _init_player_stats(name):
    return {
        "name": name,
        "actions_by_street": {street: Counter() for street in STREETS},
        "net_by_stage": Counter(),
        "showdown": {"hands": 0, "net": 0},
        "nshow": {"hands": 0, "net": 0},
        "total_net": 0,
    }


def parse_scrim(path):
    if not os.path.exists(path):
        raise SystemExit(f"log not found: {path}")

    players = []
    stats = {}
    end_stage_counts = Counter()
    hands = 0
    final = None

    current_street = "Preflop"
    round_awards = {}
    round_showdown = False
    round_end_stage = None

    with open(path, "r", encoding="utf-8", errors="ignore") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            final_match = FINAL_RE.match(line)
            if final_match:
                final = {
                    final_match.group("p1"): int(final_match.group("b1")),
                    final_match.group("p2"): int(final_match.group("b2")),
                }
                continue
            round_match = ROUND_RE.match(line)
            if round_match:
                hands += 1
                current_street = "Preflop"
                round_awards = {}
                round_showdown = False
                round_end_stage = None
                if not players:
                    p1 = round_match.group("p1")
                    p2 = round_match.group("p2")
                    players = [p1, p2]
                    stats[p1] = _init_player_stats(p1)
                    stats[p2] = _init_player_stats(p2)
                continue
            street_match = STREET_RE.match(line)
            if street_match:
                current_street = street_match.group("street")
                continue
            show_match = SHOW_RE.match(line)
            if show_match:
                round_showdown = True
                continue
            action_match = ACTION_RE.match(line)
            if action_match:
                name = action_match.group("name")
                action = action_match.group("action")
                if name in stats:
                    action_key = action
                    if action == "raises to":
                        action_key = "raises"
                    street = current_street
                    if action_key == "discards":
                        if current_street == "Flop":
                            street = "Discard 1"
                        elif current_street == "Discard 1":
                            street = "Discard 2"
                    stats[name]["actions_by_street"][street][action_key] += 1
                if action == "folds" and round_end_stage is None:
                    round_end_stage = current_street
                continue
            award_match = AWARD_RE.match(line)
            if award_match:
                name = award_match.group("name")
                delta = int(award_match.group("delta"))
                if name in stats:
                    round_awards[name] = delta
                    if len(round_awards) == 2:
                        end_stage = round_end_stage or current_street
                        end_stage_counts[end_stage] += 1
                        for pname, pdelta in round_awards.items():
                            stats[pname]["total_net"] += pdelta
                            stats[pname]["net_by_stage"][end_stage] += pdelta
                            if round_showdown:
                                stats[pname]["showdown"]["hands"] += 1
                                stats[pname]["showdown"]["net"] += pdelta
                            else:
                                stats[pname]["nshow"]["hands"] += 1
                                stats[pname]["nshow"]["net"] += pdelta
                continue

    return {
        "path": os.path.abspath(path),
        "hands": hands,
        "players": players,
        "stats": stats,
        "final": final,
        "end_stage_counts": end_stage_counts,
    }


def _format_counter(counter, order):
    parts = []
    for key in order:
        value = counter.get(key, 0)
        if value:
            parts.append(f"{key} {value}")
    return ", ".join(parts) if parts else "none"


def _print_per_file(summary):
    name = os.path.basename(summary["path"])
    print(f"{name}")
    print(f"  hands: {summary['hands']}")
    if summary.get("final"):
        final = summary["final"]
        final_text = ", ".join(f"{p} {final[p]}" for p in summary["players"])
        print(f"  final: {final_text}")
    end_counts = _format_counter(summary["end_stage_counts"], STREETS)
    print(f"  end stages: {end_counts}")
    for pname in summary["players"]:
        pstats = summary["stats"][pname]
        show = pstats["showdown"]
        nsh = pstats["nshow"]
        print(
            f"  {pname} net: {pstats['total_net']} | showdown {show['hands']} ({show['net']})"
            f" | non-show {nsh['hands']} ({nsh['net']})"
        )
    print("")


def _merge_counters(target, source):
    for key, value in source.items():
        target[key] += value


def summarize(paths):
    per_file = []
    for path in paths:
        per_file.append(parse_scrim(path))

    aggregate = {
        "hands": 0,
        "players": [],
        "stats": {},
        "final": Counter(),
        "end_stage_counts": Counter(),
    }

    for summary in per_file:
        aggregate["hands"] += summary["hands"]
        _merge_counters(aggregate["end_stage_counts"], summary["end_stage_counts"])
        if not aggregate["players"]:
            aggregate["players"] = summary["players"]
            for pname in aggregate["players"]:
                aggregate["stats"][pname] = _init_player_stats(pname)
        if summary.get("final"):
            for pname, val in summary["final"].items():
                aggregate["final"][pname] += val
        for pname in aggregate["players"]:
            src = summary["stats"][pname]
            dst = aggregate["stats"][pname]
            dst["total_net"] += src["total_net"]
            _merge_counters(dst["net_by_stage"], src["net_by_stage"])
            dst["showdown"]["hands"] += src["showdown"]["hands"]
            dst["showdown"]["net"] += src["showdown"]["net"]
            dst["nshow"]["hands"] += src["nshow"]["hands"]
            dst["nshow"]["net"] += src["nshow"]["net"]
            for street in STREETS:
                _merge_counters(dst["actions_by_street"][street], src["actions_by_street"][street])

    return per_file, aggregate


def _print_actions_by_street(summary):
    for pname in summary["players"]:
        pstats = summary["stats"][pname]
        print(f"{pname} actions:")
        for street in STREETS:
            actions = pstats["actions_by_street"][street]
            total = sum(actions.values())
            if total == 0:
                continue
            parts = []
            for key in ACTION_KEYS:
                value = actions.get(key, 0)
                if value:
                    pct = (value / float(total)) * 100.0
                    parts.append(f"{key} {value} ({pct:.1f}%)")
            print(f"  {street}: total {total} | {', '.join(parts)}")
        print("")


def main():
    parser = argparse.ArgumentParser(description="Summarize scrim game logs.")
    parser.add_argument(
        "path",
        nargs="?",
        default="scrim_logs",
        help="Directory containing scrim logs (default: scrim_logs)",
    )
    parser.add_argument(
        "--pattern",
        default="game_log*",
        help="Glob pattern within the directory (default: game_log*)",
    )
    parser.add_argument("--json-out", default=None, help="Write summary JSON to path")
    parser.add_argument(
        "--no-per-file",
        action="store_true",
        help="Skip per-file summaries",
    )
    args = parser.parse_args()

    if os.path.isdir(args.path):
        root = args.path
        glob_pattern = os.path.join(root, args.pattern)
        paths = sorted([p for p in glob.glob(glob_pattern) if os.path.isfile(p)])
    else:
        paths = [args.path]

    if not paths:
        raise SystemExit("No logs found for pattern.")

    per_file, aggregate = summarize(paths)

    if not args.no_per_file:
        for summary in per_file:
            _print_per_file(summary)

    print("Aggregate")
    print(f"  logs: {len(per_file)} | hands: {aggregate['hands']}")
    if aggregate["final"]:
        final_text = ", ".join(f"{p} {aggregate['final'][p]}" for p in aggregate["players"])
        print(f"  final totals: {final_text}")
    end_counts = _format_counter(aggregate["end_stage_counts"], STREETS)
    print(f"  end stages: {end_counts}")
    for pname in aggregate["players"]:
        pstats = aggregate["stats"][pname]
        show = pstats["showdown"]
        nsh = pstats["nshow"]
        print(
            f"  {pname} net: {pstats['total_net']} | showdown {show['hands']} ({show['net']})"
            f" | non-show {nsh['hands']} ({nsh['net']})"
        )
    print("")

    _print_actions_by_street(aggregate)

    if args.json_out:
        payload = {
            "logs": [s["path"] for s in per_file],
            "aggregate": aggregate,
            "per_file": per_file,
        }
        with open(args.json_out, "w", encoding="utf-8") as handle:
            json.dump(_to_jsonable(payload), handle, indent=2, sort_keys=True)
            handle.write("\n")


def _to_jsonable(value):
    if isinstance(value, Counter):
        return dict(value)
    if isinstance(value, dict):
        return {key: _to_jsonable(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(item) for item in value]
    return value


if __name__ == "__main__":
    main()
