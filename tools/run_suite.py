#!/usr/bin/env python3
"""
Run a suite of matches with optional seat swaps and aggregate results.
"""
import argparse
import datetime as _dt
import json
import os
import subprocess
import sys


def _timestamp() -> str:
    return _dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def _run_match(cmd, cwd, timeout):
    try:
        proc = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        out, _ = proc.communicate(timeout=timeout)
        return proc.returncode, out.decode("utf-8", errors="replace"), False
    except subprocess.TimeoutExpired:
        proc.kill()
        out, _ = proc.communicate()
        return 124, out.decode("utf-8", errors="replace"), True


def _parse_summary(summary_path):
    with open(summary_path, "r") as handle:
        return json.load(handle)


def _aggregate(all_summaries, failures):
    aggregate = {
        "matches": len(all_summaries),
        "failures": failures,
        "players": {},
        "summaries": all_summaries,
    }
    for summary in all_summaries:
        for player in summary.get("players", []):
            name = player["name"]
            info = aggregate["players"].setdefault(name, {
                "hands": 0,
                "total_delta": 0,
                "wins": 0,
                "losses": 0,
                "ties": 0,
            })
            info["hands"] += player.get("hands", 0)
            info["total_delta"] += sum(player.get("deltas", []))
            info["wins"] += player.get("wins", 0)
            info["losses"] += player.get("losses", 0)
            info["ties"] += player.get("ties", 0)
    for name, info in aggregate["players"].items():
        hands = info["hands"]
        info["ev_per_hand"] = (info["total_delta"] / float(hands)) if hands else 0.0
        info["win_rate"] = (info["wins"] / float(hands)) if hands else 0.0
    return aggregate


def main():
    parser = argparse.ArgumentParser(description="Run a suite of matches and aggregate results.")
    parser.add_argument("--bot-a", required=True, help="Path to bot A")
    parser.add_argument("--bot-b", required=True, help="Path to bot B")
    parser.add_argument("--engine-dir", default="engine-2026", help="Engine directory")
    parser.add_argument("--rounds", type=int, default=1000, help="Rounds per match")
    parser.add_argument("--matches", type=int, default=2, help="Number of seeds to run")
    parser.add_argument("--seat-swaps", action="store_true", help="Run a second match swapping seats")
    parser.add_argument("--seed-start", type=int, default=1, help="Starting seed")
    parser.add_argument("--output-dir", default="runs", help="Output directory")
    parser.add_argument("--match-timeout", type=int, default=900, help="Timeout per match in seconds")
    args = parser.parse_args()

    base_dir = os.path.abspath(args.output_dir)
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    suite_dir = os.path.join(base_dir, "suite_%s" % _timestamp())
    os.makedirs(suite_dir)

    summaries = []
    failures = []
    run_match_script = os.path.join(os.path.dirname(__file__), "run_match.py")
    parse_script = os.path.join(os.path.dirname(__file__), "parse_gamelog.py")

    for i in range(args.matches):
        seed = args.seed_start + i
        match_dir = os.path.join(suite_dir, "match_%03d" % (i + 1))
        os.makedirs(match_dir)

        cmd = [
            sys.executable,
            run_match_script,
            "--engine-dir", args.engine_dir,
            "--bot-a", args.bot_a,
            "--bot-b", args.bot_b,
            "--rounds", str(args.rounds),
            "--seed", str(seed),
            "--output-dir", match_dir,
        ]
        code, output, timed_out = _run_match(cmd, cwd=suite_dir, timeout=args.match_timeout)
        with open(os.path.join(match_dir, "run_match_stdout.txt"), "w") as handle:
            handle.write(output)
        if code != 0:
            failures.append({
                "seed": seed,
                "swap": False,
                "return_code": code,
                "timed_out": timed_out,
                "match_dir": os.path.abspath(match_dir),
            })
            continue

        match_runs = [d for d in os.listdir(match_dir) if os.path.isdir(os.path.join(match_dir, d))]
        if not match_runs:
            continue
        match_runs.sort()
        run_path = os.path.join(match_dir, match_runs[-1])
        gamelog = os.path.join(run_path, "gamelog.txt")
        summary_path = os.path.join(run_path, "summary.json")
        parse_cmd = [sys.executable, parse_script, gamelog, "--json-out", summary_path]
        parse_code, parse_out, parse_timed_out = _run_match(parse_cmd, cwd=suite_dir, timeout=60)
        with open(os.path.join(run_path, "parse_stdout.txt"), "w") as handle:
            handle.write(parse_out)
        if parse_code == 0:
            summaries.append(_parse_summary(summary_path))
        else:
            failures.append({
                "seed": seed,
                "swap": False,
                "return_code": parse_code,
                "timed_out": parse_timed_out,
                "match_dir": os.path.abspath(run_path),
            })

        if args.seat_swaps:
            swap_dir = os.path.join(match_dir, "swap")
            os.makedirs(swap_dir)
            swap_cmd = [
                sys.executable,
                run_match_script,
                "--engine-dir", args.engine_dir,
                "--bot-a", args.bot_b,
                "--bot-b", args.bot_a,
                "--rounds", str(args.rounds),
                "--seed", str(seed),
                "--output-dir", swap_dir,
            ]
            swap_code, swap_out, swap_timed_out = _run_match(swap_cmd, cwd=suite_dir, timeout=args.match_timeout)
            with open(os.path.join(swap_dir, "run_match_stdout.txt"), "w") as handle:
                handle.write(swap_out)
            if swap_code != 0:
                failures.append({
                    "seed": seed,
                    "swap": True,
                    "return_code": swap_code,
                    "timed_out": swap_timed_out,
                    "match_dir": os.path.abspath(swap_dir),
                })
                continue
            swap_runs = [d for d in os.listdir(swap_dir) if os.path.isdir(os.path.join(swap_dir, d))]
            if not swap_runs:
                continue
            swap_runs.sort()
            swap_run_path = os.path.join(swap_dir, swap_runs[-1])
            swap_gamelog = os.path.join(swap_run_path, "gamelog.txt")
            swap_summary_path = os.path.join(swap_run_path, "summary.json")
            swap_parse_cmd = [sys.executable, parse_script, swap_gamelog, "--json-out", swap_summary_path]
            swap_parse_code, swap_parse_out, swap_parse_timed_out = _run_match(
                swap_parse_cmd, cwd=suite_dir, timeout=60
            )
            with open(os.path.join(swap_run_path, "parse_stdout.txt"), "w") as handle:
                handle.write(swap_parse_out)
            if swap_parse_code == 0:
                summaries.append(_parse_summary(swap_summary_path))
            else:
                failures.append({
                    "seed": seed,
                    "swap": True,
                    "return_code": swap_parse_code,
                    "timed_out": swap_parse_timed_out,
                    "match_dir": os.path.abspath(swap_run_path),
                })

    aggregate = _aggregate(summaries, failures)
    summary_path = os.path.join(suite_dir, "suite_summary.json")
    with open(summary_path, "w") as handle:
        json.dump(aggregate, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(aggregate, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
