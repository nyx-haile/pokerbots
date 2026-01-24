#!/usr/bin/env python3
"""
Evaluate a bot against multiple baselines using the suite runner.
"""
import argparse
import datetime as _dt
import json
import os
import subprocess
import sys


def _timestamp() -> str:
    return _dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def _load_manifest(path):
    with open(path, "r") as handle:
        return json.load(handle)


def _run_suite(cmd, cwd):
    proc = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    out, _ = proc.communicate()
    return proc.returncode, out.decode("utf-8", errors="replace")


def _ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def _build_cmd(args, bot_a, bot_b, output_dir, seed_start):
    suite_script = os.path.join(os.path.dirname(__file__), "run_suite.py")
    cmd = [
        sys.executable,
        suite_script,
        "--bot-a", bot_a,
        "--bot-b", bot_b,
        "--engine-dir", args.engine_dir,
        "--rounds", str(args.rounds),
        "--matches", str(args.matches),
        "--seed-start", str(seed_start),
        "--output-dir", output_dir,
    ]
    if args.seat_swaps:
        cmd.append("--seat-swaps")
    return cmd


def main():
    parser = argparse.ArgumentParser(description="Run evaluation suites vs baselines.")
    parser.add_argument("--manifest", default=None, help="Path to evaluation manifest JSON")
    parser.add_argument("--bot", default=None, help="Path to current bot")
    parser.add_argument("--bot-name", default="Current", help="Display name for current bot")
    parser.add_argument("--opponent", action="append", default=[], help="Path to opponent bot")
    parser.add_argument("--opponent-name", action="append", default=[], help="Opponent display name")
    parser.add_argument("--engine-dir", default="engine-2026", help="Engine directory")
    parser.add_argument("--rounds", type=int, default=1000, help="Rounds per match")
    parser.add_argument("--matches", type=int, default=2, help="Seeds per opponent")
    parser.add_argument("--seat-swaps", action="store_true", help="Enable seat swaps")
    parser.add_argument("--seed-start", type=int, default=1, help="Starting seed")
    parser.add_argument("--output-dir", default="evaluations", help="Output directory")
    args = parser.parse_args()

    if args.manifest:
        manifest = _load_manifest(args.manifest)
        args.engine_dir = manifest.get("engine_dir", args.engine_dir)
        args.rounds = int(manifest.get("rounds", args.rounds))
        args.matches = int(manifest.get("matches", args.matches))
        args.seat_swaps = bool(manifest.get("seat_swaps", args.seat_swaps))
        args.seed_start = int(manifest.get("seed_start", args.seed_start))
        args.output_dir = manifest.get("output_dir", args.output_dir)
        current = manifest.get("current", {})
        args.bot = current.get("path", args.bot)
        args.bot_name = current.get("name", args.bot_name)
        args.opponent = [entry["path"] for entry in manifest.get("baselines", [])]
        args.opponent_name = [entry.get("name", "Baseline") for entry in manifest.get("baselines", [])]

    if not args.bot:
        raise SystemExit("--bot or --manifest is required")
    if not args.opponent:
        raise SystemExit("at least one --opponent or manifest baseline is required")

    eval_root = os.path.abspath(args.output_dir)
    _ensure_dir(eval_root)
    eval_dir = os.path.join(eval_root, "eval_%s" % _timestamp())
    os.makedirs(eval_dir)

    summary = {
        "current": {"name": args.bot_name, "path": os.path.abspath(args.bot)},
        "engine_dir": os.path.abspath(args.engine_dir),
        "rounds": args.rounds,
        "matches": args.matches,
        "seat_swaps": args.seat_swaps,
        "seed_start": args.seed_start,
        "evaluations": [],
    }

    seed_cursor = args.seed_start
    for idx, opponent in enumerate(args.opponent):
        name = args.opponent_name[idx] if idx < len(args.opponent_name) else "Baseline"
        run_dir = os.path.join(eval_dir, "vs_%02d" % (idx + 1))
        os.makedirs(run_dir)
        cmd = _build_cmd(args, args.bot, opponent, run_dir, seed_cursor)
        code, output = _run_suite(cmd, cwd=eval_dir)
        with open(os.path.join(run_dir, "suite_stdout.txt"), "w") as handle:
            handle.write(output)
        record = {
            "opponent": {"name": name, "path": os.path.abspath(opponent)},
            "return_code": code,
            "suite_dir": os.path.abspath(run_dir),
        }
        summary_path = os.path.join(run_dir, "suite_summary.json")
        if code == 0 and os.path.exists(summary_path):
            with open(summary_path, "r") as handle:
                record["summary"] = json.load(handle)
        summary["evaluations"].append(record)
        seed_cursor += args.matches

    with open(os.path.join(eval_dir, "evaluation_summary.json"), "w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
