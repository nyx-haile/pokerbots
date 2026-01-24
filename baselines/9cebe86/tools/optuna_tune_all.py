#!/usr/bin/env python3
"""
Run Optuna tuning across all key strategy parameters via self-play suites.
"""
import argparse
import datetime as _dt
import json
import os
import shutil
import subprocess
import sys
import tempfile

import optuna


def _timestamp() -> str:
    return _dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def _load_commands(path):
    with open(path, "r") as handle:
        return json.load(handle)


def _write_commands(path, commands):
    with open(path, "w") as handle:
        json.dump(commands, handle, indent=4, sort_keys=True)
        handle.write("\n")


def _prepare_bot_copy(bot_path, workspace):
    target = os.path.join(workspace, "bot_a")
    shutil.copytree(bot_path, target)
    return target


def _latest_suite_summary(root):
    if not os.path.isdir(root):
        return None
    suites = [d for d in os.listdir(root) if d.startswith("suite_")]
    if not suites:
        return None
    suites.sort()
    candidate = os.path.join(root, suites[-1], "suite_summary.json")
    if os.path.exists(candidate):
        return candidate
    return None


def _run_suite(args, bot_a_path, trial_dir):
    suite_script = os.path.join(os.path.dirname(__file__), "run_suite.py")
    cmd = [
        sys.executable,
        suite_script,
        "--bot-a", bot_a_path,
        "--bot-b", args.bot_b,
        "--engine-dir", args.engine_dir,
        "--rounds", str(args.rounds),
        "--matches", str(args.matches),
        "--seed-start", str(args.seed_start),
        "--output-dir", trial_dir,
        "--match-timeout", str(args.match_timeout),
    ]
    if args.engine_python:
        cmd.extend(["--engine-python", args.engine_python])
    if args.bot_python:
        cmd.extend(["--bot-python", args.bot_python])
    if args.seat_swaps:
        cmd.append("--seat-swaps")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=args.cwd)
    out, _ = proc.communicate()
    return proc.returncode, out.decode("utf-8", errors="replace")


def _score_from_summary(summary_path, player_name):
    with open(summary_path, "r") as handle:
        summary = json.load(handle)
    players = summary.get("players", {})
    entry = players.get(player_name)
    if not entry:
        return None
    return float(entry.get("ev_per_hand", 0.0))


def _score_from_output(output, player_name):
    start = output.find("{")
    end = output.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        summary = json.loads(output[start : end + 1])
    except json.JSONDecodeError:
        return None
    players = summary.get("players", {})
    entry = players.get(player_name)
    if not entry:
        return None
    return float(entry.get("ev_per_hand", 0.0))


def main():
    parser = argparse.ArgumentParser(description="Tune all strategy params with Optuna.")
    parser.add_argument("--bot-a", required=True, help="Path to tuned bot")
    parser.add_argument("--bot-b", required=True, help="Path to baseline bot")
    parser.add_argument("--engine-dir", default="engine-2026", help="Engine directory")
    parser.add_argument("--engine-python", default=None, help="Python executable for engine")
    parser.add_argument("--bot-python", default=None, help="Python executable for bots")
    parser.add_argument("--rounds", type=int, default=1000, help="Rounds per match")
    parser.add_argument("--matches", type=int, default=2, help="Seeds per trial")
    parser.add_argument("--seed-start", type=int, default=1, help="Starting seed")
    parser.add_argument("--trials", type=int, default=20, help="Optuna trials")
    parser.add_argument("--study-name", default=None, help="Study name")
    parser.add_argument("--storage", default=None, help="Optuna storage URL")
    parser.add_argument("--output-dir", default="tuning", help="Output directory")
    parser.add_argument("--match-timeout", type=int, default=900, help="Match timeout in seconds")
    parser.add_argument("--seat-swaps", action="store_true", help="Enable seat swaps")
    args = parser.parse_args()

    args.cwd = os.getcwd()
    args.engine_dir = os.path.abspath(args.engine_dir)
    bot_a = os.path.abspath(args.bot_a)
    bot_b = os.path.abspath(args.bot_b)
    args.bot_a = bot_a
    args.bot_b = bot_b

    if args.seat_swaps:
        raise SystemExit("Seat swaps are not supported for tuning (player names collide).")
    if not os.path.isdir(bot_a):
        raise SystemExit("bot-a does not exist: %s" % bot_a)
    if not os.path.isdir(bot_b):
        raise SystemExit("bot-b does not exist: %s" % bot_b)

    output_root = os.path.abspath(args.output_dir)
    if not os.path.exists(output_root):
        os.makedirs(output_root)
    tune_dir = os.path.join(output_root, "optuna_all_%s" % _timestamp())
    os.makedirs(tune_dir)

    workspace = tempfile.mkdtemp(prefix="optuna_tune_all_")
    tuned_bot = _prepare_bot_copy(bot_a, workspace)
    commands_path = os.path.join(tuned_bot, "commands.json")
    base_commands = _load_commands(commands_path)
    base_run = base_commands.get("run", [])

    def objective(trial):
        raise_light = trial.suggest_float("raise_light", 0.2, 0.6)
        raise_medium = trial.suggest_float("raise_medium", raise_light, 0.75)
        raise_strong = trial.suggest_float("raise_strong", raise_medium, 0.85)
        call_light = trial.suggest_float("call_light", 0.12, 0.5)
        call_medium = trial.suggest_float("call_medium", call_light, 0.65)
        call_strong = trial.suggest_float("call_strong", call_medium, 0.8)
        raise_size_light = trial.suggest_float("raise_size_light", 0.25, 0.7)
        raise_size_medium = trial.suggest_float("raise_size_medium", raise_size_light, 1.0)
        raise_size_strong = trial.suggest_float("raise_size_strong", raise_size_medium, 1.4)
        bluff_raise_frac = trial.suggest_float("bluff_raise_frac", 0.2, 0.6)
        raise_margin_pre = trial.suggest_float("raise_margin_pre", 0.05, 0.35)
        raise_margin_post = trial.suggest_float("raise_margin_post", 0.05, 0.25)
        raise_margin_turn = trial.suggest_float("raise_margin_turn", 0.08, 0.3)
        raise_margin_river = trial.suggest_float("raise_margin_river", 0.08, 0.3)
        call_margin_pre = trial.suggest_float("call_margin_pre", 0.0, 0.15)
        call_margin_post = trial.suggest_float("call_margin_post", 0.0, 0.12)
        call_margin_turn = trial.suggest_float("call_margin_turn", 0.0, 0.15)
        call_margin_river = trial.suggest_float("call_margin_river", 0.0, 0.18)
        raise_call_ratio = trial.suggest_float("raise_call_ratio", 0.3, 1.1)
        raise_call_penalty = trial.suggest_float("raise_call_penalty", 0.0, 0.2)
        turn_raise_ratio = trial.suggest_float("turn_raise_ratio", 0.3, 1.1)
        turn_raise_extra = trial.suggest_float("turn_raise_extra", 0.0, 0.2)
        turn_raise_sample_mult = trial.suggest_float("turn_raise_sample_mult", 1.0, 3.0)
        turn_raise_time_mult = trial.suggest_float("turn_raise_time_mult", 1.0, 3.0)

        env_run = [
            "env",
            "NEUROPOKER_PREFLOP_RAISE_THRESHOLDS=%.4f,%.4f,%.4f" % (
                raise_strong,
                raise_medium,
                raise_light,
            ),
            "NEUROPOKER_PREFLOP_CALL_THRESHOLDS=%.4f,%.4f,%.4f" % (
                call_strong,
                call_medium,
                call_light,
            ),
            "NEUROPOKER_RAISE_SIZE_FRACTIONS=%.4f,%.4f,%.4f" % (
                raise_size_strong,
                raise_size_medium,
                raise_size_light,
            ),
            "NEUROPOKER_BLUFF_RAISE_FRACTION=%.4f" % bluff_raise_frac,
            "NEUROPOKER_RAISE_MARGIN_BY_STREET=%.4f,%.4f,%.4f,%.4f" % (
                raise_margin_pre,
                raise_margin_post,
                raise_margin_turn,
                raise_margin_river,
            ),
            "NEUROPOKER_CALL_MARGIN_BY_STREET=%.4f,%.4f,%.4f,%.4f" % (
                call_margin_pre,
                call_margin_post,
                call_margin_turn,
                call_margin_river,
            ),
            "NEUROPOKER_RAISE_CALL_RATIO=%.4f" % raise_call_ratio,
            "NEUROPOKER_RAISE_CALL_PENALTY=%.4f" % raise_call_penalty,
            "NEUROPOKER_TURN_RAISE_RATIO=%.4f" % turn_raise_ratio,
            "NEUROPOKER_TURN_RAISE_EXTRA=%.4f" % turn_raise_extra,
            "NEUROPOKER_TURN_RAISE_SAMPLE_MULT=%.4f" % turn_raise_sample_mult,
            "NEUROPOKER_TURN_RAISE_TIME_MULT=%.4f" % turn_raise_time_mult,
        ] + list(base_run)
        tuned_commands = dict(base_commands)
        tuned_commands["run"] = env_run
        _write_commands(commands_path, tuned_commands)

        trial_dir = os.path.join(tune_dir, "trial_%03d" % trial.number)
        os.makedirs(trial_dir)
        code, output = _run_suite(args, tuned_bot, trial_dir)
        with open(os.path.join(trial_dir, "suite_stdout.txt"), "w") as handle:
            handle.write(output)
        if code != 0:
            trial.set_user_attr("return_code", code)
            return -1e9

        summary_path = _latest_suite_summary(trial_dir)
        score = None
        if summary_path:
            score = _score_from_summary(summary_path, "Bot A")
        if score is None:
            score = _score_from_output(output, "Bot A")
        if score is None:
            trial.set_user_attr("summary_path", summary_path or "")
            return -1e9
        trial.set_user_attr("summary_path", summary_path)
        return score

    study = optuna.create_study(
        direction="maximize",
        study_name=args.study_name,
        storage=args.storage,
        load_if_exists=bool(args.storage),
    )
    study.optimize(objective, n_trials=args.trials)

    result_path = os.path.join(tune_dir, "best_params.json")
    with open(result_path, "w") as handle:
        json.dump(
            {
                "best_value": study.best_value,
                "best_params": study.best_params,
                "study_name": study.study_name,
            },
            handle,
            indent=2,
            sort_keys=True,
        )
        handle.write("\n")

    print("Best value: %.6f" % study.best_value)
    print(json.dumps(study.best_params, indent=2, sort_keys=True))
    print("Results: %s" % result_path)


if __name__ == "__main__":
    main()
