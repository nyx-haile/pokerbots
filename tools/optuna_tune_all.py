#!/usr/bin/env python3
"""
Run Optuna tuning across all key strategy parameters via self-play suites.
"""
import argparse
import datetime as _dt
import json
import os
import random
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


def _run_suite(args, bot_a_path, bot_b_path, trial_dir):
    suite_script = os.path.join(os.path.dirname(__file__), "run_suite.py")
    cmd = [
        sys.executable,
        suite_script,
        "--bot-a", bot_a_path,
        "--bot-b", bot_b_path,
        "--engine-dir", args.engine_dir,
        "--rounds", str(args.rounds),
        "--matches", str(args.matches),
        "--output-dir", trial_dir,
        "--match-timeout", str(args.match_timeout),
    ]
    if args.bot_b_per_match and args.bot_b_pool:
        cmd.extend(["--bot-b-pool", ",".join(args.bot_b_pool)])
        cmd.extend(["--bot-b-pool-mode", args.bot_b_pool_mode])
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


def _apply_best_params(bot_path, best_params):
    output_path = os.path.join(bot_path, "best_params.json")
    payload = {"best_params": best_params}
    if os.path.exists(output_path):
        try:
            with open(output_path, "r") as handle:
                existing = json.load(handle)
            if isinstance(existing, dict):
                current = existing.get("best_params", existing)
                if isinstance(current, dict):
                    current.update(best_params)
                    payload = {"best_params": current}
        except (OSError, ValueError, TypeError):
            pass
    with open(output_path, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def main():
    parser = argparse.ArgumentParser(description="Tune all strategy params with Optuna.")
    parser.add_argument("--bot-a", required=True, help="Path to tuned bot")
    parser.add_argument("--bot-b", required=True, help="Path to baseline bot")
    parser.add_argument("--engine-dir", default="engine-2026", help="Engine directory")
    parser.add_argument("--engine-python", default=None, help="Python executable for engine")
    parser.add_argument("--bot-python", default=None, help="Python executable for bots")
    parser.add_argument("--rounds", type=int, default=1000, help="Rounds per match")
    parser.add_argument("--matches", type=int, default=2, help="Seeds per trial")
    parser.add_argument("--trials", type=int, default=50, help="Optuna trials")
    parser.add_argument("--study-name", default=None, help="Study name")
    parser.add_argument("--storage", default=None, help="Optuna storage URL")
    parser.add_argument("--output-dir", default="tuning", help="Output directory")
    parser.add_argument("--match-timeout", type=int, default=900, help="Match timeout in seconds")
    parser.add_argument("--seat-swaps", action="store_true", help="Enable seat swaps")
    parser.add_argument(
        "--bot-b-pool",
        default=None,
        help="Comma-separated list of bot-b paths; all are evaluated each trial (includes --bot-b).",
    )
    parser.add_argument(
        "--bot-b-per-match",
        action="store_true",
        help="Rotate bot-b from the pool per match instead of running one suite per bot.",
    )
    parser.add_argument(
        "--bot-b-pool-mode",
        default="cycle",
        choices=["cycle", "random"],
        help="Pool selection mode when using --bot-b-per-match.",
    )
    parser.add_argument(
        "--bot-b-weights",
        default=None,
        help="Comma-separated weights for bot-b pool (same order as pool).",
    )
    parser.add_argument(
        "--batch",
        default="all",
        choices=[
            "all",
            "preflop",
            "margins",
            "turn_defense",
            "bluff",
            "aggression",
            "fold_posture",
            "policy",
        ],
        help="Parameter batch to tune",
    )
    parser.add_argument(
        "--apply-best",
        action="store_true",
        help="Write best_params.json into bot-a after tuning",
    )
    args = parser.parse_args()

    args.cwd = os.getcwd()
    args.engine_dir = os.path.abspath(args.engine_dir)
    bot_a = os.path.abspath(args.bot_a)
    bot_b = os.path.abspath(args.bot_b)
    args.bot_a = bot_a
    args.bot_b = bot_b
    bot_b_pool = None
    bot_b_weights = None
    if args.bot_b_pool:
        pool = [bot_b]
        for entry in args.bot_b_pool.split(","):
            entry = entry.strip()
            if entry:
                pool.append(os.path.abspath(entry))
        bot_b_pool = []
        seen = set()
        for entry in pool:
            if entry not in seen:
                bot_b_pool.append(entry)
                seen.add(entry)
        if args.bot_b_weights:
            weights = [w.strip() for w in args.bot_b_weights.split(",") if w.strip()]
            try:
                bot_b_weights = [float(w) for w in weights]
            except ValueError:
                raise SystemExit("bot-b-weights must be numeric floats")
            if len(bot_b_weights) != len(bot_b_pool):
                raise SystemExit("bot-b-weights must match bot-b pool length")
        else:
            bot_b_weights = [1.0] * len(bot_b_pool)
        if args.bot_b_per_match and args.bot_b_weights:
            raise SystemExit("bot-b-weights are not supported with --bot-b-per-match")
    args.bot_b_pool = bot_b_pool
    args.bot_b_weights = bot_b_weights

    if args.seat_swaps:
        raise SystemExit("Seat swaps are not supported for tuning (player names collide).")
    if not os.path.isdir(bot_a):
        raise SystemExit("bot-a does not exist: %s" % bot_a)
    if args.bot_b_pool:
        for path in args.bot_b_pool:
            if not os.path.isdir(path):
                raise SystemExit("bot-b does not exist: %s" % path)
    elif not os.path.isdir(bot_b):
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
        env_vars = ["NEUROPOKER_ENABLE_LOCK_WIN=0"]
        batch = args.batch

        if batch in ("all", "preflop"):
            raise_light = trial.suggest_float("raise_light", 0.2, 0.6)
            raise_medium = trial.suggest_float("raise_medium", raise_light, 0.75)
            raise_strong = trial.suggest_float("raise_strong", raise_medium, 0.85)
            call_light = trial.suggest_float("call_light", 0.12, 0.5)
            call_medium = trial.suggest_float("call_medium", call_light, 0.65)
            call_strong = trial.suggest_float("call_strong", call_medium, 0.8)
            env_vars.append(
                "NEUROPOKER_PREFLOP_RAISE_THRESHOLDS=%.4f,%.4f,%.4f"
                % (raise_strong, raise_medium, raise_light)
            )
            env_vars.append(
                "NEUROPOKER_PREFLOP_CALL_THRESHOLDS=%.4f,%.4f,%.4f"
                % (call_strong, call_medium, call_light)
            )

        if batch in ("all", "margins"):
            raise_margin_pre = trial.suggest_float("raise_margin_pre", 0.05, 0.35)
            raise_margin_post = trial.suggest_float("raise_margin_post", 0.05, 0.25)
            raise_margin_turn = trial.suggest_float("raise_margin_turn", 0.08, 0.3)
            raise_margin_river = trial.suggest_float("raise_margin_river", 0.08, 0.3)
            call_margin_pre = trial.suggest_float("call_margin_pre", 0.0, 0.15)
            call_margin_post = trial.suggest_float("call_margin_post", 0.0, 0.12)
            call_margin_turn = trial.suggest_float("call_margin_turn", 0.0, 0.15)
            call_margin_river = trial.suggest_float("call_margin_river", 0.0, 0.18)
            env_vars.append(
                "NEUROPOKER_RAISE_MARGIN_BY_STREET=%.4f,%.4f,%.4f,%.4f"
                % (raise_margin_pre, raise_margin_post, raise_margin_turn, raise_margin_river)
            )
            env_vars.append(
                "NEUROPOKER_CALL_MARGIN_BY_STREET=%.4f,%.4f,%.4f,%.4f"
                % (call_margin_pre, call_margin_post, call_margin_turn, call_margin_river)
            )

        if batch in ("all", "turn_defense"):
            raise_call_ratio = trial.suggest_float("raise_call_ratio", 0.3, 1.1)
            raise_call_penalty = trial.suggest_float("raise_call_penalty", 0.0, 0.2)
            turn_raise_ratio = trial.suggest_float("turn_raise_ratio", 0.3, 1.1)
            turn_raise_extra = trial.suggest_float("turn_raise_extra", 0.0, 0.2)
            turn_raise_sample_mult = trial.suggest_float("turn_raise_sample_mult", 1.0, 3.0)
            turn_raise_time_mult = trial.suggest_float("turn_raise_time_mult", 1.0, 3.0)
            env_vars.append("NEUROPOKER_RAISE_CALL_RATIO=%.4f" % raise_call_ratio)
            env_vars.append("NEUROPOKER_RAISE_CALL_PENALTY=%.4f" % raise_call_penalty)
            env_vars.append("NEUROPOKER_TURN_RAISE_RATIO=%.4f" % turn_raise_ratio)
            env_vars.append("NEUROPOKER_TURN_RAISE_EXTRA=%.4f" % turn_raise_extra)
            env_vars.append("NEUROPOKER_TURN_RAISE_SAMPLE_MULT=%.4f" % turn_raise_sample_mult)
            env_vars.append("NEUROPOKER_TURN_RAISE_TIME_MULT=%.4f" % turn_raise_time_mult)

        if batch in ("all", "bluff"):
            discard_bluff_rate = trial.suggest_float("discard_bluff_rate", 0.0, 0.03)
            discard_bluff_raise_rate = trial.suggest_float("discard_bluff_raise_rate", 0.2, 0.9)
            discard_bluff_raise_fraction = trial.suggest_float("discard_bluff_raise_fraction", 0.4, 1.0)
            bluff_weakness_threshold = trial.suggest_float("bluff_weakness_threshold", 0.02, 0.2)
            bluff_disable_behind = trial.suggest_float("bluff_disable_behind", 0.0, 800.0)
            env_vars.append("NEUROPOKER_DISCARD_BLUFF_RATE=%.4f" % discard_bluff_rate)
            env_vars.append("NEUROPOKER_DISCARD_BLUFF_RAISE_RATE=%.4f" % discard_bluff_raise_rate)
            env_vars.append(
                "NEUROPOKER_DISCARD_BLUFF_RAISE_FRACTION=%.4f" % discard_bluff_raise_fraction
            )
            env_vars.append("NEUROPOKER_BLUFF_WEAKNESS_THRESHOLD=%.4f" % bluff_weakness_threshold)
            env_vars.append("NEUROPOKER_BLUFF_DISABLE_BEHIND=%.2f" % bluff_disable_behind)

        if batch in ("all", "aggression"):
            raise_size_light = trial.suggest_float("raise_size_light", 0.25, 0.7)
            raise_size_medium = trial.suggest_float("raise_size_medium", raise_size_light, 1.0)
            raise_size_strong = trial.suggest_float("raise_size_strong", raise_size_medium, 1.4)
            aggro_equity = trial.suggest_float("aggro_equity", 0.55, 0.8)
            aggro_raise_bonus = trial.suggest_float("aggro_raise_bonus", 0.0, 0.1)
            nut_raise_equity = trial.suggest_float("nut_raise_equity", 0.75, 0.95)
            pressure_equity_threshold = trial.suggest_float("pressure_equity_threshold", 0.55, 0.75)
            pressure_raise_bonus = trial.suggest_float("pressure_raise_bonus", 0.0, 0.12)
            pressure_foldrate_min = trial.suggest_float("pressure_foldrate_min", 0.0, 0.4)
            env_vars.append(
                "NEUROPOKER_RAISE_SIZE_FRACTIONS=%.4f,%.4f,%.4f"
                % (raise_size_strong, raise_size_medium, raise_size_light)
            )
            env_vars.append("NEUROPOKER_AGGRO_EQUITY=%.4f" % aggro_equity)
            env_vars.append("NEUROPOKER_AGGRO_RAISE_BONUS=%.4f" % aggro_raise_bonus)
            env_vars.append("NEUROPOKER_NUT_RAISE_EQUITY=%.4f" % nut_raise_equity)
            env_vars.append("NEUROPOKER_PRESSURE_EQUITY_THRESHOLD=%.4f" % pressure_equity_threshold)
            env_vars.append("NEUROPOKER_PRESSURE_RAISE_BONUS=%.4f" % pressure_raise_bonus)
            env_vars.append("NEUROPOKER_PRESSURE_FOLDRATE_MIN=%.4f" % pressure_foldrate_min)

        if batch in ("all", "fold_posture"):
            hard_fold_post = trial.suggest_float("hard_fold_post", 0.15, 0.35)
            hard_fold_turn = trial.suggest_float("hard_fold_turn", 0.2, 0.4)
            hard_fold_river = trial.suggest_float("hard_fold_river", 0.25, 0.45)
            hard_fold_pot_odds_min = trial.suggest_float("hard_fold_pot_odds_min", 0.0, 0.15)
            fold_bias_pre = trial.suggest_float("fold_bias_pre", 0.0, 0.08)
            fold_bias_post = trial.suggest_float("fold_bias_post", 0.0, 0.08)
            fold_bias_turn = trial.suggest_float("fold_bias_turn", 0.0, 0.1)
            fold_bias_river = trial.suggest_float("fold_bias_river", 0.0, 0.12)
            env_vars.append(
                "NEUROPOKER_HARD_FOLD_EQUITY_BY_STREET=%.4f,%.4f,%.4f"
                % (hard_fold_post, hard_fold_turn, hard_fold_river)
            )
            env_vars.append("NEUROPOKER_HARD_FOLD_POT_ODDS_MIN=%.4f" % hard_fold_pot_odds_min)
            env_vars.append(
                "NEUROPOKER_FOLD_BIAS_BY_STREET=%.4f,%.4f,%.4f,%.4f"
                % (fold_bias_pre, fold_bias_post, fold_bias_turn, fold_bias_river)
            )

        if batch in ("all", "policy"):
            tight_equity_threshold = trial.suggest_float("tight_equity_threshold", 0.5, 0.75, step=0.05)
            tight_fold_lr = trial.suggest_float("tight_fold_lr", 0.0, 0.35, step=0.05)
            env_vars.append("NEUROPOKER_TIGHT_EQUITY_THRESHOLD=%.2f" % tight_equity_threshold)
            env_vars.append("NEUROPOKER_TIGHT_FOLD_LR=%.2f" % tight_fold_lr)

        if batch in ("all", "lead_protection"):
            # Lead protection thresholds (coarse grid for faster convergence)
            lead_prot_t1 = trial.suggest_float("lead_prot_t1", 0.15, 0.30, step=0.05)
            lead_prot_t2 = trial.suggest_float("lead_prot_t2", 0.35, 0.50, step=0.05)
            lead_prot_t3 = trial.suggest_float("lead_prot_t3", 0.55, 0.70, step=0.05)
            lead_prot_t4 = trial.suggest_float("lead_prot_t4", 0.75, 0.90, step=0.05)
            # Adjustments
            lead_prot_a1 = trial.suggest_float("lead_prot_a1", 0.01, 0.04, step=0.01)
            lead_prot_a2 = trial.suggest_float("lead_prot_a2", 0.03, 0.07, step=0.01)
            lead_prot_a3 = trial.suggest_float("lead_prot_a3", 0.05, 0.10, step=0.01)
            lead_prot_a4 = trial.suggest_float("lead_prot_a4", 0.08, 0.14, step=0.02)
            # Size multipliers (decreasing)
            lead_prot_s1 = trial.suggest_float("lead_prot_s1", 0.85, 0.95, step=0.05)
            lead_prot_s2 = trial.suggest_float("lead_prot_s2", 0.75, 0.85, step=0.05)
            lead_prot_s3 = trial.suggest_float("lead_prot_s3", 0.65, 0.75, step=0.05)
            lead_prot_s4 = trial.suggest_float("lead_prot_s4", 0.50, 0.65, step=0.05)
            lead_prot_pot_factor = trial.suggest_float("lead_prot_pot_factor", 0.0, 0.5, step=0.1)
            env_vars.append(
                "NEUROPOKER_LEAD_PROT_THRESHOLDS=%.2f,%.2f,%.2f,%.2f"
                % (lead_prot_t1, lead_prot_t2, lead_prot_t3, lead_prot_t4)
            )
            env_vars.append(
                "NEUROPOKER_LEAD_PROT_ADJUSTMENTS=%.2f,%.2f,%.2f,%.2f"
                % (lead_prot_a1, lead_prot_a2, lead_prot_a3, lead_prot_a4)
            )
            env_vars.append(
                "NEUROPOKER_LEAD_PROT_SIZE_MULTS=%.2f,%.2f,%.2f,%.2f"
                % (lead_prot_s1, lead_prot_s2, lead_prot_s3, lead_prot_s4)
            )
            env_vars.append("NEUROPOKER_LEAD_PROT_POT_FACTOR=%.2f" % lead_prot_pot_factor)

        if batch in ("all", "opponent_model"):
            opp_passive_threshold = trial.suggest_float("opp_passive_threshold", 0.10, 0.35, step=0.05)
            opp_station_call_rate = trial.suggest_float("opp_station_call_rate", 0.40, 0.60, step=0.05)
            opp_station_fold_rate = trial.suggest_float("opp_station_fold_rate", 0.15, 0.35, step=0.05)
            opp_passive_value_mult = trial.suggest_float("opp_passive_value_mult", 1.0, 1.4, step=0.1)
            opp_station_value_mult = trial.suggest_float("opp_station_value_mult", 1.2, 1.6, step=0.1)
            env_vars.append("NEUROPOKER_OPP_PASSIVE_THRESHOLD=%.2f" % opp_passive_threshold)
            env_vars.append("NEUROPOKER_OPP_STATION_CALL_RATE=%.2f" % opp_station_call_rate)
            env_vars.append("NEUROPOKER_OPP_STATION_FOLD_RATE=%.2f" % opp_station_fold_rate)
            env_vars.append("NEUROPOKER_OPP_PASSIVE_VALUE_MULT=%.2f" % opp_passive_value_mult)
            env_vars.append("NEUROPOKER_OPP_STATION_VALUE_MULT=%.2f" % opp_station_value_mult)

        if batch in ("all", "early_boost"):
            early_boost_r1 = trial.suggest_int("early_boost_r1", 50, 150, step=25)
            early_boost_r2 = trial.suggest_int("early_boost_r2", 150, 300, step=25)
            early_boost_m1 = trial.suggest_float("early_boost_m1", 1.2, 1.6, step=0.1)
            early_boost_m2 = trial.suggest_float("early_boost_m2", 1.1, 1.4, step=0.1)
            env_vars.append("NEUROPOKER_EARLY_BOOST_ROUNDS=%d,%d" % (early_boost_r1, early_boost_r2))
            env_vars.append("NEUROPOKER_EARLY_BOOST_MULTS=%.2f,%.2f" % (early_boost_m1, early_boost_m2))

        if batch in ("all", "desperate"):
            desperate_nut_threshold = trial.suggest_float("desperate_nut_threshold", 0.65, 0.80, step=0.05)
            desperate_raise_margin = trial.suggest_float("desperate_raise_margin", 0.06, 0.16, step=0.02)
            desperate_call_penalty = trial.suggest_float("desperate_call_penalty", 0.04, 0.12, step=0.02)
            env_vars.append("NEUROPOKER_DESPERATE_NUT_THRESHOLD=%.2f" % desperate_nut_threshold)
            env_vars.append("NEUROPOKER_DESPERATE_RAISE_MARGIN=%.2f" % desperate_raise_margin)
            env_vars.append("NEUROPOKER_DESPERATE_CALL_PENALTY=%.2f" % desperate_call_penalty)

        env_run = ["env"] + env_vars + list(base_run)
        tuned_commands = dict(base_commands)
        tuned_commands["run"] = env_run
        _write_commands(commands_path, tuned_commands)

        trial_dir = os.path.join(tune_dir, "trial_%03d" % trial.number)
        os.makedirs(trial_dir)
        results = []
        bot_b_pool = args.bot_b_pool or [args.bot_b]
        bot_b_weights = args.bot_b_weights or [1.0] * len(bot_b_pool)
        if args.bot_b_pool and args.bot_b_per_match:
            run_dir = os.path.join(trial_dir, "suite")
            os.makedirs(run_dir)
            with open(os.path.join(run_dir, "bot_b.txt"), "w") as handle:
                handle.write(",".join(bot_b_pool) + "\n")
            code, output = _run_suite(args, tuned_bot, args.bot_b, run_dir)
            with open(os.path.join(run_dir, "suite_stdout.txt"), "w") as handle:
                handle.write(output)
            if code != 0:
                trial.set_user_attr("return_code", code)
                trial.set_user_attr("bot_b_pool", bot_b_pool)
                return -1e9
            summary_path = _latest_suite_summary(run_dir)
            score = None
            if summary_path:
                score = _score_from_summary(summary_path, "Bot A")
            if score is None:
                score = _score_from_output(output, "Bot A")
            if score is None:
                trial.set_user_attr("summary_path", summary_path or "")
                trial.set_user_attr("bot_b_pool", bot_b_pool)
                return -1e9
            trial.set_user_attr("bot_b_pool_mode", args.bot_b_pool_mode)
            return score

        for idx, bot_b_path in enumerate(bot_b_pool):
            run_dir = os.path.join(trial_dir, "bot_%02d" % (idx + 1))
            os.makedirs(run_dir)
            with open(os.path.join(run_dir, "bot_b.txt"), "w") as handle:
                handle.write(bot_b_path + "\n")
            code, output = _run_suite(args, tuned_bot, bot_b_path, run_dir)
            with open(os.path.join(run_dir, "suite_stdout.txt"), "w") as handle:
                handle.write(output)
            if code != 0:
                trial.set_user_attr("return_code", code)
                trial.set_user_attr("bot_b", bot_b_path)
                return -1e9
            summary_path = _latest_suite_summary(run_dir)
            score = None
            if summary_path:
                score = _score_from_summary(summary_path, "Bot A")
            if score is None:
                score = _score_from_output(output, "Bot A")
            if score is None:
                trial.set_user_attr("summary_path", summary_path or "")
                trial.set_user_attr("bot_b", bot_b_path)
                return -1e9
            results.append({"bot_b": bot_b_path, "score": score, "weight": bot_b_weights[idx]})
        total_weight = sum(item["weight"] for item in results) or 1.0
        blended = sum(item["score"] * item["weight"] for item in results) / total_weight
        trial.set_user_attr("results", results)
        return blended

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
    if args.apply_best:
        _apply_best_params(args.bot_a, study.best_params)
        print("Applied best params to %s" % os.path.join(args.bot_a, "best_params.json"))


if __name__ == "__main__":
    main()
