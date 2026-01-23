#!/usr/bin/env python3
"""
Write Optuna best_params.json into a bot-local parameter file.
"""
import argparse
import json
import os


def _load_params(path):
    with open(path, "r") as handle:
        data = json.load(handle)
    params = data.get("best_params", data)
    if not isinstance(params, dict):
        raise ValueError("best_params missing or invalid")
    return params


def main():
    parser = argparse.ArgumentParser(description="Write best params into bot config file.")
    parser.add_argument("--input", required=True, help="Path to Optuna best_params.json")
    parser.add_argument("--output", required=True, help="Output path for bot params")
    args = parser.parse_args()

    params = _load_params(args.input)
    output_dir = os.path.dirname(os.path.abspath(args.output))
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(args.output, "w") as handle:
        json.dump({"best_params": params}, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print("Wrote params to %s" % args.output)


if __name__ == "__main__":
    main()
