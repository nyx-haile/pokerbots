#!/usr/bin/env python3
"""
Export Optuna best params from a study into a JSON file.
"""
import argparse
import json

import optuna


def main():
    parser = argparse.ArgumentParser(description="Export best params from Optuna storage.")
    parser.add_argument("--storage", required=True, help="Optuna storage URL")
    parser.add_argument("--study-name", required=True, help="Optuna study name")
    parser.add_argument("--output", required=True, help="Output JSON path")
    args = parser.parse_args()

    study = optuna.load_study(study_name=args.study_name, storage=args.storage)
    data = {
        "best_value": study.best_value,
        "best_params": study.best_params,
        "study_name": study.study_name,
    }
    with open(args.output, "w") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print("Wrote %s" % args.output)


if __name__ == "__main__":
    main()
