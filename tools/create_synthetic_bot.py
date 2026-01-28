#!/usr/bin/env python3
"""
Create a synthetic bot instance by copying the main bot and applying a config.
Usage: python create_synthetic_bot.py <config.json> <output_dir>
"""
import argparse
import json
import os
import shutil
import sys


def main():
    parser = argparse.ArgumentParser(description="Create synthetic bot from config")
    parser.add_argument("config", help="Path to config JSON file")
    parser.add_argument("output", help="Output directory for the bot")
    parser.add_argument("--source", default=None, help="Source bot directory (defaults to neuropoker)")
    args = parser.parse_args()

    # Find source bot
    if args.source:
        source_dir = args.source
    else:
        # Default to neuropoker in same repo
        script_dir = os.path.dirname(os.path.abspath(__file__))
        source_dir = os.path.join(os.path.dirname(script_dir), "neuropoker")
    
    if not os.path.isdir(source_dir):
        sys.exit(f"Source directory not found: {source_dir}")
    
    # Load config
    with open(args.config, "r") as f:
        config = json.load(f)
    
    # Create output directory
    if os.path.exists(args.output):
        shutil.rmtree(args.output)
    
    # Copy source bot, excluding unnecessary files
    exclude = {".venv", "__pycache__", ".git", "*.pyc", "*.prof", "*.egg-info", "dist", "build"}
    
    def ignore_patterns(directory, files):
        ignored = []
        for f in files:
            if f in exclude or f.endswith(".pyc") or f.endswith(".prof"):
                ignored.append(f)
        return ignored
    
    shutil.copytree(source_dir, args.output, ignore=ignore_patterns)
    
    # Write the config as best_params.json
    params_file = os.path.join(args.output, "best_params.json")
    with open(params_file, "w") as f:
        json.dump(config, f, indent=2)
        f.write("\n")
    
    print(f"Created synthetic bot at {args.output}")
    print(f"Applied config from {args.config}")
    if "description" in config:
        print(f"Description: {config['description']}")


if __name__ == "__main__":
    main()
