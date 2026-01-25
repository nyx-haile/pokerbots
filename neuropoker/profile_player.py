#!/usr/bin/env python3
"""
Profiling wrapper for player.py.
Run this instead of player.py to generate a profile.

Usage:
    python profile_player.py [args...]

Output:
    - neuropoker_profile.prof: Binary profile file (view with snakeviz or pstats)
    - Profile summary printed to stderr on exit
"""
import cProfile
import io
import pstats
import sys


def main():
    # Import and run the bot with profiling
    from skeleton.runner import parse_args, run_bot
    from player import Player
    
    profiler = cProfile.Profile()
    profiler.enable()
    
    try:
        run_bot(Player(), parse_args())
    finally:
        profiler.disable()
        
        # Save binary profile
        profiler.dump_stats("neuropoker_profile.prof")
        print("[profile] Saved profile to neuropoker_profile.prof", file=sys.stderr)
        
        # Print top 50 by cumulative time
        s = io.StringIO()
        ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
        ps.print_stats(50)
        print("\n=== PROFILE SUMMARY (top 50 by cumtime) ===", file=sys.stderr)
        print(s.getvalue(), file=sys.stderr)
        
        # Print top 30 by total time
        s2 = io.StringIO()
        ps2 = pstats.Stats(profiler, stream=s2).sort_stats('tottime')
        ps2.print_stats(30)
        print("\n=== PROFILE SUMMARY (top 30 by tottime) ===", file=sys.stderr)
        print(s2.getvalue(), file=sys.stderr)


if __name__ == '__main__':
    main()
