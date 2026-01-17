# Toss or Hold'em Rules Summary

## Overview
Toss or Hold'em is a no-limit Texas Hold'em variant with a discard mechanic and a 6-card board at showdown. Each player starts with 3 hole cards, then discards one after a 2-card flop. Discarded cards become public board cards. Showdown uses the best 5-card hand from hole + board.

## Core Parameters
- Rounds per game: 1000
- Stack per round: 400 (stacks reset every round)
- Blinds: small blind 1, big blind 2

## Sequence of Play
1. Post blinds (dealer posts small blind, opponent posts big blind).
2. Deal 3 pre-discard hole cards to each player.
3. Pre-flop betting (dealer acts first).
4. Deal flop (2 board cards).
5. Discard round: out-of-position player discards first, then dealer.
6. Flop betting (after both discards, board has 4 cards).
7. Deal turn (5th board card), then betting.
8. Deal river (6th board card), then betting.
9. Showdown.

## Betting Rules (No-Limit Hold'em)
- Pre-flop: dealer acts first; can fold, call (1), or raise (to 3+).
- Post-flop/turn/river: out-of-position player acts first; can check or bet (2+).
- Minimum raise equals the previous bet/raise size; maximum raise is bounded by both stacks.
- Betting ends on fold, a call, or two consecutive checks.

## Discard Round Details
- Each player starts with 3 private cards; after the 2-card flop, each discards one.
- Discards are public and become board cards.
- The dealer discards second and sees the opponent's discard first.
- After discards, each player has 2 hole cards and the board has 4 cards.

## Showdown
- Best 5-card hand wins, using any combination of hole cards and board cards.
- Ties split the pot; bankrolls update based on round result.

Source: `documentation/variant.pdf`
