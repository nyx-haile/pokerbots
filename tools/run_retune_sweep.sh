#!/usr/bin/env bash
set -euo pipefail

BOT_A="${BOT_A:-./neuropoker}"
BOT_B="${BOT_B:-./baselines/current/neuropoker}"
ENGINE_DIR="${ENGINE_DIR:-./engine-2026}"
OUTPUT_DIR="${OUTPUT_DIR:-./tuning}"
ROUNDS="${ROUNDS:-1000}"
MATCHES="${MATCHES:-2}"
TRIALS="${TRIALS:-40}"
STORAGE="${STORAGE:-sqlite:///tuning/optuna_retune.db}"
JOBS="${JOBS:-0}"
MATCH_SUBSAMPLE="${MATCH_SUBSAMPLE:-1.0}"
MATCH_SAMPLE_MIN="${MATCH_SAMPLE_MIN:-1}"

python3.7 tools/optuna_tune_all.py \
  --bot-a "${BOT_A}" \
  --bot-b "${BOT_B}" \
  --engine-dir "${ENGINE_DIR}" \
  --rounds "${ROUNDS}" \
  --matches "${MATCHES}" \
  --trials "${TRIALS}" \
  --storage "${STORAGE}" \
  --output-dir "${OUTPUT_DIR}" \
  --jobs "${JOBS}" \
  --match-subsample "${MATCH_SUBSAMPLE}" \
  --match-sample-min "${MATCH_SAMPLE_MIN}" \
  --batch preflop

python3.7 tools/optuna_tune_all.py \
  --bot-a "${BOT_A}" \
  --bot-b "${BOT_B}" \
  --engine-dir "${ENGINE_DIR}" \
  --rounds "${ROUNDS}" \
  --matches "${MATCHES}" \
  --trials "${TRIALS}" \
  --storage "${STORAGE}" \
  --output-dir "${OUTPUT_DIR}" \
  --jobs "${JOBS}" \
  --match-subsample "${MATCH_SUBSAMPLE}" \
  --match-sample-min "${MATCH_SAMPLE_MIN}" \
  --batch margins

python3.7 tools/optuna_tune_all.py \
  --bot-a "${BOT_A}" \
  --bot-b "${BOT_B}" \
  --engine-dir "${ENGINE_DIR}" \
  --rounds "${ROUNDS}" \
  --matches "${MATCHES}" \
  --trials "${TRIALS}" \
  --storage "${STORAGE}" \
  --output-dir "${OUTPUT_DIR}" \
  --jobs "${JOBS}" \
  --match-subsample "${MATCH_SUBSAMPLE}" \
  --match-sample-min "${MATCH_SAMPLE_MIN}" \
  --batch fold_posture

python3.7 tools/optuna_tune_all.py \
  --bot-a "${BOT_A}" \
  --bot-b "${BOT_B}" \
  --engine-dir "${ENGINE_DIR}" \
  --rounds "${ROUNDS}" \
  --matches "${MATCHES}" \
  --trials "${TRIALS}" \
  --storage "${STORAGE}" \
  --output-dir "${OUTPUT_DIR}" \
  --jobs "${JOBS}" \
  --match-subsample "${MATCH_SUBSAMPLE}" \
  --match-sample-min "${MATCH_SAMPLE_MIN}" \
  --batch turn_defense

python3.7 tools/optuna_tune_all.py \
  --bot-a "${BOT_A}" \
  --bot-b "${BOT_B}" \
  --engine-dir "${ENGINE_DIR}" \
  --rounds "${ROUNDS}" \
  --matches "${MATCHES}" \
  --trials "${TRIALS}" \
  --storage "${STORAGE}" \
  --output-dir "${OUTPUT_DIR}" \
  --jobs "${JOBS}" \
  --match-subsample "${MATCH_SUBSAMPLE}" \
  --match-sample-min "${MATCH_SAMPLE_MIN}" \
  --batch aggression
