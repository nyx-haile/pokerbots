#!/usr/bin/env bash
set -euo pipefail

BOT_A="${BOT_A:-./neuropoker}"
BOT_B="${BOT_B:-}"
BOT_B_POOL="${BOT_B_POOL:-}"
ENGINE_DIR="${ENGINE_DIR:-engine-2026}"
BOT_PYTHON="${BOT_PYTHON:-/usr/bin/python3.7}"
ENGINE_PYTHON="${ENGINE_PYTHON:-/usr/bin/python3.7}"
ROUNDS="${ROUNDS:-333}"
TRIALS="${TRIALS:-100}"
STORAGE="${STORAGE:-sqlite:///tuning/optuna_tight.db}"
STUDY_NAME="${STUDY_NAME:-tight_threshold}"
OUTPUT_DIR="${OUTPUT_DIR:-tuning}"
POOL_MODE="${POOL_MODE:-random}"
BATCH="policy"
APPLY_BEST="${APPLY_BEST:-1}"
JOBS="${JOBS:-0}"
MATCH_SUBSAMPLE="${MATCH_SUBSAMPLE:-1.0}"
MATCH_SAMPLE_MIN="${MATCH_SAMPLE_MIN:-1}"

if [[ -z "${BOT_B_POOL}" ]]; then
  _default_pool=(
    "baselines/217494a/neuropoker"
    "baselines/02f38b1/neuropoker"
    "baselines/4340705/neuropoker"
  )
  BOT_B_POOL="$(IFS=,; echo "${_default_pool[*]}")"
fi

if [[ -z "${BOT_B}" && -n "${BOT_B_POOL}" ]]; then
  IFS=',' read -ra _pool_items <<< "${BOT_B_POOL}"
  if ((${#_pool_items[@]} > 0)); then
    BOT_B="${_pool_items[0]}"
  fi
fi

if [[ -n "${BOT_B_POOL}" ]]; then
  IFS=',' read -ra _pool_items <<< "${BOT_B_POOL}"
  _pool_size=${#_pool_items[@]}
  if ((_pool_size > 0)); then
    if ((_pool_size >= 3)); then
      _subset_size="$("${BOT_PYTHON}" - <<'PY'
import random
print(random.choice([2, 3]))
PY
)"
    else
      _subset_size="${_pool_size}"
    fi
    POOL="${BOT_B_POOL}" K="${_subset_size}" \
      BOT_B_POOL="$("${BOT_PYTHON}" - <<'PY'
import os
import random
items = [item for item in os.environ.get("POOL", "").split(",") if item]
k = int(os.environ.get("K", "0") or "0")
if k <= 0 or k >= len(items):
    print(",".join(items))
else:
    print(",".join(random.sample(items, k)))
PY
)"
    IFS=',' read -ra _pool_items <<< "${BOT_B_POOL}"
    if ((${#_pool_items[@]} > 0)); then
      BOT_B="${_pool_items[0]}"
      if ((${#_pool_items[@]} > 1)); then
        BOT_B_POOL="$(IFS=,; echo "${_pool_items[@]:1}")"
      else
        BOT_B_POOL=""
      fi
    fi
  fi
fi

if [[ -z "${BOT_B}" ]]; then
  echo "BOT_B is unset and no baselines were discovered. Set BOT_B or BOT_B_POOL." >&2
  exit 1
fi

if [[ -z "${MATCHES+x}" ]]; then
  if [[ -n "${BOT_B_POOL}" ]]; then
    IFS=',' read -ra _pool_items <<< "${BOT_B_POOL}"
    _match_base=$((1 + ${#_pool_items[@]}))
    if ((_match_base > 10)); then
      MATCHES=10
    else
      MATCHES=${_match_base}
    fi
  else
    MATCHES=10
  fi
fi

exec "${BOT_PYTHON}" tools/optuna_tune_all.py \
  --bot-a "${BOT_A}" \
  --bot-b "${BOT_B}" \
  --bot-b-pool "${BOT_B_POOL}" \
  --bot-b-per-match \
  --bot-b-pool-mode "${POOL_MODE}" \
  --engine-dir "${ENGINE_DIR}" \
  --bot-python "${BOT_PYTHON}" \
  --engine-python "${ENGINE_PYTHON}" \
  --rounds "${ROUNDS}" \
  --matches "${MATCHES}" \
  --trials "${TRIALS}" \
  --storage "${STORAGE}" \
  --study-name "${STUDY_NAME}" \
  --output-dir "${OUTPUT_DIR}" \
  --jobs "${JOBS}" \
  --match-subsample "${MATCH_SUBSAMPLE}" \
  --match-sample-min "${MATCH_SAMPLE_MIN}" \
  --batch "${BATCH}" \
  $(if [[ "${APPLY_BEST}" == "1" ]]; then echo "--apply-best"; fi)
