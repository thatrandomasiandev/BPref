#!/usr/bin/env bash
# Shared launcher for PEBBLE-native ablations (Oracle teacher only).
# Usage: CONDITION=full bash experiments/pebble_reward_vs_rl/run_condition.sh [hydra overrides...]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

CONDITION="${CONDITION:-full}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-1}"
STEPS="${STEPS:-100000}"
ENV="${ENV:-walker_walk}"

# Map paper conditions → Hydra overrides (see PAPER_NOTES.md).
case "$CONDITION" in
  full)
    EXTRA=(pebble_condition=full do_relabel=true num_unsup_steps=10000 max_feedback=1400)
    ;;
  no_relabel)
    EXTRA=(pebble_condition=no_relabel do_relabel=false num_unsup_steps=10000 max_feedback=1400)
    ;;
  no_pretrain)
    # Still need ≥2 complete trajs before first query; keep seed≥2000 for walker.
    EXTRA=(pebble_condition=no_pretrain do_relabel=true num_unsup_steps=0 max_feedback=1400)
    ;;
  low_budget)
    EXTRA=(pebble_condition=low_budget do_relabel=true num_unsup_steps=10000 max_feedback=400)
    ;;
  *)
    echo "Unknown CONDITION=$CONDITION (full|no_relabel|no_pretrain|low_budget)" >&2
    exit 1
    ;;
esac

export PYTHONPATH="${ROOT}:${ROOT}/custom_dmc2gym:${PYTHONPATH:-}"

echo "[run] condition=$CONDITION device=$DEVICE seed=$SEED steps=$STEPS env=$ENV"
python experiments/pebble_reward_vs_rl/train_pebble_diagnostics.py \
  device="$DEVICE" \
  seed="$SEED" \
  num_train_steps="$STEPS" \
  env="$ENV" \
  "${EXTRA[@]}" \
  "$@"
