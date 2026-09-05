#!/usr/bin/env bash
# Shared launcher for PEBBLE-native ablations (Oracle teacher only).
#
# Maps CONDITION → the single-factor Hydra overrides in CLAIM.md §2.
# Extra args are forwarded to Hydra (e.g. smoke overrides).
#
# Usage:
#   CONDITION=full DEVICE=cpu SEED=1 STEPS=100000 \
#     bash experiments/pebble_reward_vs_rl/run_condition.sh
#
# Quote list overrides under zsh:
#   ... 'diag_probe_steps=[25000,50000,100000]'
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

CONDITION="${CONDITION:-full}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-1}"
STEPS="${STEPS:-100000}"
ENV="${ENV:-walker_walk}"

# One changed factor vs `full` (except low_budget changes feedback only).
case "$CONDITION" in
  full)
    EXTRA=(pebble_condition=full do_relabel=true num_unsup_steps=10000 max_feedback=1400)
    ;;
  no_relabel)
    EXTRA=(pebble_condition=no_relabel do_relabel=false num_unsup_steps=10000 max_feedback=1400)
    ;;
  no_pretrain)
    # Unsupervised pre-train = 0 (Fig. 5a). Walker episodes are ~1000 steps, and
    # PEBBLE's first query needs ≥2 complete trajs — so use num_seed_steps=2000
    # (random actions only; NOT entropy pre-train). Documented in CLAIM.md §8.
    EXTRA=(pebble_condition=no_pretrain do_relabel=true num_unsup_steps=0 num_seed_steps=2000 max_feedback=1400)
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
export PYTHONUNBUFFERED=1

# Prefer: (1) PEBBLE_PYTHON from Colab 3.11 venv, (2) project .venv, (3) PATH python.
if [[ -n "${PEBBLE_PYTHON:-}" && -x "${PEBBLE_PYTHON}" ]]; then
  PY="${PEBBLE_PYTHON}"
elif [[ -x "${ROOT}/.venv/bin/python" ]]; then
  PY="${ROOT}/.venv/bin/python"
else
  PY="python"
fi

echo "[run] condition=$CONDITION device=$DEVICE seed=$SEED steps=$STEPS env=$ENV"
echo "[run] python=$PY"
echo "[run] REMINDER: smoke ≠ diagnostic ≠ paper-scale (CLAIM.md §6)"

# Build argv in an array so macOS bash 3.2 never re-parses EXTRA as commands
# after the python process exits (previous `"$@"` / line-continuation footgun → rc 127).
ARGS=(
  experiments/pebble_reward_vs_rl/train_pebble_diagnostics.py
  "device=${DEVICE}"
  "seed=${SEED}"
  "num_train_steps=${STEPS}"
  "env=${ENV}"
)
ARGS+=("${EXTRA[@]}")
# Forward caller overrides only when present (avoids set -u / empty "$@" issues).
if [[ $# -gt 0 ]]; then
  ARGS+=("$@")
fi

exec "$PY" "${ARGS[@]}"
