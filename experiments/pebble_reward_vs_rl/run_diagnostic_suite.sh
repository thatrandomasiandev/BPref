#!/usr/bin/env bash
# Diagnostic suite orchestrator (CLAIM.md §6).
#
# Protocol: 5 seeds × 4 conditions × 100k steps on Walker-walk.
# Output: experiments/pebble_reward_vs_rl/exp/walker_walk/<cond>/seed<seed>/
#
# This produces *tentative patterns only* — not a paper-scale claim.
#
# Usage:
#   DEVICE=cpu PARALLEL=3 bash experiments/pebble_reward_vs_rl/run_diagnostic_suite.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source .venv/bin/activate
export PYTHONPATH="${ROOT}:${ROOT}/custom_dmc2gym:${PYTHONPATH:-}"

DEVICE="${DEVICE:-cpu}"
PARALLEL="${PARALLEL:-3}"
STEPS="${STEPS:-100000}"
SEEDS="${SEEDS:-1 2 3 4 5}"
ENV="${ENV:-walker_walk}"
CONDS=(full no_relabel no_pretrain low_budget)

LOG_DIR="$ROOT/experiments/pebble_reward_vs_rl/exp/_logs"
mkdir -p "$LOG_DIR"
MASTER="$LOG_DIR/diagnostic_suite_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$MASTER") 2>&1

echo "=== DIAGNOSTIC SUITE ==="
echo "device=$DEVICE parallel=$PARALLEL steps=$STEPS seeds=[$SEEDS] env=$ENV"
echo "log=$MASTER"
echo "STATUS GATE: results = tentative patterns only (CLAIM.md §6)"

is_complete() {
  local cond="$1" seed="$2"
  local csv="$ROOT/experiments/pebble_reward_vs_rl/exp/${ENV}/${cond}/seed${seed}/diagnostics.csv"
  if [[ ! -f "$csv" ]]; then
    return 1
  fi
  # Complete if a final row exists at/near STEPS (notes=final or step>=STEPS).
  python - "$csv" "$STEPS" <<'PY'
import csv, sys
path, steps = sys.argv[1], int(float(sys.argv[2]))
rows = list(csv.DictReader(open(path)))
if not rows:
    sys.exit(1)
last = rows[-1]
ok = int(float(last["step"])) >= steps - 1 or last.get("notes", "") == "final"
sys.exit(0 if ok else 1)
PY
}

run_job() {
  local cond="$1" seed="$2"
  local tag="${cond}_seed${seed}"
  local joblog="$LOG_DIR/${tag}.log"
  if is_complete "$cond" "$seed"; then
    echo "[skip] $tag already complete"
    return 0
  fi
  echo "[start] $tag -> $joblog"
  (
    CONDITION="$cond" DEVICE="$DEVICE" SEED="$seed" STEPS="$STEPS" ENV="$ENV" \
      bash experiments/pebble_reward_vs_rl/run_condition.sh
  ) >"$joblog" 2>&1
  local rc=$?
  if [[ $rc -eq 0 ]] && is_complete "$cond" "$seed"; then
    echo "[done]  $tag"
  else
    echo "[FAIL]  $tag rc=$rc (see $joblog)"
    return $rc
  fi
}

# Build job list
JOBS=()
for seed in $SEEDS; do
  for cond in "${CONDS[@]}"; do
    JOBS+=("$cond:$seed")
  done
done

echo "Total jobs: ${#JOBS[@]} (skip completed)"

# Simple worker pool
running=0
fail=0
for spec in "${JOBS[@]}"; do
  cond="${spec%%:*}"
  seed="${spec##*:}"
  # Wait if at capacity
  while [[ $(jobs -rp | wc -l | tr -d ' ') -ge $PARALLEL ]]; do
    # Reap finished background jobs; track failures
    if ! wait -n 2>/dev/null; then
      # bash without wait -n: fall back to short sleep
      sleep 5
    fi
  done
  run_job "$cond" "$seed" &
  running=$((running + 1))
done

# Drain
fail=0
for pid in $(jobs -rp); do
  if ! wait "$pid"; then
    fail=$((fail + 1))
  fi
done

echo "=== SUITE FINISHED fail=$fail ==="
echo "Aggregate with:"
echo "  python experiments/pebble_reward_vs_rl/analyze_results.py --root experiments/pebble_reward_vs_rl/exp"
exit "$fail"
