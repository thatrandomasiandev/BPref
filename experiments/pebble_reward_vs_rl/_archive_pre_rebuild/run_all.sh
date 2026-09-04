#!/usr/bin/env bash
# Run all four PEBBLE paper ablations for one or more seeds.
# Example:
#   DEVICE=cpu STEPS=100000 SEEDS="1 2 3" bash experiments/pebble_reward_vs_rl/run_all.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

SEEDS="${SEEDS:-1}"
DEVICE="${DEVICE:-cuda}"
STEPS="${STEPS:-100000}"
ENV="${ENV:-walker_walk}"

for seed in $SEEDS; do
  for cond in full no_relabel no_pretrain low_budget; do
    echo "========== $cond seed=$seed =========="
    CONDITION="$cond" DEVICE="$DEVICE" SEED="$seed" STEPS="$STEPS" ENV="$ENV" \
      bash experiments/pebble_reward_vs_rl/run_condition.sh "$@"
  done
done

echo "Done. Aggregate with:"
echo "  python experiments/pebble_reward_vs_rl/analyze_results.py --root experiments/pebble_reward_vs_rl/exp"
