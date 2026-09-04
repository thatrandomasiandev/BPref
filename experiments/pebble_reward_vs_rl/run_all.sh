#!/usr/bin/env bash
# Run all four CLAIM.md conditions for one or more seeds.
#
# Diagnostic example (tentative patterns only — not a PhD-grade claim):
#   DEVICE=cpu STEPS=100000 SEEDS="1 2 3 4 5" \
#     bash experiments/pebble_reward_vs_rl/run_all.sh
#
# Paper-scale example (CLAIM.md §6 claim gate):
#   DEVICE=cuda STEPS=500000 SEEDS="1 2 3 4 5 6 7 8 9 10" \
#     bash experiments/pebble_reward_vs_rl/run_all.sh \
#     reward_batch=128 reward_update=200 num_eval_episodes=10
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

echo "Done. Aggregate (still check CLAIM.md §6 before claiming a finding):"
echo "  python experiments/pebble_reward_vs_rl/analyze_results.py --root experiments/pebble_reward_vs_rl/exp"
