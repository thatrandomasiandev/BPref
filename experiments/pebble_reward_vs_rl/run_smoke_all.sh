#!/usr/bin/env bash
# Wiring-only smoke for all four conditions.
# NOT EVIDENCE. CLAIM.md §6: smoke validates imports, holdout freeze, CSV, probe restore.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
source .venv/bin/activate
export PYTHONPATH="${ROOT}:${ROOT}/custom_dmc2gym:${PYTHONPATH:-}"

DEVICE="${DEVICE:-cpu}"
COMMON=(
  num_train_steps=6000
  num_interact=2000
  max_feedback=40
  reward_batch=20
  reward_update=3
  eval_frequency=6000
  num_eval_episodes=1
  diag_holdout_pairs=32
  diag_onpolicy_pairs=16
  diag_probe_gradient_steps=10
  'diag_probe_steps=[]'
  agent.batch_size=256
)

run_one() {
  local cond="$1"
  local seed="$2"
  local unsup="$3"
  local seed_steps="$4"
  echo "===== SMOKE $cond seed=$seed ====="
  CONDITION="$cond" DEVICE="$DEVICE" SEED="$seed" STEPS=6000 \
    bash experiments/pebble_reward_vs_rl/run_condition.sh \
    "${COMMON[@]}" \
    "num_seed_steps=${seed_steps}" \
    "num_unsup_steps=${unsup}" \
    "hydra.run.dir=experiments/pebble_reward_vs_rl/exp/walker_walk/${cond}/seed${seed}_smoke"
}

# full / no_relabel / low_budget: short pretrain so ≥2 trajs exist
run_one full 101 2000 1000
run_one no_relabel 102 2000 1000
run_one low_budget 103 2000 1000
# no_pretrain: longer seed so ≥2 trajs before first query (unsup=0)
run_one no_pretrain 104 0 3000

echo "SMOKE COMPLETE — wiring only. No scientific claim."
