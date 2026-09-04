# PEBBLE: reward-model quality vs RL amplification

**Paper:** Lee, Smith & Abbeel — *PEBBLE: Feedback-Efficient Interactive Reinforcement Learning via Relabeling Experience and Unsupervised Pre-training*, ICML 2021 ([arXiv:2106.05091](https://arxiv.org/abs/2106.05091)).  
**Local PDF:** [`lee2021pebble.pdf`](lee2021pebble.pdf) · **Close reading:** [`PAPER_NOTES.md`](PAPER_NOTES.md)

This experiment is **not B-Pref**. Teacher is the paper’s scripted **Oracle** (§5.1). Stress comes from PEBBLE’s own ablations (Fig. 5a / Fig. 3).

---

## Open question

PEBBLE reports only **true episode return**. When that drops under paper ablations, is it:

| ID | Hypothesis | Signature |
|----|------------|-----------|
| **H1** | Reward model ranks worse | Acc on D_holdout / D_onpolicy falls with return |
| **H2** | SAC amplifies a still-decent \(\hat r\) | Acc stays high; return ↓; causal `probe_gap` large |

Stock PEBBLE never logs Acc on held-out / on-policy pairs, so the paper’s narratives for *no-relabel* (§4.3 non-stationarity) and *no-pretrain* (“better-shaped reward”) are untested against H1.

---

## Conditions (Oracle teacher always)

| ID | Relabel | Pre-train | Feedback | Paper map |
|----|---------|-----------|----------|-----------|
| `full` | ✓ | 10k | 1400 | PEBBLE default |
| `no_relabel` | ✗ | 10k | 1400 | Fig. 5a |
| `no_pretrain` | ✓ | 0 | 1400 | Fig. 5a |
| `low_budget` | ✓ | 10k | **400** | Fig. 3 |

Primary env: **walker_walk** (Fig. 3). Optional: `ENV=quadruped_walk` to match Fig. 5a.

---

## Metrics (after every reward update)

| Metric | Meaning |
|--------|---------|
| `acc_pref_buffer` | BT Acc on preference buffer vs Oracle labels |
| `acc_holdout` | Frozen post-pretrain pairs (never trained on) |
| `acc_onpolicy` | Recent on-policy pairs |
| `acc_holdout_q1…q5` | Acc by \|ΔR\| quintile (q1 = near-ties) |
| `spearman_rhat_rtrue` | Rank corr of \(\hat r(s,a)\) vs \(r_{\mathrm{true}}\) on replay |
| `true_return` | Paper eval metric |
| `probe_*` | Same-buffer SAC under \(\hat r\) vs GT → `probe_gap` |

Learning rule unchanged; probe restores agent weights + buffer rewards.

---

## Layout

```
experiments/pebble_reward_vs_rl/
  lee2021pebble.pdf
  PAPER_NOTES.md
  README.md
  config.yaml
  metrics.py
  train_pebble_diagnostics.py
  analyze_results.py
  run_condition.sh / run_all.sh / run_{full,no_relabel,no_pretrain,low_budget}.sh
  exp/<env>/<condition>/seed<seed>/   # Hydra outputs
```

Minimal repo change outside this folder: `replay_buffer.py` stores `true_rewards` for correlations + probe.

---

## How to run

From BPref repo root (venv active):

```bash
# One condition
DEVICE=cpu STEPS=100000 SEED=1 bash experiments/pebble_reward_vs_rl/run_full.sh

# All four ablations × seeds
DEVICE=cpu STEPS=100000 SEEDS="1 2 3" bash experiments/pebble_reward_vs_rl/run_all.sh

# Aggregate
python experiments/pebble_reward_vs_rl/analyze_results.py \
  --root experiments/pebble_reward_vs_rl/exp
```

Outputs: `experiments/pebble_reward_vs_rl/exp/<env>/<condition>/seed<seed>/diagnostics.csv`

### Smoke (wiring only)

Walker episodes ≈1000 steps → keep `num_seed_steps + num_unsup_steps ≥ ~3000` so holdout can freeze:

```bash
DEVICE=cpu CONDITION=full bash experiments/pebble_reward_vs_rl/run_condition.sh \
  num_train_steps=8000 num_seed_steps=1000 num_unsup_steps=2000 \
  num_interact=2000 max_feedback=80 reward_batch=20 reward_update=5 \
  eval_frequency=4000 num_eval_episodes=1 \
  diag_holdout_pairs=64 diag_onpolicy_pairs=32 \
  diag_probe_gradient_steps=20 'diag_probe_steps=[8000]' \
  agent.batch_size=256
```

For `no_pretrain` smoke, use `num_seed_steps=3000 num_unsup_steps=0` so ≥2 trajs exist before first query.

### Paper-scale (confirmation)

```bash
DEVICE=cuda STEPS=500000 SEEDS="1 2 3 4 5" \
  bash experiments/pebble_reward_vs_rl/run_all.sh \
  reward_batch=128 reward_update=200 num_eval_episodes=10 \
  eval_frequency=10000
```

---

## Interpretation (vs `full`)

| Pattern | Reading |
|---------|---------|
| `no_relabel`: Acc high, return ↓, probe_gap large | Supports §4.3 (**H2** / non-stationarity) |
| `no_relabel`: Acc_holdout / Acc_onpolicy ↓ | Relabel also hurts \(\hat r\) itself (**H1**) |
| `no_pretrain`: Acc ↓ with return | Supports “better-shaped reward” (**H1**) |
| `no_pretrain`: Acc ≈ full, return ↓ | Pretrain helps via visitation |
| `low_budget`: Acc ↓ | Underfit \(\hat r\) (**H1**) |
| `low_budget`: Acc high, return ↓ | Amplification (**H2**) |

---

## Deliberately out of scope

B-Pref Mistake/Stoc teachers, PrefPPO, real humans (§5.4). Those are follow-ups; they are not required to answer the open question *about PEBBLE*.
