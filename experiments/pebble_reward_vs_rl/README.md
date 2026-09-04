# PEBBLE: reward-model quality vs RL amplification

**Paper:** Lee, Smith & Abbeel — PEBBLE, ICML 2021 ([arXiv:2106.05091](https://arxiv.org/abs/2106.05091))  
**PDF:** [`lee2021pebble.pdf`](lee2021pebble.pdf)  
**Pre-registration:** [`CLAIM.md`](CLAIM.md)  
**Close reading:** [`PAPER_NOTES.md`](PAPER_NOTES.md)

**Not B-Pref.** Teacher = paper §5.1 scripted **Oracle**. Stress = PEBBLE’s own ablations.

---

## Claim (one sentence)

When PEBBLE’s true return drops under `no_relabel` / `no_pretrain` / `low_budget`, is it **H1** (BT Acc of \(\hat r\) worsens) or **H2** (Acc stays high; SAC amplifies; large `probe_gap`)?

See CLAIM.md for falsifiers, confounds, and the seed plan. **No finding is claimed until paper-scale × 10 seeds.**

---

## Conditions

| ID | Relabel | Pre-train | Feedback | Paper |
|----|---------|-----------|----------|-------|
| `full` | ✓ | 10k | 1400 | default |
| `no_relabel` | ✗ | 10k | 1400 | Fig. 5a |
| `no_pretrain` | ✓ | 0 | 1400 | Fig. 5a |
| `low_budget` | ✓ | 10k | 400 | Fig. 3 |

---

## Layout

```
CLAIM.md                     # pre-registered design (read first)
PAPER_NOTES.md
lee2021pebble.pdf
config.yaml
metrics.py                   # Acc / Spearman / causal probe (commented)
train_pebble_diagnostics.py  # PEBBLE loop + hooks (commented)
analyze_results.py
run_condition.sh / run_all.sh / run_{full,no_relabel,no_pretrain,low_budget}.sh
run_smoke_all.sh             # wiring only — not evidence
exp/<env>/<condition>/seed*/ # Hydra outputs
```

Repo dependency outside this folder: `replay_buffer.py` stores `true_rewards` for correlations + probe.

---

## How to run

From `repos/BPref` (venv active):

```bash
# Wiring smoke (all four conditions) — NOT a finding
DEVICE=cpu bash experiments/pebble_reward_vs_rl/run_smoke_all.sh

# Diagnostic (tentative patterns only): 5 seeds × 100k
DEVICE=cpu STEPS=100000 SEEDS="1 2 3 4 5" \
  bash experiments/pebble_reward_vs_rl/run_all.sh

# Paper-scale (CLAIM.md claim gate): 10 seeds × ≥500k
DEVICE=cuda STEPS=500000 SEEDS="1 2 3 4 5 6 7 8 9 10" \
  bash experiments/pebble_reward_vs_rl/run_all.sh \
  reward_batch=128 reward_update=200 num_eval_episodes=10

# Aggregate
python experiments/pebble_reward_vs_rl/analyze_results.py \
  --root experiments/pebble_reward_vs_rl/exp
```

Under zsh, quote list overrides: `'diag_probe_steps=[25000,50000,100000]'`.

---

## Status vocabulary (use exactly)

| Phrase | Meaning |
|--------|---------|
| Setup complete | Code + CLAIM match; instrument exists |
| Smoke-tested | Wiring OK; **not evidence** |
| Diagnostic-scale evidence | ≥5 seeds @ ~100k; **tentative pattern only** |
| Paper-scale result | ≥10 seeds @ ≥500k (or paper match) |
| Conclusion justified | Pre-registered reading survives seed variance + confounds |

---

## Interpretation cheat-sheet (vs `full`)

| Pattern | Reading |
|---------|---------|
| Acc ↓ with return | **H1** |
| Acc stable, probe_gap large, return ↓ | **H2** |
| Acc stable, probe_gap ≈ 0, return ↓ | Visitation / exploration — neither pure H1 nor H2 |
| Return does not drop | Ablation underpowered at this scale — do not invent H1/H2 |
