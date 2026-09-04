# Close reading: PEBBLE (Lee, Smith & Abbeel, ICML 2021)

**PDF:** `lee2021pebble.pdf` (arXiv:2106.05091).  
**Design registration:** `CLAIM.md`.  
**Not B-Pref.**

---

## Method (paper §§4.1–4.3, Algorithms 1–2)

1. **EXPLORE / pre-train:** SAC on intrinsic state-entropy reward (k-NN); paper uses **10K** steps, counted on learning curves (§5.1).
2. **Reward learning:** Bradley–Terry on segment pairs; ensemble size **3**; default query selection **entropy** (§5.1: “otherwise, we use entropy-based sampling”).
3. **Agent learning:** off-policy **SAC** on \(\hat r_\psi\).
4. **Relabel:** after every reward update, rewrite **entire** replay buffer with current \(\hat r_\psi\) (§4.3) — claimed to stabilize learning under non-stationary rewards.

## Teacher (§5.1)

> Scripted teacher that provides preferences between trajectory segments **according to the true, underlying task reward**.

= **Oracle**. Evaluation = **ground-truth** episode return.  
Mistake / stochastic teachers are **B-Pref**, not this paper.

## Numbers we rely on

| Item | Paper |
|------|-------|
| Pre-train | 10K steps |
| Ensemble | 3 |
| Feedback (locomotion) | 400 / 700 / 1400 |
| Fig. 3 envs | Cheetah-run, Walker-walk, Quadruped-walk |
| Fig. 5a | Quadruped-walk @ 1400; ablate relabel / pre-train |
| Fig. 5c | segment length 50 vs 1 |
| Seeds | mean ± std over **10** runs |

## Gap this experiment fills

PEBBLE never reports reward-model **accuracy**. Ablation stories in §5.3 (“robust to changes in its reward model”; “better-shaped reward”) are therefore **unmeasured** at the \(\hat r\) level. We measure Acc + causal probe under those same ablations.
