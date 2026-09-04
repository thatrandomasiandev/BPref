# Close reading: PEBBLE (Lee, Smith & Abbeel, ICML 2021)

**Source:** `lee2021pebble.pdf` (arXiv:2106.05091) — downloaded into this folder.  
**Not B-Pref.** B-Pref is a later benchmark that *evaluates* PEBBLE under noisy teachers. This experiment stays inside PEBBLE’s own claims and protocol.

---

## What the paper actually does

### Problem (§1)
Human-in-the-loop preference RL (Christiano et al. 2017) is feedback-expensive because prior work used **on-policy** RL to cope with a **non-stationary** learned reward \(\hat r_\psi\).

### Method (§4, Algorithms 1–2)
PEBBLE = unsupervised **PrE**-training + preference-**B**ased learning via rela**B**e**L**ing **E**xperience.

| Step | Mechanism | Paper claim |
|------|-----------|-------------|
| **0. EXPLORE** | SAC on intrinsic reward \(r^{\mathrm{int}}(s_t)=\log\|s_t-s_t^{(k)}\|\) (state entropy / k-NN) | Diverse behaviors → more informative early queries |
| **1. Reward learning** | Bradley–Terry on segment pairs (Eq. 3–4); ensemble of 3 nets; uniform / disagreement / **entropy** query selection | \(\hat r_\psi\) consistent with preferences |
| **2. Agent learning** | Off-policy **SAC** on \(\hat r_\psi\) | Sample efficiency |
| **Relabel** | After every reward update, rewrite **entire** replay buffer with current \(\hat r_\psi\) | Stabilizes off-policy learning under non-stationary \(\hat r\) (§4.3) |

Loop: repeat 1–2 with feedback every \(K\) steps, \(M\) queries per session.

### Teacher (§5.1) — critical
> “scripted teacher that provides preferences between trajectory segments **according to the true, underlying task reward**.”

That is an **Oracle** preference channel. PEBBLE does **not** study Mistake/Stoc teachers (that is B-Pref).  
Evaluation metric = **ground-truth** episode return (or Meta-World success).

### Reported setup (§5.1–5.2)
- Envs: DMControl Cheetah-run, **Walker-walk**, Quadruped-walk; Meta-World manipulation
- Pre-train: **10K** steps (included in learning curves)
- Ensemble size: **3**
- Default query scheme: **entropy**
- Feedback budgets (locomotion): **400 / 700 / 1400** queries
- Ablation env (Fig. 5): **Quadruped-walk @ 1400** queries
- Segment length ablation: \(H=50\) vs \(H=1\) (Fig. 5c)
- Seeds: mean ± std over **10** runs

### Ablations that *cause* PEBBLE performance to drop (Fig. 5a)
On Quadruped-walk @ 1400 queries, removing either ingredient hurts:

1. **relabel: X** — paper: “relabeling significantly improves performance because it enables the agent to be robust to changes in its reward model” (§5.3)
2. **pre-train: X** — paper: “showing diverse behaviors to a teacher can induce a **better-shaped reward**” (§5.3)

Lower feedback (Fig. 3: 400 vs 1400) also lowers asymptotic true return.

---

## The gap (why this experiment exists)

PEBBLE **never reports reward-model accuracy** — only downstream true return.

So when return drops under `no_relabel` / `no_pretrain` / `low_budget`, the paper’s narrative is underspecified:

| Ablation | Paper’s implied story | Alternative |
|----------|----------------------|-------------|
| no_relabel | Non-stationary \(\hat r\) breaks SAC (**RL / optimization**) | \(\hat r\) itself becomes a worse ranker as policy shifts |
| no_pretrain | Worse-shaped \(\hat r\) from poor early queries (**reward model**) | Same \(\hat r\) quality, worse visitation / exploration |
| low_budget | Implicitly: \(\hat r\) underfit (**reward model**) | Small ranking errors amplified by SAC |

**Open question (this experiment):**  
When PEBBLE’s true return decreases under *PEBBLE’s own ablations*, is it because **reward-model ranking quality** got worse (**H1**), or because **downstream SAC amplifies** a still-decent \(\hat r\) (**H2**)?

That question is internal to PEBBLE. It does not require B-Pref noise models.

---

## Experimental design (derived from the paper)

### Teacher
Scripted Oracle only (paper §5.1): preferences = \(\arg\max\) of GT segment return. No Mistake/Stoc/Skip/Equal.

### Environment
Primary: **Walker-walk** (paper Fig. 3; cheaper than Quadruped for iteration).  
Optional scale-up: Quadruped-walk @ 1400 to match Fig. 5a.

### Conditions (2×2 ablations + budget; all PEBBLE-native)

| ID | relabel | pre-train | max_feedback | Maps to |
|----|---------|-----------|--------------|---------|
| `full` | ✓ | ✓ (10k) | 1400 | PEBBLE default |
| `no_relabel` | ✗ | ✓ | 1400 | Fig. 5a |
| `no_pretrain` | ✓ | ✗ (0) | 1400 | Fig. 5a |
| `low_budget` | ✓ | ✓ | **400** | Fig. 3 |

### Measurements after every reward update
PEBBLE never logged these; we add them without changing the learning rule:

1. **D_pref (train buffer)** — BT accuracy vs Oracle labels on stored preference pairs  
2. **D_holdout** — frozen pairs after pre-train (or after seed phase if no pre-train), never trained on  
3. **D_onpolicy** — pairs from recent on-policy trajectories  
4. **Spearman** \(\rho(\hat r(s,a), r_{\mathrm{true}}(s,a))\) on the replay buffer  
5. **True return** (paper’s eval metric)  
6. **Causal probe** at checkpoints: same buffer, fixed SAC steps under \(\hat r\) vs \(r_{\mathrm{true}}\) → `probe_gap`

### Interpretation (PEBBLE-specific)

| Pattern | Reading |
|---------|---------|
| `no_relabel`: Acc high, return ↓, probe_gap large | Supports paper’s §4.3 story (**H2** / non-stationarity) |
| `no_relabel`: Acc_holdout / Acc_onpolicy ↓ with return | Relabel also hurts because \(\hat r\) itself drifts (**H1** component) |
| `no_pretrain`: Acc_holdout ↓ | Supports “better-shaped reward” claim (**H1**) |
| `no_pretrain`: Acc similar, return ↓ | Pretrain helps via visitation, not ranking quality |
| `low_budget`: Acc ↓ with return | Underfitting \(\hat r\) (**H1**) |
| `low_budget`: Acc high, return ↓ | Budget hurts via amplification (**H2**) |

---

## What we deliberately exclude

- B-Pref Mistake / Stoc / Skip / Equal teachers  
- PrefPPO comparisons (out of scope for this diagnostic)  
- Real human studies (paper §5.4) — not needed for H1/H2

Those are valuable follow-ups; they are not required to answer the open question *about PEBBLE*.
