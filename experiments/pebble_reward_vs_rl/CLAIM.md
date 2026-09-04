# Pre-registered claim (PEBBLE reward-quality vs RL amplification)

**Primary source:** Lee, Smith & Abbeel, *PEBBLE*, ICML 2021 — `lee2021pebble.pdf` (arXiv:2106.05091).  
**Not B-Pref.** This study stays inside PEBBLE’s protocol and narratives (§4–§5).

**Status of this file:** design registration. Code implements this protocol.  
**Status of findings:** none yet — no diagnostic- or paper-scale results are claimed here.

---

## 1. Claim (falsifiable)

> When PEBBLE’s **ground-truth episode return** drops under PEBBLE’s own ablations  
> (`no_relabel`, `no_pretrain`, `low_budget` vs `full`; paper Fig. 5a / Fig. 3),  
> the drop is accompanied by a measurable decline in **Bradley–Terry ranking accuracy**  
> of \(\hat r\) on held-out and/or on-policy Oracle pairs (**H1**),  
> **or** ranking accuracy remains high while a same-buffer causal probe shows  
> SAC recovers under \(r_{\mathrm{true}}\) but not under \(\hat r\) (**H2**).

This is an **open mechanism question**. PEBBLE reports only true return and never Acc(\(\hat r\)).

### What would falsify a clean H1 / H2 story

| Outcome | Interpretation |
|---------|----------------|
| Return drops **and** Acc_holdout / Acc_onpolicy drop | Supports **H1** for that ablation |
| Return drops, Acc stays high (±≤0.03 vs `full`), probe_gap large | Supports **H2** |
| Return drops, Acc high, probe_gap ≈ 0 | Neither pure H1 nor H2 — visitation / exploration confound |
| Return does **not** drop under an ablation at our scale | Ablation underpowered at this budget/env — **do not** invent H1/H2 |

---

## 2. Conditions (mechanism-isolating)

Teacher always = paper §5.1 **scripted Oracle** (prefs from true segment return).

| ID | Relabel (§4.3) | Pre-train 10k (§4.1) | `max_feedback` | Paper map |
|----|----------------|----------------------|----------------|-----------|
| `full` | yes | yes | 1400 | PEBBLE default |
| `no_relabel` | **no** | yes | 1400 | Fig. 5a |
| `no_pretrain` | yes | **no** (0) | 1400 | Fig. 5a |
| `low_budget` | yes | yes | **400** | Fig. 3 |

Only one factor changes vs `full` per ablation (except budget on `low_budget`).

---

## 3. Controls / baselines a reviewer would demand

- **`full`** is the within-paper control (not SAC-with-GT, not PrefPPO).
- Optional later: SAC on GT reward for absolute return ceiling (not required to answer H1 vs H2 *relative to PEBBLE*).
- We deliberately **exclude** B-Pref Mistake/Stoc — those change the teacher, not PEBBLE’s ablations.

---

## 4. Metrics (mapped to the claim)

Logged after **every** reward-model update (instrumentation only; learning rule unchanged except `do_relabel`):

| Metric | Role |
|--------|------|
| `true_return` | Paper §5.1 downstream quantity |
| `acc_pref_buffer` | Fit quality on stored Oracle prefs |
| `acc_holdout` | Stationary ranking quality (frozen pairs) |
| `acc_onpolicy` | Ranking quality **where SAC acts now** |
| `acc_holdout_q1…q5` | Near-tie vs easy pairs (\|ΔR\| quintiles) |
| `spearman_rhat_rtrue` | Pointwise rank agreement on replay |
| `probe_gap` = \(R(\mathrm{GT})-R(\hat r)\) same buffer | Causal isolation of reward vs visitation |

---

## 5. Confounds and how we handle them

| Confound | Mitigation |
|----------|------------|
| Changing teacher (B-Pref noise) | Forbidden — Oracle only |
| Holdout contaminated by training prefs | Freeze after pretrain; never train on holdout |
| Acc on teacher labels under noise | N/A (Oracle); still report buffer Acc for fit |
| Probe changing the run | Snapshot/restore agent weights + buffer rewards |
| Env harder than paper ablation env | Primary = Walker-walk (Fig. 3); optional Quadruped for Fig. 5a match |
| Diagnostic-scale underpower | Explicit stages: smoke → diagnostic → paper-scale; no claim before stage |

---

## 6. Seed plan

| Stage | Purpose | Seeds | Steps (Walker) | Claim allowed? |
|-------|---------|-------|----------------|----------------|
| Smoke | Wiring | 1 | ~8k | **No** |
| Diagnostic | Pattern visibility | **5** | 100k | Tentative pattern only |
| Paper-scale | Claim-grade | **10** (paper §5.1) | ≥500k | Yes, if consistent |

Report **mean ± std across seeds**. Single-seed anecdotes are not findings.

---

## 7. Pre-registered interpretation (per ablation)

| Ablation | Acc ↓ with return | Acc stable + large probe_gap | Acc stable + probe_gap≈0 |
|----------|-------------------|------------------------------|---------------------------|
| `no_relabel` | H1 component (drift) | H2 / §4.3 non-stationarity | Visitation-limited |
| `no_pretrain` | H1 (“worse-shaped reward”) | H2-ish | Pretrain helps exploration only |
| `low_budget` | H1 underfit | H2 amplification | Data-limited policy |

---

## 8. Intentional deviations from paper (documented)

| Paper | This instrument default | Why OK for the claim |
|-------|-------------------------|----------------------|
| Fig. 5a on Quadruped | Walker primary | Same ablations; cheaper; optional `ENV=quadruped_walk` |
| 10 seeds, long horizon | Staged budgets | Claim gate requires paper-scale before conclusion |
| reward_batch / updates in release code | Configurable; diagnostic smaller | Does not change Oracle teacher or ablation factors |
| `no_pretrain` first query timing | `num_seed_steps=2000` (random only) | Walker episodes ≈1000 steps; PEBBLE needs ≥2 trajs before first query. This is **not** unsupervised pre-train (entropy SAC). Prefer `run_condition.sh` default. |

---

## Lineage check

- Paper: PEBBLE (Lee et al., 2021)  
- Code host: BPref repo’s `train_PEBBLE.py` (implementation vehicle only)  
- **Not** answering B-Pref teacher-noise questions
