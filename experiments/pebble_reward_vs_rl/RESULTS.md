# Results log — PEBBLE H1/H2 study

**Pre-registration:** [`CLAIM.md`](CLAIM.md)  
**Rule:** smoke ≠ diagnostic ≠ paper-scale ≠ justified conclusion.

This file is updated only from aggregated multi-seed outputs.  
Do not paste single-seed anecdotes here as findings.

---

## Current status

| Stage | State |
|-------|-------|
| Setup / instrument | Complete |
| Smoke (wiring) | Complete (archived under `exp/_smoke_archive/`) |
| Diagnostic (5×4×100k Walker) | **In progress** — see `exp/_logs/` |
| Paper-scale (10×4×≥500k) | Not started (needs CUDA; CPU ≈ multi-day) |
| Conclusion justified | **No** |

---

## Diagnostic results (fill after suite finishes)

_Command:_

```bash
python experiments/pebble_reward_vs_rl/analyze_results.py \
  --root experiments/pebble_reward_vs_rl/exp
```

Paste `summary_by_condition.csv` means here, then apply CLAIM.md §7 readings.

### Tentative pattern table (diagnostic only)

| Condition | true_return | Acc_holdout | Acc_onpolicy | probe_gap | Reading |
|-----------|-------------|-------------|--------------|-----------|---------|
| full | — | — | — | — | baseline |
| no_relabel | — | — | — | — | — |
| no_pretrain | — | — | — | — | — |
| low_budget | — | — | — | — | — |

**Allowed claim at this stage:** tentative pattern description only.  
**Forbidden:** “we showed H1/H2” without paper-scale.

---

## Paper-scale results

_Not yet run._

```bash
DEVICE=cuda PARALLEL=2 \
  python experiments/pebble_reward_vs_rl/run_paper_scale_suite.py
```
