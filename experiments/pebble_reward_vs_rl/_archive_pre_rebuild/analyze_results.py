#!/usr/bin/env python3
"""
Aggregate PEBBLE ablation diagnostics and summarize H1 vs H2 patterns.

Usage (from BPref repo root):

    python experiments/pebble_reward_vs_rl/analyze_results.py \
        --root experiments/pebble_reward_vs_rl/exp

Writes:
  - summary_by_condition.csv   (last snapshot per seed, mean±std)
  - trajectories.csv           (all rows concatenated)
  - h1_h2_overview.png         (if matplotlib available)
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple


COND_ORDER = ["full", "no_relabel", "no_pretrain", "low_budget"]

METRIC_KEYS = [
    "true_return",
    "acc_pref_buffer",
    "acc_holdout",
    "acc_onpolicy",
    "spearman_rhat_rtrue",
    "probe_gap",
    "probe_return_rhat",
    "probe_return_true",
    "total_feedback",
]


def _find_csvs(root: Path) -> List[Path]:
    return sorted(root.rglob("diagnostics.csv"))


def _read_csv(path: Path) -> List[dict]:
    with path.open() as f:
        return list(csv.DictReader(f))


def _f(row: dict, key: str) -> float:
    v = row.get(key, "")
    if v is None or v == "" or v == "nan":
        return float("nan")
    try:
        return float(v)
    except ValueError:
        return float("nan")


def _mean_std(xs: List[float]) -> Tuple[float, float]:
    ys = [x for x in xs if not math.isnan(x)]
    if not ys:
        return float("nan"), float("nan")
    m = sum(ys) / len(ys)
    if len(ys) == 1:
        return m, 0.0
    var = sum((x - m) ** 2 for x in ys) / (len(ys) - 1)
    return m, math.sqrt(var)


def _reading(cond: str, full: Optional[dict], row: dict) -> str:
    """Compare ablation end-state to `full` (when available)."""
    if full is None or cond == "full":
        return "baseline"
    d_ret = _f(row, "true_return") - _f(full, "true_return")
    d_hold = _f(row, "acc_holdout") - _f(full, "acc_holdout")
    d_on = _f(row, "acc_onpolicy") - _f(full, "acc_onpolicy")
    gap = _f(row, "probe_gap")

    ret_drop = d_ret < -20.0  # rough walker scale; adjust after looking at curves
    hold_drop = d_hold < -0.05
    on_drop = (not math.isnan(d_on)) and d_on < -0.05
    gap_large = (not math.isnan(gap)) and gap > 50.0

    if not ret_drop:
        return "no_clear_return_drop"
    if hold_drop or on_drop:
        return "H1_reward_model"
    if gap_large:
        return "H2_amplification"
    return "mixed_or_need_more_seeds"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=str,
        default="experiments/pebble_reward_vs_rl/exp",
    )
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    root = Path(args.root)
    out = Path(args.out) if args.out else root / "_analysis"
    out.mkdir(parents=True, exist_ok=True)

    csvs = _find_csvs(root)
    if not csvs:
        print(f"No diagnostics.csv under {root}")
        print("Run: CONDITION=full bash experiments/pebble_reward_vs_rl/run_condition.sh")
        return

    rows: List[dict] = []
    for p in csvs:
        for r in _read_csv(p):
            r["_source"] = str(p)
            rows.append(r)
    print(f"Loaded {len(rows)} rows from {len(csvs)} runs")

    traj_path = out / "trajectories.csv"
    if rows:
        keys = list(rows[0].keys())
        with traj_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)

    # Last snapshot per (condition, seed)
    last: Dict[str, Dict[int, dict]] = defaultdict(dict)
    for r in rows:
        cond = r["condition"]
        seed = int(float(r["seed"]))
        step = int(float(r["step"]))
        prev = last[cond].get(seed)
        if prev is None or step >= int(float(prev["step"])):
            last[cond][seed] = r

    # Mean of full condition across seeds (for relative reading)
    full_means: Optional[dict] = None
    if "full" in last and last["full"]:
        full_means = {}
        for k in METRIC_KEYS:
            vals = [_f(r, k) for r in last["full"].values()]
            full_means[k], _ = _mean_std(vals)

    summary_path = out / "summary_by_condition.csv"
    fields = ["condition", "n_seeds", "reading"]
    for k in METRIC_KEYS:
        fields += [f"{k}_mean", f"{k}_std"]

    with summary_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for cond in COND_ORDER + sorted(set(last) - set(COND_ORDER)):
            seeds = last.get(cond, {})
            if not seeds:
                continue
            # Per-seed reading vs full mean
            readings = [
                _reading(cond, full_means, r) for r in seeds.values()
            ]
            # Majority vote
            reading = max(set(readings), key=readings.count)
            row_out = {
                "condition": cond,
                "n_seeds": len(seeds),
                "reading": reading,
            }
            for k in METRIC_KEYS:
                m, s = _mean_std([_f(r, k) for r in seeds.values()])
                row_out[f"{k}_mean"] = f"{m:.4f}" if not math.isnan(m) else ""
                row_out[f"{k}_std"] = f"{s:.4f}" if not math.isnan(s) else ""
            w.writerow(row_out)
            print(
                f"{cond:12s}  n={len(seeds)}  "
                f"R={row_out['true_return_mean']:8s}  "
                f"Acc_h={row_out['acc_holdout_mean']:6s}  "
                f"Acc_o={row_out['acc_onpolicy_mean']:6s}  "
                f"gap={row_out['probe_gap_mean']:8s}  → {reading}"
            )

    print(f"Wrote {summary_path}")
    print(f"Wrote {traj_path}")

    # Optional plot
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skip plot")
        return

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    panels = [
        ("true_return", "True return (paper metric)"),
        ("acc_holdout", "Acc holdout (Oracle)"),
        ("acc_onpolicy", "Acc on-policy (Oracle)"),
        ("spearman_rhat_rtrue", "Spearman(r̂, r_true)"),
    ]
    by_cond = defaultdict(list)
    for r in rows:
        by_cond[r["condition"]].append(r)

    for ax, (key, title) in zip(axes.ravel(), panels):
        for cond in COND_ORDER:
            pts = by_cond.get(cond, [])
            if not pts:
                continue
            # Mean over seeds at each step
            by_step: Dict[int, List[float]] = defaultdict(list)
            for r in pts:
                by_step[int(float(r["step"]))].append(_f(r, key))
            steps = sorted(by_step)
            means = [_mean_std(by_step[s])[0] for s in steps]
            ax.plot(steps, means, marker="o", label=cond)
        ax.set_title(title)
        ax.set_xlabel("env step")
        ax.grid(True, alpha=0.3)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("PEBBLE ablations: reward quality vs true return")
    fig.tight_layout()
    fig_path = out / "h1_h2_overview.png"
    fig.savefig(fig_path, dpi=140)
    print(f"Wrote {fig_path}")


if __name__ == "__main__":
    main()
