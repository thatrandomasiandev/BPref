#!/usr/bin/env python3
"""
Diagnostic suite orchestrator (CLAIM.md §6).

Runs: 5 seeds × {full, no_relabel, no_pretrain, low_budget} × 100k steps.
Skips jobs whose diagnostics.csv already has a final row at the target step count.

Produces *tentative patterns only* — not a paper-scale claim.

Usage (from BPref repo root, venv active):

    DEVICE=cpu PARALLEL=3 python experiments/pebble_reward_vs_rl/run_diagnostic_suite.py
"""

from __future__ import annotations

import csv
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "experiments" / "pebble_reward_vs_rl"
CONDS = ["full", "no_relabel", "no_pretrain", "low_budget"]


def is_complete(env: str, cond: str, seed: int, steps: int) -> bool:
    csv_path = EXP / "exp" / env / cond / f"seed{seed}" / "diagnostics.csv"
    if not csv_path.is_file():
        return False
    with csv_path.open() as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return False
    last = rows[-1]
    try:
        step = int(float(last["step"]))
    except (KeyError, ValueError):
        return False
    return step >= steps - 1 or last.get("notes", "") == "final"


def run_one(
    *,
    cond: str,
    seed: int,
    device: str,
    steps: int,
    env: str,
    log_dir: Path,
) -> tuple[str, int, str]:
    tag = f"{cond}_seed{seed}"
    if is_complete(env, cond, seed, steps):
        return tag, 0, "skip"

    log_path = log_dir / f"{tag}.log"
    env_vars = os.environ.copy()
    env_vars["CONDITION"] = cond
    env_vars["DEVICE"] = device
    env_vars["SEED"] = str(seed)
    env_vars["STEPS"] = str(steps)
    env_vars["ENV"] = env
    env_vars["PYTHONUNBUFFERED"] = "1"
    env_vars["PYTHONPATH"] = f"{ROOT}:{ROOT / 'custom_dmc2gym'}:{env_vars.get('PYTHONPATH', '')}"

    cmd = ["bash", str(EXP / "run_condition.sh")]
    t0 = time.time()
    with log_path.open("w") as logf:
        proc = subprocess.run(
            cmd,
            cwd=str(ROOT),
            env=env_vars,
            stdout=logf,
            stderr=subprocess.STDOUT,
        )
    elapsed = time.time() - t0
    if proc.returncode == 0 and is_complete(env, cond, seed, steps):
        return tag, 0, f"done ({elapsed/60:.1f} min)"
    return tag, proc.returncode or 1, f"FAIL rc={proc.returncode} log={log_path}"


def main() -> int:
    device = os.environ.get("DEVICE", "cpu")
    parallel = int(os.environ.get("PARALLEL", "3"))
    steps = int(os.environ.get("STEPS", "100000"))
    env = os.environ.get("ENV", "walker_walk")
    seeds = [int(s) for s in os.environ.get("SEEDS", "1 2 3 4 5").split()]

    log_dir = EXP / "exp" / "_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    master = log_dir / f"diagnostic_suite_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    jobs = [(c, s) for s in seeds for c in CONDS]
    print(f"=== DIAGNOSTIC SUITE ===", flush=True)
    print(f"device={device} parallel={parallel} steps={steps} seeds={seeds} env={env}", flush=True)
    print(f"jobs={len(jobs)} master_log={master}", flush=True)
    print("STATUS GATE: tentative patterns only (CLAIM.md §6)", flush=True)

    # Archive smoke dirs so they do not pollute analysis means.
    smoke_arch = EXP / "exp" / "_smoke_archive"
    smoke_arch.mkdir(parents=True, exist_ok=True)
    walk = EXP / "exp" / env
    if walk.is_dir():
        for p in walk.glob("*/*"):
            if p.is_dir() and ("_smoke" in p.name or "_mps_probe" in p.name):
                dest = smoke_arch / f"{p.parent.name}__{p.name}"
                if not dest.exists():
                    p.rename(dest)
                    print(f"archived smoke {p} -> {dest}", flush=True)

    pending = [(c, s) for c, s in jobs if not is_complete(env, c, s, steps)]
    skipped = len(jobs) - len(pending)
    print(f"skip={skipped} pending={len(pending)}", flush=True)

    fails = 0
    with master.open("w") as mf:
        def _log(msg: str) -> None:
            print(msg, flush=True)
            mf.write(msg + "\n")
            mf.flush()

        with ThreadPoolExecutor(max_workers=parallel) as pool:
            futs = {
                pool.submit(
                    run_one,
                    cond=c,
                    seed=s,
                    device=device,
                    steps=steps,
                    env=env,
                    log_dir=log_dir,
                ): (c, s)
                for c, s in pending
            }
            for fut in as_completed(futs):
                tag, rc, status = fut.result()
                _log(f"[{status}] {tag}")
                if rc != 0:
                    fails += 1

    print(f"=== SUITE FINISHED fail={fails} ===", flush=True)
    print(
        "Aggregate:\n"
        "  python experiments/pebble_reward_vs_rl/analyze_results.py "
        "--root experiments/pebble_reward_vs_rl/exp",
        flush=True,
    )
    return fails


if __name__ == "__main__":
    sys.exit(main())
