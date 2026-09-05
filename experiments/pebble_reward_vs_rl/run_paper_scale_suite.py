#!/usr/bin/env python3
"""
Paper-scale suite orchestrator (CLAIM.md §6 claim gate).

Protocol: 10 seeds × 4 conditions × ≥500k steps (paper §5.1 uses 10 seeds),
with paper-ish reward_batch / reward_update / eval episodes.

Prefer CUDA (Colab A100). Resume-safe: skips completed seed×condition folders.

Usage:

    DEVICE=cuda PARALLEL=2 STEPS=500000 SEEDS=\"1 2 3 4 5 6 7 8 9 10\" \\
      python experiments/pebble_reward_vs_rl/run_paper_scale_suite.py

A100 defaults: DEVICE=cuda, PARALLEL=2 (wall-clock only; protocol unchanged).
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

# Paper defaults before importing the diagnostic orchestrator.
os.environ.setdefault("STEPS", "500000")
os.environ.setdefault("SEEDS", "1 2 3 4 5 6 7 8 9 10")
os.environ.setdefault("PARALLEL", "2")  # A100 40GB; drop to 1 on OOM
os.environ.setdefault("DEVICE", "cuda")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "pebble_reward_vs_rl"))

import run_diagnostic_suite as suite  # noqa: E402

PAPER_HYDRA = [
    "reward_batch=128",
    "reward_update=200",
    "num_eval_episodes=10",
    "eval_frequency=10000",
]


def run_one_paper(
    *,
    cond: str,
    seed: int,
    device: str,
    steps: int,
    env: str,
    log_dir: Path,
):
    """Same as diagnostic run_one, plus paper Hydra overrides."""
    tag = f"{cond}_seed{seed}"
    if suite.is_complete(env, cond, seed, steps):
        return tag, 0, "skip"

    log_path = log_dir / f"paper_{tag}.log"
    env_vars = os.environ.copy()
    env_vars["CONDITION"] = cond
    env_vars["DEVICE"] = device
    env_vars["SEED"] = str(seed)
    env_vars["STEPS"] = str(steps)
    env_vars["ENV"] = env
    env_vars["PYTHONUNBUFFERED"] = "1"
    env_vars["PYTHONPATH"] = (
        f"{suite.ROOT}:{suite.ROOT / 'custom_dmc2gym'}:{env_vars.get('PYTHONPATH', '')}"
    )
    cmd = ["bash", str(suite.EXP / "run_condition.sh")] + PAPER_HYDRA
    t0 = time.time()
    with log_path.open("w") as logf:
        proc = subprocess.run(
            cmd,
            cwd=str(suite.ROOT),
            env=env_vars,
            stdout=logf,
            stderr=subprocess.STDOUT,
        )
    elapsed = time.time() - t0
    if suite.is_complete(env, cond, seed, steps):
        note = f"done ({elapsed / 60:.1f} min)"
        if proc.returncode not in (0, None):
            note += f" [rc={proc.returncode} ignored — diagnostics complete]"
        return tag, 0, note
    return tag, proc.returncode or 1, f"FAIL rc={proc.returncode} log={log_path}"


def main() -> int:
    print("=== PAPER-SCALE SUITE (CLAIM.md §6 claim gate) ===", flush=True)
    print(f"extra hydra: {PAPER_HYDRA}", flush=True)
    print("Prefer DEVICE=cuda (Colab A100). CPU is multi-day.", flush=True)
    suite.run_one = run_one_paper
    return suite.main()


if __name__ == "__main__":
    sys.exit(main())
