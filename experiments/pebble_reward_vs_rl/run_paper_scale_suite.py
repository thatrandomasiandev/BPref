#!/usr/bin/env python3
"""
Paper-scale suite orchestrator (CLAIM.md §6 claim gate).

Protocol: 10 seeds × 4 conditions × ≥500k steps (paper §5.1 uses 10 seeds).
Raises reward_batch / reward_update / eval episodes toward paper settings.

On a CPU Mac this is multi-day. Prefer CUDA (Colab / cluster). Resume-safe:
skips completed seed×condition folders.

Usage:

    DEVICE=cuda PARALLEL=2 STEPS=500000 SEEDS="1 2 3 4 5 6 7 8 9 10" \\
      python experiments/pebble_reward_vs_rl/run_paper_scale_suite.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Reuse diagnostic runner with paper defaults via env vars.
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "pebble_reward_vs_rl"))

# Set paper-scale defaults before importing/running diagnostic main.
os.environ.setdefault("STEPS", "500000")
os.environ.setdefault("SEEDS", "1 2 3 4 5 6 7 8 9 10")
os.environ.setdefault("PARALLEL", "2")
os.environ.setdefault("DEVICE", "cuda")

# Extra Hydra overrides for paper-ish reward learning — passed via run_condition
# by setting PAPER_OVERRIDES that run_diagnostic_suite doesn't know about yet.
# We patch by wrapping run_condition through an env flag consumed below.

PAPER_HYDRA = [
    "reward_batch=128",
    "reward_update=200",
    "num_eval_episodes=10",
    "eval_frequency=10000",
]


def main() -> int:
    # Monkey-patch: inject paper hydra args into subprocess by setting env
    # that a thin wrapper reads — simplest path: call run_condition with extras
    # by temporarily replacing run_condition.sh behavior via EXTRA_HYDRA.
    os.environ["EXTRA_HYDRA"] = " ".join(PAPER_HYDRA)
    print("=== PAPER-SCALE SUITE (CLAIM.md §6 claim gate) ===", flush=True)
    print(f"extra hydra: {PAPER_HYDRA}", flush=True)
    print(
        "If DEVICE=cpu: expect multi-day wall time. Prefer CUDA.",
        flush=True,
    )

    # Import after env defaults
    from run_diagnostic_suite import main as diag_main

    # Patch run_one to forward EXTRA_HYDRA — do it by editing env for bash
    # run_condition already forwards "$@"; we need suite to pass them.
    import run_diagnostic_suite as suite

    _orig = suite.run_one

    def run_one_paper(**kwargs):
        # Inject by wrapping subprocess — patch suite.run_one body via env
        # Simplest: append to STEPS path by setting CONDITION runner extras
        extras = os.environ.get("EXTRA_HYDRA", "").split()
        # Call original but we need to modify subprocess — override here
        import csv
        import subprocess
        import time
        from pathlib import Path as P

        cond = kwargs["cond"]
        seed = kwargs["seed"]
        device = kwargs["device"]
        steps = kwargs["steps"]
        env = kwargs["env"]
        log_dir = kwargs["log_dir"]
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
        env_vars["PYTHONPATH"] = (
            f"{suite.ROOT}:{suite.ROOT / 'custom_dmc2gym'}:{env_vars.get('PYTHONPATH', '')}"
        )
        cmd = ["bash", str(suite.EXP / "run_condition.sh")] + extras
        t0 = time.time()
        with log_path.open("w") as logf:
            proc = subprocess.run(
                cmd, cwd=str(suite.ROOT), env=env_vars, stdout=logf, stderr=subprocess.STDOUT
            )
        elapsed = time.time() - t0
        if proc.returncode == 0 and suite.is_complete(env, cond, seed, steps):
            return tag, 0, f"done ({elapsed/60:.1f} min)"
        return tag, proc.returncode or 1, f"FAIL rc={proc.returncode} log={log_path}"

    suite.run_one = run_one_paper
    return diag_main()


if __name__ == "__main__":
    sys.exit(main())
