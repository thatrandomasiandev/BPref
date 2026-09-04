"""
PEBBLE diagnostics: reward-model ranking quality vs RL amplification.

Primary source
--------------
Lee, Smith & Abbeel, PEBBLE, ICML 2021 (arXiv:2106.05091).
See CLAIM.md (pre-registration) and PAPER_NOTES.md / lee2021pebble.pdf.

Scientific role of this module
------------------------------
PEBBLE's published curves report only *true episode return*. When return drops
under the paper's ablations (Fig. 5a: no-relabel / no-pretrain; Fig. 3: low
feedback), we still do not know whether:

  H1  The reward model ranks segments worse (Bradley–Terry Acc drops), or
  H2  Downstream SAC amplifies a still-decent r̂ (Acc high; probe_gap large).

This file implements the *measurements* only. It does not change PEBBLE's
learning rule. The train loop (train_pebble_diagnostics.py) calls these hooks.

Teacher
-------
Scripted Oracle only (paper §5.1). Preferences come from ground-truth segment
returns. B-Pref Mistake/Stoc channels are out of scope for this study.
"""

from __future__ import annotations

import copy
import csv
import json
import os
from dataclasses import asdict, dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch


# =============================================================================
# Preference pair containers
# =============================================================================

@dataclass
class PairSet:
    """A set of trajectory-segment pairs for Bradley–Terry evaluation.

    Geometry matches PEBBLE's RewardModel buffers:
      seg* shape = (N, H, obs_dim + act_dim)
      true_ret_* = sum of environment GT rewards over the H-step segment
                   (the Oracle ranking signal; paper §5.1).

    Why we store GT returns
    -----------------------
    Under an Oracle teacher, buffer labels equal Oracle rankings. For holdout
    and on-policy sets we *never* train on the pairs, so we must score Acc by
    re-deriving the Oracle label from GT segment returns, not from any stored
    teacher bit that could be confused with a noisy channel later.
    """

    name: str
    seg1: np.ndarray
    seg2: np.ndarray
    true_ret_1: np.ndarray  # (N, 1)
    true_ret_2: np.ndarray  # (N, 1)

    def __len__(self) -> int:
        return int(self.seg1.shape[0])

    @property
    def abs_delta_r(self) -> np.ndarray:
        """|ΔR| = |return(seg2) − return(seg1)|.

        Near-ties (small |ΔR|) are the hardest ranking cases; easy pairs
        (large |ΔR|) can look accurate even when r̂ is mediocre. Quintile
        stratification in `stratified_holdout` uses this.
        """
        return np.abs(self.true_ret_2 - self.true_ret_1).reshape(-1)

    @property
    def oracle_labels(self) -> np.ndarray:
        """Hard Oracle preference in PEBBLE's convention: 1 iff seg2 ≻ seg1."""
        return (1 * (self.true_ret_1 < self.true_ret_2)).astype(np.int64)


@dataclass
class Snapshot:
    """One longitudinal row written after a reward-model update.

    Every field exists because it maps to CLAIM.md §4. Do not add vanity
    columns without updating the claim document.
    """

    step: int
    condition: str          # full | no_relabel | no_pretrain | low_budget
    seed: int
    total_feedback: int     # cumulative preference queries (paper "pieces of feedback")

    # --- Bradley–Terry Acc vs Oracle (H1 signal) ---
    acc_pref_buffer: float  # D_pref: live preference buffer
    acc_holdout: float      # D_holdout: frozen, never trained on
    acc_onpolicy: float     # D_onpolicy: recent policy trajectories

    # Holdout Acc within |ΔR| quintiles (q1 = hardest near-ties)
    acc_holdout_q1: float
    acc_holdout_q2: float
    acc_holdout_q3: float
    acc_holdout_q4: float
    acc_holdout_q5: float

    # Pointwise agreement of r̂(s,a) with r_true on the replay buffer
    # (what the SAC critic actually regresses against after relabel).
    spearman_rhat_rtrue: float
    pearson_rhat_rtrue: float
    n_buffer: int

    # Paper §5.1 downstream metric
    true_return: float

    # Causal probe (NaN if not run this step): same buffer, SAC under r̂ vs GT
    probe_return_rhat: float = float("nan")
    probe_return_true: float = float("nan")
    probe_gap: float = float("nan")  # true − rhat; large + ⇒ reward bottleneck

    n_pref: int = 0
    n_holdout: int = 0
    n_onpolicy: int = 0
    notes: str = ""


# =============================================================================
# Trajectory store helpers (RewardModel.inputs / .targets)
# =============================================================================

def n_complete_trajs(reward_model) -> int:
    """Count *finished* episodes in RewardModel's rolling trajectory store.

    PEBBLE appends (s,a,r_true) via add_data; an in-progress episode is shorter
    than a completed one. We need ≥2 complete trajs before sampling segment
    pairs (same precondition as the paper's first query batch after explore).
    """
    if len(reward_model.inputs) == 0:
        return 0
    last = reward_model.inputs[-1]
    if len(last) == 0:
        return max(0, len(reward_model.inputs) - 1)
    if len(reward_model.inputs) >= 2 and len(last) < len(reward_model.inputs[0]):
        return len(reward_model.inputs) - 1
    return len(reward_model.inputs)


def sample_pairs_from_trajs(
    reward_model,
    n_pairs: int,
    rng: np.random.RandomState,
    traj_indices: Optional[Sequence[int]] = None,
) -> Optional[PairSet]:
    """Sample (seg1, seg2) with GT segment returns — PEBBLE query geometry.

    Mirrors RewardModel.get_queries layout so Acc is measured on the same
    segment length H = reward_model.size_segment (paper default H=50 in
    released code; Fig. 5c compares H=50 vs H=1).
    """
    n_traj = n_complete_trajs(reward_model)
    if n_traj < 2:
        return None

    if traj_indices is None:
        traj_indices = list(range(n_traj))
    else:
        traj_indices = [i for i in traj_indices if i < n_traj]
    if len(traj_indices) < 2:
        return None

    H = reward_model.size_segment
    len_traj = len(reward_model.inputs[traj_indices[0]])
    if len_traj < H:
        return None

    # Stack selected trajectories: (n_avail, T, dim)
    inputs = np.array([reward_model.inputs[i] for i in traj_indices])
    targets = np.array([reward_model.targets[i] for i in traj_indices])
    n_avail = inputs.shape[0]

    # Two independent trajectory indices per pair (with replacement, as PEBBLE).
    i2 = rng.choice(n_avail, size=n_pairs, replace=True)
    i1 = rng.choice(n_avail, size=n_pairs, replace=True)

    sa1 = inputs[i1].reshape(-1, inputs.shape[-1])
    r1 = targets[i1].reshape(-1, targets.shape[-1])
    sa2 = inputs[i2].reshape(-1, inputs.shape[-1])
    r2 = targets[i2].reshape(-1, targets.shape[-1])

    # Base index grid for length-H windows; random start offsets per pair.
    base = np.array([list(range(i * len_traj, i * len_traj + H)) for i in range(n_pairs)])
    off1 = rng.choice(len_traj - H + 1, size=n_pairs, replace=True).reshape(-1, 1)
    off2 = rng.choice(len_traj - H + 1, size=n_pairs, replace=True).reshape(-1, 1)

    seg1 = np.take(sa1, base + off1, axis=0).astype(np.float32)
    seg2 = np.take(sa2, base + off2, axis=0).astype(np.float32)
    ret1 = np.take(r1, base + off1, axis=0).sum(axis=1).astype(np.float32)
    ret2 = np.take(r2, base + off2, axis=0).sum(axis=1).astype(np.float32)

    return PairSet("sampled", seg1, seg2, ret1, ret2)


# =============================================================================
# Bradley–Terry accuracy
# =============================================================================

def _pred_prefer_seg2(reward_model, seg1: np.ndarray, seg2: np.ndarray) -> np.ndarray:
    """Ensemble hard prediction: 1 iff mean P(seg2 ≻ seg1) > 0.5.

    RewardModel.get_rank_probability returns P(seg1 ≻ seg2); we invert for
    consistency with PEBBLE's label convention (1 = seg2 preferred).
    """
    p_seg1_wins, _ = reward_model.get_rank_probability(seg1, seg2)
    return (p_seg1_wins < 0.5).astype(np.int64)


def bt_accuracy_oracle(reward_model, pairs: Optional[PairSet]) -> float:
    """Hard BT accuracy against Oracle GT segment ranking.

    Exact ties (|ΔR|=0) are dropped: the Oracle has no preference, so Acc is
    undefined on those pairs (including them would invent a label).
    """
    if pairs is None or len(pairs) == 0:
        return float("nan")
    pred = _pred_prefer_seg2(reward_model, pairs.seg1, pairs.seg2)
    labels = pairs.oracle_labels.reshape(-1)
    mask = pairs.abs_delta_r > 0
    if mask.sum() == 0:
        return float("nan")
    return float((pred[mask] == labels[mask]).mean())


def bt_accuracy_vs_buffer_labels(reward_model) -> float:
    """BT accuracy on the live preference buffer vs stored Oracle labels.

    Under paper §5.1 Oracle, stored labels *are* the Oracle ranking. This
    measures fit quality on D_pref (CLAIM.md). It is *not* a generalization
    measure — that is acc_holdout / acc_onpolicy.
    """
    n = reward_model.capacity if reward_model.buffer_full else reward_model.buffer_index
    if n <= 0:
        return float("nan")
    seg1 = reward_model.buffer_seg1[:n]
    seg2 = reward_model.buffer_seg2[:n]
    labels = reward_model.buffer_label[:n].reshape(-1).astype(np.int64)
    pred = _pred_prefer_seg2(reward_model, seg1, seg2)
    mask = labels >= 0
    if mask.sum() == 0:
        return float("nan")
    return float((pred[mask] == labels[mask]).mean())


def stratified_holdout(
    reward_model, pairs: Optional[PairSet], n_bins: int = 5
) -> List[float]:
    """Oracle BT Acc within each |ΔR| quintile (q1 = near-ties).

    If only overall Acc is high because easy pairs dominate, H1 can hide in
    near-ties. Quintiles make that visible.
    """
    if pairs is None or len(pairs) < n_bins:
        return [float("nan")] * n_bins
    abs_d = pairs.abs_delta_r
    mask = abs_d > 0
    if mask.sum() < n_bins:
        return [float("nan")] * n_bins
    abs_m = abs_d[mask]
    pred = _pred_prefer_seg2(reward_model, pairs.seg1, pairs.seg2)[mask]
    labels = pairs.oracle_labels.reshape(-1)[mask]
    edges = np.quantile(abs_m, np.linspace(0, 1, n_bins + 1))
    edges[0] -= 1e-8
    edges[-1] += 1e-8
    out: List[float] = []
    for b in range(n_bins):
        if b == n_bins - 1:
            in_bin = (abs_m >= edges[b]) & (abs_m <= edges[b + 1])
        else:
            in_bin = (abs_m >= edges[b]) & (abs_m < edges[b + 1])
        out.append(
            float("nan")
            if in_bin.sum() == 0
            else float((pred[in_bin] == labels[in_bin]).mean())
        )
    return out


def reward_correlations(
    reward_model,
    replay_buffer,
    max_n: int = 4096,
    rng: Optional[np.random.RandomState] = None,
) -> Tuple[float, float, int]:
    """Spearman / Pearson of r̂(s,a) vs r_true(s,a) on the replay buffer.

    After PEBBLE relabel (§4.3), SAC trains on scalar r̂. Rank correlation on
    visited (s,a) probes the learning signal the critic sees — complementary
    to segment-level BT Acc.
    """
    n = replay_buffer.capacity if replay_buffer.full else replay_buffer.idx
    if n < 16:
        return float("nan"), float("nan"), n
    rng = rng or np.random.RandomState(0)
    take = min(n, max_n)
    idxs = rng.choice(n, size=take, replace=False)
    x = np.concatenate([replay_buffer.obses[idxs], replay_buffer.actions[idxs]], axis=-1)
    true_r = replay_buffer.true_rewards[idxs].reshape(-1)
    pred_r = reward_model.r_hat_batch(x).reshape(-1)
    if np.std(pred_r) < 1e-12 or np.std(true_r) < 1e-12:
        return float("nan"), float("nan"), take
    pearson = float(np.corrcoef(pred_r, true_r)[0, 1])
    # Spearman via Pearson on midranks (no scipy dependency).
    spearman = float(
        np.corrcoef(
            pred_r.argsort().argsort().astype(np.float64),
            true_r.argsort().argsort().astype(np.float64),
        )[0, 1]
    )
    return spearman, pearson, take


# =============================================================================
# Causal probe — same buffer, r̂ vs r_true (CLAIM.md §4)
# =============================================================================

class _SilentLogger:
    """Swallow SAC logger calls during the probe (no TensorBoard pollution)."""

    def log(self, *args, **kwargs):
        return None

    def __getattr__(self, name):
        def _noop(*args, **kwargs):
            return None

        return _noop


def _clone_weights(agent):
    """Deep-copy SAC weights + optimizer state so the probe is non-invasive."""
    return {
        "actor": copy.deepcopy(agent.actor.state_dict()),
        "critic": copy.deepcopy(agent.critic.state_dict()),
        "critic_target": copy.deepcopy(agent.critic_target.state_dict()),
        "log_alpha": agent.log_alpha.detach().clone(),
        "actor_opt": copy.deepcopy(agent.actor_optimizer.state_dict()),
        "critic_opt": copy.deepcopy(agent.critic_optimizer.state_dict()),
        "alpha_opt": copy.deepcopy(agent.log_alpha_optimizer.state_dict()),
    }


def _restore_weights(agent, blob):
    """Undo probe updates — training must continue as if the probe never ran."""
    agent.actor.load_state_dict(blob["actor"])
    agent.critic.load_state_dict(blob["critic"])
    agent.critic_target.load_state_dict(blob["critic_target"])
    agent.log_alpha.data.copy_(blob["log_alpha"])
    agent.actor_optimizer.load_state_dict(blob["actor_opt"])
    agent.critic_optimizer.load_state_dict(blob["critic_opt"])
    agent.log_alpha_optimizer.load_state_dict(blob["alpha_opt"])


def run_causal_probe(
    agent,
    replay_buffer,
    env,
    utils_mod,
    gradient_steps: int,
    n_eval: int,
    step: int,
) -> Tuple[float, float, float]:
    """Same-buffer probe: SAC under r̂ vs under r_true; then restore.

    Interpretation (PEBBLE §4.3 + CLAIM.md):
      probe_gap = return(GT) − return(r̂)
        large positive → buffer supports a better policy if reward were correct
                         (reward-model bottleneck / consistent with H1)
        Acc high + gap large → small residual errors amplified by SAC (H2)
        ≈ 0 and both returns low → visitation / data limited (neither)

    Critical: we restore agent weights *and* buffer learning rewards so this
    does not alter PEBBLE's learning trajectory (CLAIM.md confound table).
    """
    if len(replay_buffer) < agent.batch_size:
        return float("nan"), float("nan"), float("nan")

    silent = _SilentLogger()
    reward_snap = replay_buffer.snapshot_rewards()
    weight_snap = _clone_weights(agent)

    def _eval() -> float:
        # Eval always uses environment GT reward (paper §5.1 metric).
        total = 0.0
        for _ in range(n_eval):
            obs = utils_mod.env_reset(env)
            agent.reset()
            done = False
            ep = 0.0
            while not done:
                with utils_mod.eval_mode(agent):
                    action = agent.act(obs, sample=False)
                obs, reward, done, _ = utils_mod.env_step(env, action)
                ep += reward
            total += ep
        return total / float(n_eval)

    # Arm A: current learning rewards (r̂ after relabel, or stale if no_relabel).
    agent.update(replay_buffer, silent, step, gradient_update=gradient_steps)
    ret_rhat = _eval()

    # Arm B: identical transitions, GT rewards.
    _restore_weights(agent, weight_snap)
    replay_buffer.relabel_with_true_rewards()
    agent.update(replay_buffer, silent, step, gradient_update=gradient_steps)
    ret_true = _eval()

    # Non-invasive restore.
    _restore_weights(agent, weight_snap)
    replay_buffer.restore_rewards(reward_snap)
    return float(ret_rhat), float(ret_true), float(ret_true - ret_rhat)


# =============================================================================
# Engine: owns holdout, CSV, and the post-update checklist
# =============================================================================

class PebbleDiagnostics:
    """Longitudinal logger for the H1/H2 experiment.

    Lifecycle
    ---------
    1. Construct at train start → writes diagnostics_meta.json + CSV header.
    2. freeze_holdout(...) once ≥2 complete trajs exist (post-pretrain ideally).
    3. log_update(...) after every reward-model fit (+ optional causal probe).
    """

    # Column order is part of the public analysis contract (analyze_results.py).
    FIELDS = [
        "step",
        "condition",
        "seed",
        "total_feedback",
        "acc_pref_buffer",
        "acc_holdout",
        "acc_onpolicy",
        "acc_holdout_q1",
        "acc_holdout_q2",
        "acc_holdout_q3",
        "acc_holdout_q4",
        "acc_holdout_q5",
        "spearman_rhat_rtrue",
        "pearson_rhat_rtrue",
        "n_buffer",
        "true_return",
        "probe_return_rhat",
        "probe_return_true",
        "probe_gap",
        "n_pref",
        "n_holdout",
        "n_onpolicy",
        "notes",
    ]

    def __init__(
        self,
        work_dir: str,
        condition: str,
        seed: int,
        holdout_pairs: int = 500,
        onpolicy_pairs: int = 256,
        probe_grad_steps: int = 200,
        probe_eval_episodes: int = 3,
        onpolicy_recent_trajs: int = 20,
    ):
        self.work_dir = work_dir
        self.condition = condition
        self.seed = seed
        self.holdout_pairs = holdout_pairs
        self.onpolicy_pairs = onpolicy_pairs
        self.probe_grad_steps = probe_grad_steps
        self.probe_eval_episodes = probe_eval_episodes
        self.onpolicy_recent_trajs = onpolicy_recent_trajs
        self.holdout: Optional[PairSet] = None

        os.makedirs(work_dir, exist_ok=True)
        self.csv_path = os.path.join(work_dir, "diagnostics.csv")
        with open(self.csv_path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=self.FIELDS).writeheader()

        # Meta is the reconstructibility record a reviewer would ask for.
        with open(os.path.join(work_dir, "diagnostics_meta.json"), "w") as f:
            json.dump(
                {
                    "paper": "Lee, Smith, Abbeel — PEBBLE, ICML 2021 (arXiv:2106.05091)",
                    "claim_file": "CLAIM.md",
                    "teacher": "scripted Oracle from GT segment returns (paper §5.1)",
                    "condition": condition,
                    "seed": seed,
                    "question": (
                        "When PEBBLE true return drops under paper ablations, "
                        "is it H1 (reward-model ranking) or H2 (SAC amplification)?"
                    ),
                    "datasets": {
                        "D_pref_buffer": "preference buffer vs Oracle labels",
                        "D_holdout": "frozen post-pretrain pairs vs Oracle",
                        "D_onpolicy": "recent on-policy pairs vs Oracle",
                    },
                    "not_bpref": True,
                    "finding_status": "none — instrumentation only until multi-seed runs",
                },
                f,
                indent=2,
            )

    def freeze_holdout(self, reward_model) -> int:
        """Freeze D_holdout once (≥2 complete trajs). Never train on these pairs.

        Timing: ideally right after unsupervised pre-train so the set reflects
        exploratory diversity (paper §4.1 motivation). If deferred (too few
        trajs), the train loop retries after the next reward update.
        """
        if self.holdout is not None and len(self.holdout) > 0:
            return len(self.holdout)
        if n_complete_trajs(reward_model) < 2:
            print("[pebble-diag] defer holdout freeze (need ≥2 complete trajs)")
            return 0
        # Fixed offset from run seed so holdout is reproducible but independent
        # of the preference-query RNG inside RewardModel.
        rng = np.random.RandomState(self.seed + 7919)
        self.holdout = sample_pairs_from_trajs(reward_model, self.holdout_pairs, rng)
        if self.holdout is None:
            return 0
        self.holdout.name = "holdout"
        np.savez_compressed(
            os.path.join(self.work_dir, "holdout_pairs.npz"),
            seg1=self.holdout.seg1,
            seg2=self.holdout.seg2,
            true_ret_1=self.holdout.true_ret_1,
            true_ret_2=self.holdout.true_ret_2,
        )
        print(f"[pebble-diag] froze D_holdout n={len(self.holdout)}")
        return len(self.holdout)

    def log_update(
        self,
        *,
        step: int,
        total_feedback: int,
        reward_model,
        replay_buffer,
        true_return: float,
        agent=None,
        env=None,
        utils_mod=None,
        run_probe: bool = False,
        notes: str = "",
    ) -> Snapshot:
        """Full metric suite after a PEBBLE reward update (CLAIM.md §4)."""

        acc_pref = bt_accuracy_vs_buffer_labels(reward_model)
        acc_hold = bt_accuracy_oracle(reward_model, self.holdout)
        qs = stratified_holdout(reward_model, self.holdout)

        # D_onpolicy: pairs from the most recent completed episodes — the
        # distribution where the current policy (and critic) actually act.
        n_traj = n_complete_trajs(reward_model)
        start = max(0, n_traj - self.onpolicy_recent_trajs)
        rng = np.random.RandomState(self.seed + 104729 + step)
        onpol = sample_pairs_from_trajs(
            reward_model,
            self.onpolicy_pairs,
            rng,
            traj_indices=list(range(start, n_traj)),
        )
        if onpol is not None:
            onpol.name = "onpolicy"
        acc_onpol = bt_accuracy_oracle(reward_model, onpol)

        spearman, pearson, n_buf = reward_correlations(
            reward_model, replay_buffer, rng=np.random.RandomState(self.seed + step)
        )

        probe_rhat = probe_true = probe_gap = float("nan")
        if run_probe and agent is not None and env is not None and utils_mod is not None:
            print(f"[pebble-diag] causal probe at step={step} G={self.probe_grad_steps}")
            probe_rhat, probe_true, probe_gap = run_causal_probe(
                agent,
                replay_buffer,
                env,
                utils_mod,
                self.probe_grad_steps,
                self.probe_eval_episodes,
                step,
            )
            print(
                f"[pebble-diag] probe  R(r̂)={probe_rhat:.1f}  "
                f"R(GT)={probe_true:.1f}  gap={probe_gap:.1f}"
            )

        n_pref = (
            reward_model.capacity if reward_model.buffer_full else reward_model.buffer_index
        )
        snap = Snapshot(
            step=step,
            condition=self.condition,
            seed=self.seed,
            total_feedback=total_feedback,
            acc_pref_buffer=acc_pref,
            acc_holdout=acc_hold,
            acc_onpolicy=acc_onpol,
            acc_holdout_q1=qs[0],
            acc_holdout_q2=qs[1],
            acc_holdout_q3=qs[2],
            acc_holdout_q4=qs[3],
            acc_holdout_q5=qs[4],
            spearman_rhat_rtrue=spearman,
            pearson_rhat_rtrue=pearson,
            n_buffer=n_buf,
            true_return=true_return,
            probe_return_rhat=probe_rhat,
            probe_return_true=probe_true,
            probe_gap=probe_gap,
            n_pref=int(n_pref),
            n_holdout=0 if self.holdout is None else len(self.holdout),
            n_onpolicy=0 if onpol is None else len(onpol),
            notes=notes,
        )
        with open(self.csv_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=self.FIELDS).writerow(asdict(snap))

        # Live heuristic is for monitoring only — not a formal hypothesis test.
        print(
            f"[pebble-diag] step={step}  "
            f"Acc_pref={acc_pref:.3f}  Acc_hold={acc_hold:.3f}  "
            f"Acc_onpol={acc_onpol:.3f}  ρ={spearman:.3f}  R_true={true_return:.1f}  "
            f"read={interpret(snap)}"
        )
        return snap


def interpret(snap: Snapshot) -> str:
    """Heuristic tag for live logs (CLAIM.md §7 is the formal reading).

    Thresholds are provisional for Walker-scale returns; formal claims must
    use seed-aggregated comparisons against `full`, not this tag alone.
    """
    if np.isnan(snap.acc_holdout) or np.isnan(snap.true_return):
        return "insufficient_data"
    hold_ok = snap.acc_holdout >= 0.75
    onpol_ok = np.isnan(snap.acc_onpolicy) or snap.acc_onpolicy >= 0.70
    hold_bad = snap.acc_holdout < 0.65
    onpol_bad = (not np.isnan(snap.acc_onpolicy)) and snap.acc_onpolicy < 0.65
    gap_large = (not np.isnan(snap.probe_gap)) and snap.probe_gap > 50.0

    if hold_bad and onpol_bad:
        return "H1_reward_model"
    if hold_ok and onpol_bad:
        return "H2ish_onpolicy_shift"
    if hold_ok and onpol_ok and gap_large:
        return "H2_amplification_probe"
    if hold_ok and onpol_ok:
        return "reward_model_healthy"
    return "mixed"
