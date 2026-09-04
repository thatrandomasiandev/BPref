"""
PEBBLE-native diagnostics: reward-model quality vs RL amplification.

Grounded in Lee, Smith & Abbeel (ICML 2021), arXiv:2106.05091
(see PAPER_NOTES.md and lee2021pebble.pdf in this folder).

Scientific question
-------------------
PEBBLE reports only *downstream true return*. When that return drops under
the paper's own ablations (Fig. 5a: no-relabel / no-pretrain; Fig. 3: low
feedback), is it because:

  H1  The reward model \(\hat r\) ranks segments worse (BT accuracy drops), or
  H2  Downstream SAC amplifies a still-decent \(\hat r\) (esp. under
      non-stationary rewards without relabeling — the paper's §4.3 claim)?

Teacher (paper §5.1)
--------------------
Scripted Oracle only: preferences from ground-truth segment returns.
No B-Pref Mistake / Stoc / Skip / Equal channels.
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


# ---------------------------------------------------------------------------
# Preference pair sets (three datasets requested for the open question)
# ---------------------------------------------------------------------------

@dataclass
class PairSet:
    """Segment pairs for Bradley–Terry evaluation.

    Layout matches PEBBLE RewardModel: seg shape (N, H, ds+da).
    `true_ret_*` = sum of *environment* GT rewards over the segment
    (Oracle ranking signal; paper §5.1 teacher).
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
        """|ΔR| for easy-vs-near-tie stratification."""
        return np.abs(self.true_ret_2 - self.true_ret_1).reshape(-1)

    @property
    def oracle_labels(self) -> np.ndarray:
        """Hard Oracle preference: 1 iff seg2 has higher GT return (PEBBLE convention)."""
        return (1 * (self.true_ret_1 < self.true_ret_2)).astype(np.int64)


@dataclass
class Snapshot:
    """One longitudinal row: reward quality + downstream return together."""

    step: int
    condition: str
    seed: int
    total_feedback: int

    # BT accuracy vs Oracle (paper teacher = Oracle, so "teacher" == "clean")
    acc_pref_buffer: float       # D_pref: pairs in the preference buffer
    acc_holdout: float           # D_holdout: frozen, never trained on
    acc_onpolicy: float          # D_onpolicy: recent policy trajectories

    # Holdout stratified by |ΔR| quintiles (q1 = hardest near-ties)
    acc_holdout_q1: float
    acc_holdout_q2: float
    acc_holdout_q3: float
    acc_holdout_q4: float
    acc_holdout_q5: float

    # Pointwise agreement of r̂ with r_true on the replay buffer
    spearman_rhat_rtrue: float
    pearson_rhat_rtrue: float
    n_buffer: int

    # Paper's downstream metric (§5.1): ground-truth episode return
    true_return: float

    # Causal probe (NaN if not run): same buffer, SAC under r̂ vs r_true
    probe_return_rhat: float = float("nan")
    probe_return_true: float = float("nan")
    probe_gap: float = float("nan")  # true − rhat; large + ⇒ H1

    n_pref: int = 0
    n_holdout: int = 0
    n_onpolicy: int = 0
    notes: str = ""


# ---------------------------------------------------------------------------
# Trajectory helpers (RewardModel.inputs / .targets store GT rewards)
# ---------------------------------------------------------------------------

def n_complete_trajs(reward_model) -> int:
    """Count finished episodes in RewardModel's trajectory store."""
    if len(reward_model.inputs) == 0:
        return 0
    last = reward_model.inputs[-1]
    if len(last) == 0:
        return max(0, len(reward_model.inputs) - 1)
    # In-progress episode is shorter than a completed one.
    if len(reward_model.inputs) >= 2 and len(last) < len(reward_model.inputs[0]):
        return len(reward_model.inputs) - 1
    return len(reward_model.inputs)


def sample_pairs_from_trajs(
    reward_model,
    n_pairs: int,
    rng: np.random.RandomState,
    traj_indices: Optional[Sequence[int]] = None,
) -> Optional[PairSet]:
    """Sample (seg1, seg2) with GT segment returns — same geometry as get_queries.

    Uses PEBBLE's segment length `reward_model.size_segment` (paper default H=50
    in the released BPref/PEBBLE code; Fig. 5c compares H=50 vs H=1).
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

    inputs = np.array([reward_model.inputs[i] for i in traj_indices])
    targets = np.array([reward_model.targets[i] for i in traj_indices])
    n_avail = inputs.shape[0]

    i2 = rng.choice(n_avail, size=n_pairs, replace=True)
    i1 = rng.choice(n_avail, size=n_pairs, replace=True)

    sa1 = inputs[i1].reshape(-1, inputs.shape[-1])
    r1 = targets[i1].reshape(-1, targets.shape[-1])
    sa2 = inputs[i2].reshape(-1, inputs.shape[-1])
    r2 = targets[i2].reshape(-1, targets.shape[-1])

    base = np.array([list(range(i * len_traj, i * len_traj + H)) for i in range(n_pairs)])
    off1 = rng.choice(len_traj - H + 1, size=n_pairs, replace=True).reshape(-1, 1)
    off2 = rng.choice(len_traj - H + 1, size=n_pairs, replace=True).reshape(-1, 1)

    seg1 = np.take(sa1, base + off1, axis=0).astype(np.float32)
    seg2 = np.take(sa2, base + off2, axis=0).astype(np.float32)
    ret1 = np.take(r1, base + off1, axis=0).sum(axis=1).astype(np.float32)
    ret2 = np.take(r2, base + off2, axis=0).sum(axis=1).astype(np.float32)

    return PairSet("sampled", seg1, seg2, ret1, ret2)


def pairs_from_pref_buffer(reward_model) -> Optional[PairSet]:
    """D_pref: live preference buffer rows.

    Under the paper's Oracle teacher, buffer labels equal Oracle rankings.
    We still *recompute* Oracle labels by re-scoring is not possible (buffer
    stores only (s,a) segments). For Oracle runs the stored label *is* the
    Oracle label, so we evaluate BT predictions against `buffer_label`.

    For a distribution-matched *clean* score that does not depend on stored
    labels, see `sample_pairs_from_trajs` (used for holdout / onpolicy /
    train-distribution clean in the engine).
    """
    n = reward_model.capacity if reward_model.buffer_full else reward_model.buffer_index
    if n <= 0:
        return None
    # Placeholder returns; bt_accuracy_vs_labels uses teacher labels path.
    z = np.zeros((n, 1), dtype=np.float32)
    return PairSet(
        "pref_buffer",
        reward_model.buffer_seg1[:n].copy(),
        reward_model.buffer_seg2[:n].copy(),
        z,
        z,
    )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _pred_prefer_seg2(reward_model, seg1: np.ndarray, seg2: np.ndarray) -> np.ndarray:
    """Ensemble hard prediction: 1 iff mean P(seg2 ≻ seg1) > 0.5.

    RewardModel.get_rank_probability returns P(seg1 ≻ seg2); invert.
    """
    p_seg1_wins, _ = reward_model.get_rank_probability(seg1, seg2)
    return (p_seg1_wins < 0.5).astype(np.int64)


def bt_accuracy_oracle(reward_model, pairs: Optional[PairSet]) -> float:
    """Hard BT accuracy against Oracle GT segment ranking."""
    if pairs is None or len(pairs) == 0:
        return float("nan")
    pred = _pred_prefer_seg2(reward_model, pairs.seg1, pairs.seg2)
    labels = pairs.oracle_labels.reshape(-1)
    mask = pairs.abs_delta_r > 0  # drop exact ties
    if mask.sum() == 0:
        return float("nan")
    return float((pred[mask] == labels[mask]).mean())


def bt_accuracy_vs_buffer_labels(reward_model) -> float:
    """BT accuracy on preference buffer against stored Oracle labels (§5.1)."""
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


def stratified_holdout(reward_model, pairs: Optional[PairSet], n_bins: int = 5) -> List[float]:
    """Oracle BT accuracy within each |ΔR| quintile (q1 = near-ties)."""
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
    out = []
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
    reward_model, replay_buffer, max_n: int = 4096, rng: Optional[np.random.RandomState] = None
) -> Tuple[float, float, int]:
    """Spearman / Pearson of r̂(s,a) vs r_true(s,a) on the replay buffer.

    SAC regresses against scalar r̂ (paper §4.3); rank correlation on visited
    (s,a) probes the learning signal the critic actually sees.
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
    spearman = float(
        np.corrcoef(pred_r.argsort().argsort().astype(np.float64),
                    true_r.argsort().argsort().astype(np.float64))[0, 1]
    )
    return spearman, pearson, take


# ---------------------------------------------------------------------------
# Causal probe — isolates reward quality from visitation (same buffer)
# ---------------------------------------------------------------------------

class _SilentLogger:
    """Swallow SAC's logger calls during the probe (no TensorBoard)."""

    def log(self, *args, **kwargs):
        return None

    def __getattr__(self, name):
        def _noop(*args, **kwargs):
            return None
        return _noop


def _clone_weights(agent):
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
    agent.actor.load_state_dict(blob["actor"])
    agent.critic.load_state_dict(blob["critic"])
    agent.critic_target.load_state_dict(blob["critic_target"])
    agent.log_alpha.data.copy_(blob["log_alpha"])
    agent.actor_optimizer.load_state_dict(blob["actor_opt"])
    agent.critic_optimizer.load_state_dict(blob["critic_opt"])
    agent.log_alpha_optimizer.load_state_dict(blob["alpha_opt"])


def run_causal_probe(
    agent, replay_buffer, env, utils_mod, gradient_steps: int, n_eval: int, step: int
) -> Tuple[float, float, float]:
    """Same-buffer probe: SAC under r̂ vs under r_true, then restore.

    Interpretation (PEBBLE §4.3):
      probe_gap = return(GT) − return(r̂)
        large positive → buffer supports a good policy if reward were correct
                         (reward-model bottleneck / H1)
        ≈ 0, both low  → visitation / data limited
        Acc high + gap large → small residual errors amplified by SAC (H2)
    """
    if len(replay_buffer) < agent.batch_size:
        return float("nan"), float("nan"), float("nan")

    silent = _SilentLogger()
    reward_snap = replay_buffer.snapshot_rewards()
    weight_snap = _clone_weights(agent)

    def _eval() -> float:
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
                ep += reward  # environment GT reward
            total += ep
        return total / float(n_eval)

    # Arm A: current learning rewards (r̂ after PEBBLE relabel, or stale if no_relabel)
    agent.update(replay_buffer, silent, step, gradient_update=gradient_steps)
    ret_rhat = _eval()

    # Arm B: ground-truth rewards on the *same* transitions
    _restore_weights(agent, weight_snap)
    replay_buffer.relabel_with_true_rewards()
    agent.update(replay_buffer, silent, step, gradient_update=gradient_steps)
    ret_true = _eval()

    _restore_weights(agent, weight_snap)
    replay_buffer.restore_rewards(reward_snap)
    return float(ret_rhat), float(ret_true), float(ret_true - ret_rhat)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class PebbleDiagnostics:
    """Owns D_holdout, CSV logging, and the per-reward-update checklist."""

    FIELDS = [
        "step", "condition", "seed", "total_feedback",
        "acc_pref_buffer", "acc_holdout", "acc_onpolicy",
        "acc_holdout_q1", "acc_holdout_q2", "acc_holdout_q3",
        "acc_holdout_q4", "acc_holdout_q5",
        "spearman_rhat_rtrue", "pearson_rhat_rtrue", "n_buffer",
        "true_return",
        "probe_return_rhat", "probe_return_true", "probe_gap",
        "n_pref", "n_holdout", "n_onpolicy", "notes",
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

        with open(os.path.join(work_dir, "diagnostics_meta.json"), "w") as f:
            json.dump(
                {
                    "paper": "Lee, Smith, Abbeel — PEBBLE, ICML 2021 (arXiv:2106.05091)",
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
                },
                f,
                indent=2,
            )

    def freeze_holdout(self, reward_model) -> int:
        """Freeze D_holdout once ≥2 complete trajs exist (paper pretrain → diverse set)."""
        if self.holdout is not None and len(self.holdout) > 0:
            return len(self.holdout)
        if n_complete_trajs(reward_model) < 2:
            print("[pebble-diag] defer holdout freeze (need ≥2 complete trajs)")
            return 0
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
        """Full metric suite after a PEBBLE reward update."""

        acc_pref = bt_accuracy_vs_buffer_labels(reward_model)
        acc_hold = bt_accuracy_oracle(reward_model, self.holdout)
        qs = stratified_holdout(reward_model, self.holdout)

        # D_onpolicy from recent completed episodes (where SAC acts *now*).
        n_traj = n_complete_trajs(reward_model)
        start = max(0, n_traj - self.onpolicy_recent_trajs)
        rng = np.random.RandomState(self.seed + 104729 + step)
        onpol = sample_pairs_from_trajs(
            reward_model, self.onpolicy_pairs, rng, traj_indices=list(range(start, n_traj))
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
                agent, replay_buffer, env, utils_mod,
                self.probe_grad_steps, self.probe_eval_episodes, step,
            )
            print(
                f"[pebble-diag] probe  R(r̂)={probe_rhat:.1f}  "
                f"R(GT)={probe_true:.1f}  gap={probe_gap:.1f}"
            )

        n_pref = reward_model.capacity if reward_model.buffer_full else reward_model.buffer_index
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
        print(
            f"[pebble-diag] step={step}  "
            f"Acc_pref={acc_pref:.3f}  Acc_hold={acc_hold:.3f}  "
            f"Acc_onpol={acc_onpol:.3f}  ρ={spearman:.3f}  R_true={true_return:.1f}  "
            f"read={interpret(snap)}"
        )
        return snap


def interpret(snap: Snapshot) -> str:
    """Heuristic tag for live monitoring (not a formal hypothesis test)."""
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
