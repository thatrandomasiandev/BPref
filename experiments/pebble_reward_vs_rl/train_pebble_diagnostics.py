#!/usr/bin/env python3
"""
Instrumented PEBBLE training loop for the H1/H2 mechanism study.

Primary source: Lee et al., PEBBLE, ICML 2021 — see CLAIM.md / PAPER_NOTES.md.
Implementation vehicle: BPref's train_PEBBLE.py (control flow mirrored so
results remain comparable to unmodified PEBBLE). This is *not* a B-Pref study.

What this script changes vs stock PEBBLE
----------------------------------------
  • Stores environment GT rewards in the replay buffer (for correlations + probe).
  • Freezes D_holdout; logs Acc / Spearman / true return after every reward update.
  • Optional causal probe at checkpoints (non-invasive: restore weights + rewards).
  • Ablation switches: do_relabel (Fig. 5a), num_unsup_steps (Fig. 5a), max_feedback
    (Fig. 3) — exactly the factors in CLAIM.md §2.

What this script does *not* change
----------------------------------
  • Oracle teacher (§5.1): teacher_eps_mistake must stay 0; soft beta forbidden.
  • SAC / BT learning rules, except gating buffer relabel under `no_relabel`.
"""

from __future__ import annotations

import csv
import os
import sys
import time
from collections import deque
from pathlib import Path

import hydra
import numpy as np
import torch

# ---------------------------------------------------------------------------
# Import path: works whether launched from repo root or via Hydra chdir.
# custom_dmc2gym is a local package used by utils.make_env.
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_DMC2GYM = _REPO_ROOT / "custom_dmc2gym"
if str(_DMC2GYM) not in sys.path:
    sys.path.insert(0, str(_DMC2GYM))

import utils  # noqa: E402
from logger import Logger  # noqa: E402
from replay_buffer import ReplayBuffer  # noqa: E402
from reward_model import RewardModel  # noqa: E402

from experiments.pebble_reward_vs_rl.metrics import (  # noqa: E402
    PebbleDiagnostics,
    interpret,
)


def _assert_oracle_teacher(cfg) -> None:
    """Hard-fail if someone accidentally enables B-Pref noise under this study.

    CLAIM.md forbids changing the teacher: H1/H2 here are about PEBBLE's own
    ablations, not preference-channel noise.
    """
    eps_m = float(cfg.teacher_eps_mistake)
    beta = float(cfg.teacher_beta)
    if eps_m > 0:
        raise ValueError(
            f"Oracle-only experiment (paper §5.1). Got teacher_eps_mistake={eps_m}. "
            f"B-Pref Mistake belongs in a different study."
        )
    # Released PEBBLE/BPref code uses beta=-1 for deterministic Oracle ranking.
    if beta > 0:
        raise ValueError(
            f"Soft teacher (teacher_beta={beta}) is out of scope. "
            f"Use teacher_beta=-1 for the paper's scripted Oracle."
        )


class PebbleDiagWorkspace:
    """PEBBLE Workspace with longitudinal Acc / return / probe logs."""

    def __init__(self, cfg):
        self.work_dir = os.getcwd()
        print(f"workspace: {self.work_dir}")
        self.cfg = cfg
        _assert_oracle_teacher(cfg)

        self.logger = Logger(
            self.work_dir,
            save_tb=cfg.log_save_tb,
            log_frequency=cfg.log_frequency,
            agent="sac",
        )

        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        self.log_success = False

        # Env: DMControl via dmc2gym (paper §5.2). Meta-World optional but not
        # needed for the registered Walker / Quadruped protocol.
        if "metaworld" in cfg.env:
            self.env = utils.make_metaworld_env(cfg)
            self.log_success = True
        else:
            self.env = utils.make_env(cfg)

        # Fill SAC dims from the live env (Hydra ??? placeholders).
        cfg.agent.obs_dim = self.env.observation_space.shape[0]
        cfg.agent.action_dim = self.env.action_space.shape[0]
        cfg.agent.action_range = [
            float(self.env.action_space.low.min()),
            float(self.env.action_space.high.max()),
        ]
        self.agent = hydra.utils.instantiate(cfg.agent, _recursive_=False)

        self.replay_buffer = ReplayBuffer(
            self.env.observation_space.shape,
            self.env.action_space.shape,
            int(cfg.replay_buffer_capacity),
            self.device,
        )

        self.total_feedback = 0
        self.labeled_feedback = 0
        self.step = 0

        # Bradley–Terry ensemble (paper: ensemble size 3, segment H configurable).
        self.reward_model = RewardModel(
            self.env.observation_space.shape[0],
            self.env.action_space.shape[0],
            ensemble_size=cfg.ensemble_size,
            size_segment=cfg.segment,
            activation=cfg.activation,
            lr=cfg.reward_lr,
            mb_size=cfg.reward_batch,
            large_batch=cfg.large_batch,
            label_margin=cfg.label_margin,
            teacher_beta=cfg.teacher_beta,
            teacher_gamma=cfg.teacher_gamma,
            teacher_eps_mistake=cfg.teacher_eps_mistake,
            teacher_eps_skip=cfg.teacher_eps_skip,
            teacher_eps_equal=cfg.teacher_eps_equal,
        )

        # --- Ablation tag + relabel gate (CLAIM.md §2 / Fig. 5a) ---
        condition = str(getattr(cfg, "pebble_condition", "full"))
        self.do_relabel = bool(getattr(cfg, "do_relabel", True))
        if condition == "no_relabel":
            # Force consistency even if Hydra override forgot do_relabel=false.
            self.do_relabel = False

        self.diagnostics = PebbleDiagnostics(
            work_dir=self.work_dir,
            condition=condition,
            seed=int(cfg.seed),
            holdout_pairs=int(cfg.diag_holdout_pairs),
            onpolicy_pairs=int(cfg.diag_onpolicy_pairs),
            probe_grad_steps=int(cfg.diag_probe_gradient_steps),
            probe_eval_episodes=int(cfg.diag_probe_eval_episodes),
            onpolicy_recent_trajs=int(cfg.diag_onpolicy_recent_trajs),
        )

        self.probe_steps = self._parse_probe_steps(getattr(cfg, "diag_probe_steps", []))
        # Between formal evals, track recent train episode GT returns for snapshots.
        self._recent_true_returns = deque(maxlen=10)
        self._last_eval_true_return = float("nan")

        self.eval_csv = os.path.join(self.work_dir, "eval_true_returns.csv")
        with open(self.eval_csv, "w", newline="") as f:
            csv.DictWriter(
                f, fieldnames=["step", "true_return", "episode_reward_hat", "success_rate"]
            ).writeheader()

        print(
            f"[pebble] condition={condition}  do_relabel={self.do_relabel}  "
            f"unsup={cfg.num_unsup_steps}  max_feedback={cfg.max_feedback}  "
            f"feed_type={cfg.feed_type}  teacher=Oracle  "
            f"(smoke≠evidence; see CLAIM.md §6)"
        )

    @staticmethod
    def _parse_probe_steps(raw) -> set:
        """Accept Hydra list, Python list, or comma-separated string."""
        if raw is None:
            return set()
        try:
            from omegaconf import ListConfig, OmegaConf

            if isinstance(raw, ListConfig):
                raw = OmegaConf.to_container(raw, resolve=True)
        except Exception:
            pass
        if isinstance(raw, (list, tuple)):
            return {int(x) for x in raw}
        text = str(raw).strip().strip("[]")
        if not text:
            return set()
        return {int(x.strip()) for x in text.split(",") if x.strip()}

    def _maybe_relabel(self) -> None:
        """PEBBLE §4.3 buffer rewrite — gated for the no_relabel ablation."""
        if self.do_relabel:
            self.replay_buffer.relabel_with_predictor(self.reward_model)
        else:
            print("[pebble] skipping buffer relabel (no_relabel ablation / Fig. 5a)")

    # ------------------------------------------------------------------
    # Evaluation — paper §5.1 ground-truth episode return
    # ------------------------------------------------------------------

    def evaluate(self) -> float:
        """Roll out deterministic policy; score with environment GT reward."""
        average_episode_reward = 0.0
        average_true_episode_reward = 0.0
        success_rate = 0.0

        for _ in range(self.cfg.num_eval_episodes):
            obs = utils.env_reset(self.env)
            self.agent.reset()
            done = False
            episode_reward = 0.0
            true_episode_reward = 0.0
            episode_success = 0.0

            while not done:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=False)
                obs, reward, done, extra = utils.env_step(self.env, action)
                # During eval both "episode_reward" and true return use GT env
                # reward (agent was trained on r̂; we still report GT).
                episode_reward += reward
                true_episode_reward += reward
                if self.log_success:
                    episode_success = max(episode_success, extra.get("success", 0.0))

            average_episode_reward += episode_reward
            average_true_episode_reward += true_episode_reward
            if self.log_success:
                success_rate += episode_success

        average_episode_reward /= self.cfg.num_eval_episodes
        average_true_episode_reward /= self.cfg.num_eval_episodes
        if self.log_success:
            success_rate = 100.0 * success_rate / self.cfg.num_eval_episodes

        self.logger.log("eval/episode_reward", average_episode_reward, self.step)
        self.logger.log("eval/true_episode_reward", average_true_episode_reward, self.step)
        if self.log_success:
            self.logger.log("eval/success_rate", success_rate, self.step)
        self.logger.dump(self.step)

        self._last_eval_true_return = float(average_true_episode_reward)
        with open(self.eval_csv, "a", newline="") as f:
            csv.DictWriter(
                f, fieldnames=["step", "true_return", "episode_reward_hat", "success_rate"]
            ).writerow(
                {
                    "step": self.step,
                    "true_return": average_true_episode_reward,
                    "episode_reward_hat": average_episode_reward,
                    "success_rate": success_rate if self.log_success else "",
                }
            )
        return average_true_episode_reward

    # ------------------------------------------------------------------
    # Reward learning — paper Alg. 2 (+ diagnostic hook after fit)
    # ------------------------------------------------------------------

    def learn_reward(self, first_flag=0) -> float:
        """Query Oracle, fit BT ensemble. Returns train-batch Acc (fit check)."""
        # First session: uniform (no useful disagreement/entropy yet) — PEBBLE.
        if first_flag == 1:
            labeled_queries = self.reward_model.uniform_sampling()
        else:
            # feed_type=2 = entropy (paper §5.1 default when not otherwise stated).
            if self.cfg.feed_type == 0:
                labeled_queries = self.reward_model.uniform_sampling()
            elif self.cfg.feed_type == 1:
                labeled_queries = self.reward_model.disagreement_sampling()
            elif self.cfg.feed_type == 2:
                labeled_queries = self.reward_model.entropy_sampling()
            elif self.cfg.feed_type == 3:
                labeled_queries = self.reward_model.kcenter_sampling()
            elif self.cfg.feed_type == 4:
                labeled_queries = self.reward_model.kcenter_disagree_sampling()
            elif self.cfg.feed_type == 5:
                labeled_queries = self.reward_model.kcenter_entropy_sampling()
            else:
                raise NotImplementedError(f"Unknown feed_type={self.cfg.feed_type}")

        self.total_feedback += self.reward_model.mb_size
        self.labeled_feedback += labeled_queries

        total_acc = 0.0
        if self.labeled_feedback > 0:
            for _epoch in range(self.cfg.reward_update):
                if self.cfg.label_margin > 0 or self.cfg.teacher_eps_equal > 0:
                    train_acc = self.reward_model.train_soft_reward()
                else:
                    train_acc = self.reward_model.train_reward()
                total_acc = float(np.mean(train_acc))
                # Early stop when ensemble has essentially fit the buffer.
                if total_acc > 0.97:
                    break

        print(f"Reward function is updated!! ACC(train-batch vs Oracle): {total_acc}")
        return total_acc

    def _run_diagnostics(self, notes: str = "", force_probe: bool = False):
        """Log CLAIM.md §4 metrics; optionally run the causal probe."""
        if not np.isnan(self._last_eval_true_return):
            true_ret = self._last_eval_true_return
        elif len(self._recent_true_returns) > 0:
            true_ret = float(np.mean(self._recent_true_returns))
        else:
            true_ret = float("nan")

        run_probe = force_probe or (self.step in self.probe_steps)
        snap = self.diagnostics.log_update(
            step=self.step,
            total_feedback=self.total_feedback,
            reward_model=self.reward_model,
            replay_buffer=self.replay_buffer,
            true_return=true_ret,
            agent=self.agent,
            env=self.env,
            utils_mod=utils,
            run_probe=run_probe,
            notes=notes,
        )
        # Heuristic tag for live monitoring — not a formal claim (CLAIM.md §7).
        print(f"[pebble-diag] heuristic={interpret(snap)}")
        return snap

    def _update_reward_schedule(self) -> None:
        """Optional query-batch schedule (stock PEBBLE / BPref code path)."""
        if self.cfg.reward_schedule == 1:
            frac = (self.cfg.num_train_steps - self.step) / self.cfg.num_train_steps
            frac = 0.01 if frac == 0 else frac
        elif self.cfg.reward_schedule == 2:
            frac = self.cfg.num_train_steps / (self.cfg.num_train_steps - self.step + 1)
        else:
            frac = 1.0
        self.reward_model.change_batch(frac)

    # ------------------------------------------------------------------
    # Main loop — mirrors train_PEBBLE.Workspace.run
    # ------------------------------------------------------------------

    def run(self):
        episode, episode_reward, done = 0, 0.0, True
        episode_success = 0.0
        true_episode_reward = 0.0
        avg_train_true_return = deque([], maxlen=10)
        start_time = time.time()
        interact_count = 0
        holdout_frozen = False

        # Phase boundary: end of seed (+ unsupervised pre-train if enabled).
        # no_pretrain ⇒ num_unsup_steps=0 ⇒ first prefs right after seed.
        phase_boundary = self.cfg.num_seed_steps + self.cfg.num_unsup_steps

        while self.step < self.cfg.num_train_steps:
            # ----- episode boundary -----
            if done:
                if self.step > 0:
                    self.logger.log("train/duration", time.time() - start_time, self.step)
                    start_time = time.time()
                    self.logger.dump(
                        self.step, save=(self.step > self.cfg.num_seed_steps)
                    )

                if self.step > 0 and self.step % self.cfg.eval_frequency == 0:
                    self.logger.log("eval/episode", episode, self.step)
                    self.evaluate()

                self.logger.log("train/episode_reward", episode_reward, self.step)
                self.logger.log(
                    "train/true_episode_reward", true_episode_reward, self.step
                )
                self.logger.log("train/total_feedback", self.total_feedback, self.step)
                self.logger.log(
                    "train/labeled_feedback", self.labeled_feedback, self.step
                )
                if self.log_success:
                    self.logger.log("train/episode_success", episode_success, self.step)

                obs = utils.env_reset(self.env)
                self.agent.reset()
                done = False
                episode_reward = 0.0
                avg_train_true_return.append(true_episode_reward)
                self._recent_true_returns.append(true_episode_reward)
                true_episode_reward = 0.0
                episode_success = 0.0
                episode_step = 0
                episode += 1
                self.logger.log("train/episode", episode, self.step)

            # ----- action -----
            if self.step < self.cfg.num_seed_steps:
                action = self.env.action_space.sample()
            else:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=True)

            # ============================================================
            # End of pretrain (or seed if no_pretrain) → first preference batch
            # ============================================================
            if self.step == phase_boundary:
                # Freeze holdout from exploratory (or seed) trajectories.
                if not holdout_frozen:
                    n_h = self.diagnostics.freeze_holdout(self.reward_model)
                    holdout_frozen = n_h > 0

                self._update_reward_schedule()
                # Teacher skip/equal thresholds (unused under pure Oracle zeros,
                # but kept for parity with stock PEBBLE).
                new_margin = np.mean(avg_train_true_return) * (
                    self.cfg.segment / self.env._max_episode_steps
                )
                self.reward_model.set_teacher_thres_skip(new_margin)
                self.reward_model.set_teacher_thres_equal(new_margin)

                self.learn_reward(first_flag=1)
                self._maybe_relabel()

                # Critic was on intrinsic (or random) rewards; reset before SAC on r̂.
                self.agent.reset_critic()
                self.agent.update_after_reset(
                    self.replay_buffer,
                    self.logger,
                    self.step,
                    gradient_update=self.cfg.reset_update,
                    policy_update=True,
                )
                interact_count = 0
                self._run_diagnostics(notes="first_reward_update")

            # ============================================================
            # Preference learning + off-policy SAC
            # ============================================================
            elif self.step > phase_boundary:
                if self.total_feedback < self.cfg.max_feedback:
                    if interact_count == self.cfg.num_interact:
                        self._update_reward_schedule()
                        new_margin = np.mean(avg_train_true_return) * (
                            self.cfg.segment / self.env._max_episode_steps
                        )
                        self.reward_model.set_teacher_thres_skip(
                            new_margin * self.cfg.teacher_eps_skip
                        )
                        self.reward_model.set_teacher_thres_equal(
                            new_margin * self.cfg.teacher_eps_equal
                        )

                        # Do not overshoot the feedback budget (paper pieces of feedback).
                        if (
                            self.reward_model.mb_size + self.total_feedback
                            > self.cfg.max_feedback
                        ):
                            self.reward_model.set_batch(
                                self.cfg.max_feedback - self.total_feedback
                            )

                        self.learn_reward()
                        self._maybe_relabel()
                        interact_count = 0

                        if not holdout_frozen:
                            n_h = self.diagnostics.freeze_holdout(self.reward_model)
                            holdout_frozen = n_h > 0

                        # Core measurement point for H1/H2.
                        self._run_diagnostics(notes="reward_update")

                self.agent.update(self.replay_buffer, self.logger, self.step, 1)

            # ============================================================
            # Unsupervised exploration (skipped when num_unsup_steps=0)
            # ============================================================
            elif self.step > self.cfg.num_seed_steps:
                self.agent.update_state_ent(
                    self.replay_buffer,
                    self.logger,
                    self.step,
                    gradient_update=1,
                    K=self.cfg.topK,
                )

            # ----- environment step -----
            next_obs, reward, done, extra = utils.env_step(self.env, action)
            # Learning signal is r̂; GT `reward` is kept for diagnostics + Oracle.
            reward_hat = self.reward_model.r_hat(np.concatenate([obs, action], axis=-1))

            done_float = float(done)
            done_no_max = (
                0 if episode_step + 1 == self.env._max_episode_steps else done_float
            )
            episode_reward += reward_hat
            true_episode_reward += reward
            if self.log_success:
                episode_success = max(episode_success, extra.get("success", 0.0))

            # Trajectory store for preference queries (targets = GT rewards).
            self.reward_model.add_data(obs, action, reward, done)
            # Replay: learn from r̂; remember GT for Spearman + causal probe.
            self.replay_buffer.add(
                obs,
                action,
                reward_hat,
                next_obs,
                done_float,
                done_no_max,
                true_reward=reward,
            )

            obs = next_obs
            episode_step += 1
            self.step += 1
            interact_count += 1

        # Final eval + forced probe — end-of-run checkpoint for CLAIM.md analysis.
        final_ret = self.evaluate()
        self._last_eval_true_return = float(final_ret)
        self._run_diagnostics(notes="final", force_probe=True)

        self.agent.save(self.work_dir, self.step)
        self.reward_model.save(self.work_dir, self.step)
        print(f"[pebble-diag] wrote {self.diagnostics.csv_path}")
        print(
            "[pebble-diag] STATUS: run finished. "
            "Smoke/single-seed ≠ finding. Aggregate via analyze_results.py "
            "after multi-seed diagnostic or paper-scale (CLAIM.md §6)."
        )


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg):
    workspace = PebbleDiagWorkspace(cfg)
    workspace.run()


if __name__ == "__main__":
    main()
