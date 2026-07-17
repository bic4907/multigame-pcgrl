"""
instruct_rl/utils/buffer_collector.py
=====================================
training  during  RL  in previoustext of  trajectory text  text  utility.

training 50% ~ 100% bin in  text text as
(obs, action, reward, done, env_map) data  text
experiment folder in  .npz file to  savetext.

text text text(n_collect_envs)  max_samples  text text text text also text
automatic as  text.
"""
from __future__ import annotations

import math
import os

import numpy as np

from instruct_rl.utils.log_utils import get_logger

logger = get_logger(__file__)


class BufferCollector:
    """training  during  trajectory   text npz file to  savetext  callback text.

    Parameters
    ----------
    save_dir : str
        text file  savetext directory path.
    total_updates : int
        all training update step text (NUM_UPDATES).
    max_samples : int
        text maximum transition text.
    num_steps : int
        text update step text env step text (config.num_steps).
    n_envs : int
        parallel text text (config.n_envs). timestep compute text dynamic text text text in  text for .
    collect_start_ratio : float
        text start text (0.0~1.0, default 0.5 = 50%).
    collect_end_ratio : float
        text text text (0.0~1.0, default 1.0 = 100%).
    """

    def __init__(
        self,
        save_dir: str,
        total_updates: int,
        max_samples: int,
        num_steps: int,
        n_envs: int,
        collect_start_ratio: float = 0.5,
        collect_end_ratio: float = 1.0,
    ):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        self.total_updates = total_updates
        self.max_samples = max_samples
        self.num_steps = num_steps
        self.n_envs = n_envs
        self.collect_start_ratio = collect_start_ratio
        self.collect_end_ratio = collect_end_ratio

        # ── text bin compute ───────────────────────────────────────
        self.start_step = int(total_updates * collect_start_ratio)
        self.end_step = int(total_updates * collect_end_ratio)
        collect_window = max(1, self.end_step - self.start_step)

        # ── dynamic text text text ────────────────────────────────────
        # n_collect_envs=1 text start, text available totaltext  max_samples or more  text text text
        # text available totaltext = n_collections * num_steps * n_collect_envs
        # n_collections = collect_window // collect_interval
        # collect_interval = collect_window // ceil(max_samples / (num_steps * n_collect_envs))
        self.n_collect_envs = self._compute_n_collect_envs(
            max_samples, num_steps, n_envs, collect_window
        )

        # ── interval textcompute (text n_collect_envs basis) ─────────
        transitions_per_collect = num_steps * self.n_collect_envs
        n_collections_needed = max(1, math.ceil(max_samples / transitions_per_collect))
        self.collect_interval = max(1, collect_window // n_collections_needed)

        # text text text
        self.n_collections = min(
            n_collections_needed,
            collect_window // self.collect_interval,
        )

        # timestep convert text: update_step → total_timesteps
        self._timestep_per_update = num_steps * n_envs

        self._collected = 0
        self._total_transitions = 0
        self._file_idx = 0

        logger.info(
            f"BufferCollector initialized: "
            f"save_dir={save_dir}, "
            f"total_updates={total_updates}, "
            f"max_samples={max_samples}, "
            f"n_collect_envs={self.n_collect_envs}/{n_envs}, "
            f"collect_window=[{self.start_step}, {self.end_step}), "
            f"collect_interval={self.collect_interval}, "
            f"expected_collections={self.n_collections}, "
            f"transitions_per_collection={transitions_per_collect}"
        )

    @staticmethod
    def _compute_n_collect_envs(
        max_samples: int,
        num_steps: int,
        n_envs: int,
        collect_window: int,
    ) -> int:
        """max_samples   text text text text also text n_collect_envs   text.

        env 1text to  text 1, text text text text (maximum n_envs).
        """
        for k in range(1, n_envs + 1):
            per_collect = num_steps * k
            n_needed = math.ceil(max_samples / per_collect)
            interval = max(1, collect_window // n_needed)
            actual_collections = collect_window // interval
            capacity = actual_collections * per_collect
            if capacity >= max_samples:
                return k
        return n_envs

    # ── public API ──────────────────────────────────────────────────

    def _update_to_timestep(self, update_step: int) -> int:
        """update_step   total_timesteps basis text as  convert."""
        return update_step * self._timestep_per_update

    def should_collect(self, update_step: int) -> bool:
        """current update step  in  text text text text."""
        if self._total_transitions >= self.max_samples:
            return False
        if update_step < self.start_step:
            return False
        if update_step > self.end_step:
            return False
        offset = update_step - self.start_step
        return offset % self.collect_interval == 0

    def collect_and_save(
        self,
        update_step: int,
        traj_batch,
        env_state,
    ):
        """traj_batch  in  env_idx=0..n_collect_envs-1  of  data  extracttext npz  to  save.

        Parameters
        ----------
        update_step : int
            current update step text.
        traj_batch : Transition
            shape (num_steps, n_envs, ...)  of  trajectory batch.
        env_state :
            current text text.
        """
        if not self.should_collect(update_step):
            return

        remaining = self.max_samples - self._total_transitions
        if remaining <= 0:
            return

        k = self.n_collect_envs  # text text text

        # ── env_idx=0..k-1 extract  after  (num_steps, k, ...) → (num_steps*k, ...) ──
        done = np.asarray(traj_batch.done[:, :k]).reshape(-1)
        action = np.asarray(traj_batch.action[:, :k]).reshape(-1, *traj_batch.action.shape[2:])
        value = np.asarray(traj_batch.value[:, :k]).reshape(-1)
        reward = np.asarray(traj_batch.reward[:, :k]).reshape(-1)
        log_prob = np.asarray(traj_batch.log_prob[:, :k]).reshape(-1, *traj_batch.log_prob.shape[2:])
        map_obs = np.asarray(traj_batch.obs.map_obs[:, :k]).reshape(
            -1, *traj_batch.obs.map_obs.shape[2:]
        )

        if traj_batch.env_map is not None:
            env_map = np.asarray(traj_batch.env_map[:, :k]).reshape(
                -1, *traj_batch.env_map.shape[2:]
            )
        else:
            env_map = np.asarray(env_state.env_state.env_map[:k]).reshape(
                -1, *env_state.env_state.env_map.shape[1:]
            )

        # maximum text text
        n_take = min(done.shape[0], remaining)
        done = done[:n_take]
        action = action[:n_take]
        value = value[:n_take]
        reward = reward[:n_take]
        log_prob = log_prob[:n_take]
        map_obs = map_obs[:n_take]
        env_map = env_map[:n_take]

        # ── filetext  total_timesteps basis text ──
        timestep = self._update_to_timestep(update_step)
        save_path = os.path.join(
            self.save_dir,
            f"buffer_{self._file_idx:06d}_ts{timestep}.npz",
        )
        np.savez_compressed(
            save_path,
            done=done,
            action=action,
            value=value,
            reward=reward,
            log_prob=log_prob,
            map_obs=map_obs,
            env_map=env_map,
            timestep=np.array(timestep, dtype=np.int64),
            update_step=np.array(update_step, dtype=np.int64),
        )

        self._file_idx += 1
        self._total_transitions += n_take
        self._collected += 1

        logger.info(
            f"[BufferCollector] Saved {n_take} transitions from {k} envs "
            f"(total: {self._total_transitions}/{self.max_samples}) "
            f"at timestep={timestep} (update={update_step}) → {save_path}"
        )

    @property
    def is_done(self) -> bool:
        return self._total_transitions >= self.max_samples

    @property
    def summary(self) -> dict:
        return {
            "total_transitions": self._total_transitions,
            "max_samples": self.max_samples,
            "n_collect_envs": self.n_collect_envs,
            "n_files": self._file_idx,
            "save_dir": self.save_dir,
        }

