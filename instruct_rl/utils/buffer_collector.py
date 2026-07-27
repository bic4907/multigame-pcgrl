"""
instruct_rl/utils/buffer_collector.py
=====================================
Utilities for collecting an RL agent's trajectory buffer during training.

Collect (obs, action, reward, done, env_map) at regular intervals between
50% and 100% of training and
save them as NPZ files in the experiment directory.

determine n_collect_envs automatically so max_samples can be filled.
"""
from __future__ import annotations

import math
import os

import numpy as np

from instruct_rl.utils.log_utils import get_logger

logger = get_logger(__file__)


class BufferCollector:
    """Callback that collects trajectories during training and saves NPZ files.

    Parameters
    ----------
    save_dir : str
        Directory in which to save buffer files.
    total_updates : int
        Total number of training update steps (NUM_UPDATES).
    max_samples : int
        Maximum number of transitions to collect.
    num_steps : int
        Environment steps per update step (config.num_steps).
    n_envs : int
        Parallel environments (config.n_envs), used for timestep conversion and dynamic sizing.
    collect_start_ratio : float
        Collection start ratio in [0, 1] (default: 0.5).
    collect_end_ratio : float
        Collection end ratio in [0, 1] (default: 1.0).
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

        # ── Calculate the collection window ──────────────────────
        self.start_step = int(total_updates * collect_start_ratio)
        self.end_step = int(total_updates * collect_end_ratio)
        collect_window = max(1, self.end_step - self.start_step)

        # ── Determine the dynamic environment count ──────────────
        # Start at one and increase until capacity reaches max_samples
        # capacity = n_collections * num_steps * n_collect_envs
        # n_collections = collect_window // collect_interval
        # collect_interval = collect_window // ceil(max_samples / (num_steps * n_collect_envs))
        self.n_collect_envs = self._compute_n_collect_envs(
            max_samples, num_steps, n_envs, collect_window
        )

        # ── Recalculate interval using the final n_collect_envs ──
        transitions_per_collect = num_steps * self.n_collect_envs
        n_collections_needed = max(1, math.ceil(max_samples / transitions_per_collect))
        self.collect_interval = max(1, collect_window // n_collections_needed)

        # Actual number of collections
        self.n_collections = min(
            n_collections_needed,
            collect_window // self.collect_interval,
        )

        # Conversion factor from update_step to total_timesteps
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
        """Choose n_collect_envs so max_samples can always be filled.

        Use one environment when sufficient; otherwise increase up to n_envs.
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
        """Convert update_step to a total_timesteps-based step."""
        return update_step * self._timestep_per_update

    def should_collect(self, update_step: int) -> bool:
        """Return whether collection should occur at the current update step."""
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
        """Extract env_idx=0..n_collect_envs-1 from traj_batch and save as NPZ.

        Parameters
        ----------
        update_step : int
            Current update-step number.
        traj_batch : Transition
            shape (num_steps, n_envs, ...)  of  trajectory batch.
        env_state :
            Current environment state.
        """
        if not self.should_collect(update_step):
            return

        remaining = self.max_samples - self._total_transitions
        if remaining <= 0:
            return

        k = self.n_collect_envs  # Number of environments to collect

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

        # Enforce the maximum collection size
        n_take = min(done.shape[0], remaining)
        done = done[:n_take]
        action = action[:n_take]
        value = value[:n_take]
        reward = reward[:n_take]
        log_prob = log_prob[:n_take]
        map_obs = map_obs[:n_take]
        env_map = env_map[:n_take]

        # ── Filename step is based on total_timesteps ──
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
