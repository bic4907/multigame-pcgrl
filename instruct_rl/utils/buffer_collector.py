"""
instruct_rl/utils/buffer_collector.py
=====================================
학습 중 RL 에이전트의 trajectory 버퍼를 수집하는 유틸리티.

학습 50% ~ 100% 구간에서 일정 간격으로
(obs, action, reward, done, env_map) 데이터를 수집하여
실험 폴더에 .npz 파일로 저장한다.

수집 환경 수(n_collect_envs)는 max_samples를 반드시 채울 수 있도록
자동으로 결정된다.
"""
from __future__ import annotations

import math
import os

import numpy as np

from instruct_rl.utils.log_utils import get_logger

logger = get_logger(__file__)


class BufferCollector:
    """학습 중 trajectory 를 수집하여 npz 파일로 저장하는 콜백 객체.

    Parameters
    ----------
    save_dir : str
        버퍼 파일을 저장할 디렉토리 경로.
    total_updates : int
        전체 학습 update step 수 (NUM_UPDATES).
    max_samples : int
        수집할 최대 transition 수.
    num_steps : int
        한 update step 당 env step 수 (config.num_steps).
    n_envs : int
        병렬 환경 수 (config.n_envs). timestep 계산 및 동적 환경 수 결정에 사용.
    collect_start_ratio : float
        수집 시작 시점 (0.0~1.0, 기본 0.5 = 50%).
    collect_end_ratio : float
        수집 종료 시점 (0.0~1.0, 기본 1.0 = 100%).
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

        # ── 수집 구간 계산 ───────────────────────────────────────
        self.start_step = int(total_updates * collect_start_ratio)
        self.end_step = int(total_updates * collect_end_ratio)
        collect_window = max(1, self.end_step - self.start_step)

        # ── 동적 환경 수 결정 ────────────────────────────────────
        # n_collect_envs=1 부터 시작, 수집 가능 총량이 max_samples 이상이 될 때까지 증가
        # 수집 가능 총량 = n_collections * num_steps * n_collect_envs
        # n_collections = collect_window // collect_interval
        # collect_interval = collect_window // ceil(max_samples / (num_steps * n_collect_envs))
        self.n_collect_envs = self._compute_n_collect_envs(
            max_samples, num_steps, n_envs, collect_window
        )

        # ── interval 재계산 (확정된 n_collect_envs 기준) ─────────
        transitions_per_collect = num_steps * self.n_collect_envs
        n_collections_needed = max(1, math.ceil(max_samples / transitions_per_collect))
        self.collect_interval = max(1, collect_window // n_collections_needed)

        # 실제 수집 횟수
        self.n_collections = min(
            n_collections_needed,
            collect_window // self.collect_interval,
        )

        # timestep 변환 계수: update_step → total_timesteps
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
        """max_samples 를 반드시 채울 수 있도록 n_collect_envs 를 결정.

        env 1개로 충분하면 1, 아니면 필요한 만큼 늘린다 (최대 n_envs).
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
        """update_step 을 total_timesteps 기준 스텝으로 변환."""
        return update_step * self._timestep_per_update

    def should_collect(self, update_step: int) -> bool:
        """현재 update step 에서 수집해야 하는지 판단."""
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
        """traj_batch 에서 env_idx=0..n_collect_envs-1 의 데이터를 추출하여 npz 로 저장.

        Parameters
        ----------
        update_step : int
            현재 update step 번호.
        traj_batch : Transition
            shape (num_steps, n_envs, ...) 의 trajectory 배치.
        env_state :
            현재 환경 상태.
        """
        if not self.should_collect(update_step):
            return

        remaining = self.max_samples - self._total_transitions
        if remaining <= 0:
            return

        k = self.n_collect_envs  # 수집할 환경 수

        # ── env_idx=0..k-1 추출 후 (num_steps, k, ...) → (num_steps*k, ...) ──
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

        # 최대 수집량 제한
        n_take = min(done.shape[0], remaining)
        done = done[:n_take]
        action = action[:n_take]
        value = value[:n_take]
        reward = reward[:n_take]
        log_prob = log_prob[:n_take]
        map_obs = map_obs[:n_take]
        env_map = env_map[:n_take]

        # ── 파일명은 total_timesteps 기준 스텝 ──
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

