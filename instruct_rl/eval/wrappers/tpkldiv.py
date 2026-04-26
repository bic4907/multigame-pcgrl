"""
instruct_rl/eval/wrappers/tpkldiv.py
======================================
TPKLWrapper — Tile-Pattern KL divergence across seeds per instruction.

책임 분리
---------
- TPKLWrapper  : 데이터 파이프라인 전담 (HDF5 로드 후 TPKLEvaluator 위임)
- TPKLEvaluator: 순수 알고리즘 (GT 분포 구축 + JSD 계산)
"""
from __future__ import annotations

import hashlib
import logging
import os
import pickle
import time

import numpy as np
import pandas as pd
from tqdm import tqdm

from instruct_rl.eval.hdf5_store import open_eval_store, read_state
from instruct_rl.evaluation.metrics.tpkldiv import TPKLEvaluator
from instruct_rl.evaluation.metrics.tpkl_utils import build_gt_distribution

logger = logging.getLogger(__name__)

_CACHE_DIR = os.environ.get(
    "EVAL_CACHE_DIR",
    os.path.join(os.path.dirname(__file__), "..", "..", "..", ".eval_cache"),
)


def _array_hash(arr: np.ndarray) -> str:
    return hashlib.md5(np.ascontiguousarray(arr).tobytes()).hexdigest()


def _tpkl_cache_path(gt_key: str, window_sizes: tuple, epsilon: float) -> str:
    tag = f"ws{'_'.join(map(str, window_sizes))}_eps{epsilon}"
    return os.path.join(_CACHE_DIR, "tpkl", f"{gt_key}_{tag}.pkl")


def _load_gt_dists(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def _save_gt_dists(path: str, gt_dists) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(gt_dists, f)


class TPKLWrapper:
    """Computes Tile-Pattern KL divergence across seeds per instruction."""

    def __init__(self, config):
        self.config = config

    def run(
        self,
        instruct_df: pd.DataFrame,
        gt_levels: np.ndarray,
        n_eps: int,
        **_,
    ) -> np.ndarray:
        """
        Parameters
        ----------
        instruct_df : 평가 대상 DataFrame (game, reward_enum, feature_name, condition_value)
        gt_levels   : (M, H, W) int — 호출자(metrics.py)가 이미 필터링해서 전달
        n_eps       : 시드(에피소드) 수

        Returns
        -------
        scores : np.ndarray, shape (N*n_eps,)
        """
        # ① Predicted levels 로드 (HDF5) → (N*n_eps, H, W)
        with open_eval_store(self.config.eval_dir, mode="r") as h5:
            pred_levels = np.array(
                [
                    read_state(
                        h5,
                        f"{row.get('game', 'unknown')}_re{int(row.get('reward_enum', row_i))}_{int(row_i):04d}",
                        seed_i,
                    )
                    for row_i, row in tqdm(instruct_df.iterrows(),
                                           desc="[TPKL] Loading predicted states",
                                           total=len(instruct_df))
                    for seed_i in range(n_eps)
                ]
            )

        logger.info("[TPKLWrapper] pred_levels=%s  gt_levels=%s", pred_levels.shape, gt_levels.shape)

        window_sizes = (2, 3)
        epsilon = 1e-6

        gt_key   = _array_hash(np.asarray(gt_levels, dtype=np.int32))
        cache_path = _tpkl_cache_path(gt_key, window_sizes, epsilon)

        t0 = time.perf_counter()
        with tqdm(total=2, desc="[TPKL] Computing scores") as pbar:
            if os.path.exists(cache_path):
                logger.info("[TPKLWrapper] GT dist cache HIT — loading %s", cache_path)
                gt_dists = _load_gt_dists(cache_path)
                pbar.update(1)
            else:
                logger.info("[TPKLWrapper] GT dist cache MISS — building & saving to %s", cache_path)
                gt_dists = build_gt_distribution(
                    np.asarray(gt_levels, dtype=np.int32), window_sizes, epsilon, _pbar=pbar
                )
                _save_gt_dists(cache_path, gt_dists)

            from instruct_rl.evaluation.metrics.tpkl_utils import compute_jsd_scores
            scores = compute_jsd_scores(pred_levels, gt_dists, window_sizes, epsilon, _pbar=pbar)

        elapsed = time.perf_counter() - t0
        logger.info("[TPKLWrapper] done: mean=%.4f  elapsed=%.2fs", float(np.mean(scores)), elapsed)
        return scores.reshape(-1)

