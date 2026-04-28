"""
instruct_rl/eval/wrappers/vit_score.py
========================================
ViTScoreWrapper — ViT-based human-likeness scoring.

HDF5에서 state(env_map)를 읽어 render_unified_rgb()로 동적 렌더링하고,
eval_utils가 전달한 GT 이미지와 1:1 코사인 유사도를 계산한다.

책임 분리
---------
- eval_utils.py   : GT 이미지 렌더링 (render_unified_rgb, dataset 팔레트)
- ViTScoreWrapper : HDF5 state 로드 → 동적 렌더링 → ViTEvaluator 위임
- ViTEvaluator    : run_pairwise() — 순수 ViT 임베딩 + 코사인 유사도
"""
from __future__ import annotations

import hashlib
import logging
import os
import time

import numpy as np
import pandas as pd
from tqdm import tqdm

from instruct_rl.eval.hdf5_store import open_eval_store, read_state
from instruct_rl.evaluation.metrics.vit import ViTEvaluator

logger = logging.getLogger(__name__)

_CACHE_DIR = os.environ.get(
    "EVAL_CACHE_DIR",
    os.path.join(os.path.dirname(__file__), "..", "..", "..", ".eval_cache"),
)


def _array_hash(arr: np.ndarray) -> str:
    return hashlib.md5(np.ascontiguousarray(arr).tobytes()).hexdigest()


def _vit_cache_path(gt_key: str) -> str:
    return os.path.join(_CACHE_DIR, "vit", f"{gt_key}.npy")


class ViTScoreWrapper:
    """Scores generated levels against dataset GT images via ViT embedding (1:1 pairwise)."""

    def __init__(self, config):
        self.config = config

    def run(
        self,
        instruct_df: pd.DataFrame,
        gt_images: np.ndarray,        # (N, H, W, 3) uint8 — eval_utils가 dataset 렌더링으로 전달
        n_eps: int,
        **_,
    ) -> np.ndarray:
        """
        Parameters
        ----------
        instruct_df : 평가 대상 DataFrame (game, reward_enum, ...)
        gt_images   : (N*n_eps, H*ts, W*ts, 3) uint8
        n_eps       : 시드(에피소드) 수
        """
        from envs.probs.multigame import render_multigame_maps_batch

        tile_size = getattr(self.config, 'vit_tile_size', 16)
        n_samples = len(instruct_df) * n_eps

        # ① HDF5에서 state 일괄 로드 (렌더링은 아직 안 함)
        env_map_list = []
        with open_eval_store(self.config.eval_dir, mode="r") as h5, \
             tqdm(total=n_samples, desc="[ViT] Load pred states") as pbar:
            for row_i, row in instruct_df.iterrows():
                game        = row.get("game", "unknown")
                re_val      = int(row.get("reward_enum", row_i))
                folder_name = f"{game}_re{re_val}_{int(row_i):04d}"
                for seed_i in range(n_eps):
                    env_map = read_state(h5, folder_name, seed_i)   # (H, W) uint8
                    env_map_list.append(env_map.astype(np.int32))
                    pbar.update(1)

        # ② numpy fancy-indexing 배치 렌더링 (for 루프 / PIL 없음)
        env_maps_batch = np.stack(env_map_list, axis=0)  # (N, H, W)
        logger.info("[ViTScoreWrapper] Batch rendering %d maps...", len(env_maps_batch))
        pred_images = render_multigame_maps_batch(env_maps_batch, tile_size=tile_size)  # (N, H*ts, W*ts, 3)

        logger.info(
            "[ViTScoreWrapper] pred=%s  gt=%s",
            pred_images.shape, gt_images.shape,
        )

        # ② GT 임베딩 — 디스크 캐시
        evaluator = ViTEvaluator()
        t0 = time.perf_counter()

        gt_key    = _array_hash(gt_images)
        gt_cache  = _vit_cache_path(gt_key)

        if os.path.exists(gt_cache):
            logger.info("[ViTScoreWrapper] GT embedding cache HIT — loading %s", gt_cache)
            gt_feats = np.load(gt_cache)
        else:
            logger.info("[ViTScoreWrapper] GT embedding cache MISS — computing & saving to %s", gt_cache)
            evaluator._ensure_model()
            gt_feats = np.array(evaluator.get_embeddings(gt_images, norm=True, desc="[ViT] gt  embedding"))
            os.makedirs(os.path.dirname(gt_cache), exist_ok=True)
            np.save(gt_cache, gt_feats)

        # ③ pred 임베딩 계산 후 코사인 유사도
        pred_feats = np.array(evaluator.get_embeddings(pred_images, norm=True, desc="[ViT] pred embedding"))
        scores = np.sum(pred_feats * gt_feats, axis=1).reshape(-1)  # (N,)
        elapsed = time.perf_counter() - t0

        logger.info("done: mean=%.4f  elapsed=%.2fs", float(np.mean(scores)), elapsed)
        return scores

