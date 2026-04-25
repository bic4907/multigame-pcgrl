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

import logging
import time

import numpy as np
import pandas as pd
from tqdm import tqdm

from instruct_rl.eval.hdf5_store import open_eval_store, read_state
from instruct_rl.evaluation.metrics.vit import ViTEvaluator

logger = logging.getLogger(__name__)


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
        from envs.probs.multigame import render_multigame_map

        tile_size = getattr(self.config, 'vit_tile_size', 16)
        n_samples = len(instruct_df) * n_eps

        # ① HDF5에서 state 일괄 로드 후 on-demand 렌더링
        pred_list = []
        with open_eval_store(self.config.eval_dir, mode="r") as h5, \
             tqdm(total=n_samples, desc="[ViT] Render pred states") as pbar:
            for row_i, row in instruct_df.iterrows():
                game        = row.get("game", "unknown")
                re_val      = int(row.get("reward_enum", row_i))
                folder_name = f"{game}_re{re_val}_{int(row_i):04d}"
                for seed_i in range(n_eps):
                    env_map = read_state(h5, folder_name, seed_i)   # (H, W) uint8
                    img = np.array(
                        render_multigame_map(env_map.astype(np.int32), tile_size=tile_size)
                    )  # (H*ts, W*ts, 3) uint8
                    pred_list.append(img)
                    pbar.update(1)

        pred_images = np.stack(pred_list, axis=0)  # (N*n_eps, H*ts, W*ts, 3)

        logger.info(
            "[ViTScoreWrapper] pred=%s  gt=%s",
            pred_images.shape, gt_images.shape,
        )

        # ② ViT 1:1 pairwise 유사도 계산
        evaluator = ViTEvaluator()
        t0 = time.perf_counter()
        scores = evaluator.run_pairwise(pred_images, gt_images)   # (N*n_eps,)
        elapsed = time.perf_counter() - t0

        scores = np.array(scores).reshape(-1)
        logger.info(
            "done: mean=%.4f  elapsed=%.2fs",
            float(np.mean(scores)), elapsed,
        )
        return scores

