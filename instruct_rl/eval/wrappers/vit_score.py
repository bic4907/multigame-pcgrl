"""
instruct_rl/eval/wrappers/vit_score.py
========================================
ViTScoreWrapper — ViT-based human-likeness scoring.

TPKLWrapper와 동일한 패턴으로 HDF5에서 pred 렌더링 이미지를 로드하고
eval_utils가 dataset의 render_unified_rgb로 미리 렌더링해서 전달한 GT 이미지와
1:1 코사인 유사도를 계산한다.

책임 분리
---------
- eval_utils.py   : GT 이미지 렌더링 (render_unified_rgb, dataset 팔레트)
- ViTScoreWrapper : HDF5 pred 로드 + ViTEvaluator 위임
- ViTEvaluator    : run_pairwise() — 순수 ViT 임베딩 + 코사인 유사도
"""
from __future__ import annotations

import logging
import time

import numpy as np
import pandas as pd
from tqdm import tqdm

from instruct_rl.eval.hdf5_store import open_eval_store, read_rendered_image
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
                      eval_utils에서 render_unified_rgb(sample.array) × n_eps repeat
        n_eps       : 시드(에피소드) 수
        """
        # ① Pred 렌더링 이미지 로드 (HDF5) → (N*n_eps, H, W, C)
        pred_images = []
        with open_eval_store(self.config.eval_dir, mode="r") as h5:
            for row_i, row in tqdm(
                instruct_df.iterrows(),
                desc="[ViT] Loading pred rendered images",
                total=len(instruct_df),
            ):
                game        = row.get("game", "unknown")
                re_val      = int(row.get("reward_enum", row_i))
                folder_name = f"{game}_re{re_val}_{int(row_i):04d}"
                for seed_i in range(n_eps):
                    pred_images.append(read_rendered_image(h5, folder_name, seed_i))

        pred_images = np.stack(pred_images, axis=0)  # (N*n_eps, H, W, C)

        # RGBA → RGB (필요한 경우)
        if pred_images.shape[-1] == 4:
            import cv2
            pred_images = np.stack(
                [cv2.cvtColor(img, cv2.COLOR_RGBA2RGB) for img in pred_images],
                axis=0,
            )

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

