"""
tpkldiv.py
==========
Tile-Pattern KL Divergence Evaluator.
"""
import logging

import numpy as np

from instruct_rl.evaluation.metrics.base import BaseEvaluator
from instruct_rl.evaluation.metrics.tpkl_utils import (
    build_gt_distribution,
    compute_jsd_scores,
    # re-export for backward compat
    quantize_condition as _quantize_condition,  # noqa: F401
    build_task_key as _build_task_key,          # noqa: F401
    group_states_by_task as prepare_pred_groups, # noqa: F401
)

logger = logging.getLogger(__name__)


class TPKLEvaluator(BaseEvaluator):
    """
    Tile-Pattern KL Divergence Evaluator.

    run(pred_levels, gt_levels, window_sizes, epsilon)
    --------------------------------------------------
    pred_levels  : (N, H, W) int — 평가할 레벨 배열
    gt_levels    : (M, H, W) int — GT 레벨 배열
    window_sizes : 슬라이딩 윈도우 크기 목록  e.g. (2, 3)
    epsilon      : Laplace smoothing 값
    """

    def run(
        self,
        pred_levels: np.ndarray,
        gt_levels: np.ndarray,
        window_sizes=(2, 3),
        epsilon: float = 1e-6,
        _pbar=None,
    ) -> np.ndarray:
        """
        Returns
        -------
        scores : (N,) float — 낮을수록 GT와 분포가 유사 (JSD 스타일)
        """
        window_sizes = tuple(window_sizes)
        pred_levels  = np.asarray(pred_levels, dtype=np.int32)
        gt_levels    = np.asarray(gt_levels,   dtype=np.int32)

        assert pred_levels.ndim == 3 and gt_levels.ndim == 3, (
            f"inputs must be 3-D (N,H,W) arrays. "
            f"pred={pred_levels.shape}  gt={gt_levels.shape}"
        )

        logger.info("Building GT distribution — M=%d levels, window_sizes=%s", len(gt_levels), window_sizes)
        gt_dists = build_gt_distribution(gt_levels, window_sizes, epsilon, _pbar=_pbar)
        logger.info("GT distribution ready. Computing JSD scores for N=%d pred levels ...", len(pred_levels))
        scores = compute_jsd_scores(pred_levels, gt_dists, window_sizes, epsilon, _pbar=_pbar)
        logger.info("Done. scores: min=%.4f  max=%.4f  mean=%.4f", scores.min(), scores.max(), scores.mean())
        return scores
