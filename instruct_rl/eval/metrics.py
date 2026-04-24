"""
metrics.py
==========
Post-eval metric orchestrator.
Delegates to individual wrappers in instruct_rl.eval.wrappers:
  - DiversityWrapper   — Hamming diversity across seeds
  - HumanLikenessWrapper — ViT-based human-likeness score
  - TPKLWrapper        — Tile-Pattern KL divergence
"""
import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from instruct_rl.eval.wrappers import DiversityWrapper, ViTScoreWrapper, TPKLWrapper

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    diversity: Optional[np.ndarray] = field(default=None)   # shape (N,)
    vit_score: Optional[np.ndarray] = field(default=None)   # shape (N*n_eps,)
    tpkldiv: Optional[np.ndarray] = field(default=None)     # shape (N*n_eps,)


def run_post_eval(
    config,
    instruct_df: pd.DataFrame,
    n_eps: int,
    gt_levels=None,
    gt_images=None,
) -> EvaluationResult:
    """Run all enabled post-eval metrics.

    Args:
        config      : EvalConfig (or subclass).
        instruct_df : Original instruct DataFrame.
        n_eps       : Number of episodes (seeds).
        gt_levels   : (M, H, W) int — TPKL 계산용.
        gt_images   : (M, H*ts, W*ts, 3) uint8 — ViT 계산용 dataset 렌더링 이미지.

    Returns:
        EvaluationResult — 각 metric의 scores 배열.
    """
    kwargs = dict(
        instruct_df=instruct_df,
        n_eps=n_eps,
    )

    diversity_scores = None
    vit_score_scores = None
    tpkldiv_scores = None

    if config.diversity:
        diversity_scores = DiversityWrapper(config).run(**kwargs)

    if config.vit_score:
        assert gt_images is not None, (
            "[run_post_eval] gt_images must be provided for ViT score. "
            "Pass gt_images from eval_utils via make_eval."
        )
        vit_score_scores = ViTScoreWrapper(config).run(gt_images=gt_images, **kwargs)

    if config.tpkldiv:
        assert gt_levels is not None, (
            "[run_post_eval] gt_levels must be provided for TPKL. "
            "Pass gt_levels from eval_utils via make_eval."
        )
        tpkldiv_scores = TPKLWrapper(config).run(gt_levels=gt_levels, **kwargs)

    return EvaluationResult(
        diversity=diversity_scores,
        vit_score=vit_score_scores,
        tpkldiv=tpkldiv_scores,
    )
