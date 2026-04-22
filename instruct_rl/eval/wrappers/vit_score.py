"""
instruct_rl/eval/wrappers/vit_score.py
========================================
ViTScoreWrapper — ViT-based human-likeness scoring.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ViTScoreWrapper:
    """Scores generated levels against human-authored levels via ViT embedding."""

    def __init__(self, config):
        self.config = config

    def run(
        self,
        df_output: pd.DataFrame,
        instruct_df: pd.DataFrame,
        eval_rendered: list,
        n_rows: int,
        n_eps: int,
    ) -> pd.DataFrame:
        from instruct_rl.evaluation.metrics.vit import ViTEvaluator

        eval_rendered_arr = np.concatenate(eval_rendered, axis=0)[:n_rows]
        evaluator = ViTEvaluator(normalized_vector=self.config.vit_normalize)

        index_ids = instruct_df.index.to_numpy()
        task_ids = np.repeat((index_ids // 4).astype(int), n_eps)

        scores = evaluator.run(eval_rendered_arr, task_ids)
        df_output["vit_score"] = scores
        logger.info(
            "[ViTScoreWrapper] done: mean=%.4f", float(np.mean(scores))
        )
        return df_output

