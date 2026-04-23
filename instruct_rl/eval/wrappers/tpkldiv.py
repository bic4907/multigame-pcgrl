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

import logging

import numpy as np
import pandas as pd
from tqdm import tqdm

from instruct_rl.eval.hdf5_store import open_eval_store, read_state
from instruct_rl.evaluation.metrics.tpkl_utils import load_gt_levels
from instruct_rl.evaluation.metrics.tpkldiv import TPKLEvaluator

logger = logging.getLogger(__name__)


class TPKLWrapper:
    """Computes Tile-Pattern KL divergence across seeds per instruction."""

    def __init__(self, config):
        self.config = config

    def run(
        self,
        df_output: pd.DataFrame,
        instruct_df: pd.DataFrame,
        n_eps: int,
        **_,                          # eval_rendered, n_rows 등 미사용 kwargs 흡수
    ) -> pd.DataFrame:
        """
        Parameters
        ----------
        instruct_df : 평가 대상 DataFrame (game, reward_enum, feature_name, condition_value)
        n_eps       : 시드(에피소드) 수
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
                    for seed_i in range(1, n_eps + 1)
                ]
            )

        # ② GT levels 로드 (M, H, W)
        gt_levels = load_gt_levels()

        # ③ 순수 연산 위임
        evaluator = TPKLEvaluator()
        scores = evaluator.run(pred_levels, gt_levels)

        df_output["tpkldiv"] = scores.reshape(-1)
        logger.info("[TPKLWrapper] done: mean=%.4f", float(np.mean(scores)))
        return df_output

