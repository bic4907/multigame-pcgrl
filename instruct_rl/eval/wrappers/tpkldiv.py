"""
instruct_rl/eval/wrappers/tpkldiv.py
======================================
TPKLWrapper — Tile-Pattern KL divergence across seeds per instruction.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)


class TPKLWrapper:
    """Computes Tile-Pattern KL divergence across seeds per instruction."""

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
        from instruct_rl.evaluation.metrics.tpkldiv import TPKLEvaluator
        from instruct_rl.eval.hdf5_store import open_eval_store, read_state

        with open_eval_store(self.config.eval_dir, mode="r") as h5:
            states = np.array(
                [
                    read_state(
                        h5,
                        f"{row.get('game','unknown')}_re{int(row.get('reward_enum', row_i))}_{int(row_i):04d}",
                        seed_i,
                    )
                    for row_i, row in tqdm(instruct_df.iterrows(), desc="Computing TPKLDiv")
                    for seed_i in range(1, n_eps + 1)
                ]
            )

        index_ids = instruct_df.index.to_numpy()
        task_ids = np.repeat((index_ids // 4).astype(int), n_eps)

        evaluator = TPKLEvaluator()
        scores = np.array(
            evaluator.run(states, task_ids, show_progress=True)
        ).reshape(-1)
        df_output["tpkldiv"] = scores
        logger.info(
            "[TPKLWrapper] done: mean=%.4f", float(np.mean(scores))
        )
        return df_output

