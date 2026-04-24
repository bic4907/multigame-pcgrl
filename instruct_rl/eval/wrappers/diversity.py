"""
instruct_rl/eval/wrappers/diversity.py
=======================================
DiversityWrapper — per-instruction Hamming diversity across seeds.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import wandb
from tqdm import tqdm

logger = logging.getLogger(__name__)


class DiversityWrapper:
    """Computes per-instruction Hamming diversity across seeds."""

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
        from instruct_rl.evaluation.hamming import compute_hamming_distance
        from instruct_rl.eval.hdf5_store import open_eval_store, read_state

        scores = []
        with open_eval_store(self.config.eval_dir, mode="r") as h5:
            for row_i, row in tqdm(instruct_df.iterrows(), desc="Computing Diversity"):
                game        = row.get('game', 'unknown')
                re_val      = int(row.get('reward_enum', row_i))
                folder_name = f"{game}_re{re_val}_{int(row_i):04d}"
                states = np.array([
                    read_state(h5, folder_name, seed_i)
                    for seed_i in range(1, n_eps + 1)
                ])
                scores.append(compute_hamming_distance(states))

        diversity_df = instruct_df.copy()
        diversity_df = diversity_df.loc[
            :, ~diversity_df.columns.str.startswith("embed")
        ]
        diversity_df["diversity"] = scores
        logger.info("[DiversityWrapper] done: %d instructions", len(scores))

        if wandb.run:
            wandb.log({"diversity": wandb.Table(dataframe=diversity_df)})

        return df_output

