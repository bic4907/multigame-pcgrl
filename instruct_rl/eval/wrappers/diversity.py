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

        scores = []
        for row_i, _ in tqdm(instruct_df.iterrows(), desc="Computing Diversity"):
            states = [
                np.load(
                    f"{self.config.eval_dir}/reward_{row_i}/seed_{seed_i}/state_0.npy"
                )
                for seed_i in range(1, n_eps + 1)
            ]
            scores.append(compute_hamming_distance(np.array(states)))

        diversity_df = instruct_df.copy()
        diversity_df = diversity_df.loc[
            :, ~diversity_df.columns.str.startswith("embed")
        ]
        diversity_df["diversity"] = scores
        logger.info("[DiversityWrapper] done: %d instructions", len(scores))

        if wandb.run:
            wandb.log({"diversity": wandb.Table(dataframe=diversity_df)})

        return df_output

