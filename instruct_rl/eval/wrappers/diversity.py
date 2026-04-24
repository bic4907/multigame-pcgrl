"""
instruct_rl/eval/wrappers/diversity.py
=======================================
DiversityWrapper — per-instruction Hamming diversity across seeds.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from tqdm import tqdm

from instruct_rl.evaluation.hamming import compute_hamming_distance
from instruct_rl.eval.hdf5_store import open_eval_store, read_state

logger = logging.getLogger(__name__)


class DiversityWrapper:
    """Computes per-instruction Hamming diversity across seeds."""

    def __init__(self, config):
        self.config = config

    def run(
        self,
        instruct_df: pd.DataFrame,
        n_eps: int,
        **_,
    ) -> np.ndarray:
        """Compute per-instruction Hamming diversity scores.

        Returns
        -------
        scores : np.ndarray, shape (N,) — one value per instruction row.
        """

        """
        # instruct_df : pd.DataFrame
        +----+---------+---------+-----------------------------------------+---------------+---------------+---------------+---------------+---------------+---------------+
        |    |   row_i | game    | instruction                             |   reward_enum |   condition_0 |   condition_1 |   condition_2 |   condition_3 |   condition_4 |
        |----+---------+---------+-----------------------------------------+---------------+---------------+---------------+---------------+---------------+---------------|
        |  0 |       0 | dungeon | Marginally disconnected walkable areas. |             0 |             1 |           nan |           nan |           nan |           nan |
        |  1 |       1 | dungeon | The map is heavily fragmented.          |             0 |            37 |           nan |           nan |           nan |           nan |
        +----+---------+---------+-----------------------------------------+---------------+---------------+---------------+---------------+---------------+---------------+
        """

        output_df = instruct_df.copy()
        output_df['diversity'] = np.nan

        with open_eval_store(self.config.eval_dir, mode="r") as h5:
            for row_i, row in tqdm(instruct_df.iterrows(), desc="Computing Diversity"):
                game        = row.get('game', 'unknown')
                re_val      = int(row.get('reward_enum', row_i))
                folder_name = f"{game}_re{re_val}_{int(row_i):04d}"
                states = np.array([
                    read_state(h5, folder_name, seed_i)
                    for seed_i in range(n_eps)
                ])
                output_df.loc[row_i, 'diversity'] = compute_hamming_distance(states)

        logger.info("[DiversityWrapper] done: %d instructions", len(output_df))

        return output_df
