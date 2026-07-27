"""
progress.py
===========
Progress measure utility.

progress = 1 - |condition - feat_final| / (|condition - feat_s0| + ε)
  - condition : target value (cont_value)
  - feat_final: feature measured in the final state
  - feat_s0   : feature measured in the initial state (s0)
  - Values are clipped to [0, 100].
  - Rows with condition == -1 (the null sentinel) return NaN.
"""

import numpy as np
import pandas as pd

EPS = 1e-7


def calculate_progress(condition: float, feat: float, feat_s0: float) -> float:
    """Compute scalar progress from 0 to 100, returning NaN when appropriate."""
    if np.isnan(condition) or condition == -1:
        return float("nan")
    raw = 1.0 - abs(condition - feat) / (abs(condition - feat_s0) + EPS)
    return float(np.clip(raw * 100.0, 0.0, 100.0))


class ProgressWrapper:
    """Add progress_* columns to a df_ctrl_sim DataFrame.

    Parameters
    ----------
    n_cond : int
        Number of condition_* / feat_* columns.
    """

    def __init__(self, n_cond: int):
        self.n_cond = n_cond

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        """Read condition/feat/feat_s0 at the reward_enum index and return
        a copy with a single 'progress' column."""
        df = df.copy()

        def _row_progress(row):
            i = row.get("reward_enum", float("nan"))
            if pd.isna(i):
                return float("nan")
            i = int(i)
            cond = row.get(f"condition_{i}", float("nan"))
            feat = row.get(f"feat_{i}", float("nan"))
            feat_s0 = row.get(f"feat_{i}_s0", float("nan"))
            if pd.isna(feat) or pd.isna(feat_s0):
                return float("nan")
            return calculate_progress(cond, feat, feat_s0)

        df["progress"] = df.apply(_row_progress, axis=1)
        return df
