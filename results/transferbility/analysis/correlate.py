"""Merge JS-distance features with measured performance deltas."""
from __future__ import annotations

import pandas as pd

from . import config
from .data import load_transfer_rows
from .distances import pair_feature_table


def merged_feature_table() -> pd.DataFrame:
    """Join per-enum JS-distance features with measured per-enum diff."""
    feats = pair_feature_table()
    diffs = load_transfer_rows()
    diffs = diffs[diffs["reward_enum"] != config.OVERALL_LABEL]
    return feats.merge(
        diffs[["reward_enum", "target", "source", "diff_vs_baseline"]],
        on=["reward_enum", "target", "source"], how="inner",
    )
