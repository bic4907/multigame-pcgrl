"""Data loading for the transferability analysis.

Two sources are combined:

1. The experiment result table
   (``results/transferbility/src/source_target_table_5seed.csv``) — one row per
   ``(reward_enum, target, source)`` with ``mean`` performance and
   ``diff_vs_baseline`` (gain/loss relative to the ``source==none`` baseline).

2. The per-game reward-condition distributions read from the MGPCGRL dataset
   cache (``*.ann.json``). These are the distributions the model is actually
   trained/conditioned on, so they are the right object to correlate against.
"""
from __future__ import annotations

import sys
from functools import lru_cache
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from . import config

# Reuse the battle-tested annotation loader instead of re-implementing it.
sys.path.insert(0, str(config.REPO_ROOT))
from analysis.dataset_distribution.run import load_annotations  # noqa: E402


# ── Experiment result table ─────────────────────────────────────────────────────
def load_results() -> pd.DataFrame:
    """Load the source->target result table (all rows, incl. baselines)."""
    df = pd.read_csv(config.RESULT_CSV)
    df["target"] = df["target"].str.strip()
    df["source"] = df["source"].str.strip()
    df["reward_enum"] = df["reward_enum"].str.strip()
    return df


def load_transfer_rows() -> pd.DataFrame:
    """Result rows for actual mixing runs (``source != none``).

    Adds an integer ``enum`` column (NaN for the ``overall`` aggregate).
    """
    df = load_results()
    df = df[df["source"] != config.BASELINE_SOURCE].copy()
    df["enum"] = df["reward_enum"].map(config.REWARD_LABEL_TO_ENUM)
    return df.reset_index(drop=True)


def load_baselines() -> pd.DataFrame:
    """Baseline (source==none) rows: target's no-mixing performance."""
    df = load_results()
    base = df[df["source"] == config.BASELINE_SOURCE].copy()
    return base.rename(columns={"mean": "baseline_mean", "std": "baseline_std"})[
        ["reward_enum", "target", "baseline_mean", "baseline_std"]
    ].reset_index(drop=True)


# ── Per-game / per-enum condition distributions ─────────────────────────────────
@lru_cache(maxsize=1)
def load_condition_frame() -> pd.DataFrame:
    """Long dataframe with one row per annotated sample (game, enum, condition)."""
    return load_annotations(config.CACHE_DIR, games=config.GAMES)


@lru_cache(maxsize=1)
def condition_arrays() -> Dict[Tuple[str, int], np.ndarray]:
    """Map ``(game, enum) -> np.ndarray`` of condition values."""
    df = load_condition_frame()
    out: Dict[Tuple[str, int], np.ndarray] = {}
    for (game, enum), g in df.groupby(["game", "reward_enum"]):
        out[(str(game), int(enum))] = g["condition"].to_numpy(float)
    return out


def get_array(game: str, enum: int) -> np.ndarray:
    """Condition values for a game/enum (empty array if absent)."""
    return condition_arrays().get((game, enum), np.array([], dtype=float))
