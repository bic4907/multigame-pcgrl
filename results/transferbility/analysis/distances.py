"""Source->target JS distance per (reward_enum, target, source)."""
from __future__ import annotations

import math
from typing import List

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon

from . import config
from .data import get_array


def _edges(*arrays: np.ndarray) -> np.ndarray:
    vals = np.concatenate([a for a in arrays if a.size]) if arrays else np.array([])
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.array([0.0, 1.0])
    lo, hi = math.floor(vals.min()) - 0.5, math.ceil(vals.max()) + 0.5
    if hi - lo <= 1000:
        return np.arange(lo, hi + 1.0, 1.0)
    return np.linspace(lo, hi, 61)


def _prob(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    counts, _ = np.histogram(values, bins=edges)
    total = counts.sum()
    return counts.astype(float) / total if total else np.zeros_like(counts, float)


def pair_feature_table() -> pd.DataFrame:
    """JS distance for every ordered (source, target, enum) combination."""
    rows: List[dict] = []
    for enum in config.REWARD_LABEL_TO_ENUM.values():
        for target in config.GAMES:
            if not config.feature_present(target, enum):
                continue
            for source in config.GAMES:
                if source == target:
                    continue
                s = get_array(source, enum)
                t = get_array(target, enum)
                edges = _edges(s, t)
                ps, pt = _prob(s, edges), _prob(t, edges)
                js = (float(jensenshannon(ps, pt, base=2.0))
                      if ps.sum() and pt.sum() else np.nan)
                rows.append(dict(
                    reward_enum=config.ENUM_TO_REWARD_LABEL[enum],
                    enum=enum, target=target, source=source,
                    js_distance=js,
                    source_present=float(config.feature_present(source, enum)),
                ))
    return pd.DataFrame(rows)
