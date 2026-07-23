"""Distribution descriptors and directional source->target distance features.

For every reward enum we characterise each game's condition distribution and,
for every ordered ``(source, target)`` pair, compute a set of similarity /
divergence / coverage features that can be correlated with the measured
performance delta.
"""
from __future__ import annotations

import math
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import ks_2samp, wasserstein_distance

from . import config
from .data import get_array


# ── Histogram helpers (unit-width bins for integer count features) ──────────────
def _edges(*arrays: np.ndarray) -> np.ndarray:
    vals = np.concatenate([a for a in arrays if a.size]) if arrays else np.array([])
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.array([0.0, 1.0])
    lo, hi = math.floor(vals.min()) - 0.5, math.ceil(vals.max()) + 0.5
    span = hi - lo
    if span <= 1000:
        return np.arange(lo, hi + 1.0, 1.0)
    return np.linspace(lo, hi, 61)


def _prob(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    counts, _ = np.histogram(values, bins=edges)
    total = counts.sum()
    return counts.astype(float) / total if total else np.zeros_like(counts, float)


def _entropy(values: np.ndarray, edges: np.ndarray) -> float:
    p = _prob(values, edges)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum()) if p.size else float("nan")


# ── Per-(game, enum) descriptors ────────────────────────────────────────────────
def descriptor(game: str, enum: int) -> Dict[str, float]:
    """Summary statistics of one game's condition distribution for one enum."""
    a = get_array(game, enum)
    if a.size == 0:
        return dict(n=0, mean=np.nan, std=np.nan, cv=np.nan, entropy=np.nan,
                    frac_zero=np.nan, q05=np.nan, q95=np.nan, present=False)
    edges = _edges(a)
    mean = float(a.mean())
    std = float(a.std())
    return dict(
        n=int(a.size),
        mean=mean,
        std=std,
        cv=float(std / mean) if mean else np.nan,
        entropy=_entropy(a, edges),
        frac_zero=float((a == 0).mean()),
        q05=float(np.quantile(a, 0.05)),
        q95=float(np.quantile(a, 0.95)),
        present=config.feature_present(game, enum),
    )


def descriptor_table() -> pd.DataFrame:
    """Descriptor row for every (game, enum) that has data."""
    rows: List[dict] = []
    for enum in config.REWARD_LABEL_TO_ENUM.values():
        for game in config.GAMES:
            d = descriptor(game, enum)
            if d["n"] == 0:
                continue
            rows.append(dict(game=game, enum=enum,
                             reward_label=config.ENUM_TO_REWARD_LABEL[enum], **d))
    return pd.DataFrame(rows)


# ── Directional source->target pair features ────────────────────────────────────
def _coverage(source: np.ndarray, target: np.ndarray) -> float:
    """Fraction of target samples lying within the source's observed range."""
    if source.size == 0 or target.size == 0:
        return float("nan")
    lo, hi = source.min(), source.max()
    return float(((target >= lo) & (target <= hi)).mean())


def pair_features(source: str, target: str, enum: int) -> Dict[str, float]:
    """Distribution-similarity features for mixing ``source`` into ``target``."""
    s = get_array(source, enum)
    t = get_array(target, enum)
    edges = _edges(s, t)
    ps, pt = _prob(s, edges), _prob(t, edges)

    js = float(jensenshannon(ps, pt, base=2.0)) if ps.sum() and pt.sum() else np.nan
    overlap = float(np.minimum(ps, pt).sum()) if ps.sum() and pt.sum() else np.nan
    wass = float(wasserstein_distance(s, t)) if s.size and t.size else np.nan
    ks = float(ks_2samp(s, t).statistic) if s.size and t.size else np.nan

    s_mean, t_mean = (float(s.mean()) if s.size else np.nan,
                      float(t.mean()) if t.size else np.nan)
    s_std, t_std = (float(s.std()) if s.size else np.nan,
                    float(t.std()) if t.size else np.nan)

    return dict(
        js_distance=js,
        overlap_coef=overlap,
        wasserstein=wass,
        ks_stat=ks,
        mean_diff=s_mean - t_mean,
        abs_mean_diff=abs(s_mean - t_mean),
        std_ratio=(s_std / t_std) if t_std else np.nan,
        coverage=_coverage(s, t),
        source_entropy=_entropy(s, edges) if s.size else np.nan,
        source_std=s_std,
        source_present=float(config.feature_present(source, enum)),
        target_present=float(config.feature_present(target, enum)),
    )


def pair_feature_table() -> pd.DataFrame:
    """Pair features for every ordered (source, target, enum) combination."""
    rows: List[dict] = []
    for enum in config.REWARD_LABEL_TO_ENUM.values():
        for target in config.GAMES:
            if not config.feature_present(target, enum):
                continue
            for source in config.GAMES:
                if source == target:
                    continue
                rows.append(dict(
                    reward_enum=config.ENUM_TO_REWARD_LABEL[enum],
                    enum=enum, target=target, source=source,
                    **pair_features(source, target, enum),
                ))
    return pd.DataFrame(rows)
