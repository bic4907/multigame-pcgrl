"""Correlate distribution features with measured transfer performance deltas.

Central question: *which* property of the source game's data distribution
predicts whether mixing it in helps or hurts the target game?

We test several hypotheses by correlating each distribution feature against
``diff_vs_baseline`` (target performance change vs. the no-mixing baseline).
"""
from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from . import config
from .data import load_baselines, load_transfer_rows
from .distances import descriptor, pair_feature_table

# Distribution features tested as predictors of the performance delta.
PREDICTORS = [
    "overlap_coef",    # H1 similarity: shared probability mass (higher = more similar)
    "js_distance",     # H1 similarity: Jensen-Shannon distance (higher = more different)
    "wasserstein",     # H1 similarity: earth-mover distance on raw scale
    "ks_stat",         # H1 similarity: KS statistic
    "abs_mean_diff",   # H1 similarity: |mean gap|
    "coverage",        # H3 coverage: fraction of target range covered by source
    "std_ratio",       # H2 diversity: source spread relative to target
    "source_entropy",  # H2 diversity: source histogram entropy
    "source_std",      # H2 diversity: source spread (absolute)
    "baseline_mean",   # H5 head-room: lower baseline -> more room to improve
]


# ── Merged per-feature table ────────────────────────────────────────────────────
def merged_feature_table() -> pd.DataFrame:
    """Join per-enum pair features with the measured per-enum diff and baseline."""
    feats = pair_feature_table()
    diffs = load_transfer_rows()
    diffs = diffs[diffs["reward_enum"] != config.OVERALL_LABEL]
    base = load_baselines()

    merged = feats.merge(
        diffs[["reward_enum", "target", "source", "mean", "diff_vs_baseline", "diff_std"]],
        on=["reward_enum", "target", "source"], how="inner",
    )
    merged = merged.merge(base, on=["reward_enum", "target"], how="left")
    return merged


# ── Overall (aggregate) table ───────────────────────────────────────────────────
def overall_feature_table() -> pd.DataFrame:
    """Aggregate distribution features per (source, target) for the overall score.

    Pair features are averaged over the enums that BOTH games possess, giving a
    single similarity/diversity descriptor per source->target pair. Joined with
    the ``overall`` diff.
    """
    feats = pair_feature_table()
    shared = feats[(feats["source_present"] == 1.0) & (feats["target_present"] == 1.0)]
    agg = (
        shared.groupby(["target", "source"])[
            ["overlap_coef", "js_distance", "wasserstein", "ks_stat",
             "abs_mean_diff", "coverage", "std_ratio", "source_entropy", "source_std"]
        ].mean().reset_index()
    )

    diffs = load_transfer_rows()
    overall = diffs[diffs["reward_enum"] == config.OVERALL_LABEL]
    base = load_baselines()
    base_overall = base[base["reward_enum"] == config.OVERALL_LABEL]

    merged = agg.merge(
        overall[["target", "source", "mean", "diff_vs_baseline", "diff_std"]],
        on=["target", "source"], how="inner",
    )
    merged = merged.merge(
        base_overall[["target", "baseline_mean"]], on="target", how="left"
    )
    return merged


# ── Correlation scoring ─────────────────────────────────────────────────────────
def correlations(df: pd.DataFrame, predictors: List[str],
                 response: str = "diff_vs_baseline") -> pd.DataFrame:
    """Pearson + Spearman correlation of each predictor against the response."""
    rows: List[dict] = []
    y_all = df[response]
    for p in predictors:
        if p not in df.columns:
            continue
        sub = df[[p, response]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(sub) < 4 or sub[p].nunique() < 3:
            rows.append(dict(predictor=p, n=len(sub), pearson_r=np.nan,
                             pearson_p=np.nan, spearman_r=np.nan, spearman_p=np.nan))
            continue
        pr, pp = pearsonr(sub[p], sub[response])
        sr, sp = spearmanr(sub[p], sub[response])
        rows.append(dict(predictor=p, n=len(sub), pearson_r=pr, pearson_p=pp,
                         spearman_r=sr, spearman_p=sp))
    out = pd.DataFrame(rows)
    return out.reindex(out["pearson_r"].abs().sort_values(ascending=False).index).reset_index(drop=True)


def per_enum_correlations(df: pd.DataFrame, predictors: List[str]) -> pd.DataFrame:
    """Correlation table computed separately within each reward enum."""
    parts: List[pd.DataFrame] = []
    for label, g in df.groupby("reward_enum"):
        c = correlations(g, predictors)
        c.insert(0, "reward_enum", label)
        parts.append(c)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


# ── Feature-absence effect (categorical hypothesis H4) ──────────────────────────
def absence_effect() -> pd.DataFrame:
    """Compare transfer delta when the source LACKS vs. HAS the target's feature.

    A source that structurally lacks a feature (e.g. sokoban has no hazard) can
    only dilute that feature's supervision. This contrasts the mean delta of
    such rows against rows where the source does contain the feature.
    """
    m = merged_feature_table()
    m = m.assign(source_has_feature=m["source_present"] == 1.0)
    rows: List[dict] = []
    for has, g in m.groupby("source_has_feature"):
        rows.append(dict(
            source_has_feature=bool(has),
            n=len(g),
            mean_diff=float(g["diff_vs_baseline"].mean()),
            median_diff=float(g["diff_vs_baseline"].median()),
            frac_negative=float((g["diff_vs_baseline"] < 0).mean()),
        ))
    return pd.DataFrame(rows)
