"""Deeper analyses used when the naive pooled correlation is inconclusive.

The raw pooled correlation between distribution features and the performance
delta is weak and sign-inconsistent, because each reward enum lives on a wildly
different value scale (e.g. Region ~ 0-30 vs Hazard ~ 0-250) and has a different
delta magnitude. Pooling raw features lets a few high-scale enums dominate.

This module controls for that by z-scoring predictor and response *within each
reward enum* before pooling, then quantifies the independent contribution of
each factor via partial correlation and a small OLS fit.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from .correlate import PREDICTORS, merged_feature_table


def _base_table(exclude_absent_source: bool = False) -> pd.DataFrame:
    """Merged feature/delta table, optionally dropping rows whose SOURCE game
    structurally lacks the target's feature (feature not present in the dataset)."""
    m = merged_feature_table()
    if exclude_absent_source:
        m = m[m["source_present"] == 1.0].copy()
    return m


def _zscore_within_enum(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """Return a copy with ``<col>_z`` columns z-scored within each reward enum."""
    out = df.copy()

    def _z(x: pd.Series) -> pd.Series:
        s = x.std()
        return (x - x.mean()) / s if s > 1e-9 else x * 0.0

    for c in cols:
        out[c + "_z"] = out.groupby("reward_enum")[c].transform(_z)
    return out


def scale_controlled_correlations(exclude_absent_source: bool = False) -> pd.DataFrame:
    """Pooled correlation after within-enum z-scoring of predictor and response.

    This removes the per-enum scale confound so effects from different enums are
    comparable and can be pooled. Set ``exclude_absent_source=True`` to drop rows
    where the source game structurally lacks the target's feature.
    """
    cols = PREDICTORS + ["diff_vs_baseline"]
    m = _zscore_within_enum(_base_table(exclude_absent_source), cols)
    resp = "diff_vs_baseline_z"
    rows: List[dict] = []
    for p in PREDICTORS:
        sub = m[[p + "_z", resp]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(sub) < 4 or sub[p + "_z"].nunique() < 3:
            continue
        pr, pp = pearsonr(sub[p + "_z"], sub[resp])
        sr, sp = spearmanr(sub[p + "_z"], sub[resp])
        rows.append(dict(predictor=p, n=len(sub), pearson_r=pr, pearson_p=pp,
                         spearman_r=sr, spearman_p=sp))
    out = pd.DataFrame(rows)
    return out.reindex(
        out["pearson_r"].abs().sort_values(ascending=False).index
    ).reset_index(drop=True)


def _residualise(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Residual of ``a`` after linear regression on ``b``."""
    X = np.column_stack([np.ones(len(b)), b])
    beta = np.linalg.lstsq(X, a, rcond=None)[0]
    return a - X @ beta


def partial_correlation(predictor: str, control: str = "baseline_mean",
                        exclude_absent_source: bool = False) -> dict:
    """Partial correlation of ``predictor`` with the delta, controlling ``control``.

    Answers: does distribution similarity still matter *after* accounting for the
    target's head-room (baseline performance)?
    """
    cols = [predictor, control, "diff_vs_baseline"]
    m = _zscore_within_enum(_base_table(exclude_absent_source), cols)
    sub = m[[predictor + "_z", control + "_z", "diff_vs_baseline_z"]].replace(
        [np.inf, -np.inf], np.nan).dropna()
    ry = _residualise(sub["diff_vs_baseline_z"].to_numpy(), sub[control + "_z"].to_numpy())
    rx = _residualise(sub[predictor + "_z"].to_numpy(), sub[control + "_z"].to_numpy())
    r, p = pearsonr(rx, ry)
    return dict(predictor=predictor, control=control, n=len(sub),
                partial_r=float(r), partial_p=float(p))


def ols_two_factor(predictor: str = "js_distance",
                   control: str = "baseline_mean",
                   exclude_absent_source: bool = False) -> dict:
    """Fit ``diff_z ~ control_z + predictor_z`` (within-enum z-scored) via OLS."""
    cols = [predictor, control, "diff_vs_baseline"]
    m = _zscore_within_enum(_base_table(exclude_absent_source), cols)
    sub = m[[predictor + "_z", control + "_z", "diff_vs_baseline_z"]].replace(
        [np.inf, -np.inf], np.nan).dropna()
    X = np.column_stack([np.ones(len(sub)), sub[control + "_z"], sub[predictor + "_z"]])
    y = sub["diff_vs_baseline_z"].to_numpy()
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    yhat = X @ beta
    ss_res = float(((y - yhat) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return dict(
        intercept=float(beta[0]),
        control_beta=float(beta[1]),
        predictor_beta=float(beta[2]),
        control=control, predictor=predictor,
        r2=1.0 - ss_res / ss_tot if ss_tot else float("nan"),
        n=len(sub),
    )


def target_headroom_table() -> pd.DataFrame:
    """Per-target baseline vs. mean delta — the head-room / ceiling view."""
    m = merged_feature_table()
    base = m.groupby("target")["baseline_mean"].mean()  # per-enum avg baseline
    g = (
        m.groupby("target")["diff_vs_baseline"]
        .agg(mean_diff="mean", min_diff="min", max_diff="max",
             frac_negative=lambda s: float((s < 0).mean()))
    )
    out = g.join(base.rename("avg_baseline")).reset_index()
    return out.sort_values("avg_baseline").reset_index(drop=True)


# ── Full vs. absent-source-excluded comparison (H4 robustness check) ─────────────
def absent_exclusion_comparison() -> pd.DataFrame:
    """Side-by-side scale-controlled correlations: full set vs. rows where the
    source game actually contains the target's feature.

    Isolates how much of each distribution effect is carried by the structural
    feature-absence cases (source game missing the feature in the dataset).
    """
    full = scale_controlled_correlations(exclude_absent_source=False)
    excl = scale_controlled_correlations(exclude_absent_source=True)
    merged = full.merge(excl, on="predictor", suffixes=("_full", "_excl"))
    keep = ["predictor", "n_full", "pearson_r_full", "pearson_p_full",
            "n_excl", "pearson_r_excl", "pearson_p_excl"]
    merged = merged[keep].copy()
    merged["delta_r"] = merged["pearson_r_excl"] - merged["pearson_r_full"]
    return merged.reindex(
        merged["pearson_r_excl"].abs().sort_values(ascending=False).index
    ).reset_index(drop=True)


def partial_correlations_excluding_absent() -> pd.DataFrame:
    """Partial correlations (control=baseline) on the absent-source-excluded set."""
    rows = [
        partial_correlation("js_distance", exclude_absent_source=True),
        partial_correlation("overlap_coef", exclude_absent_source=True),
        partial_correlation("coverage", exclude_absent_source=True),
        partial_correlation("ks_stat", exclude_absent_source=True),
        partial_correlation("std_ratio", exclude_absent_source=True),
    ]
    return pd.DataFrame(rows)
