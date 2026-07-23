"""Figure: source->target JS-distance boxplot vs. target-domain delta progress."""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as _fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── Register Pretendard if available on the system ────────────────────────────
_PRETENDARD_PATHS = [
    os.path.expanduser("~/Library/Fonts/Pretendard-Regular.otf"),
    os.path.expanduser("~/Library/Fonts/Pretendard-Bold.ttf"),
    os.path.expanduser("~/Library/Fonts/Pretendard-Black.ttf"),
]
for _p in _PRETENDARD_PATHS:
    if os.path.exists(_p):
        _fm.fontManager.addfont(_p)
if any("pretendard" in f.name.lower() for f in _fm.fontManager.ttflist):
    plt.rcParams["font.family"] = "Pretendard"

from .correlate import merged_feature_table


def _annotate_trend(ax, x, y, loc: str = "upper left", color: str = "k") -> None:
    """Fit + draw a linear trend line and annotate Pearson R and p-value."""
    from scipy.stats import pearsonr

    xy = pd.DataFrame({"x": np.asarray(x, float), "y": np.asarray(y, float)})
    xy = xy.replace([np.inf, -np.inf], np.nan).dropna()
    if len(xy) < 3 or xy["x"].nunique() < 2:
        return
    b = np.polyfit(xy["x"], xy["y"], 1)
    xs = np.linspace(xy["x"].min(), xy["x"].max(), 50)
    ax.plot(xs, np.polyval(b, xs), "--", lw=1.4, color=color, label="trend")
    r, p = pearsonr(xy["x"], xy["y"])
    xa, ha = (0.03, "left") if "left" in loc else (0.97, "right")
    ya = 0.97 if "upper" in loc else 0.05
    ax.text(xa, ya, f"R = {r:.3f}\np = {p:.3f}", transform=ax.transAxes,
            ha=ha, va="top" if "upper" in loc else "bottom", fontsize=9,
            bbox=dict(boxstyle="round", fc="white", ec="0.6", alpha=0.85))


def plot_similarity_boxplot(out_dir: Path, n_bins: int = 15,
                            exclude_absent_source: bool = False) -> Path:
    """Boxplot of within-enum z-scored delta across z-scored JS-distance bins.

    Predictor: source->target JS distance, standardized within each reward enum
    (removes per-enum scale confound). Response: `diff_vs_baseline` z-scored
    within each reward enum. x-axis is inverted so smaller distances are on
    the right, matching the narrative "closer distributions -> better transfer".
    """
    m = merged_feature_table().copy()
    if exclude_absent_source:
        m = m[m["source_present"] == 1.0]

    def _z(x):
        s = x.std()
        return (x - x.mean()) / s if s > 1e-9 else x * 0.0

    m["diff_z"] = m.groupby("reward_enum")["diff_vs_baseline"].transform(_z)
    m["js_z"] = m.groupby("reward_enum")["js_distance"].transform(_z)
    sub = m[["js_distance", "js_z", "diff_z"]].replace([np.inf, -np.inf], np.nan).dropna()

    sub = sub.assign(bin=pd.qcut(sub["js_z"], q=n_bins, duplicates="drop"))
    groups = list(sub.groupby("bin", observed=True))
    positions = [float(interval.mid) for interval, _ in groups]
    data = [g["diff_z"].to_numpy() for _, g in groups]
    n_boxes = len(groups)

    span = sub["js_z"].max() - sub["js_z"].min()
    width = 0.46 * span / max(n_boxes, 1)

    # coolwarm diverging: red = high JSD (left), blue = low JSD (right).
    cmap = plt.get_cmap("coolwarm")
    box_colors = [cmap(0.25 + 0.50 * i / max(n_boxes - 1, 1)) for i in range(n_boxes)]

    fig, ax = plt.subplots(figsize=(3.5, 2.7))
    bp = ax.boxplot(data, positions=positions, widths=width, showmeans=True,
                    patch_artist=True, manage_ticks=False, showfliers=False,
                    meanprops=dict(marker="D", markerfacecolor="white",
                                   markeredgecolor="0.3", markersize=4),
                    medianprops=dict(color="0.15", linewidth=1.5),
                    whiskerprops=dict(linewidth=0.9),
                    capprops=dict(linewidth=0.9))
    for patch, color in zip(bp["boxes"], box_colors):
        patch.set(facecolor=color, alpha=0.85, edgecolor="0.35", linewidth=0.8)

    _annotate_trend(ax, sub["js_z"], sub["diff_z"], loc="upper left")
    ax.axhline(0, color="grey", lw=0.7, ls="--")
    ax.set_ylim(-2.0, 2.0)
    ax.set_yticks([-2, -1, 0, 1, 2])
    x_lo, x_hi = sub["js_z"].min(), sub["js_z"].max()
    pad = 0.05 * (x_hi - x_lo)
    ax.set_xticks(np.linspace(x_lo, x_hi, 5))
    ax.xaxis.set_major_formatter(plt.FormatStrFormatter("%.1f"))
    # Inverted x-axis (high JSD left, low JSD right).
    ax.set_xlim(x_hi + pad, x_lo - pad)
    ax.set_xlabel("Source-Target JS Distance", fontsize=9)
    ax.set_ylabel(r"Target-domain $\Delta$ Progress", fontsize=9)
    ax.grid(True, axis="y", alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout(pad=0.3)
    png_path = out_dir / "fig_similarity_boxplot.png"
    pdf_path = out_dir / "fig_similarity_boxplot.pdf"
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    plt.close(fig)
    return png_path
