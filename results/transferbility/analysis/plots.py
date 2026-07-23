"""Figures for the transferability analysis. All plots are written to OUTPUT_DIR."""
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
_has_pretendard = any(
    "pretendard" in f.name.lower() for f in _fm.fontManager.ttflist
)
if _has_pretendard:
    plt.rcParams["font.family"] = "Pretendard"

from . import config
from .correlate import merged_feature_table, overall_feature_table
from .data import get_array, load_transfer_rows

_CMAP = plt.get_cmap("tab10")


def _enum_colors():
    enums = list(config.REWARD_LABEL_TO_ENUM.values())
    return {e: _CMAP(i % 10) for i, e in enumerate(enums)}


def _annotate_trend(ax, x, y, loc: str = "upper left", color: str = "k") -> None:
    """Fit and draw a linear trend line and annotate Pearson R and p-value."""
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



def plot_condition_distributions(out_dir: Path) -> Path:
    """Per-enum density overlay of every game's condition distribution."""
    enums = list(config.REWARD_LABEL_TO_ENUM.items())
    fig, axes = plt.subplots(1, len(enums), figsize=(4 * len(enums), 3.2))
    game_colors = {g: _CMAP(i % 10) for i, g in enumerate(config.GAMES)}
    for ax, (label, enum) in zip(axes, enums):
        for game in config.GAMES:
            a = get_array(game, enum)
            if a.size == 0 or not config.feature_present(game, enum):
                continue
            lo, hi = np.floor(a.min()), np.ceil(a.max())
            bins = np.arange(lo - 0.5, hi + 1.5, 1.0) if hi - lo <= 200 else 50
            ax.hist(a, bins=bins, density=True, histtype="step",
                    linewidth=1.8, color=game_colors[game], label=game)
        ax.set_title(f"{label} (enum {enum})")
        ax.set_xlabel("condition value")
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("density")
    axes[-1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    path = out_dir / "fig_condition_distributions.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_headroom(out_dir: Path) -> Path:
    """Overall performance delta vs. target baseline (head-room / ceiling)."""
    o = overall_feature_table()
    fig, ax = plt.subplots(figsize=(5.2, 4))
    for target, g in o.groupby("target"):
        ax.scatter(g["baseline_mean"], g["diff_vs_baseline"], s=45, label=target)
    ax.axhline(0, color="k", lw=0.8, ls="--")
    _annotate_trend(ax, o["baseline_mean"], o["diff_vs_baseline"], loc="upper left")
    ax.set_xlabel("target baseline (overall, no mixing)")
    ax.set_ylabel("diff vs baseline (overall)")
    ax.set_title("Head-room: low-baseline targets gain, high-baseline lose")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    fig.tight_layout()
    path = out_dir / "fig_headroom.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_similarity_boxplot(out_dir: Path, n_bins: int = 15,
                            exclude_absent_source: bool = True) -> Path:
    """Boxplot of within-enum z-scored delta across JS-distance bins.

    x-axis: source->target JS distance (raw; already bounded in [0, 1] so it is
    comparable across enums). y-axis: diff vs baseline, z-scored within each enum.
    Boxes are colored on a sequential scale (light → dark = low → high JS distance)
    using the Blues colormap, suitable for academic publication.
    A linear trend line fitted on the underlying points is overlaid with its
    Pearson R / p annotation.

    ``exclude_absent_source=True`` (default) drops rows where the source game
    structurally lacks the target's feature — those rows are categorically different
    from distribution-similarity effects and inflate variance in certain bins.
    """
    m = merged_feature_table().copy()
    if exclude_absent_source:
        m = m[m["source_present"] == 1.0]

    def _z(x):
        s = x.std()
        return (x - x.mean()) / s if s > 1e-9 else x * 0.0

    m["diff_z"] = m.groupby("reward_enum")["diff_vs_baseline"].transform(_z)
    sub = m[["js_distance", "diff_z"]].replace([np.inf, -np.inf], np.nan).dropna()

    sub = sub.assign(bin=pd.qcut(sub["js_distance"], q=n_bins, duplicates="drop"))
    groups = list(sub.groupby("bin", observed=True))
    positions = [float(interval.mid) for interval, _ in groups]
    data = [g["diff_z"].to_numpy() for _, g in groups]
    counts = [len(g) for _, g in groups]
    n_boxes = len(groups)

    span = sub["js_distance"].max() - sub["js_distance"].min()
    width = 0.46 * span / max(n_boxes, 1)

    # Sequential color palette: coolwarm diverging (red=high JSD left, blue=low JSD right).
    cmap = plt.get_cmap("coolwarm")
    # i=0 is lowest JSD bin (right side) → blue; i=n-1 is highest (left side) → red.
    # Use 0.25–0.75 range for subtle, not saturated tones.
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

    # n= labels removed for clean publication look.

    _annotate_trend(ax, sub["js_distance"], sub["diff_z"], loc="upper left")
    ax.axhline(0, color="grey", lw=0.7, ls="--")
    # y-limits: cover all whisker ends + small margin.
    whisker_vals = [l.get_ydata() for l in bp["whiskers"]]
    w_min = min(v.min() for v in whisker_vals)
    w_max = max(v.max() for v in whisker_vals)
    ax.set_ylim(-1.7, 1.7)
    ax.set_yticks([-1, 0, 1])
    # Sparse x-ticks: 5 evenly spaced across the JS distance range.
    x_lo, x_hi = sub["js_distance"].min(), sub["js_distance"].max()
    ax.set_xticks(np.linspace(x_lo, 0.9, 5))
    ax.xaxis.set_major_formatter(plt.FormatStrFormatter("%.1f"))
    # Inverted x-axis (high JSD left, low JSD right): pass limits in reversed order.
    ax.set_xlim(0.9, x_lo - 0.01)
    ax.set_xlabel("Source--Target Normalized JS Distance", fontsize=9)
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


def plot_absence_effect(out_dir: Path) -> Path:
    """Delta distribution when source has vs. lacks the target's feature."""
    m = merged_feature_table()
    m = m.assign(has=m["source_present"] == 1.0)
    groups = [("source lacks\nfeature", m.loc[~m["has"], "diff_vs_baseline"]),
              ("source has\nfeature", m.loc[m["has"], "diff_vs_baseline"])]
    fig, ax = plt.subplots(figsize=(4.6, 4))
    ax.boxplot([g.to_numpy() for _, g in groups],
               labels=[n for n, _ in groups], showmeans=True)
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.set_ylabel("diff vs baseline")
    ax.set_title("Structural feature absence hurts transfer")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    path = out_dir / "fig_absence_effect.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_experiment_bars(out_dir: Path) -> Path:
    """Per-experiment bar chart of diff_vs_baseline with +-diff_std error bars.

    One panel per reward_enum; within a panel, bars are grouped by target and
    colored by the mixed-in source game.
    """
    df = load_transfer_rows()
    labels = [config.OVERALL_LABEL] + config.FEATURE_LABELS
    source_colors = {g: _CMAP(i % 10) for i, g in enumerate(config.GAMES)}
    slot = {g: i for i, g in enumerate(config.GAMES)}      # fixed source slot order
    width = 0.15

    fig, axes = plt.subplots(2, 3, figsize=(17, 8))
    for ax, label in zip(axes.flat, labels):
        sub = df[df["reward_enum"] == label]
        targets = [t for t in config.GAMES if t in set(sub["target"])]
        for ti, target in enumerate(targets):
            tsub = sub[sub["target"] == target]
            for _, row in tsub.iterrows():
                s = row["source"]
                x = ti + (slot[s] - 2) * width
                ax.bar(x, row["diff_vs_baseline"], width=width,
                       color=source_colors[s], edgecolor="none")
                ax.errorbar(x, row["diff_vs_baseline"], yerr=row["diff_std"],
                            fmt="none", ecolor="0.25", elinewidth=0.9, capsize=2)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xticks(range(len(targets)))
        ax.set_xticklabels(targets, fontsize=9)
        ax.set_title(label)
        ax.set_ylabel("diff vs baseline")
        ax.grid(True, axis="y", alpha=0.25)

    # Shared legend for source colors.
    handles = [plt.Rectangle((0, 0), 1, 1, color=source_colors[g]) for g in config.GAMES]
    fig.legend(handles, [f"source={g}" for g in config.GAMES], ncol=len(config.GAMES),
               frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Per-experiment transfer delta (+/- std across seeds), grouped by target",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    path = out_dir / "fig_experiment_bars.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def generate_all(out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    return [
        plot_condition_distributions(out_dir),
        plot_headroom(out_dir),
        plot_similarity_boxplot(out_dir),
        plot_absence_effect(out_dir),
        plot_experiment_bars(out_dir),
    ]
