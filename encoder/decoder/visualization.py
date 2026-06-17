from __future__ import annotations

import logging
import math
import os
from os.path import basename
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from .common import _REWARD_ENUM_NAMES

log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logger = logging.getLogger(basename(__file__))
logger.setLevel(getattr(logging, log_level, logging.INFO))


# Fixed per-game color palette.
_GAME_SCATTER_COLORS = {
    "dungeon": "#4C72B0",
    "sokoban": "#DD8452",
    "zelda": "#55A868",
    "pokemon": "#C44E52",
    "doom": "#8172B3",
}

_MODALITY_MARKERS: Dict[str, str] = {
    "text": "o",
    "level": "s",
}


def _get_game_color(game: str, color_map: Optional[Dict[str, str]] = None, fallback_seed: int = 0) -> str:
    """Return the fixed color for a game."""
    if color_map is None:
        color_map = _GAME_SCATTER_COLORS
    if game in color_map:
        return color_map[game]

    # Fall back to a cyclic palette for new game names.
    fallback_colors = [
        "#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3",
        "#393b79", "#637939", "#8c6d31", "#843c39", "#8c564b",
    ]
    return fallback_colors[fallback_seed % len(fallback_colors)]


def _compute_scatter_trendline(
    x: np.ndarray,
    y: np.ndarray,
) -> Tuple[float, float, float]:
    """Return (r, slope, intercept) for finite (x, y) pairs.

    If not enough finite pairs exist, returns NaN values.
    """
    x_arr = np.asarray(x, dtype=float).reshape(-1)
    y_arr = np.asarray(y, dtype=float).reshape(-1)
    finite_mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_arr = x_arr[finite_mask]
    y_arr = y_arr[finite_mask]

    if x_arr.size < 2 or y_arr.size < 2:
        return float("nan"), float("nan"), float("nan")
    if np.std(x_arr) < 1e-12 or np.std(y_arr) < 1e-12:
        return float("nan"), float("nan"), float("nan")

    try:
        r = float(np.corrcoef(x_arr, y_arr)[0, 1])
        slope, intercept = np.polyfit(x_arr, y_arr, 1)
    except Exception:
        return float("nan"), float("nan"), float("nan")
    return r, float(slope), float(intercept)


def create_fewshot_plot(
    results: Dict[float, Dict[str, float]],
    reg_results: Dict[float, Dict[str, float]],
    unseen_game_names: Set[str],
    out_dir: str,
) -> str:
    """Visualize few-shot ratio sweep results as a single regression-loss panel.

    Reward accuracy is logged only as a W&B scalar and is not included in the image.
    """
    os.makedirs(out_dir, exist_ok=True)

    ratios = sorted([r for r in results.keys() if r < 1.0])
    all_games = sorted(
        {g for r in reg_results.values() for g in r
         if g not in ("overall", "seen_overall", "unseen_overall")}
    )

    unseen_tag = ", ".join(sorted(unseen_game_names))

    fig, ax = plt.subplots(figsize=(3.8, 2.6))

    # Seen / unseen summary lines.
    seen_ov = [reg_results[r].get("seen_overall", float("nan")) for r in ratios]
    unseen_ov = [reg_results[r].get("unseen_overall", float("nan")) for r in ratios]
    ax.plot(ratios, seen_ov, marker="s", markersize=4, linewidth=2.4,
            linestyle="--", color="#b2182b", label="Seen")
    ax.plot(ratios, unseen_ov, marker="o", markersize=4, linewidth=2.4,
            linestyle="-", color="#2166ac", label="Unseen")

    ax.set_xlabel("Few-shot Ratio", fontsize=8)
    ax.set_ylabel("Regression Loss (Huber)", fontsize=8)
    ax.set_title(f"Unseen: {unseen_tag}", fontsize=8.5)
    ax.set_xlim(-0.02, 1.02)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right", fontsize=6, framealpha=0.85)

    path = os.path.join(out_dir, "fewshot_ratio_vs_reward_accuracy.png")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    logger.info("Few-shot plot saved: %s", path)
    return path


def create_scatter_plots(
    scatter_data: Dict[int, Dict[str, np.ndarray]],
    out_dir: str,
    max_points: int = 1000,
    seed: int = 0,
    space: str = "norm",
    game_colors: Optional[Dict[str, str]] = None,
) -> Optional[str]:
    """Per reward_enum scatter plot (pred vs target).

    Parameters
    ----------
    scatter_data : return value from evaluate_per_game()
    max_points   : maximum points per subplot; random sample if exceeded
    space        : "norm" (normalized [0, 1] space) or "raw" (linear scale)
    """
    if not scatter_data:
        logger.warning("create_scatter_plots: empty scatter_data — skipping")
        return None

    os.makedirs(out_dir, exist_ok=True)
    pred_key = f"pred_{space}"
    target_key = f"target_{space}"

    enums = sorted(scatter_data.keys())
    n = len(enums)
    ncols = min(3, n)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, squeeze=False)
    fig.set_size_inches(2.9 * ncols, 2.3 * nrows)

    color_map = {**_GAME_SCATTER_COLORS, **(game_colors or {})}
    rng = np.random.RandomState(seed)
    for i, e in enumerate(enums):
        ax = axes[i // ncols][i % ncols]
        pred = np.asarray(scatter_data[e][pred_key])
        target = np.asarray(scatter_data[e][target_key])
        game_names = np.asarray(scatter_data[e].get("game_names", np.array([])), dtype=object)

        n_pts = len(pred)
        if n_pts > max_points:
            sel = rng.choice(n_pts, size=max_points, replace=False)
            pred = pred[sel]
            target = target[sel]
            if len(game_names) > 0:
                game_names = game_names[sel]

        if len(game_names) == len(pred) and len(set(game_names)) > 1:
            for gi, gname in enumerate(sorted(set(game_names))):
                gmask = game_names == gname
                if not gmask.any():
                    continue
                ax.scatter(
                    target[gmask],
                    pred[gmask],
                    s=6, alpha=0.45, edgecolors="none",
                    color=_get_game_color(gname, color_map, gi),
                    label=gname, rasterized=True,
                )

            handles, labels = ax.get_legend_handles_labels()
            if labels:
                by_label = dict(zip(labels, handles))
                ax.legend(
                    by_label.values(),
                    by_label.keys(),
                    fontsize=6,
                    loc="upper right",
                    framealpha=0.8,
                )
        else:
            ax.scatter(target, pred, s=6, alpha=0.45, edgecolors="none", color="#2166ac")

        # y=x reference line.
        lo = float(min(target.min(), pred.min())) if len(pred) else 0.0
        hi = float(max(target.max(), pred.max())) if len(pred) else 1.0
        ax.plot([lo, hi], [lo, hi], linestyle="--", color="#888", linewidth=1)
        r, slope, intercept = _compute_scatter_trendline(target, pred)
        if np.isfinite(r):
            ax.plot(
                [lo, hi],
                [slope * lo + intercept, slope * hi + intercept],
                linestyle="-",
                color="#1b9e77",
                linewidth=1,
            )

        name = _REWARD_ENUM_NAMES.get(int(e), f"enum_{e}")
        mae = float(np.mean(np.abs(pred - target))) if len(pred) else float("nan")
        r_txt = f"{r:.4f}" if np.isfinite(r) else "nan"
        ax.set_title(f"{name}\nMAE={mae:.4f} | r={r_txt}", fontsize=8)
        ax.set_xlabel("target", fontsize=7)
        ax.set_ylabel("pred", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.grid(alpha=0.25)

    # Hide unused subplots.
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].set_visible(False)

    fig.suptitle(f"Train-set Prediction Scatter ({space} space)", fontsize=10)
    fig.tight_layout()
    path = os.path.join(out_dir, f"train_scatter_{space}.png")
    fig.savefig(path, dpi=220)
    plt.close(fig)
    logger.info("Scatter plot saved: %s", path)
    return path


def create_regression_scatter_plots_per_enum(
    scatter_data: Dict[int, Dict[str, np.ndarray]],
    out_dir: str,
    max_points: int = 1000,
    seed: int = 0,
    space: str = "raw",
    game_colors: Optional[Dict[str, str]] = None,
) -> Dict[int, str]:
    """Save one regression scatter image per enum and return their paths.

    Returns
    -------
    {reward_enum: image_path}
    """
    if not scatter_data:
        logger.warning("create_regression_scatter_plots_per_enum: empty scatter_data — skipping")
        return {}

    os.makedirs(out_dir, exist_ok=True)
    pred_key = f"pred_{space}"
    target_key = f"target_{space}"

    color_map = {**_GAME_SCATTER_COLORS, **(game_colors or {})}
    pred_paths: Dict[int, str] = {}
    rng = np.random.RandomState(seed)
    for e in sorted(scatter_data.keys()):
        pred = np.asarray(scatter_data[e].get(pred_key, np.array([])))
        target = np.asarray(scatter_data[e].get(target_key, np.array([])))
        game_names = np.asarray(scatter_data[e].get("game_names", np.array([])), dtype=object)

        n_pts = len(pred)
        if n_pts == 0:
            continue

        if n_pts > max_points:
            sel = rng.choice(n_pts, size=max_points, replace=False)
            pred = pred[sel]
            target = target[sel]
            if len(game_names) > 0:
                game_names = game_names[sel]

        fig, ax = plt.subplots()
        fig.set_size_inches(2.9, 2.5)
        if len(game_names) == len(pred) and len(set(game_names)) > 1:
            for gi, gname in enumerate(sorted(set(game_names))):
                gmask = game_names == gname
                if not gmask.any():
                    continue
                ax.scatter(
                    target[gmask], pred[gmask],
                    s=6, alpha=0.45, edgecolors="none",
                    color=_get_game_color(gname, color_map, gi),
                    label=gname, rasterized=True,
                )
            handles, labels = ax.get_legend_handles_labels()
            if labels:
                by_label = dict(zip(labels, handles))
                ax.legend(
                    by_label.values(),
                    by_label.keys(),
                    fontsize=6,
                    loc="upper right",
                    framealpha=0.8,
                )
        else:
            ax.scatter(target, pred, s=6, alpha=0.45, edgecolors="none", color="#2166ac")

        lo = float(min(target.min(), pred.min()))
        hi = float(max(target.max(), pred.max()))
        ax.plot([lo, hi], [lo, hi], linestyle="--", color="#888", linewidth=1)
        r, slope, intercept = _compute_scatter_trendline(target, pred)
        if np.isfinite(r):
            ax.plot(
                [lo, hi],
                [slope * lo + intercept, slope * hi + intercept],
                linestyle="-",
                color="#1b9e77",
                linewidth=1,
            )

        name = _REWARD_ENUM_NAMES.get(int(e), f"enum_{e}")
        mae = float(np.mean(np.abs(pred - target)))
        r_txt = f"{r:.4f}" if np.isfinite(r) else "nan"
        ax.set_title(f"{name}\\nMAE={mae:.4f} | r={r_txt}", fontsize=8)
        ax.set_xlabel("target", fontsize=7)
        ax.set_ylabel("pred", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.grid(alpha=0.25)

        fig.tight_layout()
        path = os.path.join(out_dir, f"train_regression_scatter_{space}_enum_{int(e)}.png")
        fig.savefig(path, dpi=220)
        plt.close(fig)
        pred_paths[int(e)] = path

    if pred_paths:
        logger.info("Regression scatter plots (per enum, %s space) saved: %d", space, len(pred_paths))
    return pred_paths
