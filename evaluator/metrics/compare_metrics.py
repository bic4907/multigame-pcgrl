"""
evaluator/metrics/compare_metrics.py
=====================================
Metric comparison script: TPKL / SSIM / LPIPS / ShannonEntropy.
CLIPScore is excluded.

Usage (from project root):
    python -m evaluator.metrics.compare_metrics
    python -m evaluator.metrics.compare_metrics --n 50 --out outputs/metric_compare
    python -m evaluator.metrics.compare_metrics --no-lpips   # skip LPIPS (faster)

Outputs:
    <out>/score_distributions.png  -- same/diff score distributions (KDE)
    <out>/boxplot_scatter.png      -- box plot + scatter (jitter)
    <out>/roc_curves.png           -- ROC curves (all metrics overlaid)
    <out>/delta_auc_bar.png        -- Delta and AUC bar comparison
    <out>/summary.txt              -- text summary table
"""
from __future__ import annotations

import argparse
import itertools
import logging
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# Set non-interactive backend before importing pyplot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("compare_metrics")


# ─────────────────────────────────────────────────────────────────────────────
# Data loading + LevelBundle construction
# ─────────────────────────────────────────────────────────────────────────────

def load_bundles(n_per_game: int = 30, seed: int = 42) -> list:
    """
    MultiGameDataset -> list of LevelBundle.

    Parameters
    ----------
    n_per_game : max samples per game
    seed       : random seed

    Returns
    -------
    list[LevelBundle]
    """
    from dataset.multigame import MultiGameDataset
    from dataset.multigame.tile_utils import render_unified_rgb
    from evaluator.metrics import LevelBundle

    ds = MultiGameDataset(use_tile_mapping=True)
    logger.info("Dataset: %s", ds)

    rng = random.Random(seed)
    game_list = ["dungeon", "sokoban", "doom", "pokemon", "zelda"]

    bundles: list = []
    for game in game_list:
        game_samples = ds.by_game(game)
        if not game_samples:
            logger.warning("No samples found for game: %s", game)
            continue
        chosen = rng.sample(game_samples, min(n_per_game, len(game_samples)))
        for s in chosen:
            img = render_unified_rgb(s.array, tile_size=16)
            bundle = LevelBundle(
                array=s.array,
                image=img,
                text=s.instruction or "",
                game=s.game,
                meta=dict(s.meta),
            )
            bundles.append(bundle)
        logger.info("  %s: loaded %d samples", game, len(chosen))

    logger.info("Total bundles: %d", len(bundles))
    return bundles


def build_pairs(
    bundles: list,
    max_pairs_per_group: int = 400,
    seed: int = 42,
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    """
    same_pairs : pairs within the same (game, reward_enum) group
    diff_pairs : pairs across different groups

    Returns
    -------
    same_pairs, diff_pairs
    """
    rng = random.Random(seed)

    # -- group by (game, reward_enum) -----------------------------------------
    groups: Dict[Tuple[str, int], List[int]] = {}
    for i, b in enumerate(bundles):
        re_key = int(b.meta.get("reward_enum", 0))
        key = (b.game, re_key)
        groups.setdefault(key, []).append(i)

    # -- same pairs -----------------------------------------------------------
    same_pairs: List[Tuple[int, int]] = []
    for idxs in groups.values():
        pairs = list(itertools.combinations(idxs, 2))
        if len(pairs) > max_pairs_per_group:
            pairs = rng.sample(pairs, max_pairs_per_group)
        same_pairs.extend(pairs)

    # -- diff pairs -----------------------------------------------------------
    diff_pairs: List[Tuple[int, int]] = []
    group_keys = list(groups.keys())
    for ki in range(len(group_keys)):
        for kj in range(ki + 1, len(group_keys)):
            cross = list(itertools.product(groups[group_keys[ki]], groups[group_keys[kj]]))
            if len(cross) > max_pairs_per_group:
                cross = rng.sample(cross, max_pairs_per_group)
            diff_pairs.extend(cross)

    logger.info("same_pairs=%d  diff_pairs=%d", len(same_pairs), len(diff_pairs))
    return same_pairs, diff_pairs


# ─────────────────────────────────────────────────────────────────────────────
# Plotting helpers
# ─────────────────────────────────────────────────────────────────────────────

_SAME_COLOR = "#4C72B0"   # blue
_DIFF_COLOR = "#DD8452"   # orange


def _safe_kde(ax, data: np.ndarray, x_range: np.ndarray, color: str, label: str) -> None:
    """KDE curve; falls back to histogram if data is too small or has zero variance."""
    from scipy.stats import gaussian_kde  # type: ignore[import]

    try:
        if len(data) < 4 or np.std(data) < 1e-9:
            raise ValueError("zero variance")
        kde = gaussian_kde(data, bw_method="scott")
        ax.plot(x_range, kde(x_range), color=color, lw=2, label=label)
        ax.fill_between(x_range, kde(x_range), alpha=0.25, color=color)
    except Exception:
        ax.hist(data, bins=15, density=True, alpha=0.45, color=color, label=label, edgecolor="none")


def plot_score_distributions(results: list, out_dir: Path) -> None:
    """Per-metric same/diff score distribution -- KDE overlay plot."""
    out_dir.mkdir(parents=True, exist_ok=True)
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), constrained_layout=True)
    if n == 1:
        axes = [axes]

    for ax, res in zip(axes, results):
        s = np.array(res.same_scores, dtype=float)
        d = np.array(res.diff_scores, dtype=float)
        all_scores = np.concatenate([s, d])
        x_min, x_max = all_scores.min() - 0.05, all_scores.max() + 0.05
        x_range = np.linspace(x_min, x_max, 300)

        _safe_kde(ax, s, x_range, _SAME_COLOR, f"Same (mean={s.mean():.3f})")
        _safe_kde(ax, d, x_range, _DIFF_COLOR, f"Diff (mean={d.mean():.3f})")

        tag = "[OK]" if res.is_supported else "[NG]"
        ax.set_title(f"{tag} {res.name}\ndelta={res.delta:+.4f}  AUC={res.auc:.4f}", fontsize=11)
        ax.set_xlabel("Similarity Score")
        ax.set_ylabel("Density")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    fig.suptitle("Score Distributions by Metric (KDE)", fontsize=14, fontweight="bold")
    out_path = out_dir / "score_distributions.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", out_path)


def plot_boxplot_scatter(results: list, out_dir: Path) -> None:
    """Per-metric box plot + scatter (jitter)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 5), constrained_layout=True)
    if n == 1:
        axes = [axes]

    rng_np = np.random.default_rng(0)

    for ax, res in zip(axes, results):
        s = np.array(res.same_scores, dtype=float)
        d = np.array(res.diff_scores, dtype=float)

        bp = ax.boxplot(
            [s, d],
            positions=[0, 1],
            widths=0.45,
            patch_artist=True,
            medianprops=dict(color="black", linewidth=2),
            flierprops=dict(marker="o", markersize=2, alpha=0.4),
        )
        bp["boxes"][0].set(facecolor=_SAME_COLOR + "60")
        bp["boxes"][1].set(facecolor=_DIFF_COLOR + "60")

        jitter_w = 0.08
        max_pts = 200
        s_sub = s[rng_np.choice(len(s), min(max_pts, len(s)), replace=False)] if len(s) > 0 else s
        d_sub = d[rng_np.choice(len(d), min(max_pts, len(d)), replace=False)] if len(d) > 0 else d

        ax.scatter(
            rng_np.uniform(-jitter_w, jitter_w, len(s_sub)),
            s_sub, alpha=0.35, s=8, color=_SAME_COLOR, zorder=3,
        )
        ax.scatter(
            1 + rng_np.uniform(-jitter_w, jitter_w, len(d_sub)),
            d_sub, alpha=0.35, s=8, color=_DIFF_COLOR, zorder=3,
        )

        tag = "[OK]" if res.is_supported else "[NG]"
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Same\n(same group)", "Diff\n(diff group)"])
        ax.set_title(f"{tag} {res.name}\ndelta={res.delta:+.4f}  AUC={res.auc:.4f}", fontsize=11)
        ax.set_ylabel("Similarity Score")
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Score Distributions by Metric (Box + Scatter)", fontsize=14, fontweight="bold")
    out_path = out_dir / "boxplot_scatter.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", out_path)


def plot_roc_curves(results: list, out_dir: Path) -> None:
    """All metrics' ROC curves overlaid on a single axes."""
    from evaluator.metrics import roc_curve_points

    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 6))

    colors = plt.cm.tab10.colors  # type: ignore[attr-defined]
    for res, color in zip(results, colors):
        fprs, tprs = roc_curve_points(res.same_scores, res.diff_scores)
        label = f"{res.name}  AUC={res.auc:.4f}"
        ax.plot(fprs, tprs, lw=2, color=color, label=label)
        ax.fill_between(fprs, tprs, alpha=0.05, color=color)

    ax.plot([0, 1], [0, 1], "k--", lw=1.2, label="Random (AUC=0.5)")
    ax.set_xlabel("False Positive Rate (FPR)", fontsize=12)
    ax.set_ylabel("True Positive Rate (TPR)", fontsize=12)
    ax.set_title("ROC Curves -- Metric Comparison", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    ax.xaxis.set_minor_locator(mticker.MultipleLocator(0.1))
    ax.yaxis.set_minor_locator(mticker.MultipleLocator(0.1))

    out_path = out_dir / "roc_curves.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", out_path)


def plot_delta_auc_bar(results: list, out_dir: Path) -> None:
    """Side-by-side bar chart for Delta (same_mean - diff_mean) and AUC."""
    out_dir.mkdir(parents=True, exist_ok=True)
    names  = [r.name for r in results]
    deltas = [r.delta for r in results]
    aucs   = [r.auc   for r in results]
    x = np.arange(len(names))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)

    # -- Delta bar ------------------------------------------------------------
    bar_colors_d = ["#2ecc71" if v > 0 else "#e74c3c" for v in deltas]
    bars1 = ax1.bar(x, deltas, color=bar_colors_d, edgecolor="black", linewidth=0.8, zorder=3)
    ax1.axhline(0, color="black", linewidth=1.0, linestyle="--")
    ax1.bar_label(bars1, fmt="%.4f", padding=2, fontsize=9)
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=20, ha="right", fontsize=10)
    ax1.set_ylabel("Delta  (same_mean - diff_mean)", fontsize=11)
    ax1.set_title("Delta Comparison\n(positive = hypothesis supported)", fontweight="bold")
    ax1.grid(axis="y", alpha=0.3, zorder=0)

    # -- AUC bar --------------------------------------------------------------
    bar_colors_a = ["#3498db" if v >= 0.5 else "#e67e22" for v in aucs]
    bars2 = ax2.bar(x, aucs, color=bar_colors_a, edgecolor="black", linewidth=0.8, zorder=3)
    ax2.axhline(0.5, color="red", linewidth=1.4, linestyle="--", label="Random baseline (0.5)")
    ax2.bar_label(bars2, fmt="%.4f", padding=2, fontsize=9)
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=20, ha="right", fontsize=10)
    ax2.set_ylim(0, 1.08)
    ax2.set_ylabel("AUC-ROC", fontsize=11)
    ax2.set_title("AUC-ROC Comparison\n(0.5 = random,  1.0 = perfect)", fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.grid(axis="y", alpha=0.3, zorder=0)

    fig.suptitle("Delta / AUC Comparison by Metric", fontsize=14, fontweight="bold")
    out_path = out_dir / "delta_auc_bar.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", out_path)


def plot_pairwise_heatmap(results: list, out_dir: Path, bundles: list) -> None:
    """
    (N, N) similarity matrix heatmap for each metric.
    Only rendered when keep_matrix=True was passed to evaluate().
    """
    has_matrix = [r for r in results if r.matrix is not None]
    if not has_matrix:
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    n_plots = len(has_matrix)
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5.5), constrained_layout=True)
    if n_plots == 1:
        axes = [axes]

    game_labels = [b.game for b in bundles]

    for ax, res in zip(axes, has_matrix):
        im = ax.imshow(
            res.matrix,
            vmin=None,
            aspect="auto",
            cmap="viridis",
            interpolation="none",
        )
        plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)

        N = len(bundles)
        tick_step = max(1, N // 20)
        ticks = list(range(0, N, tick_step))
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels([game_labels[t][:3] for t in ticks], rotation=90, fontsize=7)
        ax.set_yticklabels([game_labels[t][:3] for t in ticks], fontsize=7)

        tag = "[OK]" if res.is_supported else "[NG]"
        ax.set_title(f"{tag} {res.name}\ndelta={res.delta:+.4f}  AUC={res.auc:.4f}", fontsize=11)

    fig.suptitle("Similarity Matrix Heatmaps by Metric", fontsize=13, fontweight="bold")
    out_path = out_dir / "similarity_heatmaps.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", out_path)


# ─────────────────────────────────────────────────────────────────────────────
# Text summary
# ─────────────────────────────────────────────────────────────────────────────

def build_summary_text(results: list, bundles: list, args) -> str:
    lines = [
        "=" * 70,
        "Metric Comparison Summary",
        "=" * 70,
        f"  Total bundles : {len(bundles)}  (max {args.n} per game)",
        f"  Random seed   : {args.seed}",
        "",
        f"{'Metric':<22} {'same_mean':>10} {'diff_mean':>10} {'delta':>10} {'AUC':>8}  verdict",
        "-" * 70,
    ]
    for r in results:
        tag = "[OK] SUPPORTED" if r.is_supported else "[NG] not supported"
        lines.append(
            f"{r.name:<22} {r.same_mean:>10.4f} {r.diff_mean:>10.4f} "
            f"{r.delta:>+10.4f} {r.auc:>8.4f}  {tag}"
        )
    lines += ["=" * 70, ""]

    # Additional stats: std dev
    lines.append(f"{'Metric':<22} {'same_std':>10} {'diff_std':>10} {'same_n':>8} {'diff_n':>8}")
    lines.append("-" * 60)
    for r in results:
        s = np.array(r.same_scores)
        d = np.array(r.diff_scores)
        lines.append(
            f"{r.name:<22} {float(s.std()):>10.4f} {float(d.std()):>10.4f} "
            f"{len(s):>8} {len(d):>8}"
        )
    lines.append("=" * 70)
    return "\n".join(lines)


def print_summary_table(results: list, bundles: list, args) -> None:
    print(build_summary_text(results, bundles, args))


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare TPKL / SSIM / LPIPS / ShannonEntropy metrics (CLIPScore excluded)"
    )
    parser.add_argument("--n",         type=int,  default=30,
                        help="samples per game (default: 30)")
    parser.add_argument("--seed",      type=int,  default=42,
                        help="random seed (default: 42)")
    parser.add_argument("--out",       type=str,  default="outputs/metric_compare",
                        help="output directory (default: outputs/metric_compare)")
    parser.add_argument("--max-pairs", type=int,  default=400,
                        help="max pairs per group (default: 400)")
    parser.add_argument("--no-lpips",  action="store_true",
                        help="skip LPIPS metric (faster runs)")
    parser.add_argument("--heatmap",   action="store_true",
                        help="also save similarity matrix heatmaps (slow for large N)")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # -- 1. Load data ---------------------------------------------------------
    bundles = load_bundles(n_per_game=args.n, seed=args.seed)
    if len(bundles) < 4:
        logger.error("Too few bundles (%d). Check your dataset.", len(bundles))
        sys.exit(1)

    same_pairs, diff_pairs = build_pairs(
        bundles, max_pairs_per_group=args.max_pairs, seed=args.seed
    )
    if not same_pairs or not diff_pairs:
        logger.error("Failed to build pairs. Try increasing --n.")
        sys.exit(1)

    # -- 2. Instantiate metrics -----------------------------------------------
    from evaluator.metrics import TPKLMetric, SSIMMetric, LPIPSMetric, ShannonEntropyMetric, BaseMetricEvaluator

    metrics: List[BaseMetricEvaluator] = [
        TPKLMetric(),
        ShannonEntropyMetric(),
        SSIMMetric(),
    ]
    if not args.no_lpips:
        try:
            metrics.append(LPIPSMetric())
            logger.info("LPIPS loaded successfully")
        except Exception as e:
            logger.warning("LPIPS load failed (%s) -- skipping.", e)

    # -- 3. Run evaluation ----------------------------------------------------
    from evaluator.metrics import MetricResult

    results: List[MetricResult] = []
    keep_matrix = args.heatmap

    for metric in metrics:
        logger.info(">> Running %s ...", metric.name)
        try:
            result = metric.evaluate(
                bundles, same_pairs, diff_pairs, keep_matrix=keep_matrix
            )
            results.append(result)
            logger.info("   %s", result.summary_line())
        except Exception as exc:
            logger.error("   %s failed: %s", metric.name, exc, exc_info=True)

    if not results:
        logger.error("All metrics failed.")
        sys.exit(1)

    # -- 4. Print + save summary ----------------------------------------------
    summary = build_summary_text(results, bundles, args)
    print("\n" + summary)
    (out_dir / "summary.txt").write_text(summary, encoding="utf-8")
    logger.info("Summary saved: %s", out_dir / "summary.txt")

    # -- 5. Plots -------------------------------------------------------------
    logger.info("Generating plots ...")
    plot_score_distributions(results, out_dir)
    plot_boxplot_scatter(results, out_dir)
    plot_roc_curves(results, out_dir)
    plot_delta_auc_bar(results, out_dir)
    if args.heatmap:
        plot_pairwise_heatmap(results, out_dir, bundles)

    print(f"\nAll results saved to: {out_dir.resolve()}")
    print("  score_distributions.png -- KDE score distributions")
    print("  boxplot_scatter.png     -- box plot + scatter")
    print("  roc_curves.png          -- ROC curve comparison")
    print("  delta_auc_bar.png       -- Delta / AUC bar chart")
    if args.heatmap:
        print("  similarity_heatmaps.png -- similarity matrix heatmaps")
    print("  summary.txt             -- text summary")


if __name__ == "__main__":
    main()

