from __future__ import annotations

import argparse
import itertools
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.distance import jensenshannon
from scipy.stats import ks_2samp, wasserstein_distance


_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[1]
_DEFAULT_CACHE_DIR = _REPO_ROOT / "dataset" / "multigame" / "cache" / "artifacts"
_DEFAULT_OUT_DIR = _HERE / "outputs"


@dataclass(frozen=True)
class AnnFile:
    game: str
    path: Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze per-game reward_enum condition distributions from "
            "dataset/multigame cache annotation files."
        )
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=_DEFAULT_CACHE_DIR,
        help="Directory containing per-game *.ann.json files.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=_DEFAULT_OUT_DIR,
        help="Directory where plots and tables are written.",
    )
    parser.add_argument(
        "--games",
        nargs="*",
        default=None,
        help="Optional game filter. Example: --games dungeon zelda pokemon",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=40,
        help="Number of bins for continuous histogram comparisons.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=30,
        help="Number of lowest-overlap pairs to include in report.md.",
    )
    parser.add_argument(
        "--plot-sample-per-game",
        type=int,
        default=1000,
        help=(
            "Maximum rows per (game, reward_enum) group used only for plots. "
            "Set 0 to plot all rows."
        ),
    )
    parser.add_argument(
        "--plot-sample-seed",
        type=int,
        default=42,
        help="Random seed for plot-only uniform sampling.",
    )
    parser.add_argument(
        "--per-reward-top-k",
        type=int,
        default=0,
        help="Rows per reward_enum ranking in report.md. 0 means all pairs.",
    )
    return parser.parse_args()


def _latest_ann_files(cache_dir: Path, games: Iterable[str] | None) -> list[AnnFile]:
    allowed = {g.lower() for g in games} if games else None
    ann_files: list[AnnFile] = []

    for game_dir in sorted(p for p in cache_dir.iterdir() if p.is_dir()):
        game = game_dir.name.lower()
        if allowed is not None and game not in allowed:
            continue
        candidates = sorted(game_dir.glob("*.ann.json"), key=lambda p: p.stat().st_mtime)
        if not candidates:
            continue
        ann_files.append(AnnFile(game=game, path=candidates[-1]))

    return ann_files


def _condition_value(row: dict) -> float | None:
    reward_enum = row.get("reward_enum")
    if reward_enum is None:
        return None

    try:
        reward_enum_i = int(reward_enum)
    except (TypeError, ValueError):
        return None

    direct_key = f"condition_{reward_enum_i}"
    value = row.get(direct_key)

    # Backward compatibility if annotations use 1-based reward_enum but 0-based condition columns.
    if value is None and 1 <= reward_enum_i <= 5:
        value = row.get(f"condition_{reward_enum_i - 1}")

    if value is None or value == "":
        return None
    try:
        value_f = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value_f):
        return None
    return value_f


def load_annotations(cache_dir: Path, games: Iterable[str] | None = None) -> pd.DataFrame:
    ann_files = _latest_ann_files(cache_dir, games)
    if not ann_files:
        game_msg = "all games" if games is None else ", ".join(games)
        raise FileNotFoundError(f"No *.ann.json files found for {game_msg} in {cache_dir}")

    rows: list[dict] = []
    for ann_file in ann_files:
        payload = json.loads(ann_file.path.read_text(encoding="utf-8"))
        game = str(payload.get("game") or ann_file.game)
        for row in payload.get("annotations", []):
            condition_value = _condition_value(row)
            if condition_value is None:
                continue
            reward_enum = int(row["reward_enum"])
            rows.append(
                {
                    "game": game,
                    "reward_enum": reward_enum,
                    "condition": condition_value,
                    "feature_name": row.get("feature_name", ""),
                    "sub_condition": row.get("sub_condition", ""),
                    "source_id": row.get("source_id", ""),
                    "key": row.get("key", ""),
                    "ann_path": str(ann_file.path),
                }
            )

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("Annotation files were found, but no condition values could be extracted.")
    return df.sort_values(["reward_enum", "game", "source_id", "key"]).reset_index(drop=True)


def _hist_edges(values: np.ndarray, bins: int) -> np.ndarray:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.array([0.0, 1.0])

    unique = np.unique(values)
    if unique.size == 1:
        v = float(unique[0])
        return np.array([v - 0.5, v + 0.5])

    is_integer_like = np.all(np.isclose(unique, np.round(unique)))
    # Most annotation measures are integer counts. Use exact unit-width bins so
    # non-overlapping integer ranges do not get merged into the same wide bin.
    if is_integer_like and (unique.max() - unique.min()) <= 1000:
        low = math.floor(float(unique.min())) - 0.5
        high = math.ceil(float(unique.max())) + 0.5
        return np.arange(low, high + 1.0, 1.0)

    return np.linspace(float(values.min()), float(values.max()), bins + 1)


def _prob_hist(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    counts, _ = np.histogram(values, bins=edges)
    total = counts.sum()
    if total == 0:
        return np.zeros_like(counts, dtype=float)
    return counts.astype(float) / float(total)


def _overlap_coefficient(a: np.ndarray, b: np.ndarray, edges: np.ndarray) -> float:
    pa = _prob_hist(a, edges)
    pb = _prob_hist(b, edges)
    return float(np.minimum(pa, pb).sum())


def _js_distance(a: np.ndarray, b: np.ndarray, edges: np.ndarray) -> float:
    pa = _prob_hist(a, edges)
    pb = _prob_hist(b, edges)
    if pa.sum() == 0 or pb.sum() == 0:
        return float("nan")
    return float(jensenshannon(pa, pb, base=2.0))


def _safe_filename(text: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", text.strip())
    safe = safe.strip("._")
    return safe or "condition"


def _smooth_density(values: np.ndarray, edges: np.ndarray, sigma: float = 1.1) -> tuple[np.ndarray, np.ndarray]:
    density, _ = np.histogram(values, bins=edges, density=True)
    centers = (edges[:-1] + edges[1:]) / 2.0
    if density.size <= 2:
        return centers, density
    return centers, gaussian_filter1d(density.astype(float), sigma=sigma, mode="nearest")


def summarize_distributions(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby(["game", "reward_enum", "feature_name", "sub_condition"], dropna=False)["condition"]
        .agg(["count", "mean", "std", "min", "median", "max"])
        .reset_index()
    )
    for q in [0.05, 0.25, 0.75, 0.95]:
        qdf = (
            df.groupby(["game", "reward_enum"], dropna=False)["condition"]
            .quantile(q)
            .rename(f"q{int(q * 100):02d}")
            .reset_index()
        )
        summary = summary.merge(qdf, on=["game", "reward_enum"], how="left")
    return summary.sort_values(["reward_enum", "game"]).reset_index(drop=True)


def compare_game_pairs(df: pd.DataFrame, bins: int) -> pd.DataFrame:
    rows: list[dict] = []
    for reward_enum, re_df in df.groupby("reward_enum"):
        edges = _hist_edges(re_df["condition"].to_numpy(float), bins=bins)
        game_values = {
            game: gdf["condition"].to_numpy(float)
            for game, gdf in re_df.groupby("game")
        }
        for game_a, game_b in itertools.combinations(sorted(game_values), 2):
            a = game_values[game_a]
            b = game_values[game_b]
            range_overlap = max(float(a.min()), float(b.min())) <= min(float(a.max()), float(b.max()))
            ks = ks_2samp(a, b)
            rows.append(
                {
                    "reward_enum": int(reward_enum),
                    "game_a": game_a,
                    "game_b": game_b,
                    "n_a": int(a.size),
                    "n_b": int(b.size),
                    "min_a": float(a.min()),
                    "max_a": float(a.max()),
                    "min_b": float(b.min()),
                    "max_b": float(b.max()),
                    "range_overlap": bool(range_overlap),
                    "overlap_coef": _overlap_coefficient(a, b, edges),
                    "ks_stat": float(ks.statistic),
                    "ks_pvalue": float(ks.pvalue),
                    "wasserstein": float(wasserstein_distance(a, b)),
                    "js_distance": _js_distance(a, b, edges),
                    "mean_diff_abs": float(abs(a.mean() - b.mean())),
                    "median_diff_abs": float(abs(np.median(a) - np.median(b))),
                }
            )

    return pd.DataFrame(rows).sort_values(
        ["range_overlap", "overlap_coef", "js_distance", "ks_stat"],
        ascending=[True, True, False, False],
    ).reset_index(drop=True)


def _sample_for_plots(df: pd.DataFrame, max_per_game: int, seed: int) -> pd.DataFrame:
    if max_per_game <= 0:
        return df

    parts: list[pd.DataFrame] = []
    for (_, _), group in df.groupby(["game", "reward_enum"], sort=False):
        if len(group) > max_per_game:
            parts.append(group.sample(n=max_per_game, random_state=seed))
        else:
            parts.append(group)
    return pd.concat(parts, ignore_index=True).sort_values(["reward_enum", "game", "source_id", "key"])


def plot_reward_enum_distributions(df: pd.DataFrame, out_dir: Path, bins: int) -> None:
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    games = sorted(df["game"].unique())
    cmap = plt.get_cmap("tab10")
    colors = {game: cmap(i % 10) for i, game in enumerate(games)}

    for reward_enum, re_df in df.groupby("reward_enum"):
        edges = _hist_edges(re_df["condition"].to_numpy(float), bins=bins)
        feature_names = sorted(str(v) for v in re_df["feature_name"].dropna().unique())
        title_feature = ", ".join(feature_names) if feature_names else "condition"

        fig, ax = plt.subplots(figsize=(5, 3))
        for game in games:
            values = re_df.loc[re_df["game"] == game, "condition"].to_numpy(float)
            if values.size == 0:
                continue
            centers, density = _smooth_density(values, edges)
            ax.fill_between(centers, density, alpha=0.10, color=colors[game])
            ax.plot(centers, density, linewidth=2.0, color=colors[game], label=game)

        ax.set_title(title_feature)
        ax.set_xlabel("condition value")
        ax.set_ylabel("density")
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(plot_dir / f"{int(reward_enum)}_{_safe_filename(title_feature)}.png", dpi=180, bbox_inches="tight")
        plt.close(fig)

    reward_enums = sorted(df["reward_enum"].unique())
    fig, axes = plt.subplots(
        len(reward_enums),
        1,
        figsize=(5, max(3, 1.4 * len(reward_enums))),
        sharex=False,
    )
    if len(reward_enums) == 1:
        axes = [axes]
    for ax, reward_enum in zip(axes, reward_enums):
        re_df = df[df["reward_enum"] == reward_enum]
        edges = _hist_edges(re_df["condition"].to_numpy(float), bins=bins)
        for game in games:
            values = re_df.loc[re_df["game"] == game, "condition"].to_numpy(float)
            if values.size == 0:
                continue
            centers, density = _smooth_density(values, edges)
            ax.plot(centers, density, linewidth=1.8, color=colors[game], label=game)
        ax.set_title(str(reward_enum))
        ax.set_ylabel("density")
        ax.grid(True, alpha=0.2)
    axes[-1].set_xlabel("condition value")
    axes[0].legend(frameon=False, ncol=min(len(games), 5))
    fig.tight_layout()
    fig.savefig(plot_dir / "reward_enum_all_distributions.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_report(
    df: pd.DataFrame,
    summary: pd.DataFrame,
    pairwise: pd.DataFrame,
    out_dir: Path,
    top_k: int,
    per_reward_top_k: int | None = None,
) -> None:
    lines: list[str] = []
    lines.append("# Dataset condition distribution report")
    lines.append("")
    lines.append(f"Total rows: {len(df):,}")
    lines.append(f"Games: {', '.join(sorted(df['game'].unique()))}")
    lines.append(f"Reward enums: {', '.join(map(str, sorted(df['reward_enum'].unique())))}")
    lines.append("")
    lines.append("## Outputs")
    lines.append("")
    lines.append("- `condition_values.csv`: extracted per-sample condition values")
    lines.append("- `distribution_summary.csv`: per-game/reward summary statistics")
    lines.append("- `game_pair_distribution_overlap.csv`: pairwise game overlap metrics")
    lines.append("- `plots/reward_enum_<N>_distribution.png`: per reward_enum overlay plot")
    lines.append("- `plots/reward_enum_all_distributions.png`: compact all reward_enum plot")
    lines.append("")
    lines.append("## Lowest-overlap game pairs")
    lines.append("")

    cols = [
        "reward_enum",
        "game_a",
        "game_b",
        "range_overlap",
        "overlap_coef",
        "ks_stat",
        "wasserstein",
        "js_distance",
        "min_a",
        "max_a",
        "min_b",
        "max_b",
    ]
    top = pairwise[cols].head(top_k).copy()
    if top.empty:
        lines.append("No pairwise comparisons were possible.")
    else:
        lines.append(top.to_markdown(index=False, floatfmt=".4g"))

    lines.append("")
    lines.append("## Reward Enum Pair Rankings")
    lines.append("")

    rank_cols = [
        "game_a",
        "game_b",
        "range_overlap",
        "overlap_coef",
        "ks_stat",
        "wasserstein",
        "js_distance",
        "min_a",
        "max_a",
        "min_b",
        "max_b",
    ]
    for reward_enum in sorted(pairwise["reward_enum"].unique()):
        re_pairs = pairwise[pairwise["reward_enum"] == reward_enum][rank_cols].copy()
        if per_reward_top_k is not None and per_reward_top_k > 0:
            re_pairs = re_pairs.head(per_reward_top_k)
        feature_names = sorted(
            str(v)
            for v in summary.loc[summary["reward_enum"] == reward_enum, "feature_name"].dropna().unique()
        )
        feature_suffix = f" ({', '.join(feature_names)})" if feature_names else ""
        lines.append(f"### reward_enum={reward_enum}{feature_suffix}")
        lines.append("")
        lines.append(re_pairs.to_markdown(index=False, floatfmt=".4g"))
        lines.append("")

    lines.append("## Interpretation")
    lines.append("")
    lines.append("- `range_overlap=False` means the two games' min-max condition ranges do not overlap for that reward_enum.")
    lines.append("- `overlap_coef` is histogram common mass. Lower means less distribution overlap; 0 means no shared bins.")
    lines.append("- `ks_stat`, `wasserstein`, and `js_distance` increase as distributions differ more.")

    out_dir.joinpath("report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = _parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = load_annotations(args.cache_dir, args.games)
    summary = summarize_distributions(df)
    pairwise = compare_game_pairs(df, bins=args.bins)

    df.to_csv(args.out_dir / "condition_values.csv", index=False)
    summary.to_csv(args.out_dir / "distribution_summary.csv", index=False)
    pairwise.to_csv(args.out_dir / "game_pair_distribution_overlap.csv", index=False)

    plot_df = _sample_for_plots(df, args.plot_sample_per_game, args.plot_sample_seed)
    plot_reward_enum_distributions(plot_df, args.out_dir, bins=args.bins)
    write_report(
        df,
        summary,
        pairwise,
        args.out_dir,
        top_k=args.top_k,
        per_reward_top_k=args.per_reward_top_k or None,
    )

    print(f"Loaded {len(df):,} condition rows from {args.cache_dir}")
    if len(plot_df) != len(df):
        print(
            f"Using {len(plot_df):,} rows for plots "
            f"(max {args.plot_sample_per_game:,} per game/reward_enum, seed={args.plot_sample_seed})"
        )
    print(f"Wrote outputs to {args.out_dir}")
    print("Lowest-overlap pairs:")
    print(pairwise.head(min(args.top_k, 10)).to_string(index=False))


if __name__ == "__main__":
    main()
