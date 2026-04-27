"""
Create condition(x) vs metric(y) plots grouped by game and reward_enum,
then export a Markdown report.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages


PROJECT_ROOT = Path(__file__).resolve().parent.parent

GAME_COLORS = {
    "doom": "#1f77b4",
    "dungeon": "#d62728",
    "pokemon": "#2ca02c",
    "sokoban": "#ff7f0e",
    "zelda": "#9467bd",
}

REWARD_ENUM_LABELS = {
    0: "Region",
    1: "Path Length",
    2: "Interactable",
    3: "Hazard",
    4: "Collectable",
}


def reward_enum_label(reward_enum: int) -> str:
    return REWARD_ENUM_LABELS.get(reward_enum, f"RE{reward_enum}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate condition-progress/feat plots by game and reward_enum."
    )
    parser.add_argument(
        "--input-root",
        default="results/wandb_download",
        help="Root directory containing downloaded ctrl_sim.csv files.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/wandb_download/condition_progress_plots",
        help="Directory to save plot images.",
    )
    parser.add_argument(
        "--output-md",
        default="results/wandb_download/condition_progress_report.md",
        help="Markdown report path.",
    )
    parser.add_argument(
        "--output-pdf",
        default=None,
        help="Optional PDF report path. If omitted, PDF is not generated.",
    )
    parser.add_argument(
        "--max-scatter-points",
        type=int,
        default=6000,
        help="Max sampled points per subplot for scatter overlay.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for scatter sampling.",
    )
    return parser.parse_args()


def resolve_path(path_arg: str, prefer_existing: bool = False) -> Path:
    raw = Path(path_arg)
    if raw.is_absolute():
        return raw.resolve()
    from_cwd = (Path.cwd() / raw).resolve()
    from_root = (PROJECT_ROOT / raw).resolve()
    if prefer_existing:
        if from_cwd.exists():
            return from_cwd
        if from_root.exists():
            return from_root
    return from_root


def resolve_input_root(path_arg: str) -> Path:
    raw = Path(path_arg)
    if raw.is_absolute():
        return raw.resolve()

    candidates = [
        (Path.cwd() / raw).resolve(),
        (PROJECT_ROOT / raw).resolve(),
        (Path(__file__).resolve().parent / raw).resolve(),
    ]
    uniq: list[Path] = []
    seen = set()
    for c in candidates:
        s = str(c)
        if s in seen:
            continue
        seen.add(s)
        uniq.append(c)

    best = (PROJECT_ROOT / raw).resolve()
    best_count = -1
    for c in uniq:
        count = sum(1 for _ in c.glob("**/ctrl_sim.csv")) if c.exists() else 0
        if count > best_count:
            best = c
            best_count = count
    return best


def collect_condition_metrics(input_root: Path) -> pd.DataFrame:
    csv_paths = sorted(input_root.glob("**/ctrl_sim.csv"))
    rows: list[pd.DataFrame] = []
    usecols = ["game", "reward_enum", "progress"]
    for i in range(5):
        usecols.append(f"condition_{i}")
        usecols.append(f"feat_{i}")
        usecols.append(f"feat_{i}_s0")

    for csv_path in csv_paths:
        try:
            df = pd.read_csv(csv_path, usecols=usecols)
        except Exception:
            continue

        if df.empty:
            continue

        reward_vals = (
            pd.to_numeric(df["reward_enum"], errors="coerce")
            .dropna()
            .astype(int)
            .unique()
            .tolist()
        )

        for reward_enum in reward_vals:
            cond_col = f"condition_{reward_enum}"
            if cond_col not in df.columns:
                continue

            feat_col = f"feat_{reward_enum}"
            feat_s0_col = f"feat_{reward_enum}_s0"
            if feat_col not in df.columns or feat_s0_col not in df.columns:
                continue

            mask = (
                (pd.to_numeric(df["reward_enum"], errors="coerce") == reward_enum)
                & pd.to_numeric(df[cond_col], errors="coerce").notna()
            )
            if not mask.any():
                continue

            sub = df.loc[mask, ["game", "progress", cond_col, feat_col, feat_s0_col]].copy()
            sub = sub.rename(
                columns={
                    cond_col: "condition",
                    feat_col: "feat",
                    feat_s0_col: "feat_s0",
                }
            )
            sub["game"] = sub["game"].astype(str)
            sub["reward_enum"] = reward_enum
            sub["condition"] = pd.to_numeric(sub["condition"], errors="coerce")
            sub["progress"] = pd.to_numeric(sub["progress"], errors="coerce")
            sub["feat"] = pd.to_numeric(sub["feat"], errors="coerce")
            sub["feat_s0"] = pd.to_numeric(sub["feat_s0"], errors="coerce")
            sub = sub.dropna(subset=["condition"])
            if not sub.empty:
                rows.append(sub)

    if not rows:
        return pd.DataFrame(
            columns=["game", "reward_enum", "condition", "progress", "feat", "feat_s0"]
        )
    return pd.concat(rows, ignore_index=True)


def plot_by_reward(
    df: pd.DataFrame,
    output_dir: Path,
    max_scatter_points: int,
    seed: int,
    y_col: str,
    y_label: str,
    draw_feat_s0_mean_line: bool = False,
) -> list[tuple[int, str]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[tuple[int, str]] = []
    games_all = sorted(df["game"].unique().tolist())

    for reward_enum in sorted(df["reward_enum"].unique().tolist()):
        sub_reward = df[df["reward_enum"] == reward_enum]

        n_games = len(games_all)
        n_cols = 3
        n_rows = math.ceil(n_games / n_cols)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.35 * n_cols, 2.1 * n_rows))
        axes_list = axes.flatten() if hasattr(axes, "flatten") else [axes]

        for i, game in enumerate(games_all):
            ax = axes_list[i]
            d = sub_reward[sub_reward["game"] == game].copy()
            line_color = GAME_COLORS.get(game, "#1f77b4")
            d = d.dropna(subset=[y_col])
            if d.empty:
                ax.axis("off")
                continue

            scatter_data = d
            if len(d) > max_scatter_points:
                scatter_data = d.sample(n=max_scatter_points, random_state=seed)

            ax.scatter(
                scatter_data["condition"],
                scatter_data[y_col],
                s=5,
                alpha=0.06,
                color=line_color,
                edgecolors="none",
            )

            trend = (
                d.groupby("condition", as_index=False)[y_col]
                .mean()
                .sort_values("condition")
            )
            ax.plot(
                trend["condition"],
                trend[y_col],
                color=line_color,
                linewidth=1.8,
                marker="o",
                markersize=2.0,
            )
            if y_col == "feat":
                x_min = float(d["condition"].min())
                x_max = float(d["condition"].max())
                y_min = float(d["feat"].min())
                y_max = float(d["feat"].max())
                lo = min(x_min, y_min)
                hi = max(x_max, y_max)
                if np.isfinite(lo) and np.isfinite(hi) and lo < hi:
                    pad = (hi - lo) * 0.03
                    lo_p = lo - pad
                    hi_p = hi + pad
                    ax.set_xlim(lo_p, hi_p)
                    ax.set_ylim(lo_p, hi_p)
                    ax.set_aspect("equal", adjustable="box")
                    ax.plot(
                        [lo_p, hi_p],
                        [lo_p, hi_p],
                        linestyle=":",
                        color="#666666",
                        linewidth=1.1,
                        alpha=0.9,
                    )
            if draw_feat_s0_mean_line:
                s0 = pd.to_numeric(d["feat_s0"], errors="coerce").dropna()
                if not s0.empty:
                    s0_mean = float(s0.mean())
                    ax.axhline(
                        y=s0_mean,
                        color="#333333",
                        linestyle="--",
                        linewidth=1.3,
                        alpha=0.9,
                    )
                    ax.text(
                        0.98,
                        0.04,
                        f"mean feat_s0={s0_mean:.2f}",
                        transform=ax.transAxes,
                        ha="right",
                        va="bottom",
                        fontsize=7.5,
                        color="#333333",
                        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
                    )
            ax.text(
                0.02,
                0.95,
                game,
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=7.8,
                color="#222222",
                bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
            )
            ax.set_xlabel("condition")
            ax.set_ylabel(y_label)
            ax.grid(alpha=0.25)

        for j in range(n_games, len(axes_list)):
            axes_list[j].axis("off")

        fig.tight_layout(pad=0.15, w_pad=0.15, h_pad=0.15)
        filename = f"reward_enum_{reward_enum}_condition_{y_col}.png"
        fig.savefig(output_dir / filename, dpi=180, bbox_inches="tight")
        plt.close(fig)
        outputs.append((reward_enum, filename))

    return outputs


def write_report(
    output_md: Path,
    progress_image_entries: list[tuple[int, str]],
    feat_image_entries: list[tuple[int, str]],
    output_dir: Path,
) -> None:
    lines: list[str] = []
    lines.append("# Condition-Based Evaluation Report")
    lines.append("")
    for reward_enum, image_name in progress_image_entries:
        label = reward_enum_label(reward_enum)
        lines.append(f"## {label}")
        lines.append("")
        feat_entry = [x for x in feat_image_entries if x[0] == reward_enum]
        feat_name = feat_entry[0][1] if feat_entry else ""
        lines.append("| Condition vs Progress | Condition vs Feat |")
        lines.append("| --- | --- |")
        lines.append(
            f"| ![{label}_progress](./{output_dir.name}/{image_name}) "
            f"| ![{label}_feat](./{output_dir.name}/{feat_name}) |"
        )
        lines.append("")

    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text("\n".join(lines), encoding="utf-8")


def write_pdf_report(
    output_pdf: Path,
    progress_image_entries: list[tuple[int, str]],
    feat_image_entries: list[tuple[int, str]],
    output_dir: Path,
) -> None:
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    feat_map = {re: name for re, name in feat_image_entries}

    with PdfPages(output_pdf) as pdf:
        for reward_enum, prog_name in progress_image_entries:
            feat_name = feat_map.get(reward_enum)
            if not feat_name:
                continue
            prog_path = output_dir / prog_name
            feat_path = output_dir / feat_name
            if not prog_path.exists() or not feat_path.exists():
                continue

            fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.0))
            for ax in axes:
                ax.axis("off")
            axes[0].imshow(mpimg.imread(prog_path))
            axes[1].imshow(mpimg.imread(feat_path))
            fig.tight_layout(pad=0.2, w_pad=0.3)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)


def main() -> None:
    args = parse_args()
    input_root = resolve_input_root(args.input_root)
    output_dir = resolve_path(args.output_dir, prefer_existing=False)
    output_md = resolve_path(args.output_md, prefer_existing=False)
    output_pdf = resolve_path(args.output_pdf, prefer_existing=False) if args.output_pdf else None

    df = collect_condition_metrics(input_root)
    if df.empty:
        raise SystemExit(f"No usable rows found under: {input_root}")

    progress_image_entries = plot_by_reward(
        df=df,
        output_dir=output_dir,
        max_scatter_points=args.max_scatter_points,
        seed=args.seed,
        y_col="progress",
        y_label="progress",
        draw_feat_s0_mean_line=False,
    )
    feat_image_entries = plot_by_reward(
        df=df,
        output_dir=output_dir,
        max_scatter_points=args.max_scatter_points,
        seed=args.seed,
        y_col="feat",
        y_label="feat",
        draw_feat_s0_mean_line=True,
    )
    write_report(
        output_md=output_md,
        progress_image_entries=progress_image_entries,
        feat_image_entries=feat_image_entries,
        output_dir=output_dir,
    )
    if output_pdf is not None:
        write_pdf_report(
            output_pdf=output_pdf,
            progress_image_entries=progress_image_entries,
            feat_image_entries=feat_image_entries,
            output_dir=output_dir,
        )

    print(f"[OK] input_root  : {input_root}")
    print(f"[OK] rows        : {len(df):,}")
    print(f"[OK] images(prog): {len(progress_image_entries)}")
    print(f"[OK] images(feat): {len(feat_image_entries)}")
    print(f"[OK] output_dir  : {output_dir}")
    print(f"[OK] output_md   : {output_md}")
    if output_pdf is not None:
        print(f"[OK] output_pdf  : {output_pdf}")


if __name__ == "__main__":
    main()
