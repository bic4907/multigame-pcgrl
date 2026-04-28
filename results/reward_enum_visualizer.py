"""
Select representative conditions per reward_enum and visualize state maps.

Inputs (default):
  - results/wandb_download/aaai27_eval_cpcgrl/cpcgrl_game-all_re-{0..4}_exp-def_s-0/ctrl_sim.csv
  - results/wandb_download/aaai27_eval_cpcgrl/cpcgrl_game-all_re-{0..4}_exp-def_s-0/eval.h5

Outputs (default):
  - results/wandb_download/reward_enum_viz/reward_enum_{re}.png
  - results/wandb_download/reward_enum_viz_report.md
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Ensure project root is importable even when run from "results/".
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from envs.probs.multigame import render_multigame_map_np


NUM_SLOTS = 4


@dataclass
class Selection:
    slot: int
    bin_low: float
    bin_high: float
    target_condition: float
    condition: float
    game: str
    row_i: int
    progress: float
    vit_score: float
    tpkldiv: float
    h5_key: str
    seed_key: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize representative 4 conditions per reward_enum."
    )
    parser.add_argument(
        "--root",
        default="results/wandb_download/aaai27_eval_cpcgrl",
        help="Root folder containing cpcgrl_game-all_re-{re}_exp-def_s-0 runs.",
    )
    parser.add_argument(
        "--reward-enums",
        nargs="+",
        type=int,
        default=[0, 1, 2, 3, 4],
        help="reward_enum ids to visualize.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/wandb_download/reward_enum_viz",
        help="Directory for PNG outputs.",
    )
    parser.add_argument(
        "--output-md",
        default="results/wandb_download/reward_enum_viz_report.md",
        help="Markdown report path.",
    )
    parser.add_argument(
        "--render-tile-size",
        type=int,
        default=16,
        help="Tile size used by env renderer (actual tile image rendering).",
    )
    return parser.parse_args()


def resolve_paths(root_arg: str, output_dir_arg: str, output_md_arg: str) -> tuple[Path, Path, Path]:
    def _resolve_project_path(path_arg: str, prefer_existing: bool = False) -> Path:
        raw = Path(path_arg)
        if raw.is_absolute():
            return raw.resolve()
        proj = (_ROOT / raw).resolve()
        cwd = (Path.cwd() / raw).resolve()
        if prefer_existing:
            if proj.exists():
                return proj
            if cwd.exists():
                return cwd
        return proj

    root = _resolve_project_path(root_arg, prefer_existing=True)
    out_dir = _resolve_project_path(output_dir_arg, prefer_existing=False)
    out_md = _resolve_project_path(output_md_arg, prefer_existing=False)
    return root, out_dir, out_md


def run_dir_for_reward(root: Path, reward_enum: int) -> Path:
    return root / f"cpcgrl_game-all_re-{reward_enum}_exp-def_s-0"


def aggregate_ctrl_sim(ctrl_sim_path: Path, reward_enum: int) -> pd.DataFrame:
    df = pd.read_csv(ctrl_sim_path)
    condition_col = f"condition_{reward_enum}"
    if condition_col not in df.columns:
        raise ValueError(f"Missing condition column: {condition_col} in {ctrl_sim_path}")

    sub = df[["game", "row_i", condition_col, "progress", "vit_score", "tpkldiv"]].copy()
    sub = sub.dropna(subset=[condition_col])
    sub["row_i"] = sub["row_i"].astype(int)
    grouped = (
        sub.groupby(["game", "row_i"], as_index=False)
        .agg(
            condition=(condition_col, "first"),
            progress=("progress", "mean"),
            vit_score=("vit_score", "mean"),
            tpkldiv=("tpkldiv", "mean"),
        )
        .sort_values(["condition", "progress"], ascending=[True, False])
        .reset_index(drop=True)
    )
    return grouped


def resolve_h5_key(h5_file: h5py.File, game: str, reward_enum: int, row_i: int) -> str | None:
    key_candidates = [
        f"{game}_re{reward_enum}_{row_i:04d}",
        f"{game}_re{reward_enum}_{row_i}",
    ]
    for key in key_candidates:
        if key in h5_file:
            return key
    return None


def choose_seed_key(group: h5py.Group) -> str:
    if "seed_0" in group:
        return "seed_0"
    seed_keys = sorted(group.keys())
    if not seed_keys:
        raise ValueError("No seed_* group found in h5 group.")
    return seed_keys[0]


def select_representative_conditions(
    grouped: pd.DataFrame,
    h5_file: h5py.File,
    reward_enum: int,
) -> list[Selection]:
    if grouped.empty:
        return []

    values = grouped["condition"].to_numpy()
    vmin = float(values.min())
    vmax = float(values.max())
    edges = np.linspace(vmin, vmax, NUM_SLOTS + 1)

    selected: list[Selection] = []
    used_idx: set[int] = set()

    for slot in range(1, NUM_SLOTS + 1):
        low = float(edges[slot - 1])
        high = float(edges[slot])
        center = float((low + high) / 2.0)

        in_bin = grouped[
            (grouped["condition"] >= low)
            & ((grouped["condition"] < high) if slot < NUM_SLOTS else (grouped["condition"] <= high))
        ].copy()

        candidates = in_bin if not in_bin.empty else grouped.copy()
        candidates["dist"] = np.abs(candidates["condition"] - center)
        candidates = candidates.sort_values(["dist", "progress"], ascending=[True, False])

        picked = None
        for idx, row in candidates.iterrows():
            if idx in used_idx:
                continue
            game = str(row["game"])
            row_i = int(row["row_i"])
            h5_key = resolve_h5_key(h5_file, game, reward_enum, row_i)
            if h5_key is None:
                continue
            seed_key = choose_seed_key(h5_file[h5_key])
            picked = Selection(
                slot=slot,
                bin_low=low,
                bin_high=high,
                target_condition=center,
                condition=float(row["condition"]),
                game=game,
                row_i=row_i,
                progress=float(row["progress"]),
                vit_score=float(row["vit_score"]),
                tpkldiv=float(row["tpkldiv"]),
                h5_key=h5_key,
                seed_key=seed_key,
            )
            used_idx.add(idx)
            break

        if picked is not None:
            selected.append(picked)

    return selected


def draw_reward_figure(
    reward_enum: int,
    grouped: pd.DataFrame,
    selections: list[Selection],
    h5_file: h5py.File,
    output_path: Path,
    render_tile_size: int,
) -> None:
    fig = plt.figure(figsize=(16, 6.5))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 1.5])

    ax_hist = fig.add_subplot(gs[0, :])
    ax_hist.hist(grouped["condition"], bins=40, color="#8FB3D9", edgecolor="#335C81")
    ax_hist.set_title(f"reward_enum={reward_enum} condition distribution")
    ax_hist.set_xlabel(f"condition_{reward_enum}")
    ax_hist.set_ylabel("count")

    colors = ["#C1121F", "#F4A261", "#2A9D8F", "#264653"]
    for i, sel in enumerate(selections):
        ax_hist.axvline(sel.condition, color=colors[i % len(colors)], linestyle="--", linewidth=2)
        ax_hist.text(
            sel.condition,
            ax_hist.get_ylim()[1] * (0.9 - i * 0.08),
            f"C{sel.slot}={sel.condition:.2f}",
            color=colors[i % len(colors)],
            ha="center",
            va="center",
            fontsize=10,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75},
        )

    for i in range(4):
        ax = fig.add_subplot(gs[1, i])
        if i >= len(selections):
            ax.axis("off")
            continue
        sel = selections[i]
        state = h5_file[sel.h5_key][sel.seed_key]["state"][()]
        rendered = render_multigame_map_np(np.asarray(state), tile_size=render_tile_size)
        ax.imshow(rendered, interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(
            (
                f"C{sel.slot} [{sel.bin_low:.1f}, {sel.bin_high:.1f}]\n"
                f"{sel.game} row={sel.row_i} seed={sel.seed_key.split('_')[-1]}\n"
                f"cond={sel.condition:.2f} prog={sel.progress:.1f}"
            ),
            fontsize=9,
        )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def build_markdown(
    report_path: Path,
    output_dir: Path,
    all_entries: list[tuple[int, pd.DataFrame, list[Selection], str]],
) -> None:
    lines: list[str] = []
    lines.append("# Reward Enum Condition Visualization")
    lines.append("")
    lines.append("선정 방식:")
    lines.append("- 각 `reward_enum`에서 `condition_{reward_enum}` 전체 범위를 4개 구간으로 균등 분할")
    lines.append("- 각 구간 중심값에 가장 가까운 샘플을 선택하고, 동률이면 `progress` 평균이 높은 샘플 우선")
    lines.append("- 각 샘플의 타일맵은 `envs.probs.multigame.render_multigame_map_np`로 실제 타일 이미지 렌더")
    lines.append("- `eval.h5`의 해당 그룹에서 `seed_0`(없으면 첫 seed) 상태를 사용")
    lines.append("")

    for reward_enum, grouped, selections, image_name in all_entries:
        cmin = grouped["condition"].min() if not grouped.empty else np.nan
        cmax = grouped["condition"].max() if not grouped.empty else np.nan
        lines.append(f"## reward_enum = {reward_enum}")
        lines.append("")
        lines.append(f"- 조건 범위: `{cmin:.2f} ~ {cmax:.2f}`")
        lines.append(f"- 후보 수(게임+row_i): `{len(grouped)}`")
        lines.append("")
        lines.append(f"![reward_enum_{reward_enum}](./{output_dir.name}/{image_name})")
        lines.append("")
        lines.append("| slot | bin_range | target_cond | selected_cond | game | row_i | progress | vit_score | tpkldiv |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        for sel in selections:
            lines.append(
                f"| C{sel.slot} | [{sel.bin_low:.2f}, {sel.bin_high:.2f}] | {sel.target_condition:.2f} | "
                f"{sel.condition:.2f} | {sel.game} | {sel.row_i} | {sel.progress:.2f} | "
                f"{sel.vit_score:.3f} | {sel.tpkldiv:.3f} |"
            )
        lines.append("")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    root, output_dir, output_md = resolve_paths(args.root, args.output_dir, args.output_md)

    all_entries: list[tuple[int, pd.DataFrame, list[Selection], str]] = []

    for reward_enum in args.reward_enums:
        run_dir = run_dir_for_reward(root, reward_enum)
        ctrl_sim_path = run_dir / "ctrl_sim.csv"
        h5_path = run_dir / "eval.h5"
        if not ctrl_sim_path.exists() or not h5_path.exists():
            print(f"[WARN] Skip reward_enum={reward_enum}: missing files in {run_dir}")
            continue

        grouped = aggregate_ctrl_sim(ctrl_sim_path, reward_enum)
        with h5py.File(h5_path, "r") as h5_file:
            selections = select_representative_conditions(grouped, h5_file, reward_enum)
            image_name = f"reward_enum_{reward_enum}.png"
            draw_reward_figure(
                reward_enum=reward_enum,
                grouped=grouped,
                selections=selections,
                h5_file=h5_file,
                output_path=output_dir / image_name,
                render_tile_size=args.render_tile_size,
            )
        all_entries.append((reward_enum, grouped, selections, image_name))
        print(
            f"[OK] reward_enum={reward_enum} range=({grouped['condition'].min():.2f}, {grouped['condition'].max():.2f}) "
            f"selected={len(selections)} image={output_dir / image_name}"
        )

    build_markdown(output_md, output_dir, all_entries)
    print(f"[OK] report={output_md}")


if __name__ == "__main__":
    main()
