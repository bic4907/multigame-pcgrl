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
import json
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


def _load_cfg() -> dict:
    cfg_path = _HERE / "config.json"
    if cfg_path.is_file():
        with cfg_path.open(encoding="utf-8") as f:
            return json.load(f)
    return {}

_CFG = _load_cfg()
_re_cfg = _CFG.get("reward_enums", {})
_paths_cfg = _CFG.get("paths", {})

_DEFAULT_REWARD_ENUMS: list[int] = sorted(
    int(k) for k in _re_cfg.get("labels", {"0": None, "1": None, "2": None, "3": None, "4": None}).keys()
)
NUM_SLOTS: int = _re_cfg.get("num_slots", 4)
_RUN_DIR_PATTERN: str = _re_cfg.get("run_dir_pattern", "cpcgrl_game-all_re-{reward_enum}_exp-def_s-0")


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
        default=_paths_cfg.get("reward_viz_root", "results/wandb_download/aaai27_eval_cpcgrl"),
        help="Root folder containing cpcgrl_game-all_re-{re}_exp-def_s-0 runs.",
    )
    parser.add_argument(
        "--reward-enums",
        nargs="+",
        type=int,
        default=_DEFAULT_REWARD_ENUMS,
        help="reward_enum ids to visualize.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for PNG outputs. Default: <run_dir>/plots/",
    )
    parser.add_argument(
        "--output-md",
        default=None,
        help="Markdown report path. Default: <run_dir>/report.md",
    )
    parser.add_argument(
        "--render-tile-size",
        type=int,
        default=16,
        help="Tile size used by env renderer (actual tile image rendering).",
    )
    parser.add_argument(
        "--experiment",
        default=None,
        help="Experiment group 이름 (현재는 로깅용; 향후 필터링에 활용 가능).",
    )
    return parser.parse_args()


def resolve_paths(root_arg: str, output_dir_arg: str, output_md_arg: str) -> tuple[Path, Path, Path]:
    def _resolve_project_path(path_arg: str, prefer_existing: bool = False) -> Path:
        raw = Path(path_arg)
        if raw.is_absolute():
            return raw.resolve()
        candidates = [
            (_HERE / raw).resolve(),   # results/ 기준 (wandb_projects/ 실제 위치)
            (_ROOT / raw).resolve(),   # 프로젝트 루트 기준
            (Path.cwd() / raw).resolve(),
        ]
        if prefer_existing:
            for c in candidates:
                if c.exists():
                    return c
        return candidates[0]  # 기본값: results/ 기준

    root = _resolve_project_path(root_arg, prefer_existing=True)
    out_dir = _resolve_project_path(output_dir_arg, prefer_existing=False)
    out_md = _resolve_project_path(output_md_arg, prefer_existing=False)
    return root, out_dir, out_md


def run_dir_for_reward(root: Path, reward_enum: int) -> Path:
    folder = _RUN_DIR_PATTERN.format(reward_enum=reward_enum)
    return root / folder


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
    lines.append("Selection method:")
    lines.append("- The full range of `condition_{reward_enum}` is split into 4 equal bins per `reward_enum`.")
    lines.append("- The sample closest to each bin center is selected; ties are broken by highest mean `progress`.")
    lines.append("- Tile maps are rendered with `envs.probs.multigame.render_multigame_map_np` using actual tile images.")
    lines.append("- The `seed_0` group (or the first seed if absent) is used from the corresponding `eval.h5` group.")
    lines.append("")

    for reward_enum, grouped, selections, image_name in all_entries:
        cmin = grouped["condition"].min() if not grouped.empty else np.nan
        cmax = grouped["condition"].max() if not grouped.empty else np.nan
        lines.append(f"## reward_enum = {reward_enum}")
        lines.append("")
        lines.append(f"- Condition range: `{cmin:.2f} ~ {cmax:.2f}`")
        lines.append(f"- Number of candidates (game+row_i): `{len(grouped)}`")
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
    import sys as _sys
    _script_dir = Path(__file__).resolve().parent
    _project_root = _script_dir.parent
    if str(_script_dir) not in _sys.path:
        _sys.path.insert(0, str(_script_dir))
    if str(_project_root) not in _sys.path:
        _sys.path.append(str(_project_root))
    from instruct_rl.utils.log_utils import get_logger
    from utils.run_output import make_run_dir, setup_logger

    args = parse_args()
    out_run_dir = make_run_dir("reward_enum_visualizer", cfg=_CFG)
    log = setup_logger(out_run_dir, name=__file__)
    log.info("run_dir    : %s", out_run_dir)
    root, _, _ = resolve_paths(args.root, str(out_run_dir / "plots"), str(out_run_dir / "report.md"))
    output_dir = Path(args.output_dir).resolve() if args.output_dir else out_run_dir / "plots"
    output_md  = Path(args.output_md).resolve()  if args.output_md  else out_run_dir / "report.md"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_entries: list[tuple[int, pd.DataFrame, list[Selection], str]] = []

    for reward_enum in args.reward_enums:
        run_dir = run_dir_for_reward(root, reward_enum)
        ctrl_sim_path = run_dir / "ctrl_sim.csv"
        h5_path = run_dir / "eval.h5"
        if not ctrl_sim_path.exists() or not h5_path.exists():
            log.warning("Skip reward_enum=%d: missing files in %s", reward_enum, run_dir)
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
        log.info(
            "reward_enum=%d range=(%.2f, %.2f) selected=%d image=%s",
            reward_enum,
            grouped["condition"].min(),
            grouped["condition"].max(),
            len(selections),
            output_dir / image_name,
        )

    build_markdown(output_md, output_dir, all_entries)
    log.info("output_dir : %s", output_dir)
    log.info("report     : %s", output_md)
    log.info("log        : %s", out_run_dir / 'run.log')


if __name__ == "__main__":
    main()
