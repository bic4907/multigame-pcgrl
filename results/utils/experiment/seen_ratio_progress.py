"""
seen_ratio_progress.py
======================
seen_ratio_progress 실험 전용 (aaai27_eval_mgpcgrl_unseen_ratios):
  train_seen_ratio (데이터 양)가 늘어남에 따라 unseen 게임의 progress 성능이
  어떻게 달라지는지를 꺾은선 그래프 1개로 시각화한다.

x 축: train_seen_ratio  (학습에 사용된 데이터 비율, e.g. 0.01 ~ 1.0)
y 축: unseen 게임의 progress (정규화)
선   : seen 게임 수(n_seen_games) — 색상 + 마커 조합으로 구분

출력:
    seen_ratio_progress.png   — 꺾은선 그래프
    seen_ratio_table.csv      — 집계 수치
    seen_ratio_table.md       — Markdown 테이블

사용법:
    python results/utils/experiment/seen_ratio_progress.py
    python results/utils/experiment/seen_ratio_progress.py --experiment seen_ratio_progress
    python results/utils/experiment/seen_ratio_progress.py --no-plot
"""

from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict
from pathlib import Path

import sys as _sys
_HERE        = Path(__file__).resolve().parent   # results/utils/experiment/
_RESULTS_DIR = _HERE.parent.parent               # results/
_ROOT        = _RESULTS_DIR.parent               # project root
if str(_RESULTS_DIR) not in _sys.path:
    _sys.path.insert(0, str(_RESULTS_DIR))
if str(_ROOT) not in _sys.path:
    _sys.path.append(str(_ROOT))

from utils.core.run_output import load_cfg, make_run_dir, setup_logger
from utils.core.io import (
    normalize_reward_enum,
    parse_run_tokens,
    iter_results_paths,
    load_run_config,
    get_game_split,
)
from utils.core.stats import safe_std, to_float
from utils.core.normalization import (
    compute_normalization_scale,
    apply_normalization,
    save_normalization_scale,
    load_normalization_scale,
)
from utils.experiment.benchmark import (
    _bar_plot_setup,
    _palette,
    _seed_agg,
    resolve_input_root,
    _get_experiment_folder_order,
)

_CFG = load_cfg()

# seen_games 수 별 마커 리스트
_MARKERS: list[str] = ["o", "s", "^", "D", "v", "P", "X", "*", "h", "+"]


# ---------------------------------------------------------------------------
# 데이터 수집
# ---------------------------------------------------------------------------

def collect_rows(
    input_root: Path,
    target_projects: list[str] | None = None,
) -> list[dict]:
    """results.csv 를 순회하며 train_seen_ratio / n_seen_games 포함 row 반환.

    train_seen_ratio: run_config["train_seen_ratio"] → 없으면 run_name sr 토큰 파싱 → 기본 1.0
    n_seen_games    : len(run_config["seen_games"])   → run_config 없으면 0
    """
    rows: list[dict] = []
    _cfg_cache: dict[Path, dict] = {}

    for results_path in iter_results_paths(input_root):
        rel = results_path.relative_to(input_root)
        if len(rel.parts) < 3:
            continue

        project   = rel.parts[0]
        run_name  = rel.parts[1]
        eval_name = rel.parts[2] if len(rel.parts) >= 4 else ""

        if target_projects and project not in target_projects:
            continue

        eval_tokens = parse_run_tokens(eval_name)
        run_tokens  = parse_run_tokens(run_name)
        seed = run_tokens.get("s", run_name)

        run_dir = results_path.parent
        if run_dir not in _cfg_cache:
            _cfg_cache[run_dir] = load_run_config(run_dir)
        run_cfg = _cfg_cache[run_dir]

        # ── train_seen_ratio ─────────────────────────────────────────────
        train_seen_ratio = run_cfg.get("train_seen_ratio")
        if train_seen_ratio is None:
            raw = run_tokens.get("sr")
            try:
                train_seen_ratio = float(raw) if raw is not None else 1.0
            except ValueError:
                train_seen_ratio = 1.0
        train_seen_ratio = float(train_seen_ratio)

        # ── n_seen_games ─────────────────────────────────────────────────
        seen_games = run_cfg.get("seen_games", [])
        n_seen_games = len(seen_games)

        with results_path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                game = (row.get("game") or "").strip() or "unknown"
                re_raw = (row.get("reward_enum") or "").strip()
                reward_enum = normalize_reward_enum(
                    re_raw or eval_tokens.get("re", "unknown")
                )
                progress = to_float(row.get("progress"))
                if progress is None:
                    continue

                game_split = get_game_split(game, run_cfg)

                rows.append({
                    "project":          project,
                    "game":             game,
                    "reward_enum":      reward_enum,
                    "game_split":       game_split,
                    "train_seen_ratio": train_seen_ratio,
                    "n_seen_games":     n_seen_games,
                    "seed":             seed,
                    "metrics":          {"progress": progress},
                })
    return rows


# ---------------------------------------------------------------------------
# 집계
# ---------------------------------------------------------------------------

def aggregate_by_seen_ratio(
    rows: list[dict],
    game_split: str = "unseen",
) -> dict[tuple[int, float], dict]:
    """(n_seen_games, train_seen_ratio) → {mean, std, n} 집계."""
    filtered = [r for r in rows if r.get("game_split") == game_split]

    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for r in filtered:
        key = (r["n_seen_games"], r["train_seen_ratio"])
        grouped[key].append(r)

    result: dict[tuple, dict] = {}
    for key, group_rows in grouped.items():
        stat = _seed_agg(group_rows, "progress")
        if stat:
            result[key] = stat
    return result


# ---------------------------------------------------------------------------
# 플롯
# ---------------------------------------------------------------------------

def write_line_plot(
    output_path: Path,
    rows: list[dict],
) -> None:
    """train_seen_ratio vs progress 꺾은선 그래프 (unseen 게임만).

    선 구분: n_seen_games (색상 + 마커)
    """
    plt = _bar_plot_setup()

    unseen_rows = [r for r in rows if r.get("game_split") == "unseen"]
    if not unseen_rows:
        return

    ratio_vals   = sorted({r["train_seen_ratio"] for r in unseen_rows})
    n_seen_vals  = sorted({r["n_seen_games"] for r in unseen_rows})
    colors       = _palette(len(n_seen_vals))

    # ratio_vals → 인덱스 매핑 (균등 간격)
    ratio_to_idx = {r: i for i, r in enumerate(ratio_vals)}

    agg = aggregate_by_seen_ratio(rows, game_split="unseen")

    fig, ax = plt.subplots(1, 1, figsize=(5, 4))

    for i, n_seen in enumerate(n_seen_vals):
        xs_idx, means, stds = [], [], []
        for ratio in ratio_vals:
            stat = agg.get((n_seen, ratio))
            if stat:
                xs_idx.append(ratio_to_idx[ratio])
                means.append(stat["mean"])
                stds.append(stat["std"])

        if not means:
            continue

        color  = colors[i % len(colors)]
        marker = _MARKERS[i % len(_MARKERS)]
        label  = f"{n_seen} seen game{'s' if n_seen != 1 else ''}"

        ax.plot(
            xs_idx, means,
            marker=marker, color=color,
            label=label, linewidth=1.8, markersize=6, zorder=3,
        )
        ax.fill_between(
            xs_idx,
            [m - s for m, s in zip(means, stds)],
            [m + s for m, s in zip(means, stds)],
            alpha=0.15, color=color,
        )

    ax.set_xlabel("Train seen ratio (data amount)", fontsize=11)
    ax.set_ylabel("Progress (normalized)", fontsize=11)
    ax.set_title("Unseen Game Progress vs. Train Seen Ratio", fontsize=12)
    ax.set_xticks(range(len(ratio_vals)))
    ax.set_xticklabels([f"{v:g}" for v in ratio_vals])
    ax.set_xlim(-0.4, len(ratio_vals) - 0.6)
    ax.grid(axis="both", alpha=0.3)
    ax.legend(title="# Seen Games", fontsize=9, title_fontsize=9,
               loc="best", framealpha=0.85)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 테이블
# ---------------------------------------------------------------------------

def write_table_csv(output_path: Path, rows: list[dict]) -> None:
    agg = aggregate_by_seen_ratio(rows, game_split="unseen")
    unseen_rows  = [r for r in rows if r.get("game_split") == "unseen"]
    ratio_vals   = sorted({r["train_seen_ratio"] for r in unseen_rows})
    n_seen_vals  = sorted({r["n_seen_games"] for r in unseen_rows})

    headers = ["n_seen_games", "train_seen_ratio",
               "progress_mean", "progress_std", "progress_n"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for n_seen in n_seen_vals:
            for ratio in ratio_vals:
                stat = agg.get((n_seen, ratio))
                writer.writerow({
                    "n_seen_games":      n_seen,
                    "train_seen_ratio":  ratio,
                    "progress_mean":     stat["mean"] if stat else "",
                    "progress_std":      stat["std"]  if stat else "",
                    "progress_n":        stat["n"]    if stat else 0,
                })


def write_table_markdown(
    output_path: Path,
    rows: list[dict],
    decimals: int = 4,
) -> None:
    agg = aggregate_by_seen_ratio(rows, game_split="unseen")
    unseen_rows  = [r for r in rows if r.get("game_split") == "unseen"]
    ratio_vals   = sorted({r["train_seen_ratio"] for r in unseen_rows})
    n_seen_vals  = sorted({r["n_seen_games"] for r in unseen_rows})

    header_cols = ["n_seen_games", "train_seen_ratio", "progress"]
    lines = [
        "| " + " | ".join(header_cols) + " |",
        "| " + " | ".join(["---"] * len(header_cols)) + " |",
    ]
    for n_seen in n_seen_vals:
        for ratio in ratio_vals:
            stat = agg.get((n_seen, ratio))
            val = (
                f"{stat['mean']:.{decimals}f} ± {stat['std']:.{decimals}f}"
                if stat else "-"
            )
            lines.append(f"| {n_seen} | {ratio:g} | {val} |")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Args / Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    _exp_names = list(_CFG.get("experiments", {}).keys())
    parser = argparse.ArgumentParser(
        description="train_seen_ratio 증가에 따른 unseen 게임 progress 변화 꺾은선 그래프"
    )
    parser.add_argument("--input", default="wandb_projects",
                        help="wandb_projects 루트 디렉토리")
    parser.add_argument("--decimals", type=int, default=4)
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument(
        "--experiment",
        choices=_exp_names if _exp_names else None,
        default="seen_ratio_progress",
        metavar="EXPERIMENT",
    )
    return parser.parse_args()


def main() -> None:
    args    = parse_args()
    run_dir = make_run_dir("seen_ratio_progress", cfg=_CFG)
    log     = setup_logger(run_dir, name=__file__)
    log.debug("run_dir : %s", run_dir)

    experiment   = args.experiment or "seen_ratio_progress"
    folder_order = _get_experiment_folder_order(experiment)
    log.info("experiment: %s  target_projects=%s", experiment, folder_order)

    input_root = resolve_input_root(args.input, _RESULTS_DIR)
    log.info("input_root: %s", input_root)

    rows = collect_rows(input_root, target_projects=folder_order or None)
    if not rows:
        msg = "No valid rows found. wandb_projects 에 aaai27_eval_mgpcgrl_unseen_ratios 데이터가 있는지 확인하세요."
        log.error(msg)
        raise SystemExit(msg)

    unseen_rows = [r for r in rows if r.get("game_split") == "unseen"]
    if not unseen_rows:
        msg = ("unseen 게임 데이터가 없습니다 — "
               "run_config.json 의 unseen_games 필드를 확인하세요.")
        log.error(msg)
        raise SystemExit(msg)

    # Normalization
    _global_scale_path = os.environ.get("PIPELINE_NORM_SCALE")
    if _global_scale_path and Path(_global_scale_path).is_file():
        norm_scale = load_normalization_scale(Path(_global_scale_path))
        log.info("norm_scale (global): %s", _global_scale_path)
    else:
        norm_scale = compute_normalization_scale(rows, ["progress"])
        scale_path = run_dir / "normalization_scale.json"
        save_normalization_scale(norm_scale, scale_path)
        log.info("norm_scale (local) : %s", scale_path)

    norm_rows = apply_normalization(rows, norm_scale, ["progress"])

    # 테이블
    write_table_csv(run_dir / "seen_ratio_table.csv", norm_rows)
    write_table_markdown(
        run_dir / "seen_ratio_table.md", norm_rows, decimals=args.decimals
    )
    log.info("table   : %s", run_dir / "seen_ratio_table.md")

    # 플롯
    if not args.no_plot:
        try:
            write_line_plot(run_dir / "seen_ratio_progress.png", norm_rows)
            log.info("plot    : %s", run_dir / "seen_ratio_progress.png")
        except RuntimeError as e:
            log.error("Plot generation failed: %s", e)
            raise SystemExit(str(e)) from e

    log.info(
        "rows: total=%d  unseen=%d  ratio_vals=%s  n_seen_vals=%s",
        len(rows),
        len(unseen_rows),
        sorted({r["train_seen_ratio"] for r in unseen_rows}),
        sorted({r["n_seen_games"] for r in unseen_rows}),
    )


if __name__ == "__main__":
    main()

