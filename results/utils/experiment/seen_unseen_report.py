"""
seen_unseen_report.py
=====================
unseen 실험 전용:
  results.csv 에서 seen / unseen 게임 성능을 분리 집계하여
  Markdown 테이블 + 비교 플롯을 생성한다.

출력:
    seen_unseen_table.md / seen_unseen_table.csv   — seen vs unseen 비교 테이블
    seen_unseen.png                                 — unseen 바 플롯 + seen 기준선 오버레이
    seen_table.md / seen_table.csv                  — seen 게임만 테이블
    unseen_table.md / unseen_table.csv              — unseen 게임만 테이블

사용법:
    python results/seen_unseen_report.py --experiment unseen
    python results/seen_unseen_report.py --experiment unseen --no-plot
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import sys as _sys
_HERE        = __import__('pathlib').Path(__file__).resolve().parent  # results/utils/experiment/
_RESULTS_DIR = _HERE.parent.parent                                     # results/
_ROOT        = _HERE.parent.parent.parent                              # project root
if str(_RESULTS_DIR) not in _sys.path:
    _sys.path.insert(0, str(_RESULTS_DIR))
if str(_ROOT) not in _sys.path:
    _sys.path.append(str(_ROOT))

from utils.core.run_output import load_cfg, make_run_dir, setup_logger
from utils.core.io import sort_key_reward_enum
from utils.core.normalization import compute_normalization_scale, apply_normalization, save_normalization_scale, load_normalization_scale
from utils.experiment.benchmark import (
    collect_plot_rows_from_results,
    resolve_input_root,
    _get_experiment_folder_order,
    _load_project_display_names,
    _project_display_name,
    _sort_folder_for_plot,
    _bar_plot_setup,
    _palette,
    _seed_agg,
    _YMIN,
    DEFAULT_METRIC_ORDER,
    METRIC_DISPLAY_NAMES,
)

_CFG = load_cfg()


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    _exp_names = list(_CFG.get("experiments", {}).keys())
    parser = argparse.ArgumentParser(
        description="seen / unseen 게임 성능 분리 리포트 (unseen 전용)"
    )
    parser.add_argument("--input", default="wandb_projects")
    parser.add_argument("--metrics", nargs="+", default=None)
    parser.add_argument("--decimals", type=int, default=4)
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument(
        "--experiment",
        choices=_exp_names if _exp_names else None,
        default="unseen",
        metavar="EXPERIMENT",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def aggregate_by_split(
    plot_rows: list[dict],
    metric_order: list[str],
    splits: list[str] | None = None,
) -> dict[tuple[str, str], dict]:
    """(folder, game_split) → {metric: {mean, std, n_seeds}} 딕셔너리 반환.

    같은 시드(seed)의 데이터를 먼저 평균낸 뒤 시드 간 표준편차를 계산한다.
    """
    splits = splits or ["seen", "unseen"]
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in plot_rows:
        if row.get("game_split") in splits:
            grouped[(row["project"], row["game_split"])].append(row)

    result: dict[tuple[str, str], dict] = {}
    for key, rows in grouped.items():
        stats: dict = {}
        for metric in metric_order:
            stat = _seed_agg(rows, metric)
            if stat:
                stats[metric] = stat
        result[key] = stats
    return result


# ---------------------------------------------------------------------------
# Table writers
# ---------------------------------------------------------------------------

def write_split_comparison_markdown(
    output_path: Path,
    plot_rows: list[dict],
    metric_order: list[str],
    decimals: int = 4,
) -> None:
    """seen 과 unseen 을 나란히 보여주는 Markdown 테이블."""
    splits = ["seen", "unseen"]
    agg = aggregate_by_split(plot_rows, metric_order, splits)
    folders = sorted({row["project"] for row in plot_rows}, key=_sort_folder_for_plot)

    header_cols = ["folder"]
    for split in splits:
        for m in metric_order:
            header_cols.append(f"{m}_{split}")

    lines = [
        "| " + " | ".join(header_cols) + " |",
        "| " + " | ".join(["---"] * len(header_cols)) + " |",
    ]
    for folder in folders:
        cells = [_project_display_name(folder)]
        for split in splits:
            stats = agg.get((folder, split), {})
            for m in metric_order:
                stat = stats.get(m)
                cells.append(
                    f"{stat['mean']:.{decimals}f} +- {stat['std']:.{decimals}f}"
                    if stat else "-"
                )
        lines.append("| " + " | ".join(cells) + " |")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_split_comparison_csv(
    output_path: Path,
    plot_rows: list[dict],
    metric_order: list[str],
) -> None:
    splits = ["seen", "unseen"]
    agg = aggregate_by_split(plot_rows, metric_order, splits)
    folders = sorted({row["project"] for row in plot_rows}, key=_sort_folder_for_plot)

    headers = ["folder"]
    for split in splits:
        for m in metric_order:
            headers += [f"{m}_{split}_mean", f"{m}_{split}_std", f"{m}_{split}_n"]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for folder in folders:
            rec: dict = {"folder": _project_display_name(folder)}
            for split in splits:
                stats = agg.get((folder, split), {})
                for m in metric_order:
                    stat = stats.get(m)
                    rec[f"{m}_{split}_mean"] = stat["mean"] if stat else ""
                    rec[f"{m}_{split}_std"]  = stat["std"]  if stat else ""
                    rec[f"{m}_{split}_n"]    = stat["n"]    if stat else 0
            writer.writerow(rec)


def write_single_split_markdown(
    output_path: Path,
    plot_rows: list[dict],
    split: str,
    metric_order: list[str],
    decimals: int = 4,
) -> None:
    """단일 split(seen 또는 unseen) 결과 Markdown 테이블."""
    agg = aggregate_by_split(plot_rows, metric_order, [split])
    folders = sorted({row["project"] for row in plot_rows}, key=_sort_folder_for_plot)

    header_cols = ["folder", "n"] + metric_order
    lines = [
        "| " + " | ".join(header_cols) + " |",
        "| " + " | ".join(["---"] * len(header_cols)) + " |",
    ]
    for folder in folders:
        stats = agg.get((folder, split), {})
        n = max((s["n"] for s in stats.values()), default=0)
        cells = [_project_display_name(folder), str(n)]
        for m in metric_order:
            stat = stats.get(m)
            cells.append(
                f"{stat['mean']:.{decimals}f} +- {stat['std']:.{decimals}f}"
                if stat else "-"
            )
        lines.append("| " + " | ".join(cells) + " |")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_single_split_csv(
    output_path: Path,
    plot_rows: list[dict],
    split: str,
    metric_order: list[str],
) -> None:
    agg = aggregate_by_split(plot_rows, metric_order, [split])
    folders = sorted({row["project"] for row in plot_rows}, key=_sort_folder_for_plot)

    headers = ["folder", "n"]
    for m in metric_order:
        headers += [f"{m}_mean", f"{m}_std", f"{m}_n"]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for folder in folders:
            stats = agg.get((folder, split), {})
            n = max((s["n"] for s in stats.values()), default=0)
            rec: dict = {"folder": _project_display_name(folder), "n": n}
            for m in metric_order:
                stat = stats.get(m)
                rec[f"{m}_mean"] = stat["mean"] if stat else ""
                rec[f"{m}_std"]  = stat["std"]  if stat else ""
                rec[f"{m}_n"]    = stat["n"]    if stat else 0
            writer.writerow(rec)


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def write_seen_unseen_plot(
    output_path: Path,
    plot_rows: list[dict],
    metric_order: list[str],
    seen_baseline_project: str | None = None,
    seen_baseline_label: str | None = None,
) -> None:
    """unseen 게임 성능 바 플롯 + seen 성능을 빨간 수평선으로 오버레이.

    레이아웃: 1행 × n_metrics열 (가로 배치, allseen re.png 와 유사한 크기)
    바 높이와 에러 바는 시드별 먼저 평균 → 시드 간 mean/std 로 계산한다.
    """
    plt = _bar_plot_setup()
    folders = sorted({r["project"] for r in plot_rows}, key=_sort_folder_for_plot)
    # unseen 데이터에 실제로 존재하는 reward_enum 만 사용 (seen-only 항목을 x축에서 제거)
    unseen_rows = [r for r in plot_rows if r.get("game_split") == "unseen"]
    rewards = sorted({r["reward_enum"] for r in unseen_rows}, key=sort_key_reward_enum)
    colors  = _palette(len(folders))

    n_metrics = len(metric_order)
    if not n_metrics:
        return

    baseline_project = seen_baseline_project or (folders[0] if folders else None)
    baseline_label   = seen_baseline_label or (
        _project_display_name(baseline_project) if baseline_project else "Seen baseline"
    )
    legend_baseline_text = f"Seen baseline ({baseline_label})"

    # (folder, game_split, reward_enum) → rows 목록
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for r in plot_rows:
        grouped[(r["project"], r.get("game_split", ""), r["reward_enum"])].append(r)

    def _seed_stats(folder, split, re, metric):
        stat = _seed_agg(grouped.get((folder, split, re), []), metric)
        if stat is None:
            return None, None
        return stat["mean"], stat["std"]

    # 가로 배치: 1행 × n_metrics열
    fig, axes = plt.subplots(
        1, n_metrics,
        figsize=(3.8 * n_metrics + 1.5, 3.5),
        squeeze=False,
    )

    x_center = list(range(len(rewards)))

    # unseen 데이터가 실제로 존재하는 폴더만 bar 위치·너비 계산에 사용
    folders_with_unseen = [
        f for f in folders
        if any(grouped.get((f, "unseen", re), []) for re in rewards)
    ]
    _BAR_SPAN = 0.55   # 전체 바 그룹 폭 (0.8 → 0.55 로 축소)
    bar_width = _BAR_SPAN / max(len(folders_with_unseen), 1)

    for ci, metric in enumerate(metric_order):
        metric_label = METRIC_DISPLAY_NAMES.get(metric, metric)
        ax = axes[0][ci]
        drew_any, y_uppers = False, []
        baseline_legend_added = False

        # ── unseen bars: 시드별 std → yerr ────────────────────────────────
        # j_eff: unseen 데이터 있는 폴더 내 인덱스 (바를 가운데 정렬)
        for j_eff, folder in enumerate(folders_with_unseen):
            j_color = folders.index(folder)  # 원래 색상 인덱스 유지
            means, stds, xs = [], [], []
            for k, re in enumerate(rewards):
                mean, std = _seed_stats(folder, "unseen", re, metric)
                if mean is None:
                    continue
                xs.append(x_center[k] - _BAR_SPAN / 2 + (j_eff + 0.5) * bar_width)
                means.append(float(mean))
                stds.append(float(std))
                y_uppers.append(float(mean) + float(std))
            if means:
                drew_any = True
                ax.bar(
                    xs, means, width=bar_width, yerr=stds, capsize=2,
                    label=_project_display_name(folder),
                    color=colors[j_color % len(colors)], edgecolor="white",
                    linewidth=0.8, alpha=0.9,
                )

        # ── seen baseline 수평선 ───────────────────────────────────────────
        if baseline_project:
            for k, re in enumerate(rewards):
                val_seen, _ = _seed_stats(baseline_project, "seen", re, metric)
                if val_seen is None:
                    continue
                y_val   = float(val_seen)
                x_left  = x_center[k] - 0.45
                x_right = x_center[k] + 0.45
                lbl = legend_baseline_text if not baseline_legend_added else None
                ax.plot(
                    [x_left, x_right], [y_val, y_val],
                    color="red", linewidth=2.0, linestyle="--",
                    zorder=5, label=lbl,
                )
                baseline_legend_added = True
                y_uppers.append(y_val)

        ax.set_title(metric_label)
        if ci == 0:
            ax.set_ylabel("Score", rotation=90, labelpad=8)
        ax.set_xticks(x_center, [f"re={r}" for r in rewards])
        ax.tick_params(axis="x", labelrotation=0)
        ax.set_xlim(-0.5, len(rewards) - 0.5)
        ax.grid(axis="y", alpha=0.3)
        if drew_any and y_uppers:
            dm  = max(y_uppers)
            pad = max(dm, 1e-6) * 0.15
            ax.set_ylim(_YMIN.get(metric, 0), dm + pad)
        else:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                    ha="center", va="center", color="gray")

    # 범례: 중복 제거 후 상단 중앙
    handles, labels = [], []
    for ax in axes[0]:
        h, l = ax.get_legend_handles_labels()
        for handle, lbl in zip(h, l):
            if lbl and lbl not in labels:
                handles.append(handle)
                labels.append(lbl)
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(len(handles), 5),
                   fontsize=8, bbox_to_anchor=(0.5, 1.0))
        fig.subplots_adjust(top=0.82)

    fig.tight_layout(rect=(0, 0, 1, 0.93))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:

    args    = parse_args()
    run_dir = make_run_dir("seen_unseen_report", cfg=_CFG)
    log     = setup_logger(run_dir, name=__file__)
    log.debug("run_dir   : %s", run_dir)

    experiment = args.experiment or "unseen"
    folder_order = _get_experiment_folder_order(experiment)
    log.info("experiment: %s  folder_order=%s", experiment, folder_order)

    from utils.experiment.benchmark import PREFERRED_PLOT_FOLDER_ORDER as _pfo  # noqa: F401
    import utils.experiment.benchmark as _bm
    _bm.PREFERRED_PLOT_FOLDER_ORDER = folder_order
    _bm._PROJECT_DISPLAY_NAMES      = _load_project_display_names(experiment)

    script_dir = _RESULTS_DIR  # wandb_projects 기본 탐색 기준은 results/
    input_root = resolve_input_root(args.input, script_dir)
    log.info("input_root: %s", input_root)

    # 메트릭 순서
    metric_order: list[str] = args.metrics or DEFAULT_METRIC_ORDER.copy()

    # results.csv 에서 로우 수집
    plot_rows = collect_plot_rows_from_results(input_root, metric_order)
    if folder_order:
        plot_rows = [r for r in plot_rows if r["project"] in folder_order]

    if not plot_rows:
        msg = "No valid rows found in results.csv under input root."
        log.error(msg)
        raise SystemExit(msg)

    has_split = any(r.get("game_split") in ("seen", "unseen") for r in plot_rows)
    if not has_split:
        msg = "seen/unseen 정보가 없습니다 — run_config.json 을 확인하세요."
        log.error(msg)
        raise SystemExit(msg)

    # Normalization — 파이프라인 전역 scale 우선, 없으면 로컬 계산
    import os as _os
    _global_scale_path = _os.environ.get("PIPELINE_NORM_SCALE")
    if _global_scale_path and Path(_global_scale_path).is_file():
        norm_scale = load_normalization_scale(Path(_global_scale_path))
        log.info("norm_scale (global): %s", _global_scale_path)
    else:
        norm_scale = compute_normalization_scale(plot_rows, metric_order)
        norm_scale_path = run_dir / "normalization_scale.json"
        save_normalization_scale(norm_scale, norm_scale_path)
        log.info("norm_scale (local) : %s", norm_scale_path)
    norm_rows = apply_normalization(plot_rows, norm_scale, metric_order)

    # ---- 비교 테이블 (seen vs unseen 한 눈에) ----
    write_split_comparison_markdown(
        run_dir / "seen_unseen_table.md", norm_rows, metric_order, args.decimals
    )
    write_split_comparison_csv(
        run_dir / "seen_unseen_table.csv", norm_rows, metric_order
    )
    log.info("table     : %s", run_dir / "seen_unseen_table.md")

    # ---- seen / unseen 각각 테이블 ----
    for split in ("seen", "unseen"):
        write_single_split_markdown(
            run_dir / f"{split}_table.md", norm_rows, split, metric_order, args.decimals
        )
        write_single_split_csv(
            run_dir / f"{split}_table.csv", norm_rows, split, metric_order
        )
        log.info("%s_table  : %s", split, run_dir / f"{split}_table.md")

    # ---- 플롯 ----
    if not args.no_plot:
        exp_cfg = _CFG.get("experiments", {}).get(experiment, {})
        sb_project = exp_cfg.get("seen_baseline_project")
        sb_label   = exp_cfg.get("seen_baseline_label")
        try:
            write_seen_unseen_plot(
                run_dir / "unseen.png", norm_rows, metric_order,
                seen_baseline_project=sb_project,
                seen_baseline_label=sb_label,
            )
            log.info("plot      : %s", run_dir / "unseen.png")
        except RuntimeError as e:
            log.error("Plot generation failed: %s", e)
            raise SystemExit(str(e)) from e

    log.info("rows_found: %d", len(plot_rows))


if __name__ == "__main__":
    main()

