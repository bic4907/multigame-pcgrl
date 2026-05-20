"""
seen_count_progress.py
======================
unseen 실험 전용:
  unseen 게임 개수별로 subplot 을 그려, 각 method 의 unseen 성능을 비교한다.

레이아웃:
    1행 × N열 (N = unseen 게임 수의 개수, 예: 2,3,4,5 → 4개)
    각 subplot 타이틀: "Unseen = N"
    x = method, y = Progress 점수, 막대 색 = method (범례)

기본 metric: progress 한 가지만 사용한다 (--metrics 로 override 가능).

출력:
    seen_count_progress.png          — reward_enum 전체 평균
    seen_count_progress_re{N}.png    — (--per-reward-enum 시) reward_enum 별
    seen_count_table.csv             — 집계 데이터
    seen_count_table.md              — Markdown 테이블

사용법:
    python results/utils/experiment/seen_count_progress.py
    python results/utils/experiment/seen_count_progress.py --per-reward-enum
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
_ROOT        = _RESULTS_DIR.parent              # project root
if str(_RESULTS_DIR) not in _sys.path:
    _sys.path.insert(0, str(_RESULTS_DIR))
if str(_ROOT) not in _sys.path:
    _sys.path.append(str(_ROOT))

from utils.core.run_output import load_cfg, make_run_dir, setup_logger
from utils.core.io import (
    normalize_reward_enum,
    parse_run_tokens,
    iter_results_paths,
    sort_key_reward_enum,
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
    _project_colors,
    _seed_agg,
    _YMIN,
    DEFAULT_METRIC_ORDER,
    METRIC_DISPLAY_NAMES,
    resolve_input_root,
    _get_experiment_folder_order,
    _project_display_name,
)

_CFG = load_cfg()

# progress 에만 ymin 고정 (나머지는 matplotlib 자동)
_YMIN_DEFAULT: float = 0.6
_FIXED_YMIN_METRICS: set[str] = {"progress"}


# ---------------------------------------------------------------------------
# 데이터 수집 — n_seen 필드 포함
# ---------------------------------------------------------------------------

def collect_rows_with_seen_count(
    input_root: Path,
    metric_order: list[str],
    target_projects: list[str] | None = None,
) -> list[dict]:
    """results.csv 를 순회하며 n_seen 필드를 포함한 row 목록을 반환.

    n_seen : run_config["seen_games"] 크기
             없으면 run_config["unseen_games"] 로 역산을 시도하되 총 게임 수를
             알 수 없는 경우 0 으로 기록한다.
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

        # ── n_seen / n_unseen ─────────────────────────────────────────────
        seen_games_list   = run_cfg.get("seen_games")
        unseen_games_list = run_cfg.get("unseen_games")
        if seen_games_list is None:
            # seen_games 없을 때 대체 수단 없음 → 건너뜀
            continue
        n_seen   = len(seen_games_list)
        n_unseen = len(unseen_games_list) if unseen_games_list is not None else 0

        with results_path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                game = (row.get("game") or "").strip() or "unknown"
                re_raw = (row.get("reward_enum") or "").strip()
                reward_enum = normalize_reward_enum(
                    re_raw or eval_tokens.get("re", "unknown")
                )
                metric_values = {
                    m: v for m in metric_order
                    if (v := to_float(row.get(m))) is not None
                }
                if not metric_values:
                    continue

                game_split = get_game_split(game, run_cfg)
                rows.append({
                    "project":     project,
                    "game":        game,
                    "reward_enum": reward_enum,
                    "game_split":  game_split,
                    "n_seen":      n_seen,
                    "n_unseen":    n_unseen,
                    "seed":        seed,
                    "metrics":     metric_values,
                })
    return rows


# ---------------------------------------------------------------------------
# 집계
# ---------------------------------------------------------------------------

def aggregate_by_n_unseen_method(
    rows: list[dict],
    metric_order: list[str],
    game_split: str = "unseen",
    reward_enum: str | None = None,
) -> dict[tuple, dict]:
    """(project, n_unseen) → {metric: {mean, std, n}} 집계."""
    filtered = [
        r for r in rows
        if (game_split is None or r.get("game_split") == game_split)
        and (reward_enum is None or r["reward_enum"] == reward_enum)
    ]

    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for r in filtered:
        grouped[(r.get("project", "unknown"), r["n_unseen"])].append(r)

    result: dict[tuple, dict] = {}
    for key, group_rows in grouped.items():
        stats: dict = {}
        for metric in metric_order:
            stat = _seed_agg(group_rows, metric)
            if stat:
                stats[metric] = stat
        result[key] = stats
    return result


# ---------------------------------------------------------------------------
# Subplot 그리드 플롯 — 1행 × n_unseen, 각 subplot 타이틀 = "Unseen = N"
# ---------------------------------------------------------------------------

def write_subplot_grid(
    output_path: Path,
    rows: list[dict],
    metric_order: list[str],
    reward_enum: str | None = None,
    title_suffix: str = "",
    ymin_progress: float = _YMIN_DEFAULT,
) -> None:
    """unseen 게임 개수별 grouped bar chart (단일 axes).

    x = n_unseen (예: 2, 3, 4, 5)
    각 그룹 안에서 method 별 막대 (색 구분), 안쪽 간격 없이 붙여서 표시
    범례: method → 색
    """
    plt = _bar_plot_setup()
    metric = metric_order[0]

    unseen_rows = [r for r in rows if r.get("game_split") == "unseen"]
    if reward_enum is not None:
        unseen_rows = [r for r in unseen_rows if r["reward_enum"] == reward_enum]

    n_unseen_vals = sorted({r["n_unseen"] for r in unseen_rows})

    # ── method 순서 고정: 지정 순서 → 나머지 정렬 ──────────────────────────
    _PREFERRED_ORDER = [
        "aaai27_eval_vipcgrl_unseen",
        "aaai27_eval_mgpcgrl_unseen",
        "aaai27_eval_mgpcgrl_all",
        "aaai27_eval_mgpcgrl_oracle",
    ]
    all_projects = {r["project"] for r in unseen_rows}
    projects = [p for p in _PREFERRED_ORDER if p in all_projects] + \
               sorted(all_projects - set(_PREFERRED_ORDER))
    if not n_unseen_vals or not projects:
        return

    colors = _project_colors(projects)
    agg = aggregate_by_n_unseen_method(
        rows, metric_order,
        game_split="unseen",
        reward_enum=reward_enum,
    )

    # ── 동적 y축 범위 계산 ────────────────────────────────────────────────
    all_lo, all_hi = [], []
    for n_unseen in n_unseen_vals:
        for proj in projects:
            stat = agg.get((proj, n_unseen), {}).get(metric)
            if stat is None:
                continue
            m, s = stat["mean"], stat["std"]
            all_lo.append(m - s)
            all_hi.append(m + s)

    if all_lo and all_hi:
        data_min, data_max = min(all_lo), max(all_hi)
        span = max(data_max - data_min, 1e-6)
        pad  = span * 0.15
        auto_bottom = max(data_min - pad, 0.0)
        auto_top    = data_max + pad
    else:
        auto_bottom, auto_top = 0.0, 1.0

    if ymin_progress is not None and metric == "progress":
        auto_bottom = ymin_progress

    # ── 단일 axes 에 grouped bar 그리기 ──────────────────────────────────
    n_groups   = len(n_unseen_vals)
    n_methods  = len(projects)
    GROUP_WIDTH = 0.60
    bar_width   = GROUP_WIDTH / n_methods

    fig, ax = plt.subplots(
        figsize=(max(1.1 * n_groups + 1.6, 4.0), 2.4),
    )

    x_center = list(range(n_groups))
    y_label = "Progress (Unseen game)"   # 고정 y축 레이블

    for j, proj in enumerate(projects):
        means, stds, xs = [], [], []
        for k, n_unseen in enumerate(n_unseen_vals):
            stat = agg.get((proj, n_unseen), {}).get(metric)
            if stat is None:
                continue
            offset = -GROUP_WIDTH / 2 + (j + 0.5) * bar_width
            xs.append(x_center[k] + offset)
            means.append(stat["mean"])
            stds.append(stat["std"])

        if means:
            bars = ax.bar(
                xs, means, width=bar_width, yerr=stds, capsize=2.5,
                color=colors[j % len(colors)],
                edgecolor="white", linewidth=0.6, alpha=0.9,
                label=_project_display_name(proj),
            )
            # 막대 위에 숫자 표시
            for bar, m in zip(bars, means):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + (auto_top - auto_bottom) * 0.015,
                    f"{m:.2f}",
                    ha="center", va="bottom",
                    fontsize=6.5, color="black",
                )

    ax.set_xticks(x_center)
    ax.set_xticklabels([f"Unseen = {n}" for n in n_unseen_vals])
    ax.set_xlim(-0.5, n_groups - 0.5)
    ax.set_ylabel(y_label + title_suffix, rotation=90, labelpad=8)
    ax.set_ylim(auto_bottom, auto_top)
    ax.grid(axis="y", alpha=0.3)
    ax.tick_params(axis="x", length=0)

    # 범례 (상단 중앙)
    fig.legend(
        loc="upper center", ncol=min(n_methods, 6), fontsize=10,
        bbox_to_anchor=(0.5, 1.02), frameon=False,
    )
    fig.subplots_adjust(top=0.84)
    fig.tight_layout(rect=(0, 0, 1, 0.90))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 테이블 출력
# ---------------------------------------------------------------------------

def write_table_csv(
    output_path: Path,
    rows: list[dict],
    metric_order: list[str],
) -> None:
    agg = aggregate_by_n_unseen_method(rows, metric_order, game_split="unseen")
    unseen_rows   = [r for r in rows if r.get("game_split") == "unseen"]
    projects      = sorted({r["project"] for r in unseen_rows})
    n_unseen_vals = sorted({r["n_unseen"] for r in unseen_rows})

    headers = ["n_unseen", "method"]
    for m in metric_order:
        headers += [f"{m}_mean", f"{m}_std", f"{m}_n"]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for n in n_unseen_vals:
            for proj in projects:
                rec: dict = {"n_unseen": n, "method": _project_display_name(proj)}
                stats = agg.get((proj, n), {})
                for m in metric_order:
                    stat = stats.get(m)
                    rec[f"{m}_mean"] = stat["mean"] if stat else ""
                    rec[f"{m}_std"]  = stat["std"]  if stat else ""
                    rec[f"{m}_n"]    = stat["n"]    if stat else 0
                writer.writerow(rec)


def write_table_markdown(
    output_path: Path,
    rows: list[dict],
    metric_order: list[str],
    decimals: int = 4,
) -> None:
    agg = aggregate_by_n_unseen_method(rows, metric_order, game_split="unseen")
    unseen_rows   = [r for r in rows if r.get("game_split") == "unseen"]
    projects      = sorted({r["project"] for r in unseen_rows})
    n_unseen_vals = sorted({r["n_unseen"] for r in unseen_rows})

    header_cols = ["n_unseen", "method"] + metric_order
    lines = [
        "| " + " | ".join(header_cols) + " |",
        "| " + " | ".join(["---"] * len(header_cols)) + " |",
    ]
    for n in n_unseen_vals:
        for proj in projects:
            stats = agg.get((proj, n), {})
            cells = [str(n), _project_display_name(proj)]
            for m in metric_order:
                stat = stats.get(m)
                cells.append(
                    f"{stat['mean']:.{decimals}f} ± {stat['std']:.{decimals}f}"
                    if stat else "-"
                )
            lines.append("| " + " | ".join(cells) + " |")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    _exp_names = list(_CFG.get("experiments", {}).keys())
    parser = argparse.ArgumentParser(
        description="unseen 게임 개수별 subplot — method 간 unseen 성능 비교"
    )
    parser.add_argument("--input", default="wandb_projects",
                        help="wandb_projects 루트 디렉토리")
    parser.add_argument("--metrics", nargs="+", default=None,
                        help="사용할 metric 목록 (기본: config.json default_order)")
    parser.add_argument("--decimals", type=int, default=4)
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument(
        "--per-reward-enum", action="store_true",
        help="reward_enum 별 개별 플롯도 추가 생성",
    )
    parser.add_argument(
        "--ymin", type=float, default=None,
        help="progress y축 하한값. 기본값 None → 데이터 기반 자동 결정. "
             "값을 지정하면 progress subplot의 하한을 강제 override.",
    )
    parser.add_argument(
        "--experiment",
        choices=_exp_names if _exp_names else None,
        default="unseen",
        metavar="EXPERIMENT",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args    = parse_args()
    run_dir = make_run_dir("seen_count_progress", cfg=_CFG)
    log     = setup_logger(run_dir, name=__file__)
    log.debug("run_dir   : %s", run_dir)

    experiment   = args.experiment or "unseen"
    folder_order = _get_experiment_folder_order(experiment)
    log.info("experiment: %s  folder_order=%s", experiment, folder_order)

    input_root = resolve_input_root(args.input, _RESULTS_DIR)
    log.info("input_root: %s", input_root)

    metric_order: list[str] = args.metrics or ["progress"]

    rows = collect_rows_with_seen_count(
        input_root, metric_order,
        target_projects=folder_order or None,
    )
    if not rows:
        msg = "No valid rows found — seen_games 정보가 없거나 결과 파일이 없습니다."
        log.error(msg)
        raise SystemExit(msg)

    has_unseen = any(r.get("game_split") == "unseen" for r in rows)
    if not has_unseen:
        msg = "unseen 게임 데이터가 없습니다 — run_config.json 의 unseen_games 를 확인하세요."
        log.error(msg)
        raise SystemExit(msg)

    # Normalization
    _global_scale_path = os.environ.get("PIPELINE_NORM_SCALE")
    if _global_scale_path and Path(_global_scale_path).is_file():
        norm_scale = load_normalization_scale(Path(_global_scale_path))
        log.info("norm_scale (global): %s", _global_scale_path)
    else:
        norm_scale = compute_normalization_scale(rows, metric_order)
        scale_path = run_dir / "normalization_scale.json"
        save_normalization_scale(norm_scale, scale_path)
        log.info("norm_scale (local) : %s", scale_path)
    norm_rows = apply_normalization(rows, norm_scale, metric_order)

    # ── 테이블 ────────────────────────────────────────────────────────────
    write_table_csv(
        run_dir / "seen_count_table.csv", norm_rows, metric_order
    )
    write_table_markdown(
        run_dir / "seen_count_table.md", norm_rows, metric_order, args.decimals
    )
    log.info("table     : %s", run_dir / "seen_count_table.md")

    # ── 플롯 ──────────────────────────────────────────────────────────────
    if not args.no_plot:
        try:
            write_subplot_grid(
                run_dir / "seen_count_progress.png",
                norm_rows, metric_order,
                ymin_progress=args.ymin,
            )
            log.info("plot (all re): %s", run_dir / "seen_count_progress.png")
        except RuntimeError as e:
            log.error("Plot generation failed: %s", e)
            raise SystemExit(str(e)) from e

        if args.per_reward_enum:
            re_vals = sorted(
                {r["reward_enum"] for r in norm_rows if r.get("game_split") == "unseen"},
                key=sort_key_reward_enum,
            )
            for re in re_vals:
                try:
                    write_subplot_grid(
                        run_dir / f"seen_count_progress_re{re}.png",
                        norm_rows, metric_order,
                        reward_enum=re, title_suffix=f" (re={re})",
                        ymin_progress=args.ymin,
                    )
                    log.info("plot (re=%s): %s", re,
                             run_dir / f"seen_count_progress_re{re}.png")
                except RuntimeError as e:
                    log.error("Plot re=%s failed: %s", re, e)

    log.info(
        "rows_found: %d  (unseen: %d)",
        len(rows),
        sum(1 for r in rows if r.get("game_split") == "unseen"),
    )


if __name__ == "__main__":
    main()

