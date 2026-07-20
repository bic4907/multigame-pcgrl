"""
seen_count_progress.py
======================
unseen 실험 전용:
  unseen 게임 개수별로 grouped bar chart 를 그려, 각 method 의
  unseen 성능과 seen 성능을 비교한다.

레이아웃:
    1행 × N열 (N = unseen 게임 수의 개수, 예: 2,3,4,5 → 4개)
    각 subplot 타이틀: "Unseen = N"
    x = method, y = Progress 점수, 막대 색 = method (범례)

기본 metric: progress 한 가지만 사용한다 (--metrics 로 override 가능).

출력:
    seen_count_progress.png          — unseen 게임 기준 reward_enum 전체 평균
    seen_count_progress_re{N}.png    — (--per-reward-enum 시) reward_enum 별
    seen_progress.png                — seen 게임 기준 reward_enum 전체 평균
                                       (unseen 게임 수 증가에 따른 seen 성능 변화)
    seen_progress_re{N}.png          — (--per-reward-enum 시) reward_enum 별
    all_progress.png                 — seen + unseen 전체 게임 기준 reward_enum 전체 평균
    all_progress_re{N}.png           — (--per-reward-enum 시) reward_enum 별
    seen_count_table.csv             — 집계 데이터
    seen_count_table.md              — Markdown 테이블
    <experiment>_table.csv           — 논문 표 형태의 pivot CSV
    <experiment>_table.tex           — 논문 표 형태의 LaTeX 테이블

사용법:
    python results/utils/experiment/seen_count_progress.py
    python results/utils/experiment/seen_count_progress.py --per-reward-enum
"""

from __future__ import annotations

import argparse
import csv
import math
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
    _load_project_display_names,
    _project_display_name,
)
from scipy import stats as scipy_stats

_CFG = load_cfg()

# progress 에만 ymin 고정 (나머지는 matplotlib 자동)
_YMIN_DEFAULT: float = 0.6
_FIXED_YMIN_METRICS: set[str] = {"progress"}
_DEFAULT_PROGRESS_PROJECT_ORDER: list[str] = [
    "aaai27_eval_ipcgrl_zeroshot",
    "aaai27_eval_mipcgrl_zeroshot",
    "aaai27_eval_vipcgrl_zeroshot",
    "aaai27_eval_mgpcgrl_zeroshot",
    "aaai27_eval_ipcgrl_fewshot",
    "aaai27_eval_mipcgrl_fewshot",
    "aaai27_eval_vipcgrl_fewshot",
    "aaai27_eval_mgpcgrl_fewshot_dw0",
    "aaai27_eval_mgpcgrl_fewshot",
    "aaai27_eval_vipcgrl_unseen",
    "aaai27_eval_mgpcgrl_unseen",
    "aaai27_eval_mgpcgrl_all",
    "aaai27_eval_mgpcgrl_oracle",
]
_DEFAULT_SIGNIFICANCE_BASELINE_PROJECT = "aaai27_eval_vipcgrl_fewshot"
_DEFAULT_SIGNIFICANCE_TARGET_PROJECTS: list[str] = [
    "aaai27_eval_mgpcgrl_fewshot_dw0",
    "aaai27_eval_mgpcgrl_fewshot",
]


def _save_figure_png_pdf(fig, output_path: Path, dpi: int = 200) -> Path:
    """Save a figure as the requested path and as a same-name PDF."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    pdf_path = output_path.with_suffix(".pdf")
    fig.savefig(pdf_path, bbox_inches="tight")
    return pdf_path


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
        unseen_game = "+".join(sorted(unseen_games_list or [])) or "unknown"

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
                    "unseen_game":  unseen_game,
                    "seed":        seed,
                    "metrics":     metric_values,
                })
    return rows


# ---------------------------------------------------------------------------
# 기준선(Baseline) 데이터 수집 및 평균 계산
# ---------------------------------------------------------------------------

def collect_baseline_rows(
    input_root: Path,
    metric_order: list[str],
    baseline_project: str,
) -> list[dict]:
    """특정 project 의 results.csv 만 읽어 row 목록을 반환.

    seen_games 여부와 무관하게 로드하며, n_unseen=0 으로 고정한다.
    CPCGRL 처럼 unseen 게임이 없는 baseline project 에 사용한다.
    """
    rows: list[dict] = []
    _cfg_cache: dict[Path, dict] = {}

    for results_path in iter_results_paths(input_root):
        rel = results_path.relative_to(input_root)
        if len(rel.parts) < 3:
            continue
        project   = rel.parts[0]
        if project != baseline_project:
            continue

        run_name  = rel.parts[1]
        eval_name = rel.parts[2] if len(rel.parts) >= 4 else ""

        eval_tokens = parse_run_tokens(eval_name)
        run_tokens  = parse_run_tokens(run_name)
        seed = run_tokens.get("s", run_name)

        run_dir = results_path.parent
        if run_dir not in _cfg_cache:
            _cfg_cache[run_dir] = load_run_config(run_dir)
        run_cfg = _cfg_cache[run_dir]

        with results_path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
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
                rows.append({
                    "project":     project,
                    "game":        (row.get("game") or "").strip() or "unknown",
                    "reward_enum": reward_enum,
                    "seed":        seed,
                    "metrics":     metric_values,
                })
    return rows


def compute_baseline_mean(
    rows: list[dict],
    metric_order: list[str],
    reward_enum: str | None = None,
) -> dict[str, float]:
    """baseline row 목록에서 metric 별 전체 평균을 반환한다."""
    filtered = rows if reward_enum is None else [
        r for r in rows if r["reward_enum"] == reward_enum
    ]
    result: dict[str, float] = {}
    for metric in metric_order:
        vals = [
            v for r in filtered
            if (v := r["metrics"].get(metric)) is not None
        ]
        if vals:
            result[metric] = sum(vals) / len(vals)
    return result


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


def aggregate_by_n_seen_method(
    rows: list[dict],
    metric_order: list[str],
    game_split: str | None = "unseen",
    reward_enum: str | None = None,
) -> dict[tuple, dict]:
    """(project, n_seen) -> {metric: {mean, std, n}} 집계."""
    filtered = [
        r for r in rows
        if (game_split is None or r.get("game_split") == game_split)
        and (reward_enum is None or r["reward_enum"] == reward_enum)
    ]

    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for r in filtered:
        grouped[(r.get("project", "unknown"), r["n_seen"])].append(r)

    result: dict[tuple, dict] = {}
    for key, group_rows in grouped.items():
        stats: dict = {}
        for metric in metric_order:
            stat = _seed_agg(group_rows, metric)
            if stat:
                stats[metric] = stat
        result[key] = stats
    return result


def aggregate_by_unseen_game_method(
    rows: list[dict],
    metric_order: list[str],
    game_split: str | None = "unseen",
    reward_enum: str | None = None,
) -> dict[tuple, dict]:
    """(project, unseen_game) -> {metric: {mean, std, n}} 집계."""
    filtered = [
        r for r in rows
        if (game_split is None or r.get("game_split") == game_split)
        and (reward_enum is None or r["reward_enum"] == reward_enum)
    ]

    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for r in filtered:
        grouped[(r.get("project", "unknown"), r.get("unseen_game", "unknown"))].append(r)

    result: dict[tuple, dict] = {}
    for key, group_rows in grouped.items():
        stats: dict = {}
        for metric in metric_order:
            stat = _seed_agg(group_rows, metric)
            if stat:
                stats[metric] = stat
        result[key] = stats
    return result


def _ordered_unseen_games(games: set[str]) -> list[str]:
    cfg_order = list(_CFG.get("games", {}).get("colors", {}).keys())
    ordered = [g for g in cfg_order if g in games]
    return ordered + sorted(games - set(ordered))


def _ordered_progress_projects(
    projects: set[str],
    experiment: str | None = None,
) -> list[str]:
    ordered: list[str] = []
    for candidate in _get_experiment_folder_order(experiment):
        if candidate in projects and candidate not in ordered:
            ordered.append(candidate)
    for candidate in _DEFAULT_PROGRESS_PROJECT_ORDER:
        if candidate in projects and candidate not in ordered:
            ordered.append(candidate)
    return ordered + sorted(projects - set(ordered))


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
    hlines: dict[str, dict[str, float]] | None = None,
) -> None:
    """unseen 게임 개수별 grouped bar chart (단일 axes).

    hlines : {label: {metric: value}} — 가로 기준선 (예: {"CPCGRL": {"progress": 0.82}})
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

    # hlines 값도 y 범위에 포함
    _hline_vals = []
    if hlines:
        for _hv in hlines.values():
            v = _hv.get(metric)
            if v is not None:
                _hline_vals.append(v)
    if _hline_vals:
        auto_bottom = min(auto_bottom, min(_hline_vals) * 0.97)
        auto_top    = max(auto_top,    max(_hline_vals) * 1.03)

    if ymin_progress is not None and metric == "progress":
        auto_bottom = ymin_progress

    # ── 단일 axes 에 grouped bar 그리기 ──────────────────────────────────
    n_groups   = len(n_unseen_vals)
    n_methods  = len(projects)
    GROUP_WIDTH = 0.60
    bar_width   = GROUP_WIDTH / n_methods

    fig, ax = plt.subplots(
        figsize=(0.6 * max(1.1 * n_groups + 1.6, 4.0), 2.4),
    )

    x_center = list(range(n_groups))
    y_label = "Progress"   # 고정 y축 레이블

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
                edgecolor="none", alpha=0.9,
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
    ax.set_xticklabels([str(n) for n in n_unseen_vals])
    ax.set_xlabel("# Unseen games", labelpad=6)
    ax.set_xlim(-0.5, n_groups - 0.5)
    ax.set_ylabel(y_label + title_suffix, rotation=90, labelpad=8)
    ax.set_ylim(auto_bottom, auto_top)
    ax.grid(axis="y", alpha=0.3)
    ax.tick_params(axis="x", length=0)

    # ── 기준선 (hlines) ────────────────────────────────────────────────────
    _hline_styles = ["--", "-.", ":"]
    _hline_colors = ["#e41a1c", "#ff7f00", "#4daf4a"]
    if hlines:
        for hi, (hlabel, hvals) in enumerate(hlines.items()):
            v = hvals.get(metric)
            if v is not None:
                _hc = _hline_colors[hi % len(_hline_colors)]
                ax.axhline(
                    v,
                    linestyle=_hline_styles[hi % len(_hline_styles)],
                    color=_hc,
                    linewidth=1.4,
                    alpha=0.85,
                    zorder=5,
                )
                # 플롯 안 오른쪽 끝에 텍스트 표시 (legend 대신)
                ax.text(
                    n_groups - 0.5 - 0.02,
                    v,
                    f"{hlabel}",
                    ha="right", va="bottom",
                    fontsize=7.5, color=_hc,
                    fontweight="bold",
                    zorder=6,
                )

    # 범례 (상단 중앙) — method 막대만 포함
    fig.legend(
        loc="upper center", ncol=min(n_methods, 6), fontsize=10,
        bbox_to_anchor=(0.5, 1.02), frameon=False,
    )
    fig.subplots_adjust(top=0.84)
    fig.tight_layout(rect=(0, 0, 1, 0.90))

    _save_figure_png_pdf(fig, output_path)
    plt.close(fig)


def write_seen_subplot_grid(
    output_path: Path,
    rows: list[dict],
    metric_order: list[str],
    reward_enum: str | None = None,
    title_suffix: str = "",
    ymin_progress: float | None = _YMIN_DEFAULT,
    hlines: dict[str, dict[str, float]] | None = None,
) -> None:
    """seen 게임 기준 Progress 그래프 — unseen 게임 수 증가에 따른 변화.

    x = n_unseen (예: 2, 3, 4, 5)
    y = Progress (Seen game)
    막대 색 = method
    """
    plt = _bar_plot_setup()
    metric = metric_order[0]

    seen_rows = [r for r in rows if r.get("game_split") == "seen"]
    if reward_enum is not None:
        seen_rows = [r for r in seen_rows if r["reward_enum"] == reward_enum]

    n_unseen_vals = sorted({r["n_unseen"] for r in seen_rows})

    _PREFERRED_ORDER = [
        "aaai27_eval_vipcgrl_unseen",
        "aaai27_eval_mgpcgrl_unseen",
        "aaai27_eval_mgpcgrl_all",
        "aaai27_eval_mgpcgrl_oracle",
    ]
    all_projects = {r["project"] for r in seen_rows}
    projects = [p for p in _PREFERRED_ORDER if p in all_projects] + \
               sorted(all_projects - set(_PREFERRED_ORDER))
    if not n_unseen_vals or not projects:
        return

    colors = _project_colors(projects)
    agg = aggregate_by_n_unseen_method(
        rows, metric_order,
        game_split="seen",
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

    _hline_vals = []
    if hlines:
        for _hv in hlines.values():
            v = _hv.get(metric)
            if v is not None:
                _hline_vals.append(v)
    if _hline_vals:
        auto_bottom = min(auto_bottom, min(_hline_vals) * 0.97)
        auto_top    = max(auto_top,    max(_hline_vals) * 1.03)

    if ymin_progress is not None and metric == "progress":
        auto_bottom = ymin_progress

    n_groups   = len(n_unseen_vals)
    n_methods  = len(projects)
    GROUP_WIDTH = 0.60
    bar_width   = GROUP_WIDTH / n_methods

    fig, ax = plt.subplots(
        figsize=(0.6 * max(1.1 * n_groups + 1.6, 4.0), 2.4),
    )

    x_center = list(range(n_groups))
    y_label = "Progress (Seen game)"

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
                edgecolor="none", alpha=0.9,
                label=_project_display_name(proj),
            )
            for bar, m in zip(bars, means):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + (auto_top - auto_bottom) * 0.015,
                    f"{m:.2f}",
                    ha="center", va="bottom",
                    fontsize=6.5, color="black",
                )

    ax.set_xticks(x_center)
    ax.set_xticklabels([str(n) for n in n_unseen_vals])
    ax.set_xlabel("# Unseen games", labelpad=6)
    ax.set_xlim(-0.5, n_groups - 0.5)
    ax.set_ylabel(y_label + title_suffix, rotation=90, labelpad=8)
    ax.set_ylim(auto_bottom, auto_top)
    ax.grid(axis="y", alpha=0.3)
    ax.tick_params(axis="x", length=0)

    # ── 기준선 (hlines) ────────────────────────────────────────────────────
    _hline_styles = ["--", "-.", ":"]
    _hline_colors = ["#e41a1c", "#ff7f00", "#4daf4a"]
    if hlines:
        for hi, (hlabel, hvals) in enumerate(hlines.items()):
            v = hvals.get(metric)
            if v is not None:
                _hc = _hline_colors[hi % len(_hline_colors)]
                ax.axhline(
                    v,
                    linestyle=_hline_styles[hi % len(_hline_styles)],
                    color=_hc,
                    linewidth=1.4,
                    alpha=0.85,
                    zorder=5,
                )
                ax.text(
                    n_groups - 0.5 - 0.02,
                    v,
                    f"{hlabel}",
                    ha="right", va="bottom",
                    fontsize=7.5, color=_hc,
                    fontweight="bold",
                    zorder=6,
                )

    _n_legend = n_methods
    fig.legend(
        loc="upper center", ncol=min(_n_legend, 6), fontsize=10,
        bbox_to_anchor=(0.5, 1.02), frameon=False,
    )
    fig.subplots_adjust(top=0.84)
    fig.tight_layout(rect=(0, 0, 1, 0.90))

    _save_figure_png_pdf(fig, output_path)
    plt.close(fig)


def write_all_subplot_grid(
    output_path: Path,
    rows: list[dict],
    metric_order: list[str],
    reward_enum: str | None = None,
    title_suffix: str = "",
    ymin_progress: float | None = _YMIN_DEFAULT,
    hlines: dict[str, dict[str, float]] | None = None,
) -> None:
    """seen + unseen 전체 게임 기준 Progress 그래프 — unseen 게임 수 증가에 따른 변화.

    hlines : {label: {metric: value}} — 가로 기준선
    """
    plt = _bar_plot_setup()
    metric = metric_order[0]

    # seen + unseen 모두 포함
    all_rows = list(rows)
    if reward_enum is not None:
        all_rows = [r for r in all_rows if r["reward_enum"] == reward_enum]

    n_unseen_vals = sorted({r["n_unseen"] for r in all_rows})

    _PREFERRED_ORDER = [
        "aaai27_eval_vipcgrl_unseen",
        "aaai27_eval_mgpcgrl_unseen",
        "aaai27_eval_mgpcgrl_all",
        "aaai27_eval_mgpcgrl_oracle",
    ]
    all_projects = {r["project"] for r in all_rows}
    projects = [p for p in _PREFERRED_ORDER if p in all_projects] + \
               sorted(all_projects - set(_PREFERRED_ORDER))
    if not n_unseen_vals or not projects:
        return

    colors = _project_colors(projects)
    agg = aggregate_by_n_unseen_method(
        rows, metric_order,
        game_split=None,   # 전체 (seen + unseen)
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

    _hline_vals = []
    if hlines:
        for _hv in hlines.values():
            v = _hv.get(metric)
            if v is not None:
                _hline_vals.append(v)
    if _hline_vals:
        auto_bottom = min(auto_bottom, min(_hline_vals) * 0.97)
        auto_top    = max(auto_top,    max(_hline_vals) * 1.03)

    if ymin_progress is not None and metric == "progress":
        auto_bottom = ymin_progress

    n_groups   = len(n_unseen_vals)
    n_methods  = len(projects)
    GROUP_WIDTH = 0.60
    bar_width   = GROUP_WIDTH / n_methods

    fig, ax = plt.subplots(
        figsize=(0.6 * max(1.1 * n_groups + 1.6, 4.0), 2.4),
    )

    x_center = list(range(n_groups))
    y_label = "Progress (All game)"

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
                edgecolor="none", alpha=0.9,
                label=_project_display_name(proj),
            )
            for bar, m in zip(bars, means):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + (auto_top - auto_bottom) * 0.015,
                    f"{m:.2f}",
                    ha="center", va="bottom",
                    fontsize=6.5, color="black",
                )

    ax.set_xticks(x_center)
    ax.set_xticklabels([str(n) for n in n_unseen_vals])
    ax.set_xlabel("# Unseen games", labelpad=6)
    ax.set_xlim(-0.5, n_groups - 0.5)
    ax.set_ylabel(y_label + title_suffix, rotation=90, labelpad=8)
    ax.set_ylim(auto_bottom, auto_top)
    ax.grid(axis="y", alpha=0.3)
    ax.tick_params(axis="x", length=0)

    # ── 기준선 (hlines) ────────────────────────────────────────────────────
    _hline_styles = ["--", "-.", ":"]
    _hline_colors = ["#e41a1c", "#ff7f00", "#4daf4a"]
    if hlines:
        for hi, (hlabel, hvals) in enumerate(hlines.items()):
            v = hvals.get(metric)
            if v is not None:
                _hc = _hline_colors[hi % len(_hline_colors)]
                ax.axhline(
                    v,
                    linestyle=_hline_styles[hi % len(_hline_styles)],
                    color=_hc,
                    linewidth=1.4,
                    alpha=0.85,
                    zorder=5,
                )
                ax.text(
                    n_groups - 0.5 - 0.02,
                    v,
                    f"{hlabel}",
                    ha="right", va="bottom",
                    fontsize=7.5, color=_hc,
                    fontweight="bold",
                    zorder=6,
                )

    _n_legend = n_methods
    fig.legend(
        loc="upper center", ncol=min(_n_legend, 6), fontsize=10,
        bbox_to_anchor=(0.5, 1.02), frameon=False,
    )
    fig.subplots_adjust(top=0.84)
    fig.tight_layout(rect=(0, 0, 1, 0.90))

    _save_figure_png_pdf(fig, output_path)
    plt.close(fig)


def write_unseen_game_grid(
    output_path: Path,
    rows: list[dict],
    metric_order: list[str],
    game_split: str | None,
    reward_enum: str | None = None,
    ylabel: str = "Progress",
    ymin_progress: float | None = _YMIN_DEFAULT,
    hlines: dict[str, dict[str, float]] | None = None,
    experiment: str | None = None,
) -> None:
    """unseen 게임 이름별 grouped bar chart.

    x = unseen_game, y = metric, bar color = method.
    game_split 이 None 이면 seen + unseen 전체 결과를 집계한다.
    """
    plt = _bar_plot_setup()
    metric = metric_order[0]

    plot_rows = [
        r for r in rows
        if (game_split is None or r.get("game_split") == game_split)
        and (reward_enum is None or r["reward_enum"] == reward_enum)
    ]
    unseen_games = _ordered_unseen_games({
        r.get("unseen_game", "unknown") for r in plot_rows
        if r.get("unseen_game", "unknown") != "unknown"
    })

    all_projects = {r["project"] for r in plot_rows}
    projects = _ordered_progress_projects(all_projects, experiment)
    if not unseen_games or not projects:
        return

    colors = _project_colors(projects)
    agg = aggregate_by_unseen_game_method(
        rows,
        metric_order,
        game_split=game_split,
        reward_enum=reward_enum,
    )

    all_lo, all_hi = [], []
    for game in unseen_games:
        for proj in projects:
            stat = agg.get((proj, game), {}).get(metric)
            if stat is None:
                continue
            m, s = stat["mean"], stat["std"]
            all_lo.append(m - s)
            all_hi.append(m + s)

    if all_lo and all_hi:
        data_min, data_max = min(all_lo), max(all_hi)
        span = max(data_max - data_min, 1e-6)
        pad = span * 0.15
        auto_bottom = max(data_min - pad, 0.0)
        auto_top = data_max + pad
    else:
        auto_bottom, auto_top = 0.0, 1.0

    _hline_vals = []
    if hlines:
        for _hv in hlines.values():
            v = _hv.get(metric)
            if v is not None:
                _hline_vals.append(v)
    if _hline_vals:
        auto_bottom = min(auto_bottom, min(_hline_vals) * 0.97)
        auto_top = max(auto_top, max(_hline_vals) * 1.03)

    if ymin_progress is not None and metric == "progress":
        auto_bottom = ymin_progress

    n_groups = len(unseen_games)
    n_methods = len(projects)
    group_width = 0.68
    bar_width = group_width / n_methods

    fig, ax = plt.subplots(figsize=(0.75 * max(1.25 * n_groups + 2.0, 5.0), 2.7))
    x_center = list(range(n_groups))

    for j, proj in enumerate(projects):
        means, stds, xs = [], [], []
        for k, game in enumerate(unseen_games):
            stat = agg.get((proj, game), {}).get(metric)
            if stat is None:
                continue
            offset = -group_width / 2 + (j + 0.5) * bar_width
            xs.append(x_center[k] + offset)
            means.append(stat["mean"])
            stds.append(stat["std"])

        if means:
            bars = ax.bar(
                xs, means, width=bar_width, yerr=stds, capsize=2.5,
                color=colors[j % len(colors)],
                edgecolor="none", alpha=0.9,
                label=_project_display_name(proj),
            )
            for bar, m in zip(bars, means):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + (auto_top - auto_bottom) * 0.015,
                    f"{m:.2f}",
                    ha="center", va="bottom",
                    fontsize=6.5, color="black",
                )

    ax.set_xticks(x_center)
    ax.set_xticklabels(unseen_games)
    ax.set_xlabel("Unseen game", labelpad=6)
    ax.set_xlim(-0.5, n_groups - 0.5)
    ax.set_ylabel(ylabel, rotation=90, labelpad=8)
    ax.set_ylim(auto_bottom, auto_top)
    ax.grid(axis="y", alpha=0.3)
    ax.tick_params(axis="x", length=0)

    _hline_styles = ["--", "-.", ":"]
    _hline_colors = ["#e41a1c", "#ff7f00", "#4daf4a"]
    if hlines:
        for hi, (hlabel, hvals) in enumerate(hlines.items()):
            v = hvals.get(metric)
            if v is None:
                continue
            _hc = _hline_colors[hi % len(_hline_colors)]
            ax.axhline(
                v,
                linestyle=_hline_styles[hi % len(_hline_styles)],
                color=_hc,
                linewidth=1.4,
                alpha=0.85,
                zorder=5,
            )
            ax.text(
                n_groups - 0.5 - 0.02,
                v,
                f"{hlabel}",
                ha="right", va="bottom",
                fontsize=7.5, color=_hc,
                fontweight="bold",
                zorder=6,
            )

    fig.legend(
        loc="upper center", ncol=min(n_methods, 5), fontsize=9,
        bbox_to_anchor=(0.5, 1.02), frameon=False,
    )
    fig.subplots_adjust(top=0.82)
    fig.tight_layout(rect=(0, 0, 1, 0.90))

    _save_figure_png_pdf(fig, output_path)
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


def _ordered_projects(projects: set[str], experiment: str | None = None) -> list[str]:
    folder_order = _get_experiment_folder_order(experiment)
    ordered = [p for p in folder_order if p in projects]
    return ordered + sorted(projects - set(ordered))


def _format_mean_std(stat: dict | None, decimals: int) -> str:
    if not stat:
        return ""
    return f"{stat['mean']:.{decimals}f} +- {stat['std']:.{decimals}f}"


def _latex_escape(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in value)


def _seed_metric_means(rows: list[dict], metric: str) -> dict[str, float]:
    seed_vals: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        value = row.get("metrics", {}).get(metric)
        if value is None:
            continue
        seed = row.get("seed") or row.get("run", str(id(row)))
        seed_vals[str(seed)].append(value)
    return {
        seed: sum(values) / len(values)
        for seed, values in seed_vals.items()
        if values
    }


def _filter_table_cell_rows(
    rows: list[dict],
    project: str,
    unseen_game: str,
    game_split: str | None,
) -> list[dict]:
    return [
        row for row in rows
        if row.get("project") == project
        and row.get("unseen_game", "unknown") == unseen_game
        and (game_split is None or row.get("game_split") == game_split)
    ]


def _paired_ttest_greater(
    baseline_by_seed: dict[str, float],
    target_by_seed: dict[str, float],
    alternative: str,
) -> dict | None:
    common_seeds = sorted(set(baseline_by_seed) & set(target_by_seed))
    if len(common_seeds) < 2:
        return None

    baseline = [baseline_by_seed[s] for s in common_seeds]
    target = [target_by_seed[s] for s in common_seeds]
    diffs = [t - b for t, b in zip(target, baseline)]
    mean_diff = sum(diffs) / len(diffs)
    if len(diffs) <= 1:
        return None

    diff_mean = mean_diff
    diff_var = sum((d - diff_mean) ** 2 for d in diffs) / (len(diffs) - 1)
    diff_std = math.sqrt(diff_var)
    if diff_std <= 1e-12:
        if abs(diff_mean) <= 1e-12:
            t_stat, p_value = 0.0, 1.0
        elif alternative == "greater":
            t_stat = math.inf if diff_mean > 0 else -math.inf
            p_value = 0.0 if diff_mean > 0 else 1.0
        elif alternative == "less":
            t_stat = math.inf if diff_mean > 0 else -math.inf
            p_value = 1.0 if diff_mean > 0 else 0.0
        else:
            t_stat = math.inf if diff_mean > 0 else -math.inf
            p_value = 0.0
    else:
        test = scipy_stats.ttest_rel(
            target,
            baseline,
            alternative=alternative,
        )
        t_stat = float(test.statistic)
        p_value = float(test.pvalue)

    return {
        "n": len(common_seeds),
        "baseline_mean": sum(baseline) / len(baseline),
        "target_mean": sum(target) / len(target),
        "mean_diff": mean_diff,
        "t": t_stat,
        "p": p_value,
        "effect_dz": mean_diff / diff_std if diff_std > 1e-12 else math.inf,
        "seeds": ",".join(common_seeds),
    }


def _holm_adjust(results: list[dict]) -> None:
    valid = [
        result for result in results
        if result.get("p") is not None and math.isfinite(result["p"])
    ]
    ranked = sorted(valid, key=lambda result: result["p"])
    m = len(ranked)
    running = 0.0
    for idx, result in enumerate(ranked, start=1):
        adjusted = min(1.0, (m - idx + 1) * result["p"])
        running = max(running, adjusted)
        result["p_holm"] = running


def _sig_stars(p_value: float | None, alpha: float = 0.05) -> str:
    if p_value is None or not math.isfinite(p_value) or p_value >= alpha:
        return ""
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    return "*"


def compute_fewshot_significance_tests(
    rows: list[dict],
    metric: str,
    experiment: str | None,
    baseline_project: str = _DEFAULT_SIGNIFICANCE_BASELINE_PROJECT,
    target_projects: list[str] | None = None,
    alternative: str = "greater",
) -> list[dict]:
    target_projects = target_projects or list(_DEFAULT_SIGNIFICANCE_TARGET_PROJECTS)
    relevant_projects = {row["project"] for row in rows}
    if baseline_project not in relevant_projects:
        return []

    unseen_rows = [row for row in rows if row.get("game_split") == "unseen"]
    unseen_games = _ordered_unseen_games({
        row.get("unseen_game", "unknown") for row in unseen_rows
    })
    projects = [
        project for project in _ordered_projects(set(target_projects), experiment)
        if project in relevant_projects
    ]

    split_labels: list[tuple[str, str | None]] = [
        ("unseen", "unseen"),
        ("seen", "seen"),
        ("all", None),
    ]
    results: list[dict] = []
    for project in projects:
        for unseen_game in unseen_games:
            for split_label, split in split_labels:
                baseline_values = _seed_metric_means(
                    _filter_table_cell_rows(rows, baseline_project, unseen_game, split),
                    metric,
                )
                target_values = _seed_metric_means(
                    _filter_table_cell_rows(rows, project, unseen_game, split),
                    metric,
                )
                stat = _paired_ttest_greater(
                    baseline_values,
                    target_values,
                    alternative=alternative,
                )
                if stat is None:
                    stat = {
                        "n": len(set(baseline_values) & set(target_values)),
                        "baseline_mean": None,
                        "target_mean": None,
                        "mean_diff": None,
                        "t": None,
                        "p": None,
                        "effect_dz": None,
                        "seeds": "",
                    }
                stat.update({
                    "baseline_project": baseline_project,
                    "target_project": project,
                    "unseen_game": unseen_game,
                    "split": split_label,
                    "alternative": alternative,
                    "metric": metric,
                })
                results.append(stat)

    _holm_adjust(results)
    for result in results:
        result.setdefault("p_holm", None)
        result["stars"] = _sig_stars(result.get("p_holm"))
    return results


def _significance_lookup(results: list[dict]) -> dict[tuple[str, str, str], dict]:
    return {
        (result["target_project"], result["unseen_game"], result["split"]): result
        for result in results
    }


def _format_p_value(value: float | None) -> str:
    if value is None or not math.isfinite(value):
        return ""
    if value < 0.001:
        return "<0.001"
    return f"{value:.3f}"


def write_fewshot_significance_tests_csv(
    output_path: Path,
    results: list[dict],
) -> None:
    headers = [
        "metric",
        "baseline_project",
        "target_project",
        "unseen_game",
        "split",
        "alternative",
        "n",
        "baseline_mean",
        "target_mean",
        "mean_diff",
        "t",
        "effect_dz",
        "p",
        "p_holm",
        "stars",
        "seeds",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for result in results:
            writer.writerow({key: result.get(key, "") for key in headers})


def write_fewshot_significance_table_markdown(
    output_path: Path,
    rows: list[dict],
    metric_order: list[str],
    experiment: str | None,
    significance_results: list[dict],
    decimals: int = 4,
) -> None:
    metric = metric_order[0]
    unseen_agg = aggregate_by_unseen_game_method(rows, [metric], game_split="unseen")
    seen_agg = aggregate_by_unseen_game_method(rows, [metric], game_split="seen")
    all_agg = aggregate_by_unseen_game_method(rows, [metric], game_split=None)
    unseen_rows = [r for r in rows if r.get("game_split") == "unseen"]
    unseen_games = _ordered_unseen_games({r.get("unseen_game", "unknown") for r in unseen_rows})
    projects = _ordered_projects(
        {r["project"] for r in unseen_rows},
        experiment,
    )
    sig = _significance_lookup(significance_results)

    header_cols = ["method"]
    for game in unseen_games:
        header_cols += [f"{game} / unseen", f"{game} / seen", f"{game} / all"]

    lines = [
        "| " + " | ".join(header_cols) + " |",
        "| " + " | ".join(["---"] * len(header_cols)) + " |",
    ]
    for proj in projects:
        cells = [_project_display_name(proj)]
        for game in unseen_games:
            for split_label, agg in (("unseen", unseen_agg), ("seen", seen_agg), ("all", all_agg)):
                cell = _format_mean_std(
                    agg.get((proj, game), {}).get(metric),
                    decimals,
                )
                test = sig.get((proj, game, split_label))
                if cell and test:
                    stars = test.get("stars", "")
                    p_holm = _format_p_value(test.get("p_holm"))
                    if stars:
                        cell = f"{cell}{stars}"
                    if p_holm:
                        cell = f"{cell} (p={p_holm})"
                cells.append(cell or "-")
        lines.append("| " + " | ".join(cells) + " |")

    lines += [
        "",
        "Paired one-sided t-test on seed-level means, alternative: target > VIPCGRL.",
        "Reported p-values are Holm-adjusted across all MGPCGRL-vs-VIPCGRL cells; * p<0.05, ** p<0.01, *** p<0.001.",
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_fewshot_table_csv(
    output_path: Path,
    rows: list[dict],
    metric_order: list[str],
    experiment: str | None,
    decimals: int = 4,
) -> None:
    metric = metric_order[0]
    unseen_agg = aggregate_by_unseen_game_method(rows, [metric], game_split="unseen")
    seen_agg = aggregate_by_unseen_game_method(rows, [metric], game_split="seen")
    all_agg = aggregate_by_unseen_game_method(rows, [metric], game_split=None)
    unseen_rows = [r for r in rows if r.get("game_split") == "unseen"]
    unseen_games = _ordered_unseen_games({r.get("unseen_game", "unknown") for r in unseen_rows})
    projects = _ordered_projects({r["project"] for r in unseen_rows}, experiment)

    headers = ["method"]
    for game in unseen_games:
        headers += [f"{game}_unseen", f"{game}_seen", f"{game}_all"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for proj in projects:
            rec = {"method": _project_display_name(proj)}
            for game in unseen_games:
                rec[f"{game}_unseen"] = _format_mean_std(
                    unseen_agg.get((proj, game), {}).get(metric),
                    decimals,
                )
                rec[f"{game}_seen"] = _format_mean_std(
                    seen_agg.get((proj, game), {}).get(metric),
                    decimals,
                )
                rec[f"{game}_all"] = _format_mean_std(
                    all_agg.get((proj, game), {}).get(metric),
                    decimals,
                )
            writer.writerow(rec)


def write_fewshot_table_markdown(
    output_path: Path,
    rows: list[dict],
    metric_order: list[str],
    experiment: str | None,
    decimals: int = 4,
) -> None:
    metric = metric_order[0]
    unseen_agg = aggregate_by_unseen_game_method(rows, [metric], game_split="unseen")
    seen_agg = aggregate_by_unseen_game_method(rows, [metric], game_split="seen")
    all_agg = aggregate_by_unseen_game_method(rows, [metric], game_split=None)
    unseen_rows = [r for r in rows if r.get("game_split") == "unseen"]
    unseen_games = _ordered_unseen_games({r.get("unseen_game", "unknown") for r in unseen_rows})
    projects = _ordered_projects({r["project"] for r in unseen_rows}, experiment)

    header_cols = ["method"]
    for game in unseen_games:
        header_cols += [f"{game} / unseen", f"{game} / seen", f"{game} / all"]

    lines = [
        "| " + " | ".join(header_cols) + " |",
        "| " + " | ".join(["---"] * len(header_cols)) + " |",
    ]
    for proj in projects:
        cells = [_project_display_name(proj)]
        for game in unseen_games:
            for agg in (unseen_agg, seen_agg, all_agg):
                cells.append(_format_mean_std(
                    agg.get((proj, game), {}).get(metric),
                    decimals,
                ) or "-")
        lines.append("| " + " | ".join(cells) + " |")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_fewshot_table_latex(
    output_path: Path,
    rows: list[dict],
    metric_order: list[str],
    experiment: str | None,
    decimals: int = 4,
    caption: str = "Few-shot generalization results.",
    label: str = "tab:fewshot",
) -> None:
    metric = metric_order[0]
    metric_label = METRIC_DISPLAY_NAMES.get(metric, metric)
    if caption == "Few-shot generalization results." and experiment == "zeroshot":
        caption = "Zero-shot generalization results."
    unseen_agg = aggregate_by_unseen_game_method(rows, [metric], game_split="unseen")
    seen_agg = aggregate_by_unseen_game_method(rows, [metric], game_split="seen")
    all_agg = aggregate_by_unseen_game_method(rows, [metric], game_split=None)
    unseen_rows = [r for r in rows if r.get("game_split") == "unseen"]
    unseen_games = _ordered_unseen_games({r.get("unseen_game", "unknown") for r in unseen_rows})
    projects = _ordered_projects({r["project"] for r in unseen_rows}, experiment)

    colspec = "l" + "ccc" * len(unseen_games)
    group_header = "Method"
    if unseen_games:
        group_header += " & " + " & ".join(
            rf"\multicolumn{{3}}{{c}}{{{_latex_escape(game)}}}" for game in unseen_games
        )
    group_header += r" \\"
    sub_header = ""
    if unseen_games:
        sub_header = " & " + " & ".join(["Unseen & Seen & All"] * len(unseen_games))
    sub_header += r" \\"
    cmidrule = ""
    if unseen_games:
        cmidrule = " ".join(
            rf"\cmidrule(lr){{{2 + 3 * i}-{4 + 3 * i}}}"
            for i in range(len(unseen_games))
        )

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{{_latex_escape(caption)}}}",
        rf"\label{{{_latex_escape(label)}}}",
        rf"\begin{{tabular}}{{{colspec}}}",
        r"\toprule",
        group_header,
        cmidrule,
        sub_header,
        r"\midrule",
    ]
    for proj in projects:
        cells = [_latex_escape(_project_display_name(proj))]
        for game in unseen_games:
            for agg in (unseen_agg, seen_agg, all_agg):
                stat = agg.get((proj, game), {}).get(metric)
                if stat:
                    cells.append(
                        rf"{stat['mean']:.{decimals}f}\std{{{stat['std']:.{decimals}f}}}"
                    )
                else:
                    cells.append("-")
        lines.append(" & ".join(cells) + r" \\")

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        rf"\vspace{{2pt}}\footnotesize{{Metric: {_latex_escape(metric_label)}.}}",
        r"\end{table}",
    ]

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
    parser.add_argument("--decimals", type=int, default=3)
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
        "--baseline-project",
        default=None,
        metavar="PROJECT",
        help=(
            "가로 기준선으로 표시할 project "
            "(기본: config의 progress_baseline_project 또는 re_oracle_project). "
            "'none' 으로 지정하면 기준선을 그리지 않는다."
        ),
    )
    parser.add_argument(
        "--baseline-label",
        default=None,
        metavar="LABEL",
        help="기준선 범례 라벨 (기본: project_display_name 사용)",
    )
    parser.add_argument(
        "--experiment",
        choices=_exp_names if _exp_names else None,
        default="unseen",
        metavar="EXPERIMENT",
    )
    parser.add_argument(
        "--no-significance",
        action="store_true",
        help="fewshot significance table/test CSV 생성을 생략한다.",
    )
    parser.add_argument(
        "--significance-baseline-project",
        default=_DEFAULT_SIGNIFICANCE_BASELINE_PROJECT,
        metavar="PROJECT",
        help="significance 비교 기준 project (기본: VIPCGRL fewshot).",
    )
    parser.add_argument(
        "--significance-target-projects",
        nargs="+",
        default=list(_DEFAULT_SIGNIFICANCE_TARGET_PROJECTS),
        metavar="PROJECT",
        help="baseline 대비 유의성 검정 대상 project 목록.",
    )
    parser.add_argument(
        "--significance-alternative",
        choices=["greater", "less", "two-sided"],
        default="greater",
        help="paired t-test alternative. 기본 greater는 target > baseline 검정.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args    = parse_args()
    run_dir = make_run_dir("progress", cfg=_CFG)
    log     = setup_logger(run_dir, name=__file__)
    log.debug("run_dir   : %s", run_dir)

    experiment   = args.experiment or "unseen"
    folder_order = _get_experiment_folder_order(experiment)
    log.info("experiment: %s  folder_order=%s", experiment, folder_order)

    import utils.experiment.benchmark as _bm
    _bm._PROJECT_DISPLAY_NAMES = _load_project_display_names(experiment)

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

    # ── 기준선(Baseline) 계산 ─────────────────────────────────────────────
    # config의 re_oracle_project (기본 aaai27_eval_cpcgrl) 데이터를 읽어
    # metric 별 전체 평균을 구하고 가로선으로 표시한다.
    hlines: dict[str, dict[str, float]] = {}
    exp_cfg = _CFG.get("experiments", {}).get(experiment, {})
    baseline_project = args.baseline_project
    if baseline_project is None:
        baseline_project = exp_cfg.get(
            "progress_baseline_project",
            exp_cfg.get("re_oracle_project", "aaai27_eval_cpcgrl"),
        )
    _bp = (baseline_project or "").strip().lower()
    if _bp and _bp != "none":
        baseline_proj = baseline_project.strip()
        baseline_label = (
            args.baseline_label
            or _project_display_name(baseline_proj)
        )
        log.info("baseline  : %s  (label=%s)", baseline_proj, baseline_label)
        _brows_raw = collect_baseline_rows(input_root, metric_order, baseline_proj)
        if _brows_raw:
            _brows_norm = apply_normalization(_brows_raw, norm_scale, metric_order)
            _bmean = compute_baseline_mean(_brows_norm, metric_order)
            if _bmean:
                hlines[baseline_label] = _bmean
                log.info("baseline mean: %s", _bmean)
            else:
                log.warning("baseline 데이터에서 metric 값을 찾지 못했습니다.")
        else:
            log.warning("baseline project '%s' 데이터가 없습니다.", baseline_proj)
    else:
        log.info("baseline  : (없음)")

    # ── 테이블 ────────────────────────────────────────────────────────────
    write_table_csv(
        run_dir / "progress_table.csv", norm_rows, metric_order
    )
    write_table_markdown(
        run_dir / "progress_table.md", norm_rows, metric_order, args.decimals
    )
    table_prefix = f"{experiment}_table" if experiment else "progress_table"
    write_fewshot_table_csv(
        run_dir / f"{table_prefix}.csv",
        norm_rows,
        metric_order,
        experiment=experiment,
        decimals=args.decimals,
    )
    write_fewshot_table_markdown(
        run_dir / f"{table_prefix}.md",
        norm_rows,
        metric_order,
        experiment=experiment,
        decimals=args.decimals,
    )
    write_fewshot_table_latex(
        run_dir / f"{table_prefix}.tex",
        norm_rows,
        metric_order,
        experiment=experiment,
        decimals=args.decimals,
    )
    significance_results: list[dict] = []
    if not args.no_significance and metric_order:
        significance_results = compute_fewshot_significance_tests(
            norm_rows,
            metric=metric_order[0],
            experiment=experiment,
            baseline_project=args.significance_baseline_project,
            target_projects=args.significance_target_projects,
            alternative=args.significance_alternative,
        )
        if significance_results:
            write_fewshot_significance_tests_csv(
                run_dir / f"{table_prefix}_significance_tests.csv",
                significance_results,
            )
            write_fewshot_significance_table_markdown(
                run_dir / f"{table_prefix}_significance_table.md",
                norm_rows,
                metric_order,
                experiment=experiment,
                significance_results=significance_results,
                decimals=args.decimals,
            )
    log.info("table     : %s", run_dir / "progress_table.md")
    log.info("table     : %s", run_dir / f"{table_prefix}.csv")
    log.info("table     : %s", run_dir / f"{table_prefix}.md")
    log.info("table     : %s", run_dir / f"{table_prefix}.tex")
    if significance_results:
        log.info("table     : %s", run_dir / f"{table_prefix}_significance_tests.csv")
        log.info("table     : %s", run_dir / f"{table_prefix}_significance_table.md")

    # ── 플롯 ──────────────────────────────────────────────────────────────
    _hl = hlines or None
    if not args.no_plot:
        by_unseen_game = all(r.get("n_unseen") == 1 for r in norm_rows)
        try:
            if by_unseen_game:
                write_unseen_game_grid(
                    run_dir / "unseen_by_game.png",
                    norm_rows, metric_order,
                    game_split="unseen",
                    ylabel="Progress (Unseen game)",
                    ymin_progress=args.ymin,
                    hlines=_hl,
                    experiment=experiment,
                )
                log.info("plot (unseen by game): %s", run_dir / "unseen_by_game.png")
            else:
                write_subplot_grid(
                    run_dir / "unseen.png",
                    norm_rows, metric_order,
                    ymin_progress=args.ymin,
                    hlines=_hl,
                )
                log.info("plot (unseen): %s", run_dir / "unseen.png")
        except RuntimeError as e:
            log.error("Plot generation failed: %s", e)
            raise SystemExit(str(e)) from e

        # ── seen 게임 progress ────────────────────────────────────────────
        has_seen = any(r.get("game_split") == "seen" for r in norm_rows)
        if has_seen:
            try:
                if by_unseen_game:
                    write_unseen_game_grid(
                        run_dir / "seen_by_game.png",
                        norm_rows, metric_order,
                        game_split="seen",
                        ylabel="Progress (Seen games)",
                        ymin_progress=args.ymin,
                        hlines=_hl,
                        experiment=experiment,
                    )
                    log.info("plot (seen by game): %s", run_dir / "seen_by_game.png")
                else:
                    write_seen_subplot_grid(
                        run_dir / "seen.png",
                        norm_rows, metric_order,
                        ymin_progress=args.ymin,
                        hlines=_hl,
                    )
                    log.info("plot (seen): %s", run_dir / "seen.png")
            except RuntimeError as e:
                log.error("Plot (seen) generation failed: %s", e)
        else:
            log.warning("seen 게임 데이터가 없어 seen.png 를 생략합니다.")

        # ── 전체 게임 progress (seen + unseen 합산) ───────────────────────
        try:
            if by_unseen_game:
                write_unseen_game_grid(
                    run_dir / "all_by_game.png",
                    norm_rows, metric_order,
                    game_split=None,
                    ylabel="Progress (All games)",
                    ymin_progress=args.ymin,
                    hlines=_hl,
                    experiment=experiment,
                )
                log.info("plot (all by game): %s", run_dir / "all_by_game.png")
            else:
                write_all_subplot_grid(
                    run_dir / "all.png",
                    norm_rows, metric_order,
                    ymin_progress=args.ymin,
                    hlines=_hl,
                )
                log.info("plot (all): %s", run_dir / "all.png")
        except RuntimeError as e:
            log.error("Plot (all) generation failed: %s", e)


    log.info(
        "rows_found: %d  (unseen: %d  seen: %d)",
        len(rows),
        sum(1 for r in rows if r.get("game_split") == "unseen"),
        sum(1 for r in rows if r.get("game_split") == "seen"),
    )


if __name__ == "__main__":
    main()
