"""
unseen_count_progress.py
========================
unseen_generalizability 실험 전용:
  unseen 게임 수(n_unseen_games)가 늘어날 때 성능이 어떻게 변화하는지를
  꺾은선 그래프로 시각화한다.

x 축: 학습에서 제외한 unseen 게임 수 (1, 2, 3, 4 …)
y 축: unseen 게임 성능 (정규화된 metric)
선  : train_seen_ratio 별 (또는 --line-key project 로 project 별)

출력:
    unseen_count_progress.png          — 전체 reward_enum 평균 꺾은선 그래프
    unseen_count_progress_re{N}.png    — (--per-reward-enum 시) reward_enum 별 그래프
    unseen_count_table.csv             — 집계 데이터
    unseen_count_table.md              — Markdown 테이블

사용법:
    python results/utils/experiment/unseen_count_progress.py
    python results/utils/experiment/unseen_count_progress.py --per-reward-enum
    python results/utils/experiment/unseen_count_progress.py --line-key project
    python results/utils/experiment/unseen_count_progress.py --experiment unseen_generalizability
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
    _seed_agg,
    _YMIN,
    DEFAULT_METRIC_ORDER,
    METRIC_DISPLAY_NAMES,
    resolve_input_root,
    _get_experiment_folder_order,
    _project_display_name,
)

_CFG = load_cfg()

# progress 메트릭에만 적용할 y축 하한 (차이를 잘 보이게)
# 나머지 메트릭(vit_score, tpkldiv, diversity)은 matplotlib 자동 결정
_YMIN_DEFAULT: float = 0.6
_FIXED_YMIN_METRICS: set[str] = {"progress"}   # 이 메트릭에만 ymin 적용


# ---------------------------------------------------------------------------
# run_name / run_config 에서 메타 정보 파싱
# ---------------------------------------------------------------------------

def _parse_n_unseen_from_name(run_name: str) -> int | None:
    """'unseen-zddmsk' 토큰을 2글자 게임 코드 단위로 분리해 unseen 게임 수 반환."""
    tokens = parse_run_tokens(run_name)
    raw = tokens.get("unseen", "")
    if not raw:
        return None
    # 각 게임 약어가 정확히 2글자이므로 전체 길이 // 2
    if len(raw) % 2 != 0:
        return None
    return len(raw) // 2


def _parse_seen_ratio_from_name(run_name: str) -> float | None:
    """'sr-0.01' 토큰에서 seen_ratio(float)를 파싱."""
    tokens = parse_run_tokens(run_name)
    raw = tokens.get("sr")
    if raw is None:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# 데이터 수집
# ---------------------------------------------------------------------------

def collect_rows_with_unseen_count(
    input_root: Path,
    metric_order: list[str],
    target_projects: list[str] | None = None,
) -> list[dict]:
    """results.csv 를 순회하며 n_unseen / train_seen_ratio 필드를 포함한 row 목록 반환.

    n_unseen        : run_config["unseen_games"] 크기 → 없으면 run_name sr 토큰 파싱
    train_seen_ratio: run_config["train_seen_ratio"] → 없으면 run_name sr 토큰 파싱
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

        # ── n_unseen ──────────────────────────────────────────────────────
        unseen_games_list = run_cfg.get("unseen_games")
        if unseen_games_list is not None:
            n_unseen = len(unseen_games_list)
        else:
            n_unseen = _parse_n_unseen_from_name(run_name) or 0

        # ── train_seen_ratio ──────────────────────────────────────────────
        train_seen_ratio = run_cfg.get("train_seen_ratio")
        if train_seen_ratio is None:
            train_seen_ratio = _parse_seen_ratio_from_name(run_name)
        if train_seen_ratio is None:
            train_seen_ratio = 1.0

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
                    "project":          project,
                    "game":             game,
                    "reward_enum":      reward_enum,
                    "game_split":       game_split,
                    "n_unseen":         n_unseen,
                    "train_seen_ratio": train_seen_ratio,
                    "seed":             seed,
                    "metrics":          metric_values,
                })
    return rows


# ---------------------------------------------------------------------------
# 집계
# ---------------------------------------------------------------------------

def _get_line_fn(line_key: str):
    """line_key 문자열 → row에서 선 구분 값을 추출하는 함수."""
    if line_key == "project_seen_ratio":
        return lambda r: (r.get("project", "unknown"), float(r.get("train_seen_ratio", 1.0)))
    return lambda r: r.get(line_key, "unknown")


def _line_label(line_key: str, val) -> str:
    """선 구분 값 → legend 라벨 문자열."""
    if line_key == "project_seen_ratio":
        project, sr = val
        return f"{_project_display_name(project)} sr={sr:.2g}"
    if line_key == "train_seen_ratio":
        return f"sr={float(val):.2g}"
    return str(val)


def aggregate_by_n_unseen(
    rows: list[dict],
    metric_order: list[str],
    line_key: str = "train_seen_ratio",
    game_split: str = "unseen",
    reward_enum: str | None = None,
) -> dict[tuple, dict]:
    """(line_key_value, n_unseen) → {metric: {mean, std, n}} 집계.

    Parameters
    ----------
    rows        : collect_rows_with_unseen_count 결과
    line_key    : 선 구분 기준 ("train_seen_ratio" 또는 "project")
    game_split  : "unseen" | "seen" | None (None = 모두)
    reward_enum : None 이면 전 reward_enum 평균
    """
    filtered = [
        r for r in rows
        if (game_split is None or r.get("game_split") == game_split)
        and (reward_enum is None or r["reward_enum"] == reward_enum)
    ]

    line_fn = _get_line_fn(line_key)
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for r in filtered:
        key_val = line_fn(r)
        grouped[(key_val, r["n_unseen"])].append(r)

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
# 꺾은선 그래프
# ---------------------------------------------------------------------------

def write_line_plot(
    output_path: Path,
    rows: list[dict],
    metric_order: list[str],
    line_key: str = "project_seen_ratio",
    reward_enum: str | None = None,
    title_suffix: str = "",
    ymin: float = _YMIN_DEFAULT,
) -> None:
    """unseen 게임 수 vs metric 꺾은선 그래프 (오차 밴드 포함)."""
    plt = _bar_plot_setup()

    unseen_rows = [r for r in rows if r.get("game_split") == "unseen"]
    n_unseen_vals = sorted({r["n_unseen"] for r in unseen_rows})
    if not n_unseen_vals:
        return

    line_fn   = _get_line_fn(line_key)
    line_vals = sorted({line_fn(r) for r in unseen_rows})
    colors    = _palette(len(line_vals))

    agg = aggregate_by_n_unseen(
        rows, metric_order,
        line_key=line_key,
        game_split="unseen",
        reward_enum=reward_enum,
    )

    n_metrics = len(metric_order)
    fig, axes = plt.subplots(
        1, n_metrics,
        figsize=(3.8 * n_metrics + 1.5, 3.5),
        squeeze=False,
    )

    for ci, metric in enumerate(metric_order):
        ax = axes[0][ci]
        metric_label = METRIC_DISPLAY_NAMES.get(metric, metric)
        ax.set_title(metric_label + title_suffix)
        drew_any, y_uppers = False, []

        for i, lv in enumerate(line_vals):
            xs, means, stds = [], [], []
            for n in n_unseen_vals:
                stat = agg.get((lv, n), {}).get(metric)
                if stat:
                    xs.append(n)
                    means.append(stat["mean"])
                    stds.append(stat["std"])
                    y_uppers.append(stat["mean"] + stat["std"])

            if not means:
                continue
            drew_any = True
            color = colors[i % len(colors)]
            label = _line_label(line_key, lv)
            ax.plot(xs, means, marker="o", label=label, color=color, linewidth=1.8, zorder=3)
            ax.fill_between(
                xs,
                [m - s for m, s in zip(means, stds)],
                [m + s for m, s in zip(means, stds)],
                alpha=0.15, color=color,
            )

        ax.set_xlabel("# unseen games")
        if ci == 0:
            ax.set_ylabel("Score", rotation=90, labelpad=8)
        ax.set_xticks(n_unseen_vals)
        ax.grid(axis="y", alpha=0.3)

        if drew_any and y_uppers:
            dm  = max(y_uppers)
            pad = max(dm, 1e-6) * 0.15
            # progress 등 지정 메트릭만 ymin 적용, 나머지는 matplotlib 자동
            if metric in _FIXED_YMIN_METRICS:
                lo = _YMIN.get(metric, ymin)
                ax.set_ylim(lo, dm + pad)
            else:
                ax.set_ylim(top=dm + pad)
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
# 테이블 출력
# ---------------------------------------------------------------------------

def write_table_csv(
    output_path: Path,
    rows: list[dict],
    metric_order: list[str],
    line_key: str = "project_seen_ratio",
) -> None:
    agg = aggregate_by_n_unseen(rows, metric_order, line_key=line_key, game_split="unseen")
    unseen_rows   = [r for r in rows if r.get("game_split") == "unseen"]
    line_fn       = _get_line_fn(line_key)
    line_vals     = sorted({line_fn(r) for r in unseen_rows})
    n_unseen_vals = sorted({r["n_unseen"] for r in unseen_rows})

    headers = ["line_group", "n_unseen"]
    for m in metric_order:
        headers += [f"{m}_mean", f"{m}_std", f"{m}_n"]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for lv in line_vals:
            for n in n_unseen_vals:
                rec: dict = {"line_group": _line_label(line_key, lv), "n_unseen": n}
                stats = agg.get((lv, n), {})
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
    line_key: str = "project_seen_ratio",
    decimals: int = 4,
) -> None:
    agg = aggregate_by_n_unseen(rows, metric_order, line_key=line_key, game_split="unseen")
    unseen_rows   = [r for r in rows if r.get("game_split") == "unseen"]
    line_fn       = _get_line_fn(line_key)
    line_vals     = sorted({line_fn(r) for r in unseen_rows})
    n_unseen_vals = sorted({r["n_unseen"] for r in unseen_rows})

    header_cols = ["line_group", "n_unseen"] + metric_order
    lines = [
        "| " + " | ".join(header_cols) + " |",
        "| " + " | ".join(["---"] * len(header_cols)) + " |",
    ]
    for lv in line_vals:
        for n in n_unseen_vals:
            stats = agg.get((lv, n), {})
            cells = [_line_label(line_key, lv), str(n)]
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
        description="unseen 게임 수 증가에 따른 성능 변화 꺾은선 그래프 (unseen_generalizability 전용)"
    )
    parser.add_argument("--input", default="wandb_projects",
                        help="wandb_projects 루트 디렉토리")
    parser.add_argument("--metrics", nargs="+", default=None,
                        help="사용할 metric 목록 (기본: config.json default_order)")
    parser.add_argument("--decimals", type=int, default=4)
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument(
        "--line-key",
        default="project_seen_ratio",
        choices=["project_seen_ratio", "train_seen_ratio", "project"],
        help="꺾은선 그룹 기준 (기본: project_seen_ratio → 'mgpcgrl sr=0.01' 형식)",
    )
    parser.add_argument(
        "--per-reward-enum", action="store_true",
        help="reward_enum 별 개별 플롯도 추가 생성",
    )
    parser.add_argument(
        "--ymin", type=float, default=_YMIN_DEFAULT,
        help=f"y축 하한값 (기본: {_YMIN_DEFAULT}). 차이가 잘 보이도록 0 이상으로 설정.",
    )
    parser.add_argument(
        "--experiment",
        choices=_exp_names if _exp_names else None,
        default="unseen_generalizability",
        metavar="EXPERIMENT",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args    = parse_args()
    run_dir = make_run_dir("unseen_count_progress", cfg=_CFG)
    log     = setup_logger(run_dir, name=__file__)
    log.debug("run_dir   : %s", run_dir)

    experiment   = args.experiment or "unseen_generalizability"
    folder_order = _get_experiment_folder_order(experiment)
    log.info("experiment: %s  folder_order=%s", experiment, folder_order)

    script_dir = _RESULTS_DIR
    input_root = resolve_input_root(args.input, script_dir)
    log.info("input_root: %s", input_root)

    metric_order: list[str] = args.metrics or DEFAULT_METRIC_ORDER.copy()

    rows = collect_rows_with_unseen_count(
        input_root, metric_order,
        target_projects=folder_order or None,
    )
    if not rows:
        msg = "No valid rows found."
        log.error(msg)
        raise SystemExit(msg)

    has_unseen = any(r.get("game_split") == "unseen" for r in rows)
    if not has_unseen:
        msg = "unseen 게임 데이터가 없습니다 — run_config.json 의 unseen_games 를 확인하세요."
        log.error(msg)
        raise SystemExit(msg)

    # Normalization — 파이프라인 전역 scale 우선, 없으면 로컬 계산
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

    line_key = args.line_key

    # ── 테이블 ────────────────────────────────────────────────────────────
    write_table_csv(
        run_dir / "unseen_count_table.csv", norm_rows, metric_order, line_key
    )
    write_table_markdown(
        run_dir / "unseen_count_table.md", norm_rows, metric_order, line_key, args.decimals
    )
    log.info("table     : %s", run_dir / "unseen_count_table.md")

    # ── 플롯 ──────────────────────────────────────────────────────────────
    if not args.no_plot:
        # 전체 reward_enum 평균 플롯
        try:
            write_line_plot(
                run_dir / "unseen_count_progress.png",
                norm_rows, metric_order, line_key=line_key,
                ymin=args.ymin,
            )
            log.info("plot (all re): %s", run_dir / "unseen_count_progress.png")
        except RuntimeError as e:
            log.error("Plot generation failed: %s", e)
            raise SystemExit(str(e)) from e

        # reward_enum 별 플롯 (옵션)
        if args.per_reward_enum:
            re_vals = sorted(
                {r["reward_enum"] for r in norm_rows if r.get("game_split") == "unseen"},
                key=sort_key_reward_enum,
            )
            for re in re_vals:
                try:
                    write_line_plot(
                        run_dir / f"unseen_count_progress_re{re}.png",
                        norm_rows, metric_order, line_key=line_key,
                        reward_enum=re, title_suffix=f" (re={re})",
                        ymin=args.ymin,
                    )
                    log.info("plot (re=%s): %s", re,
                             run_dir / f"unseen_count_progress_re{re}.png")
                except RuntimeError as e:
                    log.error("Plot re=%s failed: %s", re, e)

    log.info(
        "rows_found: %d  (unseen: %d)",
        len(rows),
        sum(1 for r in rows if r.get("game_split") == "unseen"),
    )


if __name__ == "__main__":
    main()

