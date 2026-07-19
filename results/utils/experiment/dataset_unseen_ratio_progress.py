"""
dataset_unseen_ratio_progress.py
================================
predictive_reward 실험 전용:
  dataset_unseen_ratio 가 늘어남에 따라 unseen 게임의 progress 성능이
  어떻게 달라지는지를 꺾은선 그래프로 시각화한다.

x 축: dataset_unseen_ratio
y 축: unseen 게임의 progress (정규화)
선   : encoder ckpt 그룹(default) 또는 project
필터: 기본적으로 unseen game 이 doom(dm)인 row 만 사용

출력:
    dataset_unseen_ratio_progress.png
    dataset_unseen_ratio_table.csv
    dataset_unseen_ratio_table.md
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
from collections import defaultdict
from pathlib import Path

import sys as _sys
_HERE         = Path(__file__).resolve().parent
_RESULTS_DIR  = _HERE.parent.parent
_ROOT         = _RESULTS_DIR.parent
if str(_RESULTS_DIR) not in _sys.path:
    _sys.path.insert(0, str(_RESULTS_DIR))
if str(_ROOT) not in _sys.path:
    _sys.path.append(str(_ROOT))

from conf.game_utils import CANONICAL_GAMES, parse_unseen_game_names
from utils.core.io import (
    get_game_split,
    iter_results_paths,
    load_run_config,
    normalize_reward_enum,
    parse_run_tokens,
)
from utils.core.normalization import (
    apply_normalization,
    compute_normalization_scale,
    load_normalization_scale,
    save_normalization_scale,
)
from utils.core.run_output import load_cfg, make_run_dir, setup_logger
from utils.core.stats import to_float
from utils.experiment.benchmark import (
    _bar_plot_setup,
    _get_experiment_folder_order,
    _project_display_name,
    _seed_agg,
    resolve_input_root,
)

_CFG = load_cfg()
_MARKERS: list[str] = ["o", "s", "^", "D", "v", "P", "X", "*", "h", "+"]
_VIPCGRL_LABEL = "VIPCGRL"
_MGPCGRL_NO_DEC_LABEL = r"MGPCGRK ($-\mathcal{L}_{\mathrm{dec}}$)"
_MGPCGRL_LABEL = "MGPCGRL"
_GROUP_ORDER: dict[str, int] = {
    _VIPCGRL_LABEL: 0,
    _MGPCGRL_NO_DEC_LABEL: 1,
    _MGPCGRL_LABEL: 2,
}
_GOOGLE_GROUP_COLORS: dict[str, str] = {
    _VIPCGRL_LABEL: "#DB4437",          # Google red
    _MGPCGRL_NO_DEC_LABEL: "#0F9D58",   # Google green
    _MGPCGRL_LABEL: "#4285F4",          # Google blue
}
_GOOGLE_FALLBACK_COLORS: list[str] = ["#F4B400", "#AB47BC", "#00ACC1", "#FF7043"]
_PROJECT_RATIO_OVERRIDES: dict[str, float] = {
    "aaai27_eval_mgpcgrl_zeroshot_dw0": 0.0,
    "aaai27_eval_mgpcgrl_fewshot_dw0": 1.0,
    "aaai27_eval_mgpcgrl_fewshot": 1.0,
}
_PRETENDARD_MEDIUM_CANDIDATES: tuple[Path, ...] = (
    Path(os.environ.get("PRETENDARD_MEDIUM_PATH", "")),
    Path("/Users/inchang/Desktop/MuCap-fin2/mucap/fonts/Pretendard-Medium.otf"),
    Path("/Users/inchang/Library/Fonts/Pretendard-Medium.otf"),
    Path("/Library/Fonts/Pretendard-Medium.otf"),
)


def _apply_pretendard_medium(plt) -> None:
    from matplotlib import font_manager

    for font_path in _PRETENDARD_MEDIUM_CANDIDATES:
        if not str(font_path) or not font_path.is_file():
            continue
        font_manager.fontManager.addfont(str(font_path))
        family = font_manager.FontProperties(fname=str(font_path)).get_name()
        plt.rcParams.update({
            "font.family": family,
            "font.weight": "medium",
            "axes.labelweight": "medium",
        })
        return

    plt.rcParams.update({
        "font.weight": "medium",
        "axes.labelweight": "medium",
    })


def _canonical_game(game: str) -> str:
    return "doom" if game == "doom2" else game


def _parse_game_filter(value: str | None) -> set[str]:
    if value is None:
        return set()
    text = value.strip()
    if not text or text.lower() in {"all", "*"}:
        return set()
    tokens = re.split(r"[,/\s]+", text)
    games: set[str] = set()
    for token in tokens:
        if not token:
            continue
        parsed = parse_unseen_game_names(token)
        if parsed:
            games.update(_canonical_game(g) for g in parsed)
        else:
            games.add(_canonical_game(token))
    return games


def _parse_ratio_filter(value: str | None) -> set[float]:
    if value is None:
        return set()
    text = value.strip()
    if not text:
        return set()
    ratios: set[float] = set()
    for token in re.split(r"[,/\s]+", text):
        ratio = _to_float_token(token)
        if ratio is not None:
            ratios.add(ratio)
    return ratios


def _to_float_token(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().replace("p", ".")
    try:
        return float(text)
    except ValueError:
        return None


def _extract_unseen_from_text(text: str) -> list[str]:
    match = re.search(r"(?:^|_)unseen-([a-z]+)(?:_|$)", text)
    if not match:
        match = re.search(r"(?:^|_)un-([a-z]+)(?:_|$)", text)
    if not match:
        return []
    names = parse_unseen_game_names(match.group(1))
    return sorted({"doom" if g == "doom2" else g for g in names})


def _resolve_game_split(game: str, run_cfg: dict, run_name: str) -> tuple[str, list[str]]:
    split = get_game_split(game, run_cfg)
    unseen_games = run_cfg.get("unseen_games") or []
    seen_games = run_cfg.get("seen_games") or []
    canonical_unseen = sorted({"doom" if g == "doom2" else g for g in unseen_games})
    canonical_seen = sorted({"doom" if g == "doom2" else g for g in seen_games})
    if split != "unknown":
        return split, canonical_unseen
    if game in canonical_unseen:
        return "unseen", canonical_unseen
    if game in canonical_seen:
        return "seen", canonical_unseen

    train_unseen_abbr = str(run_cfg.get("train_unseen_abbr") or "").strip()
    if train_unseen_abbr:
        inferred = sorted({"doom" if g == "doom2" else g for g in parse_unseen_game_names(train_unseen_abbr)})
        if inferred:
            return ("unseen" if game in inferred else "seen"), inferred

    for key in ("encoder.ckpt_name", "ckpt_name", "encoder.ckpt_dir", "encoder.ckpt_path"):
        inferred = _extract_unseen_from_text(str(run_cfg.get(key, "")))
        if inferred:
            return ("unseen" if game in inferred else "seen"), inferred

    inferred = _extract_unseen_from_text(run_name)
    if inferred:
        return ("unseen" if game in inferred else "seen"), inferred

    return "unknown", []


def _extract_dw_label(text: str) -> str | None:
    match = re.search(r"(?:^|_)dw-([0-9p.]+)(?:_|$)", text)
    if not match:
        return None
    value = match.group(1).replace("p", ".")
    if value == "0.03":
        return _MGPCGRL_LABEL
    return f"dw={value}"


def _group_label(project: str, run_cfg: dict, run_name: str, group_by: str) -> str:
    if "vipcgrl" in project or "vipcgrl" in run_name:
        return _VIPCGRL_LABEL
    if project == "aaai27_eval_mgpcgrl_zeroshot_dw0":
        return _MGPCGRL_NO_DEC_LABEL
    if project == "aaai27_eval_mgpcgrl_fewshot_dw0":
        return _MGPCGRL_NO_DEC_LABEL

    if group_by == "none":
        return "All"
    if group_by == "project":
        return _project_display_name(project)

    ckpt_name = str(run_cfg.get("encoder.ckpt_name") or run_cfg.get("ckpt_name") or "").strip()
    if group_by == "encoder_ckpt" and ckpt_name:
        dw_label = _extract_dw_label(ckpt_name)
        if dw_label:
            return dw_label
        return _MGPCGRL_NO_DEC_LABEL

    if group_by == "encoder_ckpt":
        delta_weight = _to_float_token(run_cfg.get("encoder_delta_weight"))
        if delta_weight is not None:
            if abs(delta_weight - 0.03) < 1e-9:
                return _MGPCGRL_LABEL
            return f"dw={delta_weight:g}"
        dw_label = _extract_dw_label(run_name)
        if dw_label:
            return dw_label
        if "zeroshot" in project or "zeroshot" in run_name:
            return _MGPCGRL_NO_DEC_LABEL

    return _project_display_name(project)


def _group_sort_key(group: str) -> tuple[int, str]:
    return (_GROUP_ORDER.get(group, len(_GROUP_ORDER)), group)


def collect_rows(
    input_root: Path,
    target_projects: list[str] | None = None,
    group_by: str = "encoder_ckpt",
    unseen_game_filter: str | None = "dm",
    exclude_ratios: set[float] | None = None,
) -> list[dict]:
    rows: list[dict] = []
    cfg_cache: dict[Path, dict] = {}
    target_unseen_games = _parse_game_filter(unseen_game_filter)
    excluded = exclude_ratios or set()

    for results_path in iter_results_paths(input_root):
        rel = results_path.relative_to(input_root)
        if len(rel.parts) < 3:
            continue

        project = rel.parts[0]
        run_name = rel.parts[1]
        eval_name = rel.parts[2] if len(rel.parts) >= 4 else ""
        if target_projects and project not in target_projects:
            continue

        eval_tokens = parse_run_tokens(eval_name)
        run_tokens = parse_run_tokens(run_name)
        seed = run_tokens.get("s", run_name)

        run_dir = results_path.parent
        if run_dir not in cfg_cache:
            cfg_cache[run_dir] = load_run_config(run_dir)
        run_cfg = cfg_cache[run_dir]

        ratio = _PROJECT_RATIO_OVERRIDES.get(project)
        if ratio is None:
            ratio = _to_float_token(run_cfg.get("dataset_unseen_ratio"))
        if ratio is None:
            ratio = _to_float_token(run_cfg.get("train_unseen_ratio"))
        if ratio is None:
            ratio = _to_float_token(run_cfg.get("unseen_ratio"))
        if ratio is None:
            ratio = _to_float_token(run_tokens.get("dur"))
        if ratio is None:
            ratio = _to_float_token(run_tokens.get("ur"))
        if ratio is None and ("zeroshot" in project or "zeroshot" in run_name):
            ratio = 0.0
        if ratio is None:
            continue
        if any(abs(ratio - x) < 1e-9 for x in excluded):
            continue

        label = _group_label(project, run_cfg, run_name, group_by)

        with results_path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                game = _canonical_game((row.get("game") or "").strip() or "unknown")
                split, inferred_unseen = _resolve_game_split(game, run_cfg, run_name)
                if split != "unseen":
                    continue
                if target_unseen_games and game not in target_unseen_games:
                    continue
                progress = to_float(row.get("progress"))
                if progress is None:
                    continue

                re_raw = (row.get("reward_enum") or "").strip()
                reward_enum = normalize_reward_enum(re_raw or eval_tokens.get("re", "unknown"))
                rows.append({
                    "project": project,
                    "group": label,
                    "game": game,
                    "reward_enum": reward_enum,
                    "game_split": split,
                    "dataset_unseen_ratio": ratio,
                    "unseen_game": "+".join(inferred_unseen or run_cfg.get("unseen_games", []) or []),
                    "n_unseen": len(inferred_unseen or run_cfg.get("unseen_games", []) or []),
                    "seed": seed,
                    "metrics": {"progress": progress},
                })
    return rows


def aggregate_by_dataset_unseen_ratio(
    rows: list[dict],
    game_split: str = "unseen",
) -> dict[tuple[str, float], dict]:
    filtered = [r for r in rows if r.get("game_split") == game_split]
    grouped: dict[tuple[str, float], list[dict]] = defaultdict(list)
    for row in filtered:
        grouped[(row["group"], row["dataset_unseen_ratio"])].append(row)

    result: dict[tuple[str, float], dict] = {}
    for key, group_rows in grouped.items():
        stat = _seed_agg(group_rows, "progress")
        if stat:
            result[key] = stat
    return result


def _log_axis_bounds(values: list[float], fallback_min: float = 1e-3) -> tuple[float, float]:
    positive = [v for v in values if v > 0]
    if not positive:
        return fallback_min, 1.0

    left = 10 ** math.floor(math.log10(min(positive)))
    right = 10 ** math.ceil(math.log10(max(positive)))
    if math.isclose(left, right):
        right *= 10
    return left, right


def _log_tick_values(left: float, right: float, include: list[float]) -> list[float]:
    start = math.floor(math.log10(left))
    stop = math.ceil(math.log10(right))
    ticks = [10 ** exp for exp in range(start, stop + 1)]
    ticks.extend(v for v in include if v > 0)
    return sorted({v for v in ticks if left <= v <= right})


def _zero_log_proxy(values: list[float], fallback: float = 1e-2) -> float:
    positive = [v for v in values if v > 0]
    if not positive:
        return fallback

    proxy = 10 ** math.floor(math.log10(min(positive)))
    if math.isclose(proxy, min(positive)):
        proxy /= 10
    return proxy * 0.5


def write_line_plot(output_path: Path, rows: list[dict]) -> None:
    plt = _bar_plot_setup()
    from matplotlib.ticker import MaxNLocator

    _apply_pretendard_medium(plt)
    unseen_rows = [r for r in rows if r.get("game_split") == "unseen"]
    if not unseen_rows:
        return

    ratio_vals = sorted({r["dataset_unseen_ratio"] for r in unseen_rows})
    group_vals = sorted({r["group"] for r in unseen_rows}, key=_group_sort_key)
    agg = aggregate_by_dataset_unseen_ratio(rows, game_split="unseen")
    split_ratio = 0.03
    zero_x = 0.0
    low_ratio_x = 0.025
    split_x = 0.16
    ratio_01_x = 0.26
    ratio_max_x = 1.0

    def x_pos(ratio: float) -> float:
        if abs(ratio) < 1e-12:
            return zero_x
        if abs(ratio - 0.01) < 1e-12:
            return low_ratio_x
        if abs(ratio - split_ratio) < 1e-12:
            return split_x
        if ratio >= 0.1:
            return ratio_01_x + (ratio - 0.1) / 0.9 * (ratio_max_x - ratio_01_x)
        return ratio

    fig, ax = plt.subplots(1, 1, figsize=(2.75, 2.05))
    for i, group in enumerate(group_vals):
        xs, means, stds = [], [], []
        for ratio in ratio_vals:
            stat = agg.get((group, ratio))
            if stat:
                xs.append(x_pos(ratio))
                means.append(stat["mean"])
                stds.append(stat["std"])
        if not means:
            continue

        color = _GOOGLE_GROUP_COLORS.get(
            group,
            _GOOGLE_FALLBACK_COLORS[i % len(_GOOGLE_FALLBACK_COLORS)],
        )
        marker = _MARKERS[i % len(_MARKERS)]
        ax.plot(
            xs, means,
            marker=marker, color=color, label=group,
            linewidth=1.4, markersize=4.2, zorder=3,
        )
        ax.fill_between(
            xs,
            [m - s for m, s in zip(means, stds)],
            [m + s for m, s in zip(means, stds)],
            alpha=0.15, color=color,
        )

    plotted_xs = [x_pos(ratio) for ratio in ratio_vals]
    x_left = min(plotted_xs) - 0.03
    x_right = max(plotted_xs) + 0.04
    if split_ratio in ratio_vals:
        ax.axvspan(
            x_left,
            split_x,
            facecolor="#f1f1f1",
            edgecolor="#d0d0d0",
            hatch="///",
            linewidth=0.0,
            alpha=0.35,
            zorder=0,
        )
        ax.axvline(split_x, color="#444444", linestyle="--", linewidth=1.0, alpha=0.75, zorder=2)

    ax.set_xlabel("Unseen-Domain Data Usage", fontsize=8, fontweight="medium")
    ax.set_ylabel("Progress (Unseen domain)", fontsize=8, fontweight="medium")
    tick_pairs = [
        (zero_x, "0"),
        (split_x, "0.03"),
        (x_pos(0.1), "0.1"),
        (x_pos(0.4), "0.4"),
        (x_pos(0.7), "0.7"),
        (x_pos(1.0), "1.0"),
    ]
    tick_candidates = [x for x, _ in tick_pairs]
    tick_values = [v for v in tick_candidates if x_left <= v <= x_right]
    tick_labels = [label for x, label in tick_pairs if x_left <= x <= x_right]
    ax.set_xticks(tick_values)
    ax.set_xticklabels(tick_labels)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.tick_params(axis="both", labelsize=7)
    ax.set_xlim(x_left, x_right)
    ax.grid(axis="both", alpha=0.3)

    baseline_stat = agg.get((_VIPCGRL_LABEL, split_ratio))
    target_stat = agg.get((_MGPCGRL_NO_DEC_LABEL, 1.0))
    if baseline_stat and target_stat:
        baseline_y = baseline_stat["mean"]
        target_y = target_stat["mean"]
        diff = target_y - baseline_y
        pct = diff / abs(baseline_y) * 100 if abs(baseline_y) > 1e-12 else None
        baseline_x = x_pos(split_ratio)
        target_x = x_pos(1.0)
        compare_x = target_x
        ax.plot(
            [baseline_x, compare_x],
            [baseline_y, baseline_y],
            color="#222222",
            linewidth=0.9,
            linestyle="--",
            alpha=0.75,
            zorder=4,
        )
        ax.annotate(
            "",
            xy=(compare_x, target_y),
            xytext=(compare_x, baseline_y),
            arrowprops={
                "arrowstyle": "->",
                "color": "#222222",
                "linewidth": 1.4,
                "mutation_scale": 12,
                "shrinkA": 1,
                "shrinkB": 1,
            },
            zorder=5,
        )
        improvement_value = f"{diff:+.3f}" if pct is None else f"{diff:+.3f} ({pct:+.1f}%)"
        improvement_label = f"Reward prediction\n{improvement_value}"
        ax.text(
            compare_x - 0.02,
            baseline_y + (target_y - baseline_y) * 0.52,
            improvement_label,
            ha="right",
            va="center",
            fontsize=7,
            fontweight="medium",
            color="#222222",
            bbox={
                "boxstyle": "round,pad=0.18",
                "facecolor": "white",
                "edgecolor": "#dddddd",
                "linewidth": 0.6,
                "alpha": 0.88,
            },
            zorder=6,
        )

    mgpcgrl_stat = agg.get((_MGPCGRL_NO_DEC_LABEL, 0.4))
    da_stat = agg.get((_MGPCGRL_LABEL, 0.4))
    if mgpcgrl_stat and da_stat:
        baseline_y = mgpcgrl_stat["mean"]
        target_y = da_stat["mean"]
        diff = target_y - baseline_y
        pct = diff / abs(baseline_y) * 100 if abs(baseline_y) > 1e-12 else None
        compare_x = x_pos(0.4)
        ax.annotate(
            "",
            xy=(compare_x, target_y),
            xytext=(compare_x, baseline_y),
            arrowprops={
                "arrowstyle": "->",
                "color": "#222222",
                "linewidth": 1.4,
                "mutation_scale": 12,
                "shrinkA": 1,
                "shrinkB": 1,
            },
            zorder=5,
        )
        improvement_value = f"{diff:+.3f}" if pct is None else f"{diff:+.3f} ({pct:+.1f}%)"
        improvement_label = f"Domain alignment\n{improvement_value}"
        y0, y1 = ax.get_ylim()
        ax.text(
            x_pos(0.3),
            min(baseline_y, target_y) - (y1 - y0) * 0.06,
            improvement_label,
            ha="center",
            va="top",
            fontsize=7,
            fontweight="medium",
            color="#222222",
            bbox={
                "boxstyle": "round,pad=0.18",
                "facecolor": "white",
                "edgecolor": "#dddddd",
                "linewidth": 0.6,
                "alpha": 0.88,
            },
            zorder=6,
        )

    if split_ratio in ratio_vals:
        y0, y1 = ax.get_ylim()
        y_arrow = y1 - (y1 - y0) * 0.13
        y_text = y_arrow + (y1 - y0) * 0.018
        labeled_text_x = (x_left + split_x) * 0.5
        ax.annotate(
            "",
            xy=(split_x - 0.018, y_arrow),
            xytext=(x_left + 0.035, y_arrow),
            arrowprops={
                "arrowstyle": "->",
                "color": "#444444",
                "linewidth": 0.8,
                "shrinkA": 0,
                "shrinkB": 0,
            },
        )
        ax.annotate(
            "",
            xy=(min(split_x + 0.36, x_right - 0.03), y_arrow),
            xytext=(split_x + 0.012, y_arrow),
            arrowprops={
                "arrowstyle": "->",
                "color": "#444444",
                "linewidth": 0.8,
                "shrinkA": 0,
                "shrinkB": 0,
            },
        )
        ax.text(
            labeled_text_x,
            y_text,
            "Labeled",
            ha="center",
            va="bottom",
            rotation=0,
            fontsize=6.5,
            fontweight="medium",
            color="#444444",
        )
        ax.text(
            split_x + 0.02,
            y_text,
            "Predicted reward",
            ha="left",
            va="bottom",
            fontsize=6.5,
            fontweight="medium",
            color="#444444",
        )
    if len(group_vals) > 1:
        ax.legend(
            prop={"size": 7, "weight": "medium"},
            loc="best",
            framealpha=0.85,
        )

    fig.tight_layout(pad=0.0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight", pad_inches=0)
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def write_table_csv(output_path: Path, rows: list[dict]) -> None:
    agg = aggregate_by_dataset_unseen_ratio(rows, game_split="unseen")
    unseen_rows = [r for r in rows if r.get("game_split") == "unseen"]
    ratio_vals = sorted({r["dataset_unseen_ratio"] for r in unseen_rows})
    group_vals = sorted({r["group"] for r in unseen_rows})

    headers = ["group", "dataset_unseen_ratio", "progress_mean", "progress_std", "progress_n"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for group in group_vals:
            for ratio in ratio_vals:
                stat = agg.get((group, ratio))
                writer.writerow({
                    "group": group,
                    "dataset_unseen_ratio": ratio,
                    "progress_mean": stat["mean"] if stat else "",
                    "progress_std": stat["std"] if stat else "",
                    "progress_n": stat["n"] if stat else 0,
                })


def write_table_markdown(output_path: Path, rows: list[dict], decimals: int = 4) -> None:
    agg = aggregate_by_dataset_unseen_ratio(rows, game_split="unseen")
    unseen_rows = [r for r in rows if r.get("game_split") == "unseen"]
    ratio_vals = sorted({r["dataset_unseen_ratio"] for r in unseen_rows})
    group_vals = sorted({r["group"] for r in unseen_rows})

    lines = [
        "| group | dataset_unseen_ratio | progress |",
        "| --- | --- | --- |",
    ]
    for group in group_vals:
        for ratio in ratio_vals:
            stat = agg.get((group, ratio))
            value = f"{stat['mean']:.{decimals}f} +/- {stat['std']:.{decimals}f}" if stat else "-"
            lines.append(f"| {group} | {ratio:g} | {value} |")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _game_sort_key(game: str) -> tuple[int, str]:
    try:
        return (CANONICAL_GAMES.index(game), game)
    except ValueError:
        return (len(CANONICAL_GAMES), game)


def _write_outputs(
    output_dir: Path,
    rows: list[dict],
    *,
    decimals: int,
    no_plot: bool,
    log,
) -> None:
    write_table_csv(output_dir / "dataset_unseen_ratio_table.csv", rows)
    write_table_markdown(
        output_dir / "dataset_unseen_ratio_table.md",
        rows,
        decimals=decimals,
    )
    log.info("table   : %s", output_dir / "dataset_unseen_ratio_table.md")

    if not no_plot:
        try:
            write_line_plot(output_dir / "dataset_unseen_ratio_progress.png", rows)
            log.info("plot    : %s", output_dir / "dataset_unseen_ratio_progress.png")
        except RuntimeError as e:
            log.error("Plot generation failed: %s", e)
            raise SystemExit(str(e)) from e


def parse_args() -> argparse.Namespace:
    exp_names = list(_CFG.get("experiments", {}).keys())
    parser = argparse.ArgumentParser(
        description="dataset_unseen_ratio 증가에 따른 unseen 게임 progress 변화 꺾은선 그래프"
    )
    parser.add_argument("--input", default="wandb_projects", help="wandb_projects 루트 디렉토리")
    parser.add_argument("--decimals", type=int, default=4)
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument(
        "--unseen-game",
        default="dm",
        help="사용할 unseen game 필터. 기본값: dm(doom). all 지정 시 필터 해제",
    )
    parser.add_argument(
        "--exclude-ratios",
        default="0.2",
        help="제외할 dataset_unseen_ratio 목록. 기본값: 0.2",
    )
    parser.add_argument(
        "--group-by",
        choices=["encoder_ckpt", "project", "none"],
        default="encoder_ckpt",
        help="line grouping 기준",
    )
    parser.add_argument(
        "--use-global-norm",
        action="store_true",
        help=(
            "PIPELINE_NORM_SCALE 이 있으면 전역 min-max scale 을 사용합니다. "
            "기본값은 predictive_reward 플롯에 들어간 row만으로 local scale 을 계산합니다."
        ),
    )
    parser.add_argument(
        "--experiment",
        choices=exp_names if exp_names else None,
        default="predictive_reward",
        metavar="EXPERIMENT",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = make_run_dir("dataset_unseen_ratio_progress", cfg=_CFG)
    log = setup_logger(run_dir, name=__file__)
    log.debug("run_dir : %s", run_dir)

    experiment = args.experiment or "predictive_reward"
    folder_order = _get_experiment_folder_order(experiment)
    log.info("experiment: %s  target_projects=%s", experiment, folder_order)

    input_root = resolve_input_root(args.input, _RESULTS_DIR)
    log.info("input_root: %s", input_root)

    rows = collect_rows(
        input_root,
        target_projects=folder_order or None,
        group_by=args.group_by,
        unseen_game_filter=args.unseen_game,
        exclude_ratios=_parse_ratio_filter(args.exclude_ratios),
    )
    if not rows:
        msg = "No valid rows found. dataset_unseen_ratio 가 저장된 predictive_reward 결과가 있는지 확인하세요."
        log.error(msg)
        raise SystemExit(msg)

    unseen_rows = [r for r in rows if r.get("game_split") == "unseen"]
    if not unseen_rows:
        msg = (
            f"unseen 게임 데이터가 없습니다(unseen_game={args.unseen_game}). run_config.json 의 unseen_games 또는 "
            "encoder.ckpt_name 의 _unseen-XX_ 토큰을 확인하세요."
        )
        log.error(msg)
        raise SystemExit(msg)

    global_scale_path = os.environ.get("PIPELINE_NORM_SCALE")
    if args.use_global_norm and global_scale_path and Path(global_scale_path).is_file():
        norm_scale = load_normalization_scale(Path(global_scale_path))
        log.info("norm_scale (global): %s", global_scale_path)
    else:
        norm_scale = compute_normalization_scale(rows, ["progress"])
        scale_path = run_dir / "normalization_scale.json"
        save_normalization_scale(norm_scale, scale_path)
        log.info("norm_scale (local, filtered rows): %s", scale_path)

    norm_rows = apply_normalization(rows, norm_scale, ["progress"])

    output_games = sorted({r["game"] for r in norm_rows if r.get("game_split") == "unseen"}, key=_game_sort_key)
    if len(output_games) > 1:
        for game in output_games:
            game_rows = [r for r in norm_rows if r.get("game") == game]
            game_dir = run_dir / game
            log.info("game output: %s  rows=%d", game, len(game_rows))
            _write_outputs(
                game_dir,
                game_rows,
                decimals=args.decimals,
                no_plot=args.no_plot,
                log=log,
            )
    else:
        _write_outputs(
            run_dir,
            norm_rows,
            decimals=args.decimals,
            no_plot=args.no_plot,
            log=log,
        )

    log.info(
        "rows: total=%d  unseen=%d  unseen_game_filter=%s  exclude_ratios=%s  ratios=%s  groups=%s  known_games=%s",
        len(rows),
        len(unseen_rows),
        args.unseen_game,
        args.exclude_ratios,
        sorted({r["dataset_unseen_ratio"] for r in unseen_rows}),
        sorted({r["group"] for r in unseen_rows}),
        CANONICAL_GAMES,
    )


if __name__ == "__main__":
    main()
