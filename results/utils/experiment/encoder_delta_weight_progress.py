"""
Plot progress against encoder_delta_weight for ablation runs.

Expected input layout:
    <input_root>/<project>/<run_name>/<eval_name>/results.csv
    <input_root>/<project>/<run_name>/<eval_name>/run_config.json

The target project is selected from config.json:
    experiments.<experiment>.target_projects excluding re_oracle_project

Outputs:
    encoder_delta_weight_progress_macro.csv
    encoder_delta_weight_progress_summary.csv
    progress_vs_encoder_delta_weight_overall.png/.pdf
    progress_vs_encoder_delta_weight_by_reward_enum.png/.pdf
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path

import sys as _sys

_HERE = Path(__file__).resolve().parent
_RESULTS_DIR = _HERE.parent.parent
_ROOT = _RESULTS_DIR.parent
if str(_RESULTS_DIR) not in _sys.path:
    _sys.path.insert(0, str(_RESULTS_DIR))
if str(_ROOT) not in _sys.path:
    _sys.path.append(str(_ROOT))

from utils.core.io import iter_results_paths, load_run_config, parse_run_tokens
from utils.core.run_output import load_cfg, make_run_dir, setup_logger
from utils.core.stats import to_float
from utils.experiment.benchmark import resolve_input_root

_CFG = load_cfg()


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def _std(xs: list[float]) -> float:
    return statistics.stdev(xs) if len(xs) > 1 else 0.0


def _sem(xs: list[float]) -> float:
    return _std(xs) / math.sqrt(len(xs)) if len(xs) > 1 else 0.0


def _as_float(value: object) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return to_float(value)


def _parse_weight_from_name(text: str) -> float | None:
    """Parse dw-0p03 / dw-0p1 tokens from run names as fallback."""
    match = re.search(r"(?:^|_)dw-([^_]+)", text)
    if not match:
        return None
    raw = match.group(1).replace("p", ".")
    try:
        return float(raw)
    except ValueError:
        return None


def _reward_labels() -> dict[int, str]:
    raw = _CFG.get("reward_enums", {}).get("labels", {})
    labels: dict[int, str] = {}
    for key, value in raw.items():
        try:
            labels[int(key)] = str(value)
        except (TypeError, ValueError):
            continue
    return labels


def _project_display_names(experiment: str | None) -> dict[str, str]:
    names = dict(_CFG.get("project_display_names", {}))
    if experiment:
        names.update(
            _CFG.get("experiments", {})
            .get(experiment, {})
            .get("project_display_names", {})
        )
    return names


def _display_name(project: str, experiment: str | None) -> str:
    return _project_display_names(experiment).get(project, project)


def _resolve_projects(
    experiment: str,
    target_project: str | None,
    oracle_project: str | None,
) -> tuple[str, str, str]:
    exp_cfg = _CFG.get("experiments", {}).get(experiment, {})
    configured_targets = list(exp_cfg.get("target_projects", []))

    oracle = oracle_project or exp_cfg.get("re_oracle_project", "aaai27_eval_cpcgrl")
    oracle_label = exp_cfg.get("re_oracle_label") or _display_name(oracle, experiment)

    if target_project:
        return target_project, oracle, oracle_label

    candidates = [p for p in configured_targets if p != oracle]
    if not candidates:
        raise SystemExit(
            f"No target project found for experiment '{experiment}'. "
            "Set target_projects and re_oracle_project in results/config.json."
        )
    return candidates[0], oracle, oracle_label


def _collect_progress_rows(
    input_root: Path,
    project: str,
    include_weight: bool,
) -> list[dict]:
    rows: list[dict] = []
    project_root = input_root / project
    if not project_root.is_dir():
        return rows

    for results_path in iter_results_paths(project_root):
        rel = results_path.relative_to(project_root)
        if len(rel.parts) < 3:
            continue
        run_name = rel.parts[0]
        eval_name = rel.parts[1] if len(rel.parts) >= 3 else ""
        run_dir = results_path.parent
        run_cfg = load_run_config(run_dir)
        run_tokens = parse_run_tokens(run_name)
        eval_tokens = parse_run_tokens(eval_name)

        weight: float | None = None
        if include_weight:
            weight = _as_float(run_cfg.get("encoder_delta_weight"))
            if weight is None:
                weight = _parse_weight_from_name(run_name)
            if weight is None:
                continue

        seed = run_cfg.get("seed", run_tokens.get("s", run_name))
        cfg_re = run_cfg.get("re", eval_tokens.get("re", run_tokens.get("re")))

        with results_path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                progress = to_float(row.get("progress"))
                if progress is None:
                    continue
                re_raw = row.get("reward_enum", cfg_re)
                try:
                    reward_enum = int(float(re_raw))
                except (TypeError, ValueError):
                    continue
                rows.append(
                    {
                        "project": project,
                        "encoder_delta_weight": weight,
                        "seed": seed,
                        "reward_enum": reward_enum,
                        "game": (row.get("game") or "").strip() or "unknown",
                        "progress": progress,
                        "run": run_name,
                        "eval": eval_name,
                    }
                )
    return rows


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _format_float(value: float, decimals: int) -> str:
    if math.isnan(value):
        return "-"
    return f"{value:.{decimals}f}"


def _write_markdown(
    path: Path,
    macro_rows: list[dict],
    summary_rows: list[dict],
    decimals: int,
) -> None:
    lines = [
        "# encoder_delta_weight progress",
        "",
        "## Macro",
        "",
        "| encoder_delta_weight | progress | oracle | n_reward_enum |",
        "|---:|---:|---:|---:|",
    ]
    for row in macro_rows:
        lines.append(
            "| "
            f"{row['encoder_delta_weight']:.4g} | "
            f"{_format_float(row['progress_macro_mean'], decimals)} | "
            f"{_format_float(row['oracle_progress_macro_mean'], decimals)} | "
            f"{row['n_reward_enum']} |"
        )

    lines += [
        "",
        "## By Reward Enum",
        "",
        "| encoder_delta_weight | reward_enum | reward_label | progress | oracle | n_rows |",
        "|---:|---:|---|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            "| "
            f"{row['encoder_delta_weight']:.4g} | "
            f"{row['reward_enum']} | "
            f"{row['reward_label']} | "
            f"{_format_float(row['progress_mean'], decimals)} | "
            f"{_format_float(row['oracle_progress_mean'], decimals)} | "
            f"{row['n_rows']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _save_png_pdf(fig, output_path: Path, dpi: int = 220) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")


def _progress_ymin(values: list[float], requested_ymin: float | None) -> float:
    if requested_ymin is None:
        return 0.0
    finite_values = [v for v in values if not math.isnan(v)]
    if not finite_values:
        return requested_ymin
    # Keep low bars visible while still zooming the axis when possible.
    return min(requested_ymin, math.floor(min(finite_values) / 10.0) * 10.0)


def _weight_colors(weights: list[float]) -> list:
    import matplotlib.pyplot as plt

    cmap = plt.get_cmap("Blues")
    if len(weights) <= 1:
        return [cmap(0.65)]
    return [cmap(0.35 + 0.5 * i / (len(weights) - 1)) for i in range(len(weights))]


def _plot_overall(
    output_path: Path,
    macro_rows: list[dict],
    target_label: str,
    oracle_label: str,
    ymin: float | None,
) -> None:
    import matplotlib.pyplot as plt

    weights = [r["encoder_delta_weight"] for r in macro_rows]
    means = [r["progress_macro_mean"] for r in macro_rows]
    oracle = macro_rows[0]["oracle_progress_macro_mean"] if macro_rows else float("nan")

    plt.rcParams.update(
        {"font.size": 11, "axes.spines.top": False, "axes.spines.right": False}
    )
    fig, ax = plt.subplots(figsize=(3.2, 2.7))
    xs = list(range(len(weights)))
    colors = _weight_colors(weights)
    bars = ax.bar(
        xs,
        means,
        color=colors,
        edgecolor="none",
        alpha=0.9,
        label=target_label,
    )
    y_uppers = list(means)
    for bar, mean_value in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(mean_value, 1e-6) * 0.02,
            f"{mean_value:.2f}",
            ha="center",
            va="bottom",
            fontsize=6,
            color="black",
        )
    if not math.isnan(oracle):
        ax.axhline(
            oracle,
            color="red",
            linestyle="--",
            linewidth=1.5,
            zorder=5,
            label=oracle_label.replace(" (Oracle)", ""),
        )
        y_uppers.append(oracle)
    ax.set_xlabel("Delta Weight Scale", fontsize=9)
    ax.set_ylabel("Progress")
    ax.set_xticks(xs, [f"{w:g}" for w in weights])
    ax.tick_params(axis="both", labelsize=8)
    ax.grid(axis="y", alpha=0.3)
    if y_uppers:
        dm = max(y_uppers)
        ax.set_ylim(_progress_ymin(means, ymin), dm + max(dm, 1e-6) * 0.15)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(len(handles), 4), fontsize=7)
        fig.subplots_adjust(top=0.82)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    _save_png_pdf(fig, output_path)
    plt.close(fig)


def _weight_label(weight: float) -> str:
    if abs(weight) < 1e-12:
        return "MGPCGRL"
    return f"dw={weight:g}"


def _plot_by_reward_enum(
    output_path: Path,
    summary_rows: list[dict],
    reward_enums: list[int],
    weights: list[float],
    ymin: float | None,
) -> None:
    import matplotlib.pyplot as plt

    by_key = {(r["encoder_delta_weight"], r["reward_enum"]): r for r in summary_rows}

    plt.rcParams.update(
        {"font.size": 11, "axes.spines.top": False, "axes.spines.right": False}
    )
    fig, ax = plt.subplots(figsize=(5.8, 3.0))

    x_center = list(range(len(reward_enums)))
    n_bars = max(len(weights), 1)
    bar_width = 0.6 / n_bars
    colors = _weight_colors(weights)
    y_uppers: list[float] = []

    for weight_idx, weight in enumerate(weights):
        xs, ys = [], []
        for reward_idx, reward_enum in enumerate(reward_enums):
            row = by_key.get((weight, reward_enum))
            if not row:
                continue
            xs.append(x_center[reward_idx] - 0.3 + (weight_idx + 0.5) * bar_width)
            ys.append(row["progress_mean"])
            y_uppers.append(row["progress_mean"])
        if not ys:
            continue
        bars = ax.bar(
            xs,
            ys,
            width=bar_width,
            label=_weight_label(weight),
            color=colors[weight_idx % len(colors)],
            edgecolor="none",
            alpha=0.9,
        )
        for bar, mean_value in zip(bars, ys):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(mean_value, 1e-6) * 0.02,
                f"{mean_value:.1f}",
                ha="center",
                va="bottom",
                fontsize=5.2,
                color="black",
            )

    baseline_legend_added = False
    for reward_idx, reward_enum in enumerate(reward_enums):
        oracle = float("nan")
        for weight in weights:
            row = by_key.get((weight, reward_enum))
            if row:
                oracle = row["oracle_progress_mean"]
                break
        if math.isnan(oracle):
            continue
        label = "CPCGRL" if not baseline_legend_added else None
        ax.plot(
            [x_center[reward_idx] - 0.45, x_center[reward_idx] + 0.45],
            [oracle, oracle],
            color="red",
            linewidth=1.5,
            linestyle="--",
            zorder=5,
            label=label,
        )
        baseline_legend_added = True
        y_uppers.append(oracle)

    xtick_labels = []
    for reward_enum in reward_enums:
        label = str(reward_enum)
        for weight in weights:
            row = by_key.get((weight, reward_enum))
            if row:
                label = row["reward_label"]
                break
        xtick_labels.append(f"re={reward_enum}\n{label}")
    ax.set_ylabel("Progress", rotation=90, labelpad=8)
    ax.set_xlabel("Delta Weight Scale", fontsize=9)
    ax.set_xticks(x_center, xtick_labels)
    ax.tick_params(axis="x", labelrotation=0, labelsize=8)
    ax.tick_params(axis="y", labelsize=8)
    ax.set_xlim(-0.5, len(reward_enums) - 0.5)
    ax.grid(axis="y", alpha=0.3)
    if y_uppers:
        dm = max(y_uppers)
        plotted_values = [row["progress_mean"] for row in summary_rows]
        ax.set_ylim(_progress_ymin(plotted_values, ymin), dm + max(dm, 1e-6) * 0.15)

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(len(handles), 6), fontsize=7)
        fig.subplots_adjust(top=0.82)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    _save_png_pdf(fig, output_path)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    exp_names = list(_CFG.get("experiments", {}).keys())
    parser = argparse.ArgumentParser(
        description="Plot progress by encoder_delta_weight."
    )
    parser.add_argument("--input", default="wandb_projects")
    parser.add_argument("--experiment", choices=exp_names if exp_names else None, default="encoder_delta_weight_progress")
    parser.add_argument("--target-project", default=None)
    parser.add_argument("--oracle-project", default=None)
    parser.add_argument("--decimals", type=int, default=3)
    parser.add_argument(
        "--ymin",
        type=float,
        default=40.0,
        help="Preferred y-axis lower bound for progress plots. "
        "If data goes below this, the bound is lowered to keep bars visible.",
    )
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = make_run_dir("encoder_delta_weight", cfg=_CFG)
    log = setup_logger(run_dir, name=__file__)

    experiment = args.experiment
    target_project, oracle_project, oracle_label = _resolve_projects(
        experiment,
        target_project=args.target_project,
        oracle_project=args.oracle_project,
    )
    target_label = _display_name(target_project, experiment)
    input_root = resolve_input_root(args.input, _RESULTS_DIR)

    log.info("experiment    : %s", experiment)
    log.info("input_root    : %s", input_root)
    log.info("target_project: %s", target_project)
    log.info("oracle_project: %s", oracle_project)

    target_rows = _collect_progress_rows(input_root, target_project, include_weight=True)
    oracle_rows = _collect_progress_rows(input_root, oracle_project, include_weight=False)
    if not target_rows:
        raise SystemExit(f"No target rows found for project '{target_project}'.")
    if not oracle_rows:
        raise SystemExit(f"No oracle rows found for project '{oracle_project}'.")

    reward_labels = _reward_labels()

    by_weight_re: dict[tuple[float, int], list[float]] = defaultdict(list)
    for row in target_rows:
        by_weight_re[(row["encoder_delta_weight"], row["reward_enum"])].append(row["progress"])

    oracle_by_re: dict[int, list[float]] = defaultdict(list)
    for row in oracle_rows:
        oracle_by_re[row["reward_enum"]].append(row["progress"])

    weights = sorted({weight for weight, _ in by_weight_re})
    reward_enums = sorted({re_id for _, re_id in by_weight_re} | set(oracle_by_re))

    summary_rows: list[dict] = []
    for weight in weights:
        for reward_enum in reward_enums:
            values = by_weight_re.get((weight, reward_enum), [])
            if not values:
                continue
            oracle_values = oracle_by_re.get(reward_enum, [])
            summary_rows.append(
                {
                    "project": target_project,
                    "encoder_delta_weight": weight,
                    "reward_enum": reward_enum,
                    "reward_label": reward_labels.get(reward_enum, str(reward_enum)),
                    "n_rows": len(values),
                    "progress_mean": _mean(values),
                    "progress_std": _std(values),
                    "progress_sem": _sem(values),
                    "oracle_project": oracle_project,
                    "oracle_label": oracle_label,
                    "oracle_n_rows": len(oracle_values),
                    "oracle_progress_mean": _mean(oracle_values) if oracle_values else float("nan"),
                }
            )

    oracle_re_means = [
        _mean(oracle_by_re[reward_enum])
        for reward_enum in reward_enums
        if oracle_by_re.get(reward_enum)
    ]
    oracle_macro = _mean(oracle_re_means)

    macro_rows: list[dict] = []
    for weight in weights:
        re_means = [
            _mean(by_weight_re[(weight, reward_enum)])
            for reward_enum in reward_enums
            if (weight, reward_enum) in by_weight_re
        ]
        macro_rows.append(
            {
                "encoder_delta_weight": weight,
                "progress_macro_mean": _mean(re_means),
                "progress_macro_std_across_reward_enum": _std(re_means),
                "progress_macro_sem_across_reward_enum": _sem(re_means),
                "n_reward_enum": len(re_means),
                "oracle_project": oracle_project,
                "oracle_label": oracle_label,
                "oracle_progress_macro_mean": oracle_macro,
            }
        )

    _write_csv(run_dir / "encoder_delta_weight_progress_summary.csv", summary_rows)
    _write_csv(run_dir / "encoder_delta_weight_progress_macro.csv", macro_rows)
    _write_markdown(run_dir / "encoder_delta_weight_progress.md", macro_rows, summary_rows, args.decimals)

    if not args.no_plot:
        _plot_overall(
            run_dir / "progress_vs_encoder_delta_weight_overall.png",
            macro_rows,
            target_label=target_label,
            oracle_label=oracle_label,
            ymin=args.ymin,
        )
        _plot_by_reward_enum(
            run_dir / "progress_vs_encoder_delta_weight_by_reward_enum.png",
            summary_rows,
            reward_enums=reward_enums,
            weights=weights,
            ymin=args.ymin,
        )

    log.info("rows target/oracle: %d / %d", len(target_rows), len(oracle_rows))
    log.info("weights: %s", weights)
    log.info("reward_enums: %s", reward_enums)
    log.info("macro csv: %s", run_dir / "encoder_delta_weight_progress_macro.csv")


if __name__ == "__main__":
    main()
