"""
domain_identity_progress.py
===========================
Domain identity ablation 전용 리포트.

출력:
    domain_identity_seen_unseen.png/.pdf
    domain_identity_summary_table.csv
    domain_identity_summary_table.md
    domain_identity_summary_table.tex

Seen/Unseen split 성능을 5개 leave-one-game fold 전체로 합쳐 리포트한다.
"""

from __future__ import annotations

import argparse
import csv
import os
import textwrap
from pathlib import Path

import sys as _sys

_HERE = Path(__file__).resolve().parent
_RESULTS_DIR = _HERE.parent.parent
_ROOT = _RESULTS_DIR.parent
if str(_RESULTS_DIR) not in _sys.path:
    _sys.path.insert(0, str(_RESULTS_DIR))
if str(_ROOT) not in _sys.path:
    _sys.path.append(str(_ROOT))

from utils.core.normalization import (
    apply_normalization,
    compute_normalization_scale,
    load_normalization_scale,
    save_normalization_scale,
)
from utils.core.run_output import load_cfg, make_run_dir, setup_logger
from utils.experiment.benchmark import (
    METRIC_DISPLAY_NAMES,
    _get_experiment_folder_order,
    _load_project_display_names,
    _project_display_name,
    resolve_input_root,
)
from utils.experiment.seen_count_progress import (
    _format_metric_cell,
    _latex_escape,
    _metric_best_keys,
    aggregate_by_project_split,
    collect_rows_with_seen_count,
)

_CFG = load_cfg()
_DEFAULT_EXPERIMENT = "domain_identity"
_METRIC_LATEX_LABELS = {
    "progress": r"Progress$\uparrow$",
    "vit_score": r"ViTScore$\uparrow$",
    "tpkldiv": r"TPKL-Div$\downarrow$",
    "diversity": r"Diversity$\uparrow$",
}
_METRIC_MARKDOWN_LABELS = {
    "progress": "Progress ↑",
    "vit_score": "ViTScore ↑",
    "tpkldiv": "TPKL-Div ↓",
    "diversity": "Diversity ↑",
}
_PRETENDARD_CANDIDATES: tuple[Path, ...] = (
    Path(os.environ.get("PRETENDARD_MEDIUM_PATH", "")),
    Path(os.environ.get("PRETENDARD_REGULAR_PATH", "")),
    Path.home() / "Library/Fonts/Pretendard-Medium.otf",
    Path.home() / "Library/Fonts/Pretendard-Regular.otf",
    Path("/Library/Fonts/Pretendard-Medium.otf"),
    Path("/Library/Fonts/Pretendard-Regular.otf"),
)


def _save_figure_png_pdf(fig, output_path: Path, dpi: int = 350) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", pad_inches=0.015)
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.015)


def _apply_plot_style(plt, sns) -> None:
    font_family = "Pretendard"
    try:
        from matplotlib import font_manager

        for font_path in _PRETENDARD_CANDIDATES:
            if not str(font_path) or not font_path.is_file():
                continue
            font_manager.fontManager.addfont(str(font_path))
            font_family = font_manager.FontProperties(fname=str(font_path)).get_name()
            break
    except Exception:
        pass

    sns.set_theme(
        style="whitegrid",
        font=font_family,
        rc={
            "font.family": font_family,
            "font.sans-serif": [font_family, "Pretendard", "Arial", "Helvetica", "DejaVu Sans"],
            "font.weight": "medium",
            "axes.labelweight": "medium",
        },
    )
    plt.rcParams.update({
        "hatch.linewidth": 0.75,
        "axes.axisbelow": True,
    })


def _ordered_projects(projects: set[str], experiment: str) -> list[str]:
    order = _get_experiment_folder_order(experiment)
    if order:
        ordered = [p for p in order if p in projects]
        ordered += sorted(projects - set(ordered))
        return ordered
    return sorted(projects)


def _plot_label(project: str) -> str:
    label = _project_display_name(project)
    fixed = {
        "Domain name": "Domain\nname",
        "Domain description": "Domain\ndescription",
        "Name + description": "Name +\ndescription",
        "No identity": "No\nidentity",
        "No domain identity": "No domain\nidentity",
        "Domain-specific condition": "Domain-specific\ncondition",
        "Domain-general condition": "Domain-general\ncondition",
        "Domain-specific + general condition": "Domain-specific +\ngeneral condition",
    }
    if label in fixed:
        return fixed[label]
    if " + " in label:
        return label.replace(" + ", " +\n")
    return "\n".join(textwrap.wrap(label, width=18, break_long_words=False)) or label


def _display_metric_label(metric: str) -> str:
    if metric == "progress":
        return "Normalized Progress"
    return METRIC_DISPLAY_NAMES.get(metric, metric)


def _summary_records(rows: list[dict], metric: str, experiment: str) -> list[dict]:
    agg = aggregate_by_project_split(rows, [metric])
    projects = _ordered_projects({r["project"] for r in rows}, experiment)
    records: list[dict] = []
    for project in projects:
        rec = {
            "project": project,
            "method": _project_display_name(project),
            "plot_label": _plot_label(project),
        }
        for split in ("seen", "unseen", "all"):
            stat = agg.get((project, split), {}).get(metric)
            rec[f"{split}_mean"] = float(stat["mean"]) if stat else float("nan")
            rec[f"{split}_std"] = float(stat["std"]) if stat else 0.0
            rec[f"{split}_n"] = int(stat["n"]) if stat else 0
        records.append(rec)
    return records


def write_summary_csv(output_path: Path, records: list[dict], decimals: int) -> None:
    headers = [
        "method",
        "seen_mean",
        "seen_std",
        "seen_n",
        "unseen_mean",
        "unseen_std",
        "unseen_n",
        "all_mean",
        "all_std",
        "all_n",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for rec in records:
            row = {"method": rec["method"]}
            for split in ("seen", "unseen", "all"):
                row[f"{split}_mean"] = f"{rec[f'{split}_mean']:.{decimals}f}"
                row[f"{split}_std"] = f"{rec[f'{split}_std']:.{decimals}f}"
                row[f"{split}_n"] = rec[f"{split}_n"]
            writer.writerow(row)


def write_summary_markdown(
    output_path: Path,
    rows: list[dict],
    records: list[dict],
    metric: str,
    experiment: str,
    decimals: int,
) -> None:
    agg = aggregate_by_project_split(rows, [metric])
    projects = [r["project"] for r in records]
    splits = ["seen", "unseen", "all"]
    best = _metric_best_keys(agg, projects, splits, [metric])
    label = _METRIC_MARKDOWN_LABELS.get(metric, METRIC_DISPLAY_NAMES.get(metric, metric))
    lines = [
        f"| Method | Seen {label} | Unseen {label} | All {label} |",
        "| --- | ---: | ---: | ---: |",
    ]
    for rec in records:
        project = rec["project"]
        cells = [rec["method"]]
        for split in splits:
            cells.append(
                _format_metric_cell(
                    metric,
                    agg.get((project, split), {}).get(metric),
                    decimals,
                    bold=(project, split, metric) in best,
                )
            )
        lines.append("| " + " | ".join(cells) + " |")
    lines += [
        "",
        "Rows aggregate all five leave-one-game folds. Seen and Unseen columns merge every corresponding split row before reporting.",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_summary_latex(
    output_path: Path,
    rows: list[dict],
    records: list[dict],
    metric: str,
    experiment: str,
    decimals: int,
) -> None:
    agg = aggregate_by_project_split(rows, [metric])
    projects = [r["project"] for r in records]
    splits = ["seen", "unseen", "all"]
    best = _metric_best_keys(agg, projects, splits, [metric])
    metric_label = _METRIC_LATEX_LABELS.get(metric, _latex_escape(METRIC_DISPLAY_NAMES.get(metric, metric)))
    caption_title = {
        "domain_identity": "Domain identity ablation.",
        "domain_condition": "Domain condition ablation.",
    }.get(experiment, f"{experiment.replace('_', ' ').title()} ablation.")
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        rf"\caption{{\textbf{{{_latex_escape(caption_title)}}} Five-game aggregated Seen, Unseen, and All performance.}}",
        rf"\label{{tab:{experiment}_summary}}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        rf"Method & Seen {metric_label} & Unseen {metric_label} & All {metric_label} \\",
        r"\midrule",
    ]
    for rec in records:
        project = rec["project"]
        cells = [_latex_escape(rec["method"])]
        for split in splits:
            cells.append(
                _format_metric_cell(
                    metric,
                    agg.get((project, split), {}).get(metric),
                    decimals,
                    bold=(project, split, metric) in best,
                    latex=True,
                )
            )
        lines.append(" & ".join(cells) + r" \\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_seen_unseen_plot(
    output_path: Path,
    records: list[dict],
    metric: str,
    experiment: str,
    ymin: float | None,
) -> None:
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns
    from matplotlib.patches import Patch
    from matplotlib.ticker import MaxNLocator

    _apply_plot_style(plt, sns)

    plot_rows = []
    for rec in records:
        for split in ("seen", "unseen"):
            plot_rows.append({
                "Method": rec["plot_label"],
                "Split": "Seen" if split == "seen" else "Unseen",
                "mean": rec[f"{split}_mean"],
                "std": rec[f"{split}_std"],
            })
    df = pd.DataFrame(plot_rows)

    values = [
        value
        for rec in records
        for split in ("seen", "unseen")
        for value in (rec[f"{split}_mean"] - rec[f"{split}_std"], rec[f"{split}_mean"] + rec[f"{split}_std"])
    ]
    data_min, data_max = min(values), max(values)
    span = max(data_max - data_min, 1e-6)
    bottom = max(0.0, data_min - span * 0.18)
    top = data_max + span * 0.24
    if ymin is not None and metric == "progress":
        bottom = ymin

    fig_width = 5.0 if experiment == "domain_condition" else 4.7
    fig, ax = plt.subplots(figsize=(fig_width, 3.0))
    method_colors = sns.color_palette("Set2", n_colors=max(len(records), 3))
    bar_width = 0.50 if experiment == "domain_condition" else 0.62
    sns.barplot(
        data=df,
        x="Method",
        y="mean",
        hue="Split",
        hue_order=["Seen", "Unseen"],
        palette=["#8c8c8c", "#8c8c8c"],
        width=bar_width,
        errorbar=None,
        ax=ax,
    )

    split_by_container = [("seen", 0.92, None), ("unseen", 0.78, "//////")]
    for container, (split, alpha, hatch) in zip(ax.containers, split_by_container):
        for i, (patch, rec) in enumerate(zip(container.patches, records)):
            center = patch.get_x() + patch.get_width() / 2
            mean = float(rec[f"{split}_mean"])
            std = float(rec[f"{split}_std"])
            patch.set_y(bottom)
            patch.set_height(max(0.0, mean - bottom))
            patch.set_facecolor(method_colors[i % len(method_colors)])
            patch.set_edgecolor("#333333")
            patch.set_linewidth(0.35)
            patch.set_alpha(alpha)
            patch.set_clip_on(True)
            if hatch:
                patch.set_hatch(hatch)
            ax.errorbar(
                center,
                mean,
                yerr=std,
                fmt="none",
                ecolor="#333333",
                elinewidth=0.85,
                capsize=2.0,
                capthick=0.85,
                clip_on=True,
                zorder=4,
            )

    ax.set_xlabel("")
    ax.set_ylabel(_display_metric_label(metric))
    ax.set_ylim(bottom, top)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))

    handles = [
        Patch(facecolor="#8c8c8c", edgecolor="#333333", linewidth=0.35, alpha=0.92, label="Seen"),
        Patch(facecolor="#8c8c8c", edgecolor="#333333", linewidth=0.35, alpha=0.78, hatch="//////", label="Unseen"),
    ]
    labels = ["Seen", "Unseen"]
    ax.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.18),
        ncol=2,
        frameon=False,
        title=None,
    )

    fig.tight_layout()
    _save_figure_png_pdf(fig, output_path)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    exp_names = list(_CFG.get("experiments", {}).keys())
    parser = argparse.ArgumentParser(description="Domain ablation progress plot and table.")
    parser.add_argument("--input", default="wandb_projects", help="Input root under results/, or an absolute path.")
    parser.add_argument("--experiment", choices=exp_names if exp_names else None, default=_DEFAULT_EXPERIMENT)
    parser.add_argument("--metrics", nargs="+", default=None, help="Metric list. The first metric is plotted.")
    parser.add_argument("--decimals", type=int, default=3)
    parser.add_argument("--ymin", type=float, default=None)
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    experiment = args.experiment or _DEFAULT_EXPERIMENT
    output_prefix = experiment
    run_dir = make_run_dir(f"{output_prefix}_progress", cfg=_CFG)
    log = setup_logger(run_dir, name=__file__)

    import utils.experiment.benchmark as _bm

    _bm._PROJECT_DISPLAY_NAMES = _load_project_display_names(experiment)
    input_root = resolve_input_root(args.input, _RESULTS_DIR)
    exp_cfg = _CFG.get("experiments", {}).get(experiment, {})
    metric_order = args.metrics or exp_cfg.get("metrics") or ["progress"]
    if isinstance(metric_order, str):
        metric_order = [metric_order]
    metric = metric_order[0]
    folder_order = _get_experiment_folder_order(experiment)

    log.info("experiment: %s  folder_order=%s", experiment, folder_order)
    log.info("input_root: %s", input_root)

    rows = collect_rows_with_seen_count(input_root, metric_order, target_projects=folder_order or None)
    if not rows:
        msg = f"No valid rows found for {experiment}."
        log.error(msg)
        raise SystemExit(msg)
    if not any(r.get("game_split") == "unseen" for r in rows):
        msg = f"No unseen split rows found for {experiment}."
        log.error(msg)
        raise SystemExit(msg)

    global_scale = os.environ.get("PIPELINE_NORM_SCALE")
    if global_scale and Path(global_scale).is_file():
        norm_scale = load_normalization_scale(Path(global_scale))
        log.info("norm_scale (global): %s", global_scale)
    else:
        norm_scale = compute_normalization_scale(rows, metric_order)
        save_normalization_scale(norm_scale, run_dir / "normalization_scale.json")
        log.info("norm_scale (local) : %s", run_dir / "normalization_scale.json")
    norm_rows = apply_normalization(rows, norm_scale, metric_order)

    records = _summary_records(norm_rows, metric, experiment)
    summary_csv = run_dir / f"{output_prefix}_summary_table.csv"
    summary_md = run_dir / f"{output_prefix}_summary_table.md"
    summary_tex = run_dir / f"{output_prefix}_summary_table.tex"
    plot_path = run_dir / f"{output_prefix}_seen_unseen.png"
    write_summary_csv(summary_csv, records, args.decimals)
    write_summary_markdown(
        summary_md,
        norm_rows,
        records,
        metric,
        experiment,
        args.decimals,
    )
    write_summary_latex(
        summary_tex,
        norm_rows,
        records,
        metric,
        experiment,
        args.decimals,
    )
    log.info("table: %s", summary_csv)
    log.info("table: %s", summary_md)
    log.info("table: %s", summary_tex)

    if not args.no_plot:
        write_seen_unseen_plot(
            plot_path,
            records,
            metric,
            experiment,
            args.ymin,
        )
        log.info("plot : %s", plot_path)

    log.info(
        "rows_found: %d  (unseen: %d  seen: %d)",
        len(rows),
        sum(1 for r in rows if r.get("game_split") == "unseen"),
        sum(1 for r in rows if r.get("game_split") == "seen"),
    )


if __name__ == "__main__":
    main()
