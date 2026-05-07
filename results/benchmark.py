"""
Build benchmark tables from downloaded summary.csv files.

Expected folder layout:
    <input_root>/<project>/<run_name>/summary.csv
or
    <input_root>/<project>/<run_name>/<eval_name>/summary.csv

Each summary.csv should have:
    metric,mean
    progress,...
    vit_score,...
    ...
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# sys.path bootstrap — must happen before any local imports
# ---------------------------------------------------------------------------
import sys as _sys
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
if str(_HERE) not in _sys.path:
    _sys.path.insert(0, str(_HERE))
if str(_ROOT) not in _sys.path:
    _sys.path.append(str(_ROOT))

from utils.run_output import load_cfg
from utils.stats import safe_std, to_float
from utils.io import (
    normalize_reward_enum,
    parse_run_tokens,
    iter_summary_paths,
    iter_results_paths,
    read_summary,
    sort_key_reward_enum,
)
from utils.normalization import (
    compute_normalization_scale,
    apply_normalization,
    save_normalization_scale,
)

_CFG = load_cfg()

DEFAULT_METRIC_ORDER: list[str] = _CFG.get("metrics", {}).get(
    "default_order", ["progress", "vit_score", "tpkldiv", "diversity"]
)
METRIC_DISPLAY_NAMES: dict[str, str] = _CFG.get("metrics", {}).get(
    "display_names", {
        "progress": "Progress",
        "vit_score": "ViTScore",
        "tpkldiv": "TPKL-Div",
        "diversity": "Diversity",
    }
)


def _get_experiment_folder_order(experiment: str | None = None) -> list[str]:
    experiments = _CFG.get("experiments", {})
    if experiment and experiment in experiments:
        return experiments[experiment].get("target_projects", [])
    seen: set[str] = set()
    merged: list[str] = []
    for exp in experiments.values():
        for p in exp.get("target_projects", []):
            if p not in seen:
                seen.add(p)
                merged.append(p)
    return merged or ["aaai27_eval_cpcgrl"]


_PROJECT_DISPLAY_NAMES: dict[str, str] = _CFG.get("project_display_names", {})


def _load_project_display_names(experiment: str | None) -> dict[str, str]:
    return _CFG.get("project_display_names", {})


def _project_display_name(folder: str) -> str:
    return _PROJECT_DISPLAY_NAMES.get(folder, folder)


PREFERRED_PLOT_FOLDER_ORDER: list[str] = _get_experiment_folder_order()


def _count_summary_files(root: Path) -> int:
    return sum(1 for _ in root.rglob("summary.csv"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create benchmark table from wandb_download summary.csv files."
    )
    parser.add_argument("--input", default="wandb_projects",
        help="Root directory that contains <project>/<run>/summary.csv files.")
    parser.add_argument("--group-by",
        choices=["folder", "project_game", "project", "game", "reward_enum", "folder_game_reward_enum"],
        default="folder")
    parser.add_argument("--metrics", nargs="+", default=None)
    parser.add_argument("--output-md", default=None)
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--decimals", type=int, default=4)
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--output-folder-md", default=None)
    parser.add_argument("--output-folder-csv", default=None)
    _exp_names = list(_CFG.get("experiments", {}).keys())
    parser.add_argument("--experiment",
        choices=_exp_names if _exp_names else None,
        default=None, metavar="EXPERIMENT")
    return parser.parse_args()


def resolve_input_root(input_arg: str, script_dir: Path) -> Path:
    raw = Path(input_arg)
    if raw.is_absolute():
        return raw.resolve()

    candidates = [
        (Path.cwd() / raw).resolve(),
        (script_dir / raw).resolve(),
        (script_dir.parent / raw).resolve(),
    ]
    seen_set: set[str] = set()
    unique: list[Path] = []
    for c in candidates:
        if str(c) not in seen_set:
            seen_set.add(str(c))
            unique.append(c)

    best, best_count = unique[0], -1
    for c in unique:
        count = _count_summary_files(c)
        if count > best_count:
            best, best_count = c, count
    return best


# ---------------------------------------------------------------------------
# Data discovery & aggregation
# ---------------------------------------------------------------------------

def discover_rows(input_root: Path, group_by: str) -> tuple[list[dict], set[str]]:
    rows: list[dict] = []
    metric_names: set[str] = set()

    for summary_path in iter_summary_paths(input_root):
        rel = summary_path.relative_to(input_root)
        if len(rel.parts) < 3:
            continue
        project  = rel.parts[0]
        run_name = rel.parts[1]
        eval_name = rel.parts[2] if len(rel.parts) >= 4 else ""

        run_tokens  = parse_run_tokens(run_name)
        eval_tokens = parse_run_tokens(eval_name)
        game = run_tokens.get("game", "unknown")
        reward_enum = normalize_reward_enum(eval_tokens.get(
            "re", run_tokens.get("re", run_tokens.get("reward_enum", "unknown"))
        ))
        metrics = read_summary(summary_path)
        if not metrics:
            continue

        metric_names.update(metrics.keys())
        if group_by == "folder_game_reward_enum":
            group_key = (project, game, reward_enum)
        elif group_by == "project_game":
            group_key = (project, game)
        elif group_by in ("project", "folder"):
            group_key = (project,)
        elif group_by == "reward_enum":
            group_key = (reward_enum,)
        else:
            group_key = (game,)

        rows.append({"group": group_key, "project": project, "game": game,
                     "reward_enum": reward_enum, "run": run_name, "eval": eval_name,
                     "metrics": metrics})
    return rows, metric_names


def resolve_metric_order(selected: list[str] | None, discovered: set[str]) -> list[str]:
    if selected:
        return selected
    ordered = [m for m in DEFAULT_METRIC_ORDER if m in discovered]
    return ordered + sorted(m for m in discovered if m not in ordered)


def aggregate(rows: list[dict], metric_order: list[str]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["group"]].append(row)
    result: list[dict] = []
    for group_key in sorted(grouped.keys()):
        group_rows = grouped[group_key]
        agg: dict = {"group": group_key, "n_runs": len(group_rows), "stats": {}}
        for metric in metric_order:
            values = [r["metrics"][metric] for r in group_rows if metric in r["metrics"]]
            if values:
                agg["stats"][metric] = {"mean": sum(values)/len(values),
                                        "std": safe_std(values), "n": len(values)}
        result.append(agg)
    return result


def aggregate_folder_game_reward(rows: list[dict], metric_order: list[str]) -> dict:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["project"], row["game"], row["reward_enum"])].append(row)
    result = {}
    for key, group_rows in grouped.items():
        stats: dict = {}
        for metric in metric_order:
            values = [r["metrics"][metric] for r in group_rows if metric in r["metrics"]]
            if values:
                stats[metric] = {"mean": sum(values)/len(values),
                                 "std": safe_std(values), "n": len(values)}
        result[key] = stats
    return result


def aggregate_folder_reward_overall(rows: list[dict], metric_order: list[str]) -> dict:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["project"], row["reward_enum"])].append(row)
    result = {}
    for key, group_rows in grouped.items():
        stats: dict = {}
        for metric in metric_order:
            values = [r["metrics"][metric] for r in group_rows if metric in r["metrics"]]
            if values:
                stats[metric] = {"mean": sum(values)/len(values),
                                 "std": safe_std(values), "n": len(values)}
        result[key] = stats
    return result


def collect_plot_rows_from_results(input_root: Path, metric_order: list[str]) -> list[dict]:
    rows: list[dict] = []
    for results_path in iter_results_paths(input_root):
        rel = results_path.relative_to(input_root)
        if len(rel.parts) < 3:
            continue
        project   = rel.parts[0]
        eval_name = rel.parts[2] if len(rel.parts) >= 4 else ""
        eval_tokens = parse_run_tokens(eval_name)

        with results_path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                game = (row.get("game") or "").strip() or "unknown"
                re_raw = (row.get("reward_enum") or "").strip()
                reward_enum = normalize_reward_enum(re_raw or eval_tokens.get("re", "unknown"))
                metric_values = {
                    m: v for m in metric_order
                    if (v := to_float(row.get(m))) is not None
                }
                if metric_values:
                    rows.append({"project": project, "game": game,
                                 "reward_enum": reward_enum, "metrics": metric_values})
    return rows


def aggregate_folder_only(rows: list[dict], metric_order: list[str]) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["project"]].append(row)
    out: list[dict] = []
    for folder in sorted(grouped.keys()):
        folder_rows = grouped[folder]
        rec: dict = {"folder": folder, "n_rows": len(folder_rows), "stats": {}}
        for metric in metric_order:
            values = [r["metrics"][metric] for r in folder_rows if metric in r["metrics"]]
            if values:
                rec["stats"][metric] = {"mean": sum(values)/len(values),
                                        "std": safe_std(values), "n": len(values)}
        out.append(rec)
    return out


# ---------------------------------------------------------------------------
# Sort helpers
# ---------------------------------------------------------------------------

def _sort_folder_for_plot(value: str) -> tuple[int, int | str]:
    try:
        return (0, PREFERRED_PLOT_FOLDER_ORDER.index(value))
    except ValueError:
        return (1, value)


# ---------------------------------------------------------------------------
# Table writers
# ---------------------------------------------------------------------------

def group_headers(group_by: str) -> list[str]:
    if group_by == "folder_game_reward_enum": return ["folder", "game", "reward_enum"]
    if group_by == "project_game":            return ["folder", "game"]
    if group_by in ("project", "folder"):     return ["folder"]
    if group_by == "reward_enum":             return ["reward_enum"]
    return ["game"]


def group_cells(group_by: str, group: tuple) -> dict[str, str]:
    headers = group_headers(group_by)
    values  = list(group) + [""] * max(0, len(headers) - len(group))
    return {k: v for k, v in zip(headers, values)}


def write_markdown_table(output_path: Path, grouped_rows: list[dict],
                         metric_order: list[str], group_by: str, decimals: int) -> None:
    headers = group_headers(group_by) + ["n_runs"] + metric_order
    lines = ["| " + " | ".join(headers) + " |",
             "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in grouped_rows:
        cells = list(group_cells(group_by, row["group"]).values()) + [str(row["n_runs"])]
        for metric in metric_order:
            stat = row["stats"].get(metric)
            cells.append(f"{stat['mean']:.{decimals}f} +- {stat['std']:.{decimals}f}" if stat else "-")
        lines.append("| " + " | ".join(cells) + " |")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_csv_table(output_path: Path, grouped_rows: list[dict],
                    metric_order: list[str], group_by: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    headers = group_headers(group_by) + ["n_runs"]
    for m in metric_order:
        headers += [f"{m}_mean", f"{m}_std", f"{m}_n"]
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in grouped_rows:
            rec: dict = {**group_cells(group_by, row["group"]), "n_runs": row["n_runs"]}
            for m in metric_order:
                stat = row["stats"].get(m)
                rec[f"{m}_mean"] = stat["mean"] if stat else ""
                rec[f"{m}_std"]  = stat["std"]  if stat else ""
                rec[f"{m}_n"]    = stat["n"]    if stat else 0
            writer.writerow(rec)


def write_folder_only_markdown(output_path: Path, folder_rows: list[dict],
                                metric_order: list[str], decimals: int) -> None:
    headers = ["folder", "n_rows"] + metric_order
    lines = ["| " + " | ".join(headers) + " |",
             "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in folder_rows:
        cells = [row["folder"], str(row["n_rows"])]
        for m in metric_order:
            stat = row["stats"].get(m)
            cells.append(f"{stat['mean']:.{decimals}f} +- {stat['std']:.{decimals}f}" if stat else "-")
        lines.append("| " + " | ".join(cells) + " |")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_folder_only_csv(output_path: Path, folder_rows: list[dict],
                           metric_order: list[str]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    headers = ["folder", "n_rows"]
    for m in metric_order:
        headers += [f"{m}_mean", f"{m}_std", f"{m}_n"]
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in folder_rows:
            rec: dict = {"folder": row["folder"], "n_rows": row["n_rows"]}
            for m in metric_order:
                stat = row["stats"].get(m)
                rec[f"{m}_mean"] = stat["mean"] if stat else ""
                rec[f"{m}_std"]  = stat["std"]  if stat else ""
                rec[f"{m}_n"]    = stat["n"]    if stat else 0
            writer.writerow(rec)


# ---------------------------------------------------------------------------
# Plot functions
# ---------------------------------------------------------------------------

def _bar_plot_setup():
    try:
        import matplotlib.pyplot as plt
        return plt
    except Exception as e:
        raise RuntimeError("Failed to import matplotlib. Use --no-plot.") from e


def _palette(n: int):
    try:
        import seaborn as sns
        sns.set_theme(style="whitegrid", context="notebook")
        return sns.color_palette("Set2", n_colors=max(n, 3))
    except Exception:
        import matplotlib.pyplot as plt
        return plt.cm.Set2.colors


def write_game_reward_subplots(output_path: Path, plot_rows: list[dict],
                                metric_order: list[str]) -> None:
    plt = _bar_plot_setup()
    grouped_stats         = aggregate_folder_game_reward(plot_rows, metric_order)
    grouped_stats_overall = aggregate_folder_reward_overall(plot_rows, metric_order)
    folders = sorted({f for f, _, _ in grouped_stats}, key=_sort_folder_for_plot)
    colors  = _palette(len(folders))
    games   = sorted({g for _, g, _ in grouped_stats})
    rewards = sorted({r for _, _, r in grouped_stats}, key=sort_key_reward_enum)

    n_metrics = len(metric_order)
    if not n_metrics:
        return
    column_keys: list[str | None] = [None] + games
    fig, axes = plt.subplots(n_metrics, len(column_keys),
                             figsize=(3.6 * len(column_keys), 2.7 * n_metrics), squeeze=False)
    axes_flat = [ax for row in axes for ax in row]
    x_center  = list(range(len(rewards)))

    for ri, metric in enumerate(metric_order):
        metric_label = METRIC_DISPLAY_NAMES.get(metric, metric)
        for ci, game_key in enumerate(column_keys):
            ax    = axes[ri][ci]
            width = 0.8 / max(len(folders), 1)
            drew_any, y_uppers = False, []
            for j, folder in enumerate(folders):
                means, stds, xs = [], [], []
                for k, re in enumerate(rewards):
                    stat = (grouped_stats_overall.get((folder, re), {})
                            if game_key is None
                            else grouped_stats.get((folder, game_key, re), {})).get(metric)
                    if not stat:
                        continue
                    xs.append(x_center[k] - 0.4 + (j + 0.5) * width)
                    means.append(float(stat["mean"]))
                    stds.append(float(stat["std"]))
                    y_uppers.append(float(stat["mean"]) + float(stat["std"]))
                if not means:
                    continue
                drew_any = True
                ax.bar(xs, means, width=width, yerr=stds, capsize=2,
                       label=_project_display_name(folder),
                       color=colors[j % len(colors)], edgecolor="white", linewidth=0.8, alpha=0.9)
            if ri == 0:
                ax.set_title("overall" if game_key is None else game_key)
            if ci == 0:
                ax.set_ylabel(metric_label, rotation=90, labelpad=8)
            ax.set_xticks(x_center, [f"re={r}" for r in rewards])
            ax.tick_params(axis="x", labelrotation=0)
            ax.set_xlim(-0.5, len(rewards) - 0.5)
            ax.grid(axis="y", alpha=0.3)
            if drew_any and y_uppers:
                dm = max(y_uppers)
                pad = max(dm, 1e-6) * 0.12
                ax.set_ylim(0, dm + pad)
            else:
                ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center")

    handles, labels = [], []
    for ax in axes_flat:
        h, l = ax.get_legend_handles_labels()
        if h:
            handles, labels = h, l
            break
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(len(labels), 6))
        fig.subplots_adjust(top=0.88)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_overall_simple_plot(output_path: Path, plot_rows: list[dict],
                               metric_order: list[str]) -> None:
    plt = _bar_plot_setup()
    grouped_overall = aggregate_folder_reward_overall(plot_rows, metric_order)
    folders = sorted({f for f, _ in grouped_overall}, key=_sort_folder_for_plot)
    rewards = sorted({r for _, r in grouped_overall}, key=sort_key_reward_enum)
    colors  = _palette(len(folders))

    n_metrics = len(metric_order)
    if not n_metrics:
        return
    fig, axes = plt.subplots(1, n_metrics, figsize=(3.8 * n_metrics, 3.0), squeeze=False)
    x_center  = list(range(len(rewards)))

    for ci, metric in enumerate(metric_order):
        ax    = axes[0][ci]
        width = 0.8 / max(len(folders), 1)
        drew_any, y_uppers = False, []
        for j, folder in enumerate(folders):
            means, stds, xs = [], [], []
            for k, re in enumerate(rewards):
                stat = grouped_overall.get((folder, re), {}).get(metric)
                if not stat:
                    continue
                xs.append(x_center[k] - 0.4 + (j + 0.5) * width)
                means.append(float(stat["mean"]))
                stds.append(float(stat["std"]))
                y_uppers.append(float(stat["mean"]) + float(stat["std"]))
            if not means:
                continue
            drew_any = True
            ax.bar(xs, means, width=width, yerr=stds, capsize=2,
                   label=_project_display_name(folder),
                   color=colors[j % len(colors)], edgecolor="white", linewidth=0.8, alpha=0.9)
        ax.set_title(METRIC_DISPLAY_NAMES.get(metric, metric))
        if ci == 0:
            ax.set_ylabel("overall", rotation=90, labelpad=8)
        ax.set_xticks(x_center, [f"re={r}" for r in rewards])
        ax.tick_params(axis="x", labelrotation=0)
        ax.set_xlim(-0.5, len(rewards) - 0.5)
        ax.grid(axis="y", alpha=0.3)
        if drew_any and y_uppers:
            dm  = max(y_uppers)
            pad = max(dm, 1e-6) * 0.12
            ax.set_ylim(0, dm + pad)
        else:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center")

    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(len(labels), 6))
        fig.subplots_adjust(top=0.82)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    from utils.run_output import make_run_dir, setup_logger
    from instruct_rl.utils.log_utils import get_logger  # noqa: F401 (side-effect import)

    args    = parse_args()
    run_dir = make_run_dir("benchmark", cfg=_CFG)
    log     = setup_logger(run_dir, name=__file__)
    log.debug("run_dir   : %s", run_dir)

    folder_order = _get_experiment_folder_order(args.experiment)
    if args.experiment:
        log.info("experiment: %s  folder_order=%s", args.experiment, folder_order)
    global PREFERRED_PLOT_FOLDER_ORDER, _PROJECT_DISPLAY_NAMES
    PREFERRED_PLOT_FOLDER_ORDER = folder_order
    _PROJECT_DISPLAY_NAMES      = _load_project_display_names(args.experiment)

    script_dir = Path(__file__).resolve().parent
    input_root = resolve_input_root(args.input, script_dir)
    output_md         = Path(args.output_md).resolve()         if args.output_md         else run_dir / "benchmark_table.md"
    output_csv        = Path(args.output_csv).resolve()        if args.output_csv        else run_dir / "benchmark_table.csv"
    output_folder_md  = Path(args.output_folder_md).resolve()  if args.output_folder_md  else run_dir / "benchmark_folder_mean.md"
    output_folder_csv = Path(args.output_folder_csv).resolve() if args.output_folder_csv else run_dir / "benchmark_folder_mean.csv"

    rows, discovered_metrics = discover_rows(input_root=input_root, group_by=args.group_by)

    # experiment 지정 시 target_projects 이외 프로젝트 제거
    target_projects: list[str] = folder_order
    if target_projects:
        rows = [r for r in rows if r["project"] in target_projects]
        discovered_metrics = {m for r in rows for m in r["metrics"]}

    if not rows:
        msg = f"No valid summary.csv rows found under: {input_root}"
        log.error(msg)
        raise SystemExit(msg)

    metric_order = resolve_metric_order(args.metrics, discovered_metrics)
    grouped_rows = aggregate(rows=rows, metric_order=metric_order)

    write_markdown_table(output_md,  grouped_rows, metric_order, args.group_by, args.decimals)
    write_csv_table(output_csv, grouped_rows, metric_order, args.group_by)
    log.debug("table_md  : %s", output_md)
    log.debug("table_csv : %s", output_csv)

    plot_rows = collect_plot_rows_from_results(input_root, metric_order)
    if target_projects:
        plot_rows = [r for r in plot_rows if r["project"] in target_projects]

    if plot_rows:
        folder_rows = aggregate_folder_only(plot_rows, metric_order)
        write_folder_only_markdown(output_folder_md,  folder_rows, metric_order, args.decimals)
        write_folder_only_csv(output_folder_csv, folder_rows, metric_order)
        log.debug("folder_md : %s", output_folder_md)
        log.debug("folder_csv: %s", output_folder_csv)
    else:
        log.warning("No valid rows in results.csv — skipped folder-only mean outputs.")

    if not args.no_plot and not plot_rows:
        msg = "No valid plot rows found from results.csv under input root."
        log.error(msg)
        raise SystemExit(msg)

    # ---- Normalization + plots ----
    if plot_rows:
        norm_scale     = compute_normalization_scale(plot_rows, metric_order)
        norm_scale_path = run_dir / "normalization_scale.json"
        save_normalization_scale(norm_scale, norm_scale_path)
        log.info("norm_scale: %s", norm_scale_path)

        norm_rows = apply_normalization(plot_rows, norm_scale, metric_order)

        norm_folder_rows = aggregate_folder_only(norm_rows, metric_order)
        write_folder_only_markdown(run_dir / "normalized_folder_mean.md",
                                   norm_folder_rows, metric_order, args.decimals)
        write_folder_only_csv(run_dir / "normalized_folder_mean.csv",
                               norm_folder_rows, metric_order)
        log.debug("norm_md   : %s", run_dir / "normalized_folder_mean.md")
        log.debug("norm_csv  : %s", run_dir / "normalized_folder_mean.csv")

        if not args.no_plot:
            try:
                write_overall_simple_plot(run_dir / "re.png",      norm_rows, DEFAULT_METRIC_ORDER.copy())
                write_game_reward_subplots(run_dir / "re_game.png", norm_rows, metric_order)
                log.info("plot      : %s", run_dir / "re.png")
                log.info("plot_game : %s", run_dir / "re_game.png")
            except RuntimeError as e:
                log.error("Plot generation failed: %s", e)
                raise SystemExit(str(e)) from e

    log.info("input_root: %s", input_root)
    log.info("rows_found: %d  groups=%d", len(rows), len(grouped_rows))


if __name__ == "__main__":
    main()
