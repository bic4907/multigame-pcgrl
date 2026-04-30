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

DEFAULT_METRIC_ORDER = ["progress", "vit_score", "tpkldiv", "diversity"]
METRIC_DISPLAY_NAMES = {
    "progress": "Progress",
    "vit_score": "ViTScore",
    "tpkldiv": "TPKL-Div",
    "diversity": "Diversity",
}
PREFERRED_PLOT_FOLDER_ORDER = [
    "aaai27_eval_cpcgrl",
    "aaai27_eval_cpcgrl_gamegroup",
    "aaai27_eval_cpcgrl_all",
]


def _count_summary_files(root: Path) -> int:
    return sum(1 for _ in root.rglob("summary.csv"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create benchmark table from wandb_download summary.csv files."
    )
    parser.add_argument(
        "--input",
        default="results/eval",
        help="Root directory that contains <project>/<run>/summary.csv files.",
    )
    parser.add_argument(
        "--group-by",
        choices=[
            "folder",
            "project_game",
            "project",
            "game",
            "reward_enum",
            "folder_game_reward_enum",
        ],
        default="folder",
        help=(
            "Grouping key for benchmark rows. "
            "'folder' groups by top-level folder name under input root "
            "(e.g., aaai27_eval_cpcgrl)."
        ),
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=None,
        help="Metrics to include. Default: auto-detected with preferred order.",
    )
    parser.add_argument(
        "--output-md",
        default=None,
        help="Markdown output path. Default: <input>/benchmark_table.md",
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        help="CSV output path. Default: <input>/benchmark_table.csv",
    )
    parser.add_argument(
        "--decimals",
        type=int,
        default=4,
        help="Decimal places for mean/std values.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip subplot figure generation.",
    )
    parser.add_argument(
        "--plot-file",
        default=None,
        help="Plot output path. Default: <input>/benchmark_game_reward_enum.png",
    )
    parser.add_argument(
        "--plot-file-simple",
        default=None,
        help="Simple overall-only plot path. Default: <input>/benchmark_overall_simple.png",
    )
    parser.add_argument(
        "--output-folder-md",
        default=None,
        help="Folder-only markdown output path. Default: <input>/benchmark_folder_mean.md",
    )
    parser.add_argument(
        "--output-folder-csv",
        default=None,
        help="Folder-only CSV output path. Default: <input>/benchmark_folder_mean.csv",
    )
    return parser.parse_args()


def resolve_input_root(input_arg: str, script_dir: Path) -> Path:
    raw = Path(input_arg)
    if raw.is_absolute():
        return raw.resolve()

    candidates: list[Path] = []
    cwd = Path.cwd()
    candidates.append((cwd / raw).resolve())
    candidates.append((script_dir / raw).resolve())
    candidates.append((script_dir.parent / raw).resolve())

    # Common case: running this script inside "results/" with default input.
    # "results/wandb_download" should map to "<repo>/results/wandb_download".
    if len(raw.parts) >= 2 and raw.parts[0] == "results":
        candidates.append((script_dir / Path(*raw.parts[1:])).resolve())
    if raw.name == "wandb_download":
        # Newer downloader defaults to results/eval, and running from "results/"
        # may create nested paths like results/results/eval.
        candidates.extend(
            [
                (cwd / "results/eval").resolve(),
                (cwd / "results/results/eval").resolve(),
                (script_dir / "eval").resolve(),
                (script_dir / "results/eval").resolve(),
                (script_dir.parent / "results/eval").resolve(),
            ]
        )

    unique_candidates: list[Path] = []
    seen = set()
    for c in candidates:
        key = str(c)
        if key in seen:
            continue
        seen.add(key)
        unique_candidates.append(c)

    best = unique_candidates[0]
    best_count = -1
    for c in unique_candidates:
        count = _count_summary_files(c)
        if count > best_count:
            best = c
            best_count = count
    return best


def parse_run_tokens(run_name: str) -> dict[str, str]:
    tokens: dict[str, str] = {}
    for part in run_name.split("_"):
        if "-" not in part:
            continue
        key, value = part.split("-", 1)
        tokens[key] = value
    return tokens


def _iter_summary_paths(input_root: Path) -> list[Path]:
    return sorted(p for p in input_root.rglob("summary.csv") if p.is_file())


def _iter_results_paths(input_root: Path) -> list[Path]:
    return sorted(p for p in input_root.rglob("results.csv") if p.is_file())


def read_summary(summary_path: Path) -> dict[str, float]:
    metrics: dict[str, float] = {}
    with summary_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            metric = (row.get("metric") or "").strip()
            mean_text = (row.get("mean") or "").strip()
            if not metric or not mean_text:
                continue
            try:
                metrics[metric] = float(mean_text)
            except ValueError:
                continue
    return metrics


def _to_float(text: str | None) -> float | None:
    if text is None:
        return None
    s = text.strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def safe_std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    return math.sqrt(variance)


def discover_rows(input_root: Path, group_by: str) -> tuple[list[dict], set[str]]:
    rows: list[dict] = []
    metric_names: set[str] = set()

    for summary_path in _iter_summary_paths(input_root):
        rel = summary_path.relative_to(input_root)
        # Need at least: <project>/<run>/.../summary.csv
        if len(rel.parts) < 3:
            continue

        project = rel.parts[0]
        run_name = rel.parts[1]
        eval_name = rel.parts[2] if len(rel.parts) >= 4 else ""

        run_tokens = parse_run_tokens(run_name)
        eval_tokens = parse_run_tokens(eval_name)
        game = run_tokens.get("game", "unknown")
        reward_enum = eval_tokens.get(
            "re",
            run_tokens.get("re", run_tokens.get("reward_enum", "unknown")),
        )
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

        rows.append(
            {
                "group": group_key,
                "project": project,
                "game": game,
                "reward_enum": reward_enum,
                "run": run_name,
                "eval": eval_name,
                "metrics": metrics,
            }
        )

    return rows, metric_names


def resolve_metric_order(selected: list[str] | None, discovered: set[str]) -> list[str]:
    if selected:
        return selected
    ordered: list[str] = [m for m in DEFAULT_METRIC_ORDER if m in discovered]
    leftovers = sorted(m for m in discovered if m not in ordered)
    return ordered + leftovers


def aggregate(rows: list[dict], metric_order: list[str]) -> list[dict]:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["group"]].append(row)

    result: list[dict] = []
    for group_key in sorted(grouped.keys()):
        group_rows = grouped[group_key]
        agg = {"group": group_key, "n_runs": len(group_rows), "stats": {}}
        for metric in metric_order:
            values = [
                r["metrics"][metric]
                for r in group_rows
                if metric in r["metrics"]
            ]
            if not values:
                continue
            agg["stats"][metric] = {
                "mean": sum(values) / len(values),
                "std": safe_std(values),
                "n": len(values),
            }
        result.append(agg)
    return result


def aggregate_folder_game_reward(
    rows: list[dict],
    metric_order: list[str],
) -> dict[tuple[str, str, str], dict]:
    grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["project"], row["game"], row["reward_enum"])].append(row)

    result: dict[tuple[str, str, str], dict] = {}
    for key, group_rows in grouped.items():
        stats: dict[str, dict[str, float | int]] = {}
        for metric in metric_order:
            values = [r["metrics"][metric] for r in group_rows if metric in r["metrics"]]
            if not values:
                continue
            stats[metric] = {
                "mean": sum(values) / len(values),
                "std": safe_std(values),
                "n": len(values),
            }
        result[key] = stats
    return result


def aggregate_folder_reward_overall(
    rows: list[dict],
    metric_order: list[str],
) -> dict[tuple[str, str], dict]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["project"], row["reward_enum"])].append(row)

    result: dict[tuple[str, str], dict] = {}
    for key, group_rows in grouped.items():
        stats: dict[str, dict[str, float | int]] = {}
        for metric in metric_order:
            values = [r["metrics"][metric] for r in group_rows if metric in r["metrics"]]
            if not values:
                continue
            stats[metric] = {
                "mean": sum(values) / len(values),
                "std": safe_std(values),
                "n": len(values),
            }
        result[key] = stats
    return result


def collect_plot_rows_from_results(
    input_root: Path,
    metric_order: list[str],
) -> list[dict]:
    rows: list[dict] = []
    for results_path in _iter_results_paths(input_root):
        rel = results_path.relative_to(input_root)
        if len(rel.parts) < 3:
            continue
        project = rel.parts[0]
        eval_name = rel.parts[2] if len(rel.parts) >= 4 else ""
        eval_tokens = parse_run_tokens(eval_name)

        with results_path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                game = (row.get("game") or "").strip() or "unknown"
                reward_enum = (row.get("reward_enum") or "").strip()
                if not reward_enum:
                    reward_enum = eval_tokens.get("re", "unknown")

                metric_values: dict[str, float] = {}
                for metric in metric_order:
                    v = _to_float(row.get(metric))
                    if v is None:
                        continue
                    metric_values[metric] = v
                if not metric_values:
                    continue

                rows.append(
                    {
                        "project": project,
                        "game": game,
                        "reward_enum": reward_enum,
                        "metrics": metric_values,
                    }
                )
    return rows


def _sort_reward_enum(value: str) -> tuple[int, float | str]:
    if value == "all":
        return (0, -1.0)
    try:
        return (0, float(value))
    except ValueError:
        return (1, value)


def _sort_folder_for_plot(value: str) -> tuple[int, int | str]:
    try:
        return (0, PREFERRED_PLOT_FOLDER_ORDER.index(value))
    except ValueError:
        return (1, value)


def write_game_reward_subplots(
    output_path: Path,
    plot_rows: list[dict],
    metric_order: list[str],
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        raise RuntimeError(
            "Failed to import matplotlib. Install compatible matplotlib/numpy "
            "or run with --no-plot."
        ) from e

    grouped_stats = aggregate_folder_game_reward(plot_rows, metric_order)
    grouped_stats_overall = aggregate_folder_reward_overall(plot_rows, metric_order)
    folders = sorted({folder for folder, _, _ in grouped_stats.keys()}, key=_sort_folder_for_plot)
    try:
        import seaborn as sns

        sns.set_theme(style="whitegrid", context="notebook")
        colors = sns.color_palette("Set2", n_colors=max(len(folders), 3))
    except Exception:
        colors = plt.cm.Set2.colors

    games = sorted({game for _, game, _ in grouped_stats.keys()})
    reward_values_all = {reward for _, _, reward in grouped_stats.keys()}
    rewards_global = sorted(reward_values_all, key=_sort_reward_enum)

    n_metrics = len(metric_order)
    if n_metrics == 0:
        return
    n_rows = n_metrics
    column_keys: list[str | None] = [None] + games
    n_cols = max(len(column_keys), 1)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.6 * n_cols, 2.7 * n_rows), squeeze=False)
    axes_flat = [ax for row_axes in axes for ax in row_axes]

    for row_idx, metric in enumerate(metric_order):
        metric_label = METRIC_DISPLAY_NAMES.get(metric, metric)
        for col_idx, game_key in enumerate(column_keys):
            ax = axes[row_idx][col_idx]
            rewards = rewards_global
            x_center = list(range(len(rewards)))
            width = 0.8 / max(len(folders), 1)

            drew_any = False
            y_lowers: list[float] = []
            y_uppers: list[float] = []
            for j, folder in enumerate(folders):
                means: list[float] = []
                stds: list[float] = []
                xs: list[float] = []
                for k, reward_enum in enumerate(rewards):
                    if game_key is None:
                        stat = grouped_stats_overall.get((folder, reward_enum), {}).get(metric)
                    else:
                        stat = grouped_stats.get((folder, game_key, reward_enum), {}).get(metric)
                    if not stat:
                        continue
                    x = x_center[k] - 0.4 + (j + 0.5) * width
                    xs.append(x)
                    means.append(float(stat["mean"]))
                    stds.append(float(stat["std"]))
                    y_lowers.append(float(stat["mean"]) - float(stat["std"]))
                    y_uppers.append(float(stat["mean"]) + float(stat["std"]))
                if not means:
                    continue
                drew_any = True
                ax.bar(
                    xs,
                    means,
                    width=width,
                    yerr=stds,
                    capsize=2,
                    label=folder,
                    color=colors[j % len(colors)],
                    edgecolor="white",
                    linewidth=0.8,
                    alpha=0.9,
                )

            if row_idx == 0:
                ax.set_title("overall" if game_key is None else game_key)
            if col_idx == 0:
                ax.set_ylabel(metric_label, rotation=90, labelpad=8)
            ax.set_xticks(x_center, [f"re={r}" for r in rewards])
            ax.tick_params(axis="x", labelrotation=0)
            ax.set_xlim(-0.5, len(rewards) - 0.5)
            ax.grid(axis="y", alpha=0.3)
            if drew_any and y_lowers and y_uppers:
                data_min = min(y_lowers)
                data_max = max(y_uppers)
                span = data_max - data_min
                if span <= 0:
                    span = max(abs(data_max), 1.0) * 0.1
                pad = span * 0.12
                ax.set_ylim(data_min - pad, data_max + pad)
            else:
                ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center")

    legend_handles = []
    legend_labels = []
    for ax in axes_flat:
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            legend_handles = handles
            legend_labels = labels
            break
    if legend_handles:
        fig.legend(legend_handles, legend_labels, loc="upper center", ncol=min(len(legend_labels), 6))
        fig.subplots_adjust(top=0.88)
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    else:
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_overall_simple_plot(
    output_path: Path,
    plot_rows: list[dict],
    metric_order: list[str],
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        raise RuntimeError(
            "Failed to import matplotlib. Install compatible matplotlib/numpy "
            "or run with --no-plot."
        ) from e

    grouped_stats_overall = aggregate_folder_reward_overall(plot_rows, metric_order)
    folders = sorted({folder for folder, _ in grouped_stats_overall.keys()}, key=_sort_folder_for_plot)
    reward_values_all = {reward for _, reward in grouped_stats_overall.keys()}
    rewards_global = sorted(reward_values_all, key=_sort_reward_enum)

    try:
        import seaborn as sns

        sns.set_theme(style="whitegrid", context="notebook")
        colors = sns.color_palette("Set2", n_colors=max(len(folders), 3))
    except Exception:
        colors = plt.cm.Set2.colors

    n_metrics = len(metric_order)
    if n_metrics == 0:
        return

    fig, axes = plt.subplots(1, n_metrics, figsize=(3.8 * n_metrics, 3.0), squeeze=False)
    axes_row = axes[0]

    for col_idx, metric in enumerate(metric_order):
        ax = axes_row[col_idx]
        metric_label = METRIC_DISPLAY_NAMES.get(metric, metric)
        rewards = rewards_global
        x_center = list(range(len(rewards)))
        width = 0.8 / max(len(folders), 1)

        drew_any = False
        y_lowers: list[float] = []
        y_uppers: list[float] = []
        for j, folder in enumerate(folders):
            means: list[float] = []
            stds: list[float] = []
            xs: list[float] = []
            for k, reward_enum in enumerate(rewards):
                stat = grouped_stats_overall.get((folder, reward_enum), {}).get(metric)
                if not stat:
                    continue
                x = x_center[k] - 0.4 + (j + 0.5) * width
                xs.append(x)
                means.append(float(stat["mean"]))
                stds.append(float(stat["std"]))
                y_lowers.append(float(stat["mean"]) - float(stat["std"]))
                y_uppers.append(float(stat["mean"]) + float(stat["std"]))
            if not means:
                continue
            drew_any = True
            ax.bar(
                xs,
                means,
                width=width,
                yerr=stds,
                capsize=2,
                label=folder,
                color=colors[j % len(colors)],
                edgecolor="white",
                linewidth=0.8,
                alpha=0.9,
            )

        ax.set_title(metric_label)
        if col_idx == 0:
            ax.set_ylabel("overall", rotation=90, labelpad=8)
        ax.set_xticks(x_center, [f"re={r}" for r in rewards])
        ax.tick_params(axis="x", labelrotation=0)
        ax.set_xlim(-0.5, len(rewards) - 0.5)
        ax.grid(axis="y", alpha=0.3)

        if drew_any and y_lowers and y_uppers:
            data_min = min(y_lowers)
            data_max = max(y_uppers)
            span = data_max - data_min
            if span <= 0:
                span = max(abs(data_max), 1.0) * 0.1
            pad = span * 0.12
            ax.set_ylim(data_min - pad, data_max + pad)
        else:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center")

    handles, labels = axes_row[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(len(labels), 6))
        fig.subplots_adjust(top=0.82)
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    else:
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def group_headers(group_by: str) -> list[str]:
    if group_by == "folder_game_reward_enum":
        return ["folder", "game", "reward_enum"]
    if group_by == "project_game":
        return ["folder", "game"]
    if group_by in ("project", "folder"):
        return ["folder"]
    if group_by == "reward_enum":
        return ["reward_enum"]
    return ["game"]


def group_cells(group_by: str, group: tuple[str, ...]) -> dict[str, str]:
    headers = group_headers(group_by)
    values = list(group)
    if len(values) < len(headers):
        values.extend([""] * (len(headers) - len(values)))
    return {k: v for k, v in zip(headers, values)}


def write_markdown_table(
    output_path: Path,
    grouped_rows: list[dict],
    metric_order: list[str],
    group_by: str,
    decimals: int,
) -> None:
    headers = group_headers(group_by)
    headers.append("n_runs")
    headers.extend(metric_order)

    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

    for row in grouped_rows:
        row_group_cells = group_cells(group_by, row["group"])
        values = []
        for h in group_headers(group_by):
            values.append(row_group_cells.get(h, ""))
        values.append(str(row["n_runs"]))

        for metric in metric_order:
            stat = row["stats"].get(metric)
            if not stat:
                values.append("-")
                continue
            mean_text = f"{stat['mean']:.{decimals}f}"
            std_text = f"{stat['std']:.{decimals}f}"
            values.append(f"{mean_text} +- {std_text}")

        lines.append("| " + " | ".join(values) + " |")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_csv_table(
    output_path: Path,
    grouped_rows: list[dict],
    metric_order: list[str],
    group_by: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    headers = group_headers(group_by)
    headers.append("n_runs")

    for metric in metric_order:
        headers.extend([f"{metric}_mean", f"{metric}_std", f"{metric}_n"])

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in grouped_rows:
            rec: dict[str, str | int | float] = {}
            rec.update(group_cells(group_by, row["group"]))
            rec["n_runs"] = row["n_runs"]

            for metric in metric_order:
                stat = row["stats"].get(metric)
                if not stat:
                    rec[f"{metric}_mean"] = ""
                    rec[f"{metric}_std"] = ""
                    rec[f"{metric}_n"] = 0
                    continue
                rec[f"{metric}_mean"] = stat["mean"]
                rec[f"{metric}_std"] = stat["std"]
                rec[f"{metric}_n"] = stat["n"]

            writer.writerow(rec)


def aggregate_folder_only(rows: list[dict], metric_order: list[str]) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["project"]].append(row)

    out: list[dict] = []
    for folder in sorted(grouped.keys()):
        folder_rows = grouped[folder]
        rec = {"folder": folder, "n_rows": len(folder_rows), "stats": {}}
        for metric in metric_order:
            values = [r["metrics"][metric] for r in folder_rows if metric in r["metrics"]]
            if not values:
                continue
            rec["stats"][metric] = {
                "mean": sum(values) / len(values),
                "std": safe_std(values),
                "n": len(values),
            }
        out.append(rec)
    return out


def write_folder_only_markdown(
    output_path: Path,
    folder_rows: list[dict],
    metric_order: list[str],
    decimals: int,
) -> None:
    headers = ["folder", "n_rows"] + metric_order
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

    for row in folder_rows:
        values = [row["folder"], str(row["n_rows"])]
        for metric in metric_order:
            stat = row["stats"].get(metric)
            if not stat:
                values.append("-")
                continue
            mean_text = f"{stat['mean']:.{decimals}f}"
            std_text = f"{stat['std']:.{decimals}f}"
            values.append(f"{mean_text} +- {std_text}")
        lines.append("| " + " | ".join(values) + " |")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_folder_only_csv(
    output_path: Path,
    folder_rows: list[dict],
    metric_order: list[str],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    headers = ["folder", "n_rows"]
    for metric in metric_order:
        headers.extend([f"{metric}_mean", f"{metric}_std", f"{metric}_n"])

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in folder_rows:
            rec: dict[str, str | int | float] = {
                "folder": row["folder"],
                "n_rows": row["n_rows"],
            }
            for metric in metric_order:
                stat = row["stats"].get(metric)
                if not stat:
                    rec[f"{metric}_mean"] = ""
                    rec[f"{metric}_std"] = ""
                    rec[f"{metric}_n"] = 0
                    continue
                rec[f"{metric}_mean"] = stat["mean"]
                rec[f"{metric}_std"] = stat["std"]
                rec[f"{metric}_n"] = stat["n"]
            writer.writerow(rec)


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    input_root = resolve_input_root(args.input, script_dir)
    output_md = Path(args.output_md).resolve() if args.output_md else input_root / "benchmark_table.md"
    output_csv = Path(args.output_csv).resolve() if args.output_csv else input_root / "benchmark_table.csv"
    output_plot = (
        Path(args.plot_file).resolve()
        if args.plot_file
        else input_root / "benchmark_game_reward_enum.png"
    )
    output_plot_simple = (
        Path(args.plot_file_simple).resolve()
        if args.plot_file_simple
        else input_root / "benchmark_overall_simple.png"
    )
    output_folder_md = (
        Path(args.output_folder_md).resolve()
        if args.output_folder_md
        else input_root / "benchmark_folder_mean.md"
    )
    output_folder_csv = (
        Path(args.output_folder_csv).resolve()
        if args.output_folder_csv
        else input_root / "benchmark_folder_mean.csv"
    )

    rows, discovered_metrics = discover_rows(input_root=input_root, group_by=args.group_by)
    if not rows:
        hint_paths = [
            Path("results/wandb_download"),
            Path("results/eval"),
            Path("results/results/eval"),
        ]
        existing_hints: list[str] = []
        for p in hint_paths:
            abs_p = (Path.cwd() / p).resolve()
            count = _count_summary_files(abs_p)
            if count > 0:
                existing_hints.append(f"{abs_p} ({count} summaries)")
        hint_text = ""
        if existing_hints:
            hint_text = "\nDetected summary.csv files under: " + ", ".join(existing_hints)
        raise SystemExit(
            f"No valid summary.csv rows found under: {input_root}{hint_text}\n"
            "Try: --input <path containing <project>/<run>/summary.csv>"
        )

    metric_order = resolve_metric_order(args.metrics, discovered_metrics)
    grouped_rows = aggregate(rows=rows, metric_order=metric_order)

    write_markdown_table(
        output_path=output_md,
        grouped_rows=grouped_rows,
        metric_order=metric_order,
        group_by=args.group_by,
        decimals=args.decimals,
    )
    write_csv_table(
        output_path=output_csv,
        grouped_rows=grouped_rows,
        metric_order=metric_order,
        group_by=args.group_by,
    )

    plot_rows = collect_plot_rows_from_results(
        input_root=input_root,
        metric_order=metric_order,
    )
    if plot_rows:
        folder_rows = aggregate_folder_only(plot_rows, metric_order)
        write_folder_only_markdown(
            output_path=output_folder_md,
            folder_rows=folder_rows,
            metric_order=metric_order,
            decimals=args.decimals,
        )
        write_folder_only_csv(
            output_path=output_folder_csv,
            folder_rows=folder_rows,
            metric_order=metric_order,
        )
    else:
        print("[WARN] No valid rows in results.csv; skipped folder-only mean outputs.")

    if not args.no_plot:
        if not plot_rows:
            raise SystemExit(
                "No valid plot rows found from results.csv under input root. "
                "Expected rows with game/reward_enum and metric columns."
            )
        try:
            write_game_reward_subplots(
                output_path=output_plot,
                plot_rows=plot_rows,
                metric_order=metric_order,
            )
            simple_metric_order = DEFAULT_METRIC_ORDER.copy()
            write_overall_simple_plot(
                output_path=output_plot_simple,
                plot_rows=plot_rows,
                metric_order=simple_metric_order,
            )
        except RuntimeError as e:
            raise SystemExit(str(e)) from e

    print(f"[OK] Input root : {input_root}")
    print(f"[OK] Rows found  : {len(rows)}")
    print(f"[OK] Group count : {len(grouped_rows)}")
    print(f"[OK] Markdown    : {output_md}")
    print(f"[OK] CSV         : {output_csv}")
    if plot_rows:
        print(f"[OK] Folder MD  : {output_folder_md}")
        print(f"[OK] Folder CSV : {output_folder_csv}")
    if not args.no_plot:
        print(f"[OK] Plot       : {output_plot}")
        print(f"[OK] PlotSimple : {output_plot_simple}")


if __name__ == "__main__":
    main()
