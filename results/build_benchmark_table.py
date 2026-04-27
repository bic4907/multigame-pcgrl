"""
Build benchmark tables from downloaded summary.csv files.

Expected folder layout:
    <input_root>/<project>/<run_name>/summary.csv

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create benchmark table from wandb_download summary.csv files."
    )
    parser.add_argument(
        "--input",
        default="results/wandb_download",
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
        count = sum(1 for _ in c.glob("*/*/summary.csv"))
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


def safe_std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    return math.sqrt(variance)


def discover_rows(input_root: Path, group_by: str) -> tuple[list[dict], set[str]]:
    rows: list[dict] = []
    metric_names: set[str] = set()

    for summary_path in sorted(input_root.glob("*/*/summary.csv")):
        run_dir = summary_path.parent
        project_dir = run_dir.parent
        project = project_dir.name
        run_name = run_dir.name

        run_tokens = parse_run_tokens(run_name)
        game = run_tokens.get("game", "unknown")
        reward_enum = run_tokens.get("re", run_tokens.get("reward_enum", "unknown"))
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


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    input_root = resolve_input_root(args.input, script_dir)
    output_md = Path(args.output_md).resolve() if args.output_md else input_root / "benchmark_table.md"
    output_csv = Path(args.output_csv).resolve() if args.output_csv else input_root / "benchmark_table.csv"

    rows, discovered_metrics = discover_rows(input_root=input_root, group_by=args.group_by)
    if not rows:
        raise SystemExit(f"No valid summary.csv rows found under: {input_root}")

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

    print(f"[OK] Input root : {input_root}")
    print(f"[OK] Rows found  : {len(rows)}")
    print(f"[OK] Group count : {len(grouped_rows)}")
    print(f"[OK] Markdown    : {output_md}")
    print(f"[OK] CSV         : {output_csv}")


if __name__ == "__main__":
    main()
