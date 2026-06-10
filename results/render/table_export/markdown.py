from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from .csv_rows import CONDITION_BUCKETS, condition_bucket_key, condition_bucket_label, row_key, row_label
from .models import RunResult
from .render_assets import render_image_table_assets
from .utils import markdown_escape, reward_enum_section_title, reward_enum_value, unique_methods


def _markdown_image(path: Path | None, width: int = 160) -> str:
    if path is None:
        return "-"
    return f'<img src="{path.as_posix()}" width="{width}">'


def export_markdown_table(config, run_results: list[RunResult], output_path: Path | str = "table.md"):
    output_path = Path(output_path)
    table_cfg = config if isinstance(config, dict) else {}

    max_rows_per_condition = int(table_cfg.get("table_max_rows_per_condition", 4))
    seed_i = int(table_cfg.get("table_seed", 0))
    tile_size = int(table_cfg.get("table_tile_size", 12))
    rows, method_images, dataset_images = render_image_table_assets(
        run_results,
        output_path.parent,
        max_rows_per_condition=max_rows_per_condition,
        seed_i=seed_i,
        tile_size=tile_size,
    )

    lines = ["# W&B Eval Render Table", ""]

    methods = unique_methods(run_results)
    header = ["Game / Task / Condition / Instruction", *methods, "Dataset"]

    if not rows:
        lines.append("## Rendered Samples")
        lines.append("")
        lines.append("| " + " | ".join(header) + " |")
        lines.append("|" + "|".join(["---"] * len(header)) + "|")
        lines.append("| No rows available. eval_csv/results.csv is required. | " + " | ".join(["-"] * len(methods)) + " | - |")
        lines.append("")
    else:
        rows_by_reward_enum_condition: dict[int, dict[str, list[dict[str, str]]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for row in rows:
            rows_by_reward_enum_condition[reward_enum_value(row)][condition_bucket_key(row)].append(row)

        for reward_enum in sorted(rows_by_reward_enum_condition):
            lines.append(f"## {reward_enum_section_title(reward_enum)}")
            lines.append("")
            bucket_rows = rows_by_reward_enum_condition[reward_enum]
            for bucket, bucket_label in CONDITION_BUCKETS:
                if not bucket_rows.get(bucket):
                    continue
                lines.append(f"### {bucket_label}")
                lines.append("")
                lines.append("| " + " | ".join(header) + " |")
                lines.append("|" + "|".join(["---"] * len(header)) + "|")
                for row in bucket_rows[bucket]:
                    key = row_key(row)
                    cells = [row_label(row)]
                    for method in methods:
                        cells.append(_markdown_image(method_images.get((key, method))))
                    cells.append(_markdown_image(dataset_images.get(key)))
                    lines.append("| " + " | ".join(cells) + " |")
                lines.append("")

            if bucket_rows.get("unknown"):
                lines.append(f"### {condition_bucket_label(bucket_rows['unknown'][0])}")
                lines.append("")
                lines.append("| " + " | ".join(header) + " |")
                lines.append("|" + "|".join(["---"] * len(header)) + "|")
                for row in bucket_rows["unknown"]:
                    key = row_key(row)
                    cells = [row_label(row)]
                    for method in methods:
                        cells.append(_markdown_image(method_images.get((key, method))))
                    cells.append(_markdown_image(dataset_images.get(key)))
                    lines.append("| " + " | ".join(cells) + " |")
                lines.append("")

    lines.append("## Artifact Status")
    lines.append("")
    lines.append("| Method | Task | W&B Run | H5 Artifact | CSV Artifact | Status |")
    lines.append("|---|---|---|---|---|---|")
    for item in run_results:
        status = "OK" if item.error is None else f"ERROR: {item.error}"
        run_link = f"[Link]({item.run_url})" if item.run_url else "N/A"
        task = reward_enum_section_title(item.reward_enum) if item.reward_enum is not None else "-"
        lines.append(
            f"| {item.method} | {task} | {run_link} | {item.artifact_name or '-'} | "
            f"{item.csv_artifact_name or '-'} | {markdown_escape(status)} |"
        )
    lines.append("")

    content = "\n".join(lines)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)
    return content
