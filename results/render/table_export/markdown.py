from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import shutil

from .csv_rows import (
    CONDITION_BUCKETS,
    annotate_condition_percentiles,
    condition_bucket_key,
    condition_bucket_label,
    row_key,
    row_label,
)
from .models import RunResult
from .render_assets import render_image_table_assets
from .utils import reward_enum_section_title, reward_enum_value, safe_slug, unique_methods


def _markdown_image(path: Path | None, width: int = 160) -> str:
    if path is None:
        return "-"
    return f'<img src="{path.as_posix()}" width="{width}">'


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


def _latex_member_label(row: dict) -> str:
    labels = []
    for member in row.get("_pair_members", []):
        instruction = member.get("instruction") or "-"
        reward_enum = reward_enum_value(member)
        value = member.get(f"condition_{reward_enum}", "")
        try:
            numeric = float(value)
            c_value = str(int(numeric)) if numeric.is_integer() else f"{numeric:g}"
        except (TypeError, ValueError):
            c_value = str(value)
        labels.append(f"{instruction} (c={c_value})")
    return " / ".join(labels)


def _export_latex_rows(
    rows: list[dict],
    methods: list[str],
    output_dir: Path,
) -> None:
    pair_rows = [row for row in rows if row.get("_pair_members")]
    if not pair_rows:
        return

    latex_dir = output_dir / "latex"
    qualitative_dir = latex_dir / "experiment" / "qualitative"
    latex_dir.mkdir(parents=True, exist_ok=True)
    qualitative_dir.mkdir(parents=True, exist_ok=True)
    lines = []

    for row in pair_rows:
        key = row_key(row)
        src_dir = output_dir / "images" / key
        episode_dir = qualitative_dir / key
        dst_image_dir = episode_dir / key
        dst_image_dir.mkdir(parents=True, exist_ok=True)

        cells = [rf"\vspace{{-0.6cm}}\texttt{{\scriptsize{{{_latex_escape(_latex_member_label(row))}}}}}"]
        episode_cells = [cells[0]]
        for method in methods:
            method_slug = safe_slug(method)
            paths = []
            episode_paths = []
            for side_i in [0, 1]:
                src = src_dir / f"{method_slug}_{side_i}.png"
                dst = dst_image_dir / src.name
                if src.exists():
                    shutil.copy2(src, dst)
                tex_path = (Path("experiment") / "qualitative" / key / key / src.name).as_posix()
                paths.append(tex_path)
                episode_paths.append(tex_path)
            cells.append(rf"\twinimage{{{paths[0]}}}{{{paths[1]}}}")
            episode_cells.append(rf"\twinimage{{{episode_paths[0]}}}{{{episode_paths[1]}}}")

        dataset_paths = []
        episode_dataset_paths = []
        for side_i in [0, 1]:
            src = src_dir / f"dataset_{side_i}.png"
            dst = dst_image_dir / src.name
            if src.exists():
                shutil.copy2(src, dst)
            tex_path = (Path("experiment") / "qualitative" / key / key / src.name).as_posix()
            dataset_paths.append(tex_path)
            episode_dataset_paths.append(tex_path)
        cells.append(rf"\twinimage{{{dataset_paths[0]}}}{{{dataset_paths[1]}}}")
        episode_cells.append(rf"\twinimage{{{episode_dataset_paths[0]}}}{{{episode_dataset_paths[1]}}}")
        row_tex = " & ".join(cells) + r" \\"
        lines.append(row_tex)

        episode_row_tex = " & ".join(episode_cells) + r" \\"
        (episode_dir / "row.tex").write_text(episode_row_tex + "\n", encoding="utf-8")

    (qualitative_dir / "table_rows.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def export_markdown_table(config, run_results: list[RunResult], output_path: Path | str = "table.md"):
    output_path = Path(output_path)
    table_cfg = config if isinstance(config, dict) else {}

    max_rows_per_condition = int(table_cfg.get("table_max_rows_per_condition", 4))
    seed_i = int(table_cfg.get("table_seed", 0))
    tile_size = int(table_cfg.get("table_tile_size", 12))
    condition_targets = table_cfg.get("condition_contrast_targets")
    num_episodes = int(table_cfg.get("table_num_episodes", 10))
    image_width = int(table_cfg.get("table_image_width", 320 if condition_targets else 160))
    rows, method_images, dataset_images = render_image_table_assets(
        run_results,
        output_path.parent,
        max_rows_per_condition=max_rows_per_condition,
        seed_i=seed_i,
        tile_size=tile_size,
        condition_targets=condition_targets,
        num_episodes=num_episodes,
    )
    annotate_condition_percentiles(rows)

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
                        cells.append(_markdown_image(method_images.get((key, method)), width=image_width))
                    cells.append(_markdown_image(dataset_images.get(key), width=image_width))
                    lines.append("| " + " | ".join(cells) + " |")
                lines.append("")

            for unknown_bucket in sorted(k for k in bucket_rows if k not in dict(CONDITION_BUCKETS)):
                lines.append(f"### {condition_bucket_label(bucket_rows[unknown_bucket][0])}")
                lines.append("")
                lines.append("| " + " | ".join(header) + " |")
                lines.append("|" + "|".join(["---"] * len(header)) + "|")
                for row in bucket_rows[unknown_bucket]:
                    key = row_key(row)
                    cells = [row_label(row)]
                    for method in methods:
                        cells.append(_markdown_image(method_images.get((key, method)), width=image_width))
                    cells.append(_markdown_image(dataset_images.get(key), width=image_width))
                    lines.append("| " + " | ".join(cells) + " |")
                lines.append("")

    content = "\n".join(lines)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)
    if condition_targets:
        _export_latex_rows(rows, methods, output_path.parent)
    return content
