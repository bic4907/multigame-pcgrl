"""
Export per-game rendered MultiGame samples with reward metadata.

Example:
    python dataset/scripts/export_game_render_samples.py \
        --out-dir results/game_render_samples_export_table_tiles \
        --count 100 \
        --tile-size 16 \
        --tile-mode export-table
"""
from __future__ import annotations

import argparse
import csv
import json
import shlex
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset.multigame import MultiGameDataset
from dataset.multigame.base import GameTag, GameSample
from dataset.multigame.render import render_game_level
from dataset.multigame.tile_utils import (
    CATEGORY_COLORS,
    UNIFIED_CATEGORIES,
    game_mapping_rows,
    render_unified_rgb,
    to_unified,
)
from dataset.reward_annotations.instruction_config import FEATURE_DESCRIPTIONS


DEFAULT_GAMES = [
    GameTag.DUNGEON,
    GameTag.SOKOBAN,
    GameTag.ZELDA,
    GameTag.POKEMON,
    GameTag.DOOM,
]


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _format_value(value: Any) -> str:
    value = _json_safe(value)
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _write_kv_txt(path: Path, data: dict[str, Any]) -> None:
    lines = []
    for key, value in data.items():
        if isinstance(value, dict):
            lines.append(f"[{key}]")
            for sub_key, sub_value in value.items():
                lines.append(f"{sub_key}: {_format_value(sub_value)}")
            lines.append("")
        elif isinstance(value, list):
            lines.append(f"{key}: {_format_value(value)}")
        else:
            lines.append(f"{key}: {_format_value(value)}")
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _write_rows_txt(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("\t".join(fieldnames) + "\n")
        for row in rows:
            f.write("\t".join(_format_value(row.get(field, "")) for field in fieldnames) + "\n")


def _write_metadata_txt(path: Path, metadata: dict[str, Any]) -> None:
    lines = [
        f"game: {metadata['game']}",
        f"sample_index: {metadata['sample_index']}",
        f"source_id: {metadata['source_id']}",
        f"order: {metadata['order']}",
        f"shape: {_format_value(metadata['shape'])}",
        f"tile_mode: {metadata['tile_mode']}",
        f"tile_size: {metadata['tile_size']}",
        "",
        "[files]",
    ]
    for key, value in metadata["files"].items():
        lines.append(f"{key}: {value}")

    lines.extend(["", "[annotations]"])
    ann_fields = [
        "reward_enum",
        "feature_name",
        "sub_condition",
        "condition",
        "key",
        "instruction",
        "instruction_raw",
        "instruction_uni",
    ]
    lines.append("\t".join(ann_fields))
    for ann in metadata["annotations"]:
        lines.append("\t".join(_format_value(ann.get(field, "")) for field in ann_fields))

    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _sample_sort_key(sample: GameSample) -> tuple[int, str]:
    order = sample.order if sample.order is not None else 10**12
    return int(order), str(sample.source_id)


def _group_unique_sources(samples: list[GameSample]) -> list[list[GameSample]]:
    grouped: dict[str, list[GameSample]] = defaultdict(list)
    for sample in samples:
        grouped[str(sample.source_id)].append(sample)
    groups = list(grouped.values())
    groups.sort(key=lambda group: _sample_sort_key(group[0]))
    for group in groups:
        group.sort(key=lambda sample: int(sample.meta.get("reward_enum", 10**9)))
    return groups


def _samples_by_game_for_mode(tile_mode: str) -> dict[str, list[GameSample]]:
    if tile_mode == "export-table":
        from instruct_rl.utils.dataset_loader_helpers.preprocessing import (
            apply_tile_offset,
            preprocess_samples,
        )

        samples = list(MultiGameDataset(use_cache=True, use_tile_mapping=True))
        samples = preprocess_samples(samples, longtail_cut=True)
        samples = apply_tile_offset(samples, 1)
    else:
        samples = list(MultiGameDataset(use_cache=True, use_tile_mapping=False))

    by_game: dict[str, list[GameSample]] = defaultdict(list)
    for sample in samples:
        by_game[sample.game].append(sample)
    return by_game


def _annotation_groups_by_game_source() -> dict[tuple[str, str], list[GameSample]]:
    samples = list(MultiGameDataset(use_cache=True, use_tile_mapping=False))
    groups: dict[tuple[str, str], list[GameSample]] = defaultdict(list)
    for sample in samples:
        groups[(sample.game, str(sample.source_id))].append(sample)
    for group in groups.values():
        group.sort(key=lambda sample: int(sample.meta.get("reward_enum", 10**9)))
    return groups


def _annotation_record(sample: GameSample) -> dict[str, Any]:
    reward_enum = int(sample.meta.get("reward_enum", -1))
    conditions = sample.meta.get("conditions", {})
    condition_value = conditions.get(reward_enum)
    return {
        "key": sample.meta.get("key"),
        "reward_enum": reward_enum,
        "feature_name": sample.meta.get("feature_name"),
        "sub_condition": sample.meta.get("sub_condition", ""),
        "condition": condition_value,
        "conditions": conditions,
        "instruction_raw": sample.meta.get("instruction_raw"),
        "instruction_uni": sample.meta.get("instruction_uni"),
        "instruction": sample.instruction,
    }


def _render_png(
    sample: GameSample,
    unified: np.ndarray,
    out_path: Path,
    *,
    tile_mode: str,
    tile_size: int,
) -> None:
    if tile_mode == "export-table":
        render_game_level(
            sample.game,
            sample.array,
            tile_size=tile_size,
            save_path=out_path,
        )
        return

    if tile_mode == "raw-image":
        render_game_level(
            sample.game,
            sample.array,
            tile_size=tile_size,
            save_path=out_path,
            tile_ims_dir=ROOT / "dataset" / "multigame" / "tile_ims",
        )
        return

    if tile_mode == "unified":
        rgb = render_unified_rgb(unified, tile_size=tile_size)
        Image.fromarray(rgb, mode="RGB").save(out_path)
        return

    from dataset.multigame.render import render_sample_pil

    render_sample_pil(sample, tile_size=tile_size).save(out_path)


def export(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    samples_by_game = _samples_by_game_for_mode(args.tile_mode)
    annotation_groups = _annotation_groups_by_game_source()
    command = " ".join(shlex.quote(part) for part in [sys.executable, *sys.argv])
    recommended_command = (
        "python dataset/scripts/export_game_render_samples.py "
        f"--out-dir {shlex.quote(str(out_dir))} "
        f"--count {args.count} --tile-size {args.tile_size} "
        f"--tile-mode {args.tile_mode}"
    )

    run_metadata = {
        "executed_command": command,
        "recommended_command": recommended_command,
        "count_per_game": args.count,
        "tile_mode": args.tile_mode,
        "tile_size": args.tile_size,
        "games": args.games,
        "rendering_note": (
            "export-table mode matches results/render/table_export/render_assets.py: "
            "MultiGameDataset(use_tile_mapping=True), preprocess_samples(longtail_cut=True), "
            "apply_tile_offset(samples, 1), then GameLevelRenderer.render()."
            if args.tile_mode == "export-table"
            else ""
        ),
        "unified_categories": UNIFIED_CATEGORIES,
        "category_colors_rgb": CATEGORY_COLORS,
        "reward_enum_legend": {
            str(i): {
                "feature_name": name,
                "description": FEATURE_DESCRIPTIONS.get(name, ""),
            }
            for i, name in enumerate(
                [
                    "region",
                    "path_length",
                    "interactable_count",
                    "hazard_count",
                    "collectable_count",
                ]
            )
        },
    }
    (out_dir / "command.txt").write_text(recommended_command + "\n", encoding="utf-8")
    _write_kv_txt(out_dir / "run_metadata.txt", run_metadata)

    summary_rows: list[dict[str, Any]] = []
    for game in args.games:
        game_dir = out_dir / game
        game_dir.mkdir(parents=True, exist_ok=True)
        (game_dir / "command.txt").write_text(recommended_command + "\n", encoding="utf-8")
        tile_mapping_rows = game_mapping_rows(game)
        _write_rows_txt(
            game_dir / "tile_mapping.txt",
            tile_mapping_rows,
            ["raw_id", "raw_name", "unified_id", "unified_name"],
        )

        groups = _group_unique_sources(samples_by_game.get(game, []))
        if len(groups) < args.count:
            raise RuntimeError(f"{game}: requested {args.count}, only {len(groups)} unique source_id samples found")

        manifest_rows: list[dict[str, Any]] = []
        selected = groups[: args.count]
        for sample_idx, group in enumerate(selected):
            base_sample = group[0]
            render_level = np.asarray(base_sample.array, dtype=np.int32)
            if args.tile_mode == "export-table":
                raw = render_level - 1
                unified = raw
            else:
                raw = render_level
                unified = to_unified(raw, base_sample.game, warn_unmapped=False)
            stem = f"{sample_idx:03d}_{game}"

            png_path = game_dir / f"{stem}.png"
            raw_path = game_dir / f"{stem}.raw.npy"
            unified_path = game_dir / f"{stem}.unified.npy"
            render_level_path = game_dir / f"{stem}.render_level.npy"
            metadata_path = game_dir / f"{stem}.metadata.txt"

            _render_png(base_sample, unified, png_path, tile_mode=args.tile_mode, tile_size=args.tile_size)
            np.save(raw_path, raw)
            np.save(unified_path, unified.astype(np.int32))
            np.save(render_level_path, render_level)

            annotation_group = annotation_groups.get((game, str(base_sample.source_id)), group)
            annotations = [_annotation_record(sample) for sample in annotation_group]
            metadata = {
                "game": game,
                "sample_index": sample_idx,
                "source_id": base_sample.source_id,
                "order": base_sample.order,
                "shape": list(raw.shape),
                "tile_mode": args.tile_mode,
                "tile_size": args.tile_size,
                "files": {
                    "png": png_path.name,
                    "raw_npy": raw_path.name,
                    "unified_npy": unified_path.name,
                    "render_level_npy": render_level_path.name,
                },
                "annotations": annotations,
            }
            _write_metadata_txt(metadata_path, metadata)

            for ann in annotations:
                row = {
                    "game": game,
                    "sample_index": sample_idx,
                    "source_id": base_sample.source_id,
                    "png": png_path.name,
                    "raw_npy": raw_path.name,
                    "unified_npy": unified_path.name,
                    "render_level_npy": render_level_path.name,
                    "metadata_txt": metadata_path.name,
                    "reward_enum": ann["reward_enum"],
                    "feature_name": ann["feature_name"],
                    "sub_condition": ann["sub_condition"],
                    "condition": ann["condition"],
                    "instruction": ann["instruction"],
                }
                manifest_rows.append(row)
                summary_rows.append(row)

        fieldnames = [
            "game",
            "sample_index",
            "source_id",
            "png",
            "raw_npy",
            "unified_npy",
            "render_level_npy",
            "metadata_txt",
            "reward_enum",
            "feature_name",
            "sub_condition",
            "condition",
            "instruction",
        ]
        with (game_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(manifest_rows)
        _write_rows_txt(game_dir / "manifest.txt", manifest_rows, fieldnames)

    with (out_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    _write_rows_txt(out_dir / "manifest.txt", summary_rows, list(summary_rows[0].keys()))
    return {
        "out_dir": str(out_dir),
        "games": args.games,
        "samples_per_game": args.count,
        "rendered_pngs": len(args.games) * args.count,
        "manifest_rows": len(summary_rows),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="results/game_render_samples_export_table_tiles")
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--tile-size", type=int, default=16)
    parser.add_argument(
        "--tile-mode",
        choices=["unified", "raw", "raw-image", "export-table"],
        default="export-table",
    )
    parser.add_argument("--games", nargs="+", default=DEFAULT_GAMES)
    return parser.parse_args()


def main() -> None:
    result = export(parse_args())
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
