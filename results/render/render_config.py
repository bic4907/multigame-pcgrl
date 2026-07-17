"""Render Table 2 from a fixed render_config.json.

This script only loads the fixed row_i/seed panels from render_config.json,
reads the matching eval artifacts, and writes the render/LaTeX/preview bundle.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from table_export.models import RenderCell  # noqa: E402
from blender.manifest_renderer import (  # noqa: E402
    BlenderRenderRequest,
    render_levels,
)
from table_export.semantic.artifacts import (  # noqa: E402
    _build_candidates,
    _download_runs,
    _read_state,
)
from table_export.semantic.config import (  # noqa: E402
    _load_render_config,
    _render_config_candidate,
)
from table_export.semantic.constants import (  # noqa: E402
    DEFAULT_FEATURES,
    DEFAULT_GAMES,
    DEFAULT_LATEX_GAMES,
    DEFAULT_PROJECTS,
    DEFAULT_TILE_SIZE,
    ENTITY,
    METHOD_ORDER,
    _fmt_num,
    _reward_enum_for_feature_game,
    _safe_slug,
    _side_labels_for_feature,
)
from table_export.semantic.output import (  # noqa: E402
    _build_latex,
    _make_overleaf_bundle,
    _make_preview_png_pdf,
    _write_manifest,
)
from table_export.semantic.render import (  # noqa: E402
    _combine_triplet,
    _draw_level_image,
    _overlay_metric,
)
from table_export.semantic.metrics import _path_metric_and_coords  # noqa: E402
from table_export.semantic.tile_renderer import (  # noqa: E402
    DEFAULT_MAPPED_TILE_DIR,
    SemanticTileRenderer,
)


DEFAULT_RENDER_CONFIG = SCRIPT_DIR / "render_config.json"
def _missing_cached_projects(entity: str, projects: dict[str, str]) -> list[str]:
    cache_root = SCRIPT_DIR / ".wandb_download" / _safe_slug(entity)
    missing: list[str] = []
    for project in sorted(set(projects.values())):
        project_root = cache_root / _safe_slug(project)
        if not project_root.is_dir() or not any(project_root.glob("*/eval.h5")):
            missing.append(project)
    return missing


def _missing_run_keys(
    runs: dict[tuple[str, int], Any],
    projects: dict[str, str],
    reward_enums: list[int],
) -> list[tuple[str, int]]:
    missing: list[tuple[str, int]] = []
    for method in projects:
        for reward_enum in reward_enums:
            run = runs.get((method, reward_enum))
            if run is None or run.h5_path is None or run.csv_dir is None:
                missing.append((method, reward_enum))
    return missing


def _split_csv(value: str | None) -> list[str]:
    return [item.strip().lower() for item in str(value or "").split(",") if item.strip()]


def _scope_list(scope: dict[str, Any], key: str, default: list[str], override: str | None) -> list[str]:
    if override:
        return _split_csv(override)
    raw = scope.get(key)
    if isinstance(raw, list) and raw:
        return [str(item).strip().lower() for item in raw if str(item).strip()]
    if isinstance(raw, str) and raw.strip():
        return _split_csv(raw)
    return list(default)


def _copy_script_snapshot(output_dir: Path) -> Path:
    dst = output_dir / "scripts" / Path(__file__).name
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(__file__), dst)
    return dst


def _resolve_blender_resolution(args: argparse.Namespace) -> tuple[int, int]:
    square_side = int(args.tile_size) * 16
    width = int(args.blender_resolution_x or 0)
    height = int(args.blender_resolution_y or 0)
    if width <= 0 and height <= 0:
        return square_side, square_side
    if width <= 0:
        return height, height
    if height <= 0:
        return width, width
    return width, height


def _overlay_metric_label(
    image_path: Path,
    metric_value: float,
    target_value: float | None,
    out_path: Path,
    tile_size: int,
) -> Path:
    from PIL import Image, ImageDraw, ImageFont

    def load_font(size: int) -> ImageFont.ImageFont:
        candidates = [
            str(PROJECT_ROOT / "debug" / "Pretendard-Regular.ttf"),
            "/System/Library/Fonts/Supplemental/Pretendard-Regular.otf",
            "/System/Library/Fonts/Helvetica.ttc",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        ]
        for path in candidates:
            if path and Path(path).exists():
                try:
                    return ImageFont.truetype(path, size)
                except Exception:
                    pass
        return ImageFont.load_default()

    img = Image.open(image_path).convert("RGBA")
    draw = ImageDraw.Draw(img, "RGBA")
    label_lines = [f"Target={_fmt_num(target_value)}", f"Result={_fmt_num(metric_value)}"]
    pad_x = max(16, int(tile_size * 0.50))
    pad_y = max(12, int(tile_size * 0.38))
    inset_x = max(10, int(tile_size * 0.36))
    inset_y = max(10, int(tile_size * 0.36))
    radius = max(8, int(tile_size * 0.35))
    line_gap = max(4, int(tile_size * 0.16))
    font_size = max(54, int(tile_size * 1.78))
    font = load_font(font_size)

    def line_boxes() -> list[tuple[int, int, int, int]]:
        return [draw.textbbox((0, 0), line, font=font) for line in label_lines]

    bboxes = line_boxes()
    text_w = max(bbox[2] - bbox[0] for bbox in bboxes)
    text_h = sum(bbox[3] - bbox[1] for bbox in bboxes) + line_gap * (len(label_lines) - 1)
    while text_w + 2 * pad_x + inset_x > img.width and font_size > 36:
        font_size -= 2
        font = load_font(font_size)
        bboxes = line_boxes()
        text_w = max(bbox[2] - bbox[0] for bbox in bboxes)
        text_h = sum(bbox[3] - bbox[1] for bbox in bboxes) + line_gap * (len(label_lines) - 1)

    box = (
        inset_x,
        inset_y,
        inset_x + text_w + 2 * pad_x,
        inset_y + text_h + 2 * pad_y,
    )
    draw.rounded_rectangle(box, radius=radius, fill=(255, 255, 255, 230), outline=(20, 24, 30, 210), width=2)
    y = inset_y + pad_y
    for line, bbox in zip(label_lines, bboxes):
        text_xy = (inset_x + pad_x, y - bbox[1])
        draw.text(text_xy, line, fill=(20, 24, 30, 255), font=font)
        draw.text((text_xy[0] + 1, text_xy[1]), line, fill=(20, 24, 30, 255), font=font)
        y += bbox[3] - bbox[1] + line_gap

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.convert("RGB").save(str(out_path))
    return out_path


def _render_cell_images(
    *,
    render_mode: str,
    game: str,
    stem: str,
    low_state: Any,
    mid_state: Any,
    high_state: Any,
    raw_dir: Path,
    tile_size: int,
    renderer: Any,
    blender: str | None,
    blender_resolution: tuple[int, int],
    reward_enum: int,
) -> tuple[Path, Path, Path]:
    low_img = raw_dir / f"{stem}_low.png"
    mid_img = raw_dir / f"{stem}_mid.png"
    high_img = raw_dir / f"{stem}_high.png"
    if render_mode == "blender":
        low_path = _path_coords_for_blender(low_state, reward_enum)
        mid_path = _path_coords_for_blender(mid_state, reward_enum)
        high_path = _path_coords_for_blender(high_state, reward_enum)
        render_levels(
            [
                BlenderRenderRequest(game, low_state, low_img, "low", path_coords=low_path),
                BlenderRenderRequest(game, mid_state, mid_img, "mid", path_coords=mid_path),
                BlenderRenderRequest(game, high_state, high_img, "high", path_coords=high_path),
            ],
            raw_dir / "_blender_manifests" / f"{stem}.json",
            blender=blender,
            resolution=blender_resolution,
        )
        return low_img, mid_img, high_img

    low_img = _draw_level_image(game, low_state, low_img, tile_size, renderer)
    mid_img = _draw_level_image(game, mid_state, mid_img, tile_size, renderer)
    high_img = _draw_level_image(game, high_state, high_img, tile_size, renderer)
    return low_img, mid_img, high_img


def _path_coords_for_blender(state: Any, reward_enum: int) -> list[list[int]] | None:
    if reward_enum != 1:
        return None
    _, coords = _path_metric_and_coords(state)
    if len(coords) < 2:
        return None
    return [[int(row), int(col)] for row, col in coords]


def render_table(args: argparse.Namespace) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) if args.output_dir else SCRIPT_DIR / "outputs" / f"table2_semantic_render_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    render_config_path = Path(args.render_config)
    panels, meta = _load_render_config(render_config_path)
    if not panels:
        raise ValueError(f"No row_i/seed panels found in {render_config_path}")

    scope = meta.get("scope", {})
    games = _scope_list(scope, "games", DEFAULT_GAMES, args.games)
    latex_games = _scope_list(scope, "latex_games", DEFAULT_LATEX_GAMES, args.latex_games)
    features = _scope_list(scope, "features", DEFAULT_FEATURES, args.features)
    projects = dict(scope.get("projects") or DEFAULT_PROJECTS)

    missing_latex_games = [game for game in latex_games if game not in games]
    if missing_latex_games:
        raise ValueError("latex_games must be a subset of games: " + ",".join(missing_latex_games))

    reward_enums = sorted({
        _reward_enum_for_feature_game(feature, game)
        for feature in features
        for game in games
    })
    use_cache_only = args.cache_only
    runs = _download_runs(
        entity=args.entity,
        projects=projects,
        reward_enums=reward_enums,
        output_dir=output_dir,
        use_cache_only=use_cache_only,
    )
    missing_projects = _missing_cached_projects(args.entity, projects) if args.cache_only else []
    missing_run_keys = _missing_run_keys(runs, projects, reward_enums) if args.cache_only else []
    if missing_run_keys and args.download_missing_cache:
        missing_text = ", ".join(f"{method}/re{reward_enum}" for method, reward_enum in missing_run_keys)
        message = f"Missing local W&B artifacts for {missing_text}."
        if missing_projects:
            message += " Missing project cache(s): " + ", ".join(missing_projects) + "."
        print(message + " Downloading required artifacts...")
        use_cache_only = False
        runs = _download_runs(
            entity=args.entity,
            projects=projects,
            reward_enums=reward_enums,
            output_dir=output_dir,
            use_cache_only=use_cache_only,
        )
    candidates_by_method_re = {
        key: _build_candidates(run, games)
        for key, run in runs.items()
    }

    renderer = SemanticTileRenderer(DEFAULT_MAPPED_TILE_DIR)
    cells: dict[tuple[str, str, str], RenderCell] = {}
    raw_dir = output_dir / "renders"
    overlay_dir = output_dir / "overlays"
    blender_resolution = _resolve_blender_resolution(args)

    for feature in features:
        side_labels = _side_labels_for_feature(feature)
        for game in games:
            reward_enum = _reward_enum_for_feature_game(feature, game)
            for method in METHOD_ORDER:
                run = runs.get((method, reward_enum))
                method_candidates = candidates_by_method_re.get((method, reward_enum), {})
                if run is None or run.h5_path is None:
                    raise RuntimeError(f"Missing eval artifacts for {method}/{game}/{feature} reward_enum={reward_enum}")

                low, low_seed = _render_config_candidate(panels, method_candidates, method, game, feature, "low") or (None, None)
                mid, mid_seed = _render_config_candidate(panels, method_candidates, method, game, feature, "mid") or (None, None)
                high, high_seed = _render_config_candidate(panels, method_candidates, method, game, feature, "high") or (None, None)
                if low is None or mid is None or high is None:
                    raise RuntimeError(f"render_config is missing {method}/{game}/{feature} low/mid/high entries")

                low_state = _read_state(run.h5_path, low.h5_group, low_seed)
                mid_state = _read_state(run.h5_path, mid.h5_group, mid_seed)
                high_state = _read_state(run.h5_path, high.h5_group, high_seed)
                if low_state is None or mid_state is None or high_state is None:
                    raise RuntimeError(f"Missing H5 state for {method}/{game}/{feature}")

                stem = f"{method.lower()}_{game}_{feature}"
                low_img, mid_img, high_img = _render_cell_images(
                    render_mode=args.render_mode,
                    game=game,
                    stem=stem,
                    low_state=low_state,
                    mid_state=mid_state,
                    high_state=high_state,
                    raw_dir=raw_dir,
                    tile_size=args.tile_size,
                    renderer=renderer,
                    blender=args.blender,
                    blender_resolution=blender_resolution,
                    reward_enum=reward_enum,
                )
                if args.render_mode == "blender":
                    low_overlay = _overlay_metric_label(
                        low_img,
                        low.seed_metrics[low_seed],
                        low.target,
                        overlay_dir / f"{stem}_low_overlay.png",
                        args.tile_size,
                    )
                    mid_overlay = _overlay_metric_label(
                        mid_img,
                        mid.seed_metrics[mid_seed],
                        mid.target,
                        overlay_dir / f"{stem}_mid_overlay.png",
                        args.tile_size,
                    )
                    high_overlay = _overlay_metric_label(
                        high_img,
                        high.seed_metrics[high_seed],
                        high.target,
                        overlay_dir / f"{stem}_high_overlay.png",
                        args.tile_size,
                    )
                else:
                    low_overlay = _overlay_metric(
                        low_img,
                        low_state,
                        reward_enum,
                        low.seed_metrics[low_seed],
                        low.target,
                        overlay_dir / f"{stem}_low_overlay.png",
                        args.tile_size,
                    )
                    mid_overlay = _overlay_metric(
                        mid_img,
                        mid_state,
                        reward_enum,
                        mid.seed_metrics[mid_seed],
                        mid.target,
                        overlay_dir / f"{stem}_mid_overlay.png",
                        args.tile_size,
                    )
                    high_overlay = _overlay_metric(
                        high_img,
                        high_state,
                        reward_enum,
                        high.seed_metrics[high_seed],
                        high.target,
                        overlay_dir / f"{stem}_high_overlay.png",
                        args.tile_size,
                    )
                triplet_overlay = _combine_triplet(
                    low_overlay,
                    mid_overlay,
                    high_overlay,
                    overlay_dir / f"{stem}_triplet_overlay.png",
                    side_labels[0],
                    side_labels[1],
                    side_labels[2],
                )
                cells[(method, game, feature)] = RenderCell(
                    method=method,
                    game=game,
                    feature=feature,
                    low=low,
                    mid=mid,
                    high=high,
                    low_seed=low_seed,
                    mid_seed=mid_seed,
                    high_seed=high_seed,
                    low_image=low_img,
                    mid_image=mid_img,
                    high_image=high_img,
                    low_overlay=low_overlay,
                    mid_overlay=mid_overlay,
                    high_overlay=high_overlay,
                    triplet_overlay=triplet_overlay,
                )

    latex = _build_latex(cells, features, latex_games, output_dir)
    (output_dir / "tbl_qualitative.tex").write_text(latex, encoding="utf-8")
    _make_preview_png_pdf(cells, features, games, output_dir)
    _write_manifest(cells, output_dir)
    overleaf_dir = _make_overleaf_bundle(cells, output_dir)
    _copy_script_snapshot(output_dir)
    shutil.copy2(render_config_path, output_dir / "render_config.json")

    used_config = {
        "source_render_config": str(render_config_path),
        "entity": args.entity,
        "projects": projects,
        "games": games,
        "latex_games": latex_games,
        "features": features,
        "tile_size": args.tile_size,
        "render_mode": args.render_mode,
        "mapped_tile_dir": str(DEFAULT_MAPPED_TILE_DIR),
        "blender": args.blender,
        "blender_resolution": [blender_resolution[0], blender_resolution[1]],
        "cache_only": args.cache_only,
        "download_missing_cache": args.download_missing_cache,
        "resolved_cache_only": use_cache_only,
        "overleaf_dir": overleaf_dir.relative_to(output_dir).as_posix(),
    }
    (output_dir / "used_render_config.json").write_text(
        json.dumps(used_config, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(output_dir)
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render-config", default=str(DEFAULT_RENDER_CONFIG))
    parser.add_argument("--entity", default=ENTITY)
    parser.add_argument("--games", default="", help="Optional comma-separated override. Defaults to render_config scope.")
    parser.add_argument("--latex-games", default="", help="Optional comma-separated override. Defaults to render_config scope.")
    parser.add_argument("--features", default="", help="Optional comma-separated override. Defaults to render_config scope.")
    parser.add_argument("--tile-size", type=int, default=DEFAULT_TILE_SIZE)
    parser.add_argument("--render-mode", choices=("2d", "blender"), default="2d")
    parser.add_argument("--blender", default="", help="Optional path to Blender executable.")
    parser.add_argument("--blender-resolution-x", type=int, default=0, help="0 uses tile_size * 16.")
    parser.add_argument("--blender-resolution-y", type=int, default=0, help="0 uses the resolved width for square output.")
    parser.add_argument("--cache-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--download-missing-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When --cache-only is set, download from W&B if a required local project cache is missing.",
    )
    parser.add_argument("--output-dir", default="")
    return parser.parse_args()


if __name__ == "__main__":
    render_table(parse_args())
