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

from dataset.multigame.render import GameLevelRenderer  # noqa: E402
from table_export.models import RenderCell  # noqa: E402
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
    _reward_enum_for_feature_game,
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


DEFAULT_RENDER_CONFIG = SCRIPT_DIR / "render_config.json"


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
    runs = _download_runs(
        entity=args.entity,
        projects=projects,
        reward_enums=reward_enums,
        output_dir=output_dir,
        use_cache_only=args.cache_only,
    )
    candidates_by_method_re = {
        key: _build_candidates(run, games)
        for key, run in runs.items()
    }

    renderer = GameLevelRenderer()
    cells: dict[tuple[str, str, str], RenderCell] = {}
    raw_dir = output_dir / "renders"
    overlay_dir = output_dir / "overlays"

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
                low_img = _draw_level_image(game, low_state, raw_dir / f"{stem}_low.png", args.tile_size, renderer)
                mid_img = _draw_level_image(game, mid_state, raw_dir / f"{stem}_mid.png", args.tile_size, renderer)
                high_img = _draw_level_image(game, high_state, raw_dir / f"{stem}_high.png", args.tile_size, renderer)
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
        "cache_only": args.cache_only,
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
    parser.add_argument("--cache-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output-dir", default="")
    return parser.parse_args()


if __name__ == "__main__":
    render_table(parse_args())
