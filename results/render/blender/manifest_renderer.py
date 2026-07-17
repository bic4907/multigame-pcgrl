from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from dataset.multigame.tile_utils import to_unified


BLENDER_DIR = Path(__file__).resolve().parent
RENDER_ROOT = BLENDER_DIR.parent
DEFAULT_BLENDER_SCRIPT = BLENDER_DIR / "blender_render_multigame_manifest.py"
DEFAULT_ASSET_DIR = RENDER_ROOT / "assets"
DEFAULT_BLENDER_MACOS = "/Applications/Blender.app/Contents/MacOS/Blender"

@dataclass(frozen=True)
class BlenderRenderRequest:
    game: str
    level: np.ndarray
    output: Path
    label: str = ""
    changed_cells: list[list[int]] | None = None
    path_coords: list[list[int]] | None = None


def find_blender(explicit: str | None = None) -> str:
    candidates = [explicit] if explicit else []
    found = shutil.which("blender")
    if found:
        candidates.append(found)
    candidates.append(DEFAULT_BLENDER_MACOS)
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return candidate
    raise FileNotFoundError("Blender executable not found. Pass --blender or add blender to PATH.")


def to_blender_unified(level: np.ndarray, game: str) -> np.ndarray:
    arr = np.asarray(level)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D level array for Blender rendering, got shape {arr.shape}")
    unified = to_unified(arr.astype(np.int32, copy=False), game, warn_unmapped=False)
    return np.clip(unified, 0, 4).astype(np.int32, copy=False)


def build_manifest(
    requests: Iterable[BlenderRenderRequest],
    manifest_path: Path,
    *,
    resolution: tuple[int, int] = (640, 520),
) -> Path:
    samples = []
    for request in requests:
        unified = to_blender_unified(request.level, request.game)
        request.output.parent.mkdir(parents=True, exist_ok=True)
        stage = {
            "label": request.label or request.output.stem,
            "output": str(request.output),
            "game": request.game,
            "unified": unified.astype(int).tolist(),
        }
        if request.changed_cells:
            stage["changed_cells"] = request.changed_cells
        if request.path_coords:
            stage["path_coords"] = request.path_coords
        samples.append(
            {
                "title": request.label or request.output.stem,
                "game": request.game,
                "output": str(request.output),
                "stages": [stage],
            }
        )

    manifest = {
        "resolution": [int(resolution[0]), int(resolution[1])],
        "samples": samples,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path


def render_manifest(
    manifest_path: Path,
    *,
    blender: str | None = None,
    blender_script: Path = DEFAULT_BLENDER_SCRIPT,
    asset_dir: Path = DEFAULT_ASSET_DIR,
) -> None:
    executable = find_blender(blender)
    cmd = [
        executable,
        "--background",
        "--python",
        str(blender_script),
        "--",
        "--manifest",
        str(manifest_path),
        "--asset-dir",
        str(asset_dir),
    ]
    subprocess.run(cmd, check=True)


def render_levels(
    requests: Iterable[BlenderRenderRequest],
    manifest_path: Path,
    *,
    blender: str | None = None,
    resolution: tuple[int, int] = (640, 520),
) -> Path:
    manifest = build_manifest(
        list(requests),
        manifest_path,
        resolution=resolution,
    )
    render_manifest(manifest, blender=blender)
    return manifest
