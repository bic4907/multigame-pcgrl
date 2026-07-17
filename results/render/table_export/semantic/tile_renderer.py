from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_MAPPED_TILE_DIR = Path(__file__).resolve().parents[2] / "assets" / "mapped_tiles"


class SemanticTileRenderer:
    """Render levels from semantic tile exports such as mapped_tiles/<game>/empty.png."""

    def __init__(self, tile_dir: Path | str = DEFAULT_MAPPED_TILE_DIR):
        self.tile_dir = Path(tile_dir)
        manifest_path = self.tile_dir / "manifest.json"
        manifest: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.tile_names = {
            int(tile_id): str(name)
            for tile_id, name in manifest.get("render_tile_ids", {}).items()
        }
        self._tile_cache: dict[tuple[str, int, int], np.ndarray] = {}

    def _load_tile_image(self, game: str, tile_id: int, tile_size: int) -> np.ndarray:
        from PIL import Image

        cache_key = (game, tile_id, tile_size)
        if cache_key in self._tile_cache:
            return self._tile_cache[cache_key]

        semantic_name = self.tile_names.get(int(tile_id), "empty")
        tile_path = self.tile_dir / game / f"{semantic_name}.png"
        if not tile_path.exists():
            tile_path = self.tile_dir / game / "empty.png"

        img = Image.open(tile_path).convert("RGB")
        if img.size != (tile_size, tile_size):
            img = img.resize((tile_size, tile_size), Image.NEAREST)
        tile_img = np.array(img)
        self._tile_cache[cache_key] = tile_img
        return tile_img

    def render(
        self,
        game: str,
        level: np.ndarray,
        tile_size: int = 16,
        save_path: Path | str | None = None,
        show_tile_numbers: bool = False,
    ):
        from PIL import Image

        if show_tile_numbers:
            raise ValueError("SemanticTileRenderer does not support show_tile_numbers=True")
        if not isinstance(level, np.ndarray):
            level = np.array(level)
        if level.ndim != 2:
            raise ValueError(f"Level must be 2D array, got shape {level.shape}")

        height, width = level.shape
        canvas = np.zeros((height * tile_size, width * tile_size, 3), dtype=np.uint8)
        for y in range(height):
            for x in range(width):
                tile_img = self._load_tile_image(game, int(level[y, x]), tile_size)
                y0 = y * tile_size
                x0 = x * tile_size
                canvas[y0 : y0 + tile_size, x0 : x0 + tile_size] = tile_img

        img = Image.fromarray(canvas, mode="RGB")
        if save_path is not None:
            out = Path(save_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            img.save(str(out))
        return img
