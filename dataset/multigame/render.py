"""
dataset/multigame/render.py
===========================
Tile-grid rendering utilities.

- Palette-based rendering (array_to_rgb, render_sample, etc.)
- tile image based rendering (GameLevelRenderer)

External dependencies: NumPy and Pillow (PIL).
When Pillow is unavailable, only NumPy arrays are returned.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from .base import GameSample
from .handlers.vglc_games import PALETTES
from .handlers.dungeon_handler import DUNGEON_PALETTE

# ── Combined palette ────────────────────────────────────────────────────────────
_ALL_PALETTES: Dict[str, Dict[int, Tuple[int, int, int]]] = {
    **PALETTES,
    "dungeon": DUNGEON_PALETTE,
}

_DEFAULT_UNKNOWN_COLOR = (255, 0, 255)   # Magenta when no mapping exists
_DEFAULT_TILE_SIZE     = 16              # Tile-cell size


def get_palette(game: str) -> Dict[int, Tuple[int, int, int]]:
    """Return a palette dictionary for a game tag."""
    return _ALL_PALETTES.get(game, {})


def array_to_rgb(
    array: np.ndarray,
    palette: Dict[int, Tuple[int, int, int]],
    unknown_color: Tuple[int, int, int] = _DEFAULT_UNKNOWN_COLOR,
) -> np.ndarray:
    """
    (H, W) int array → (H, W, 3) uint8 RGB array.

    Parameters
    ----------
    array         : (H, W) int32/int64 tile ID array
    palette       : mapping from tile_id to (R, G, B)
    unknown_color : color for tiles absent from the palette
    """
    h, w = array.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for tile_id, color in palette.items():
        mask = array == tile_id
        rgb[mask] = color
    # Tiles not registered in the palette
    registered = set(palette.keys())
    for r in range(h):
        for c in range(w):
            if int(array[r, c]) not in registered:
                rgb[r, c] = unknown_color
    return rgb


def render_sample(
    sample: GameSample,
    tile_size: int = _DEFAULT_TILE_SIZE,
    unknown_color: Tuple[int, int, int] = _DEFAULT_UNKNOWN_COLOR,
) -> np.ndarray:
    """
    GameSample → (H*tile_size, W*tile_size, 3) uint8 RGB array.

    Returns
    -------
    NumPy ndarray (works without Pillow).
    """
    palette = get_palette(sample.game)
    small   = array_to_rgb(sample.array, palette, unknown_color)
    if tile_size == 1:
        return small
    # Upscale
    return np.repeat(np.repeat(small, tile_size, axis=0), tile_size, axis=1)


def render_sample_pil(
    sample: GameSample,
    tile_size: int = _DEFAULT_TILE_SIZE,
    unknown_color: Tuple[int, int, int] = _DEFAULT_UNKNOWN_COLOR,
):
    """
    GameSample → PIL Image.
    Raise ImportError if Pillow is not installed.
    """
    from PIL import Image
    rgb = render_sample(sample, tile_size=tile_size, unknown_color=unknown_color)
    return Image.fromarray(rgb, mode="RGB")


def save_rendered(
    sample: GameSample,
    save_path: Path | str,
    tile_size: int = _DEFAULT_TILE_SIZE,
    unknown_color: Tuple[int, int, int] = _DEFAULT_UNKNOWN_COLOR,
) -> Path:
    """
    GameSample  PNG to  save.

    Returns
    -------
    Path of the saved file.
    """
    from PIL import Image
    img = render_sample_pil(sample, tile_size=tile_size, unknown_color=unknown_color)
    out = Path(save_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(out))
    return out


def render_grid(
    samples: List[GameSample],
    cols: int = 4,
    tile_size: int = _DEFAULT_TILE_SIZE,
    gap: int = 2,
    bg_color: Tuple[int, int, int] = (30, 30, 30),
) -> np.ndarray:
    """
    Return an image containing multiple GameSamples arranged in a grid.

    Parameters
    ----------
    samples   : list of GameSamples
    cols      : number of columns
    tile_size : tile-cell size
    gap       : spacing between cells in pixels
    bg_color  : background color as (R, G, B)

    Returns
    -------
    (total_H, total_W, 3) uint8 RGB array
    """
    if not samples:
        return np.zeros((1, 1, 3), dtype=np.uint8)

    rendered = [render_sample(s, tile_size=tile_size) for s in samples]
    max_h = max(r.shape[0] for r in rendered)
    max_w = max(r.shape[1] for r in rendered)

    rows = (len(samples) + cols - 1) // cols
    total_h = rows * max_h + (rows + 1) * gap
    total_w = cols * max_w + (cols + 1) * gap

    canvas = np.full((total_h, total_w, 3), bg_color, dtype=np.uint8)
    for idx, img in enumerate(rendered):
        row_i = idx // cols
        col_i = idx % cols
        y = gap + row_i * (max_h + gap)
        x = gap + col_i * (max_w + gap)
        h, w = img.shape[:2]
        canvas[y:y + h, x:x + w] = img

    return canvas


def save_grid(
    samples: List[GameSample],
    save_path: Path | str,
    cols: int = 4,
    tile_size: int = _DEFAULT_TILE_SIZE,
    gap: int = 2,
    bg_color: Tuple[int, int, int] = (30, 30, 30),
) -> Path:
    """render_grid result  PNG to  save."""
    from PIL import Image
    canvas = render_grid(samples, cols=cols, tile_size=tile_size,
                         gap=gap, bg_color=bg_color)
    out = Path(save_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(canvas, mode="RGB").save(str(out))
    return out


# ── tile image based rendering ───────────────────────────────────────────────────────

class GameLevelRenderer:
    """
    Render game levels from tile images.

    Render each tile using its actual tile image from tile_mapping.json
    and the tile_ims/ directory.
    """

    def __init__(self, tile_ims_dir: Optional[Union[str, Path]] = None):
        """
        Parameters
        ----------
        tile_ims_dir : tile image directory path (default value: current directory of  tile_ims)
        """
        if tile_ims_dir is None:
            current_dir = Path(__file__).parent
            tile_ims_dir = current_dir / "tile_ims"

        self.tile_ims_dir = Path(tile_ims_dir)
        self.mapping_file = self.tile_ims_dir.parent / "tile_mapping.json"

        # tile_mapping.json load
        with open(self.mapping_file, 'r', encoding='utf-8') as f:
            self.tile_mapping = json.load(f)

        # tile image cache
        self._tile_cache = {}

    def _load_tile_image(self, game: str, tile_id: int, tile_size: int = 16) -> np.ndarray:
        """
        Load the image corresponding to a tile ID for a given game.

        Parameters
        ----------
        game : game name (dungeon, doom, pokemon, sokoban, zelda)
        tile_id : tile ID
        tile_size : tile-cell size

        Returns
        -------
        np.ndarray : (tile_size, tile_size, 3) RGB image
        """
        cache_key = (game, tile_id, tile_size)
        if cache_key in self._tile_cache:
            return self._tile_cache[cache_key]

        # Find the tile-image path in tile_mapping.json
        game_config = self.tile_mapping.get(game, {})
        tile_images = game_config.get("_tile_images", {})

        # tile ID  string to  convert
        tile_id_str = str(tile_id)

        # Find the corresponding tile-image filename
        if tile_id_str in tile_images:
            tile_filename = tile_images[tile_id_str]
        else:
            tile_filename = "empty.png"

        # Build the file path
        if tile_filename.startswith("pokemon/"):
            tile_path = self.tile_ims_dir / tile_filename
        else:
            tile_path = self.tile_ims_dir / game / tile_filename

        # image load
        if not tile_path.exists():
            # Create a gray placeholder
            tile_img = np.full((tile_size, tile_size, 3), 128, dtype=np.uint8)
        else:
            from PIL import Image
            img = Image.open(tile_path).convert('RGB')
            if img.size != (tile_size, tile_size):
                img = img.resize((tile_size, tile_size), Image.NEAREST)
            tile_img = np.array(img)

        # cache in  save
        self._tile_cache[cache_key] = tile_img
        return tile_img

    def render(
        self,
        game: str,
        level: np.ndarray,
        tile_size: int = 16,
        save_path: Optional[Union[str, Path]] = None,
        show_tile_numbers: bool = False
    ):
        """
        Render a game level with tile images.

        Parameters
        ----------
        game : game name (dungeon, doom, pokemon, sokoban, zelda)
        level : game level array (2D numpy array, each cell  tile ID)
        tile_size : cell size of each tile
        save_path : output path; do not save when None
        show_tile_numbers : whether to overlay tile IDs on the image

        Returns
        -------
        PIL.Image.Image : rendered image
        """
        from PIL import Image, ImageDraw, ImageFont

        if not isinstance(level, np.ndarray):
            level = np.array(level)

        if level.ndim != 2:
            raise ValueError(f"Level must be 2D array, got shape {level.shape}")

        height, width = level.shape

        # Create the rendering canvas
        canvas = np.zeros((height * tile_size, width * tile_size, 3), dtype=np.uint8)

        # each tile rendering
        for y in range(height):
            for x in range(width):
                tile_id = int(level[y, x])
                tile_img = self._load_tile_image(game, tile_id, tile_size)

                # Place the tile on the canvas
                y_start = y * tile_size
                y_end = y_start + tile_size
                x_start = x * tile_size
                x_end = x_start + tile_size
                canvas[y_start:y_end, x_start:x_end] = tile_img

        # PIL image to  convert
        img = Image.fromarray(canvas, mode='RGB')

        # Display tile IDs
        if show_tile_numbers:
            draw = ImageDraw.Draw(img)

            # Set the font size
            font_size = max(8, int(tile_size * 0.45))
            font = None

            try:
                font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", font_size)
            except Exception:
                try:
                    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
                except:
                    try:
                        font = ImageFont.load_default()
                    except:
                        return img

            # Draw an ID over each tile
            for y in range(height):
                for x in range(width):
                    tile_id = int(level[y, x])
                    text = str(tile_id)

                    x_pos = x * tile_size + tile_size // 2
                    y_pos = y * tile_size + tile_size // 2

                    try:
                        if hasattr(font, 'getbbox'):
                            bbox = font.getbbox(text)
                            text_width = bbox[2] - bbox[0]
                            text_height = bbox[3] - bbox[1]
                        else:
                            text_width = len(text) * font_size // 2
                            text_height = font_size
                    except:
                        text_width = len(text) * font_size // 2
                        text_height = font_size

                    text_x = x_pos - text_width // 2
                    text_y = y_pos - text_height // 2

                    # Background rectangle
                    padding = 1
                    rect_bbox = [
                        text_x - padding,
                        text_y - padding,
                        text_x + text_width + padding,
                        text_y + text_height + padding
                    ]
                    draw.rectangle(rect_bbox, fill=(0, 0, 0, 180))

                    # Draw the text
                    try:
                        draw.text((text_x, text_y), text, fill=(255, 255, 255), font=font)
                    except Exception:
                        pass

        # save
        if save_path is not None:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            img.save(save_path)

        return img


def render_game_level(
    game: str,
    level: np.ndarray,
    tile_size: int = 16,
    save_path: Optional[Union[str, Path]] = None,
    show_tile_numbers: bool = False,
    tile_ims_dir: Optional[Union[str, Path]] = None
):
    """
    Convenience function for rendering a game level with tile images.

    Parameters
    ----------
    game : game name
    level : 2D numpy array
    tile_size : tile-cell size
    save_path : save path
    show_tile_numbers : whether to display tile IDs
    tile_ims_dir : tile image directory (select)

    Returns
    -------
    PIL.Image.Image : rendered image
    """
    renderer = GameLevelRenderer(tile_ims_dir=tile_ims_dir)
    return renderer.render(
        game,
        level,
        tile_size=tile_size,
        save_path=save_path,
        show_tile_numbers=show_tile_numbers
    )
