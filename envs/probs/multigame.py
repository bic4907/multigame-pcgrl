"""envs/probs/multigame.py  (updated)

MultigameProblem and make_multigame_env() use the unified categories in the
tile_mapping.json _categories section as the tile enum.

_categories (5 categories)
-----------------
  0  EMPTY        - background / traversable empty space
  1  WALL         - solid, impassable obstacle
  2  INTERACTIVE  - interactive entity or structure
  3  HAZARD       - environmental damage, trap, or hostile entity
  4  COLLECTABLE  - item, pickup, or collectible

This module alone can create an environment whose action space matches the
tile_mapping specification.

Usage
-----
    from envs.probs.multigame import make_multigame_env

    env, env_params = make_multigame_env()          # default: narrow, 16x16
    env, env_params = make_multigame_env(
        representation="narrow",
        map_shape=(16, 16),
        rf_shape=(31, 31),
    )

    # Confirm that n_editable_tiles matches NUM_CATEGORIES (currently 5)
    assert env.rep.n_editable_tiles == 5
"""
from __future__ import annotations

import json
from enum import IntEnum
from functools import partial
from pathlib import Path
from typing import Optional, Tuple

import chex
from flax import struct
import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image

from envs.probs.problem import Placeholder, Problem, ProblemState

# ── tile_mapping.json load ──────────────────────────────────────────────────────
_MAPPING_FILE = Path(__file__).parent.parent.parent / "dataset" / "multigame" / "tile_mapping.json"
with _MAPPING_FILE.open("r", encoding="utf-8") as _f:
    _MAPPING_CONFIG: dict = json.load(_f)

_CATEGORIES: dict[int, str] = {
    int(k): v for k, v in _MAPPING_CONFIG["_categories"].items()
}
_CATEGORY_COLORS: dict[int, tuple] = {
    int(k): tuple(v) for k, v in _MAPPING_CONFIG.get("_category_colors_rgb", {}).items()
}
NUM_CATEGORIES: int = len(_CATEGORIES)   # Currently 5

# ── Tile-image mapping loaded from JSON _category_tile_images ─────────────────
# key "border" → BORDER tile (index 0)
# key "0".."N"  → category index (MultigameTiles index = cat+1)
_TILE_IMS_DIR = Path(__file__).parent / "tile_ims"

_raw_tile_images: dict = _MAPPING_CONFIG.get("_category_tile_images", {})
_BORDER_IMAGE: str       = _raw_tile_images.get("border", "solid.png")
_CATEGORY_IMAGE_FILES: dict[int, str] = {
    int(k): v
    for k, v in _raw_tile_images.items()
    if k not in ("_comment", "border")
}

_TILE_SIZE = 16


def _make_color_tile(rgb: tuple, size: int = _TILE_SIZE) -> Image.Image:
    """Create a solid-color RGBA tile image."""
    r, g, b = rgb
    arr = np.full((size, size, 4), [r, g, b, 255], dtype=np.uint8)
    return Image.fromarray(arr, mode="RGBA")


def _load_tile_image(filename: str, size: int = _TILE_SIZE) -> Image.Image:
    """Load envs/probs/tile_ims/<filename>, falling back to purple if absent."""
    path = _TILE_IMS_DIR / filename
    if path.exists():
        return Image.open(path).convert("RGBA").resize((size, size))
    import warnings
    warnings.warn(f"[multigame] tile image not found: {path}", stacklevel=2)
    return _make_color_tile((200, 0, 200), size)   # Purple marks a missing image


def _load_or_color_tile(cat_idx: int, size: int = _TILE_SIZE) -> Image.Image:
    """category index → tile image.
    Use the file specified by _CATEGORY_IMAGE_FILES (JSON), or create a
    solid-color tile from _CATEGORY_COLORS when no file is configured."""
    fname = _CATEGORY_IMAGE_FILES.get(cat_idx)
    if fname:
        return _load_tile_image(fname, size)
    color = _CATEGORY_COLORS.get(cat_idx, (128, 128, 128))
    return _make_color_tile(color, size)


# ── tile_mapping._categories → IntEnum ─────────────────────────────────────────
# The environment convention requires BORDER at index 0, so category indices
# are shifted by one: BORDER=0 and categories=1..NUM_CATEGORIES.
#
#   MultigameTiles
#   --------------
#   BORDER = 0            (internal boundary tile; unavailable as an action)
#   EMPTY  = 1          (category 0)
#   WALL   = 2          (category 1)
#   INTERACTABLE = 3      (category 2)
#   HAZARD = 4          (category 3)
#   COLLECTIBLE = 5      (category 4)

MultigameTiles = IntEnum(
    "MultigameTiles",
    {"BORDER": 0, **{name.upper(): idx + 1 for idx, name in _CATEGORIES.items()}},
)
"""Tile enum generated automatically from tile_mapping._categories.

BORDER=0, EMPTY=1, WALL=2, INTERACTIVE=3, HAZARD=4, COLLECTABLE=5.
Excluding BORDER leaves 5 editable tiles, equal to NUM_CATEGORIES.
"""

# Tiles considered passable by pathfinding, region, and path-length metrics.
# Include only names that exist so category changes remain backward compatible.
_PASSABLE_TILE_NAMES = (
    "EMPTY",
    "FLOOR",
    "INTERACTIVE",
    "INTERACTABLE",
    "HAZARD",
    "OBJECT",
    "SPAWN",
    "COLLECTABLE",
    "COLLECTIBLE",
)
_passable_tiles = [getattr(MultigameTiles, n) for n in _PASSABLE_TILE_NAMES if hasattr(MultigameTiles, n)]
if not _passable_tiles and hasattr(MultigameTiles, "EMPTY"):
    _passable_tiles = [MultigameTiles.EMPTY]
MultigamePassable = jnp.array(_passable_tiles, dtype=jnp.int32)


class MultigameMetrics(IntEnum):
    """The multigame environment has no separate metrics; use one dummy element."""
    DUMMY = 0


@struct.dataclass
class MultigameState(ProblemState):
    """MultigameProblem  for  dummy state."""
    pass


class MultigameProblem(Problem):
    """Problem whose action space directly uses the five unified categories.

    - tile_enum = MultigameTiles (BORDER + 5 categories = 6 entries)
    - editable = 5 (= NUM_CATEGORIES, excluding BORDER)
    - stats/reward are null (0); subclasses may override them for reward shaping.
    """

    tile_enum = MultigameTiles
    metrics_enum = MultigameMetrics
    region_metrics_enum = Placeholder

    # tile create probability: BORDER=0, EMPTY=0.30, WALL=0.40, remaining each 0.10 (normalize)
    _p_norm = 0.30 + 0.40 + 0.10 * (NUM_CATEGORIES - 2)
    tile_probs = tuple(
        [0.0, 0.30 / _p_norm, 0.40 / _p_norm]
        + [0.10 / _p_norm] * (NUM_CATEGORIES - 2)
    )

    # No fixed counts; every tile can be placed freely
    tile_nums = tuple([0] * len(MultigameTiles))

    # stat weights / trgs / ctrl_threshes: shape (1,) — dummy, no reward
    stat_weights  = np.zeros(1)
    stat_trgs     = jnp.zeros(1)   # Must be a jnp.array for Problem.__init__
    ctrl_threshes = np.zeros(1)

    tile_size = _TILE_SIZE
    unavailable_tiles: list = []
    passable_tiles = MultigamePassable

    def __init__(self, map_shape: Tuple[int, int], ctrl_metrics: Tuple, pinpoints: bool):
        super().__init__(map_shape, ctrl_metrics, pinpoints)

    def get_metric_bounds(self, map_shape: Tuple[int, int]):
        """Return a dummy (1, 2) array because there are no metrics."""
        return np.zeros((1, 2), dtype=np.float32)

    def get_curr_stats(self, env_map: chex.Array) -> MultigameState:
        """Return zero statistics because there are no metrics."""
        stats = jnp.zeros(len(MultigameMetrics))
        return MultigameState(stats=stats)

    def get_stats(self, env_map, prob_state: ProblemState):
        """Return zeros with shape (1,) because there are no metrics."""
        return np.zeros(1)

    def get_path_coords(self, env_map: chex.Array, prob_state: ProblemState):
        """Return an empty tuple because there are no paths (render-compatible)."""
        return ()

    def draw_path(self, lvl_img, env_map, border_size, path_coords_tpl, tile_size):
        """path none → image as-is return."""
        return lvl_img

    @partial(jax.jit, static_argnums=(0, 3))
    def get_cont_obs(self, env_map, condition, raw_obs: bool = False) -> jnp.array:
        """CPCGRL condition → observation convert.

        All condition values are numeric, so mask unused -1 values to 0 and return them.
        total output shape: (5,)  — vec_input_dim  and  same.
        """
        mask = jnp.not_equal(condition, -1).astype(jnp.float32)
        obs = jnp.where(mask == 1, condition, 0.0)
        return obs

    def init_graphics(self):
        """Initialize tile images from tile_mapping.json _category_tile_images.

        MultigameTiles index:
          BORDER = 0  → _category_tile_images["border"]
          EMPTY  = 1  → _category_tile_images["0"]
          WALL   = 2  → _category_tile_images["1"]
          ...
          HAZARD = 7  → _category_tile_images["6"]  (lava.png)
        """
        from envs.utils import idx_dict_to_arr

        graphics: dict = {}

        # BORDER (index 0): load from the JSON "border" key
        graphics[0] = _load_tile_image(_BORDER_IMAGE)

        # category tiles: MultigameTiles index = cat_idx + 1
        for cat_idx in _CATEGORIES:
            graphics[cat_idx + 1] = _load_or_color_tile(cat_idx)

        self.graphics = jnp.array(idx_dict_to_arr(graphics))
        super().init_graphics()


def render_multigame_map(env_map: np.ndarray, tile_size: int = _TILE_SIZE) -> Image.Image:
    """Render an (H, W) int32 env_map as a PIL image using _category_tile_images.

    Parameters
    ----------
    env_map  : (H, W) NumPy array containing MultigameTiles integers
    tile_size: tile cell size (default: 16)

    Returns
    -------
    PIL.Image.Image  (RGB)
    """
    arr = render_multigame_map_np(env_map, tile_size)  # (H*ts, W*ts, 3) uint8
    return Image.fromarray(arr, mode="RGB")


# ── tile array cache ─────────────────────────────────────────────────────────────
_tile_array_cache: dict[int, np.ndarray] = {}


def _get_tile_array(tile_size: int = _TILE_SIZE) -> np.ndarray:
    """Return an RGBA NumPy array by tile index, cached per tile_size.

    Returns
    -------
    tile_array : (num_tiles, tile_size, tile_size, 4) uint8
                 index 0 = BORDER, 1..NUM_CATEGORIES = categories
    """
    if tile_size in _tile_array_cache:
        return _tile_array_cache[tile_size]

    num_tiles = 1 + NUM_CATEGORIES  # BORDER + categories
    tile_arr = np.zeros((num_tiles, tile_size, tile_size, 4), dtype=np.uint8)

    border_img = _load_tile_image(_BORDER_IMAGE, tile_size)
    tile_arr[0] = np.array(border_img.convert("RGBA"))

    for cat_idx in _CATEGORIES:
        img = _load_or_color_tile(cat_idx, tile_size)
        tile_arr[cat_idx + 1] = np.array(img.convert("RGBA"))

    _tile_array_cache[tile_size] = tile_arr
    return tile_arr


def render_multigame_map_np(env_map: np.ndarray, tile_size: int = _TILE_SIZE) -> np.ndarray:
    """env_map (H, W) → numpy RGB array (H*ts, W*ts, 3).

    Assemble in O(1) with NumPy fancy indexing instead of a PIL paste loop.
    Tile arrays are cached by tile_size, avoiding reload costs on repeated calls.
    """
    tile_arr = _get_tile_array(tile_size)  # (T, ts, ts, 4)

    H, W = env_map.shape
    # Clamp out-of-range indices to the fallback entry at index 0
    idx = np.clip(env_map.astype(np.int32), 0, len(tile_arr) - 1)  # (H, W)

    # fancy indexing: (H, W, ts, ts, 4) → transpose → (H*ts, W*ts, 4)
    canvas = tile_arr[idx]                       # (H, W, ts, ts, 4)
    canvas = canvas.transpose(0, 2, 1, 3, 4)    # (H, ts, W, ts, 4)
    canvas = canvas.reshape(H * tile_size, W * tile_size, 4)

    return canvas[:, :, :3]  # RGB


def render_multigame_maps_batch(
    env_maps: np.ndarray,
    tile_size: int = _TILE_SIZE,
) -> np.ndarray:
    """Render an (N, H, W) array to (N, H*ts, W*ts, 3) uint8 in one operation.

    Uses only NumPy fancy indexing and reshape, with no Python loop.
    """
    tile_arr = _get_tile_array(tile_size)  # (T, ts, ts, 4)

    N, H, W = env_maps.shape
    idx = np.clip(env_maps.astype(np.int32), 0, len(tile_arr) - 1)  # (N, H, W)

    canvas = tile_arr[idx]                           # (N, H, W, ts, ts, 4); can be memory-intensive
    canvas = canvas.transpose(0, 1, 3, 2, 4, 5)    # (N, H, ts, W, ts, 4)
    canvas = canvas.reshape(N, H * tile_size, W * tile_size, 4)

    return canvas[:, :, :, :3]  # RGB, (N, H*ts, W*ts, 3)


# ── Factory function ───────────────────────────────────────────────────────────

def make_multigame_env(
    representation: str = "narrow",
    map_shape: Tuple[int, int] = (16, 16),
    rf_shape: Tuple[int, int] | None = None,
    act_shape: Tuple[int, int] = (1, 1),
    max_board_scans: float = 3.0,
):
    """Return a PCGRLEnv whose action space matches tile_mapping._categories.

    Parameters
    ----------
    representation  : "narrow" | "wide" | "turtle" | "nca"
    map_shape       : (H, W) map size (default 16x16)
    rf_shape        : receptive-field size; defaults to 2*map_width-1 when None
    act_shape       : action-patch size ((1, 1) for narrow/turtle)
    max_board_scans : maximum number of board scans

    Returns
    -------
    (env, env_params) : (PCGRLEnv, PCGRLEnvParams)

    Guarantees
    ----------
    env.rep.n_editable_tiles == NUM_CATEGORIES  (== 7)
    """
    from envs.pcgrl_env import PCGRLEnv, PCGRLEnvParams, ProbEnum, RepEnum, PROB_CLASSES

    # Register MultigameProblem in PROB_CLASSES, always refreshing the entry
    _MULTIGAME_KEY = max(ProbEnum) + 1
    PROB_CLASSES[_MULTIGAME_KEY] = MultigameProblem

    # rf_shape automatic compute
    if rf_shape is None:
        rf_size = 2 * map_shape[0] - 1
        rf_shape = (rf_size, rf_size)

    rep_key = RepEnum[representation.upper()]

    env_params = PCGRLEnvParams(
        problem=_MULTIGAME_KEY,
        representation=int(rep_key),
        map_shape=map_shape,
        rf_shape=rf_shape,
        act_shape=act_shape,
        max_board_scans=max_board_scans,
    )
    env = PCGRLEnv(env_params)
    return env, env_params
