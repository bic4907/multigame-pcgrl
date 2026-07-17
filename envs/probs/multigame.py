"""envs/probs/multigame.py  (updated)

tile_mapping.json  of  unified categories(_categories text)  tile_enum as  text
MultigameProblem text make_multigame_env() text.

_categories (7text)
-----------------
  0  EMPTY   – background / void
  1  WALL    – solid, impassable obstacle
  2  FLOOR   – traversable ground
  3  ENEMY   – hostile entity
  4  OBJECT  – item / pickup / collectible
  5  SPAWN   – player start / exit / door
  6  HAZARD  – environmental damage / trap

  file text as  "tile_mapping text and  sametext action text   text env"   maketext text text.

Usage
-----
    from envs.probs.multigame import make_multigame_env

    env, env_params = make_multigame_env()          # default: narrow, 16x16
    env, env_params = make_multigame_env(
        representation="narrow",
        map_shape=(16, 16),
        rf_shape=(31, 31),
    )

    # n_editable_tiles   NUM_CATEGORIES(7)  and  text  check
    assert env.rep.n_editable_tiles == 7
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
NUM_CATEGORIES: int = len(_CATEGORIES)   # 7

# ── tile image file text: JSON _category_tile_images  in  load ─────────────────
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
    """text RGBA tile image create."""
    r, g, b = rgb
    arr = np.full((size, size, 4), [r, g, b, 255], dtype=np.uint8)
    return Image.fromarray(arr, mode="RGBA")


def _load_tile_image(filename: str, size: int = _TILE_SIZE) -> Image.Image:
    """envs/probs/tile_ims/<filename>   load. if missing text fallback."""
    path = _TILE_IMS_DIR / filename
    if path.exists():
        return Image.open(path).convert("RGBA").resize((size, size))
    import warnings
    warnings.warn(f"[multigame] tile image not found: {path}", stacklevel=2)
    return _make_color_tile((200, 0, 200), size)   # text = missing tabletext


def _load_or_color_tile(cat_idx: int, size: int = _TILE_SIZE) -> Image.Image:
    """category index → tile image.
    _CATEGORY_IMAGE_FILES(JSON) in  file  text text text file text for ,
    if missing _CATEGORY_COLORS  to  text tile create."""
    fname = _CATEGORY_IMAGE_FILES.get(cat_idx)
    if fname:
        return _load_tile_image(fname, size)
    color = _CATEGORY_COLORS.get(cat_idx, (128, 128, 128))
    return _make_color_tile(color, size)


# ── tile_mapping._categories → IntEnum ─────────────────────────────────────────
# BORDER(0)   env text index 0  text text to , category index  1-shift text
# BORDER=0, categories=1..NUM_CATEGORIES  to  batchtext.
#
#   MultigameTiles
#   --------------
#   BORDER = 0          (env internal text tile, action text )
#   EMPTY  = 1          (category 0)
#   WALL   = 2          (category 1)
#   INTERACTABLE = 3      (category 2)
#   HAZARD = 4          (category 3)
#   COLLECTIBLE = 5      (category 4)

MultigameTiles = IntEnum(
    "MultigameTiles",
    {"BORDER": 0, **{name.upper(): idx + 1 for idx, name in _CATEGORIES.items()}},
)
"""tile_mapping._categories  in  automatic createtext tile enum.

BORDER=0, EMPTY=1, WALL=2, ..., HAZARD=7  (total 8text)
 in  BORDER   text editable = 7 = NUM_CATEGORIES.
"""

# text/region/path-length text in  "text and  available" as  text tile.
# textgame text  text also  name  text  text automatic text.
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
    """textgame env   separate text texttable  text for text text text. dummy 1-element."""
    DUMMY = 0


@struct.dataclass
class MultigameState(ProblemState):
    """MultigameProblem  for  dummy state."""
    pass


class MultigameProblem(Problem):
    """tile_mapping.json  of  unified 7-category   as-is action text as  text  Problem.

    - tile_enum  = MultigameTiles  (BORDER + 7 categories = 8text)
    - editable   = 7 (= NUM_CATEGORIES, BORDER text)
    - stat/reward   null (0) — reward shaping   text textclass in  text text.
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

    # fixed count none (text text batch)
    tile_nums = tuple([0] * len(MultigameTiles))

    # stat weights / trgs / ctrl_threshes: shape (1,) — dummy, no reward
    stat_weights  = np.zeros(1)
    stat_trgs     = jnp.zeros(1)   # jnp.array text Problem.__init__  in  text text
    ctrl_threshes = np.zeros(1)

    tile_size = _TILE_SIZE
    unavailable_tiles: list = []
    passable_tiles = MultigamePassable

    def __init__(self, map_shape: Tuple[int, int], ctrl_metrics: Tuple, pinpoints: bool):
        super().__init__(map_shape, ctrl_metrics, pinpoints)

    def get_metric_bounds(self, map_shape: Tuple[int, int]):
        """text texttable none → dummy (1, 2) array."""
        return np.zeros((1, 2), dtype=np.float32)

    def get_curr_stats(self, env_map: chex.Array) -> MultigameState:
        """text texttable none → zeros stats."""
        stats = jnp.zeros(len(MultigameMetrics))
        return MultigameState(stats=stats)

    def get_stats(self, env_map, prob_state: ProblemState):
        """text texttable none → zeros (1,)."""
        return np.zeros(1)

    def get_path_coords(self, env_map: chex.Array, prob_state: ProblemState):
        """path none → empty tuple (render text)."""
        return ()

    def draw_path(self, lvl_img, env_map, border_size, path_coords_tpl, tile_size):
        """path none → image as-is return."""
        return lvl_img

    @partial(jax.jit, static_argnums=(0, 3))
    def get_cont_obs(self, env_map, condition, raw_obs: bool = False) -> jnp.array:
        """CPCGRL condition → observation convert.

        text condition text  text text to , -1(text for )  0 as  text as-is return.
        total output shape: (5,)  — vec_input_dim  and  same.
        """
        mask = jnp.not_equal(condition, -1).astype(jnp.float32)
        obs = jnp.where(mask == 1, condition, 0.0)
        return obs

    def init_graphics(self):
        """tile_mapping.json  of  _category_tile_images   text tile image  initializetext.

        MultigameTiles index:
          BORDER = 0  → _category_tile_images["border"]
          EMPTY  = 1  → _category_tile_images["0"]
          WALL   = 2  → _category_tile_images["1"]
          ...
          HAZARD = 7  → _category_tile_images["6"]  (lava.png)
        """
        from envs.utils import idx_dict_to_arr

        graphics: dict = {}

        # BORDER (index 0): JSON "border" text in  load
        graphics[0] = _load_tile_image(_BORDER_IMAGE)

        # category tiles: MultigameTiles index = cat_idx + 1
        for cat_idx in _CATEGORIES:
            graphics[cat_idx + 1] = _load_or_color_tile(cat_idx)

        self.graphics = jnp.array(idx_dict_to_arr(graphics))
        super().init_graphics()


def render_multigame_map(env_map: np.ndarray, tile_size: int = _TILE_SIZE) -> Image.Image:
    """tile_mapping._category_tile_images  in  text env_map (H×W int32)   PIL Image  to  renderingtext.

    Parameters
    ----------
    env_map  : (H, W) numpy array, text  MultigameTiles integer
    tile_size: tile textcell size (default 16)

    Returns
    -------
    PIL.Image.Image  (RGB)
    """
    arr = render_multigame_map_np(env_map, tile_size)  # (H*ts, W*ts, 3) uint8
    return Image.fromarray(arr, mode="RGB")


# ── tile array cache ─────────────────────────────────────────────────────────────
_tile_array_cache: dict[int, np.ndarray] = {}


def _get_tile_array(tile_size: int = _TILE_SIZE) -> np.ndarray:
    """tile indextext RGBA numpy array  return. tile_sizeby text.

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

    PIL paste text text numpy fancy-indexing  as  O(1) text.
    tile array  tile_sizeby text repetition call text  to text text for  none.
    """
    tile_arr = _get_tile_array(tile_size)  # (T, ts, ts, 4)

    H, W = env_map.shape
    # range outside index  fallback (text = index 0 as  clamp, text to   text  index)
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
    """(N, H, W) array  text text in  rendering → (N, H*ts, W*ts, 3) uint8.

    numpy fancy-indexing + reshape text text for text to  for text none.
    """
    tile_arr = _get_tile_array(tile_size)  # (T, ts, ts, 4)

    N, H, W = env_maps.shape
    idx = np.clip(env_maps.astype(np.int32), 0, len(tile_arr) - 1)  # (N, H, W)

    canvas = tile_arr[idx]                           # (N, H, W, ts, ts, 4) — text text text warning
    canvas = canvas.transpose(0, 1, 3, 2, 4, 5)    # (N, H, ts, W, ts, 4)
    canvas = canvas.reshape(N, H * tile_size, W * tile_size, 4)

    return canvas[:, :, :, :3]  # RGB, (N, H*ts, W*ts, 3)


# ── text function ─────────────────────────────────────────────────────────────────

def make_multigame_env(
    representation: str = "narrow",
    map_shape: Tuple[int, int] = (16, 16),
    rf_shape: Tuple[int, int] | None = None,
    act_shape: Tuple[int, int] = (1, 1),
    max_board_scans: float = 3.0,
):
    """tile_mapping._categories text and  sametext action text   text PCGRLEnv   returntext.

    Parameters
    ----------
    representation  : "narrow" | "wide" | "turtle" | "nca"
    map_shape       : (H, W) map size (default 16x16)
    rf_shape        : receptive field size. None  text 2*map_width-1  to  automatic config.
    act_shape       : action patch size (narrow/turtle  of  text (1,1))
    max_board_scans : text  maximum text text text

    Returns
    -------
    (env, env_params) : (PCGRLEnv, PCGRLEnvParams)

    Guarantees
    ----------
    env.rep.n_editable_tiles == NUM_CATEGORIES  (== 7)
    """
    from envs.pcgrl_env import PCGRLEnv, PCGRLEnvParams, ProbEnum, RepEnum, PROB_CLASSES

    # MultigameProblem   PROB_CLASSES  in  text (always latest text to  text)
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
