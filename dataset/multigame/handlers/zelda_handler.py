"""
dataset/multigame/handlers/zelda_handler.py
===========================================
The Legend of Zelda (TheVGLC) handler.

The VGLC Zelda data stores every dungeon map of a level in one file.
Each character is a tile id, and a room is an 11x16 (WxH) patch split out of the map.
 Patches that are entirely void ('-') are discarded.

Map structure (NES The Legend of Zelda)
----------------------------------
  - One room = 11 characters wide x 16 rows tall
  - Layout: a 2-row wall band (WW...), a 2-character wall margin (WW), and a 7x12 interior
  - Adjacent rooms share their walls (WWWW = room 1's wall + room 2's wall)
  - Levels are separated by a line of 11 dashes ('-----------')

preprocessing:
  1. Strip one row/column of border wall from the 11x16 patch → 9x14 (interior + one wall ring)
  2. Nearest-neighbour stretch along the short axis (width 9) → 14x14 square
  3. Centre the 14x14 patch in a 16x16 grid (WALL padding)
  4. Double the data with 90-degree rotation augmentation
  5. In half of the maps, randomly place 1-5 MOB/OBJECT tiles on FLOOR/EMPTY cells (seed=42)

Tile ids (from vglc_games/zelda.py)
-------------------------------------
0  : EMPTY   (-, void)
1  : WALL    (W)
2  : FLOOR   (F)
3  : DOOR    (D)
4  : BLOCK   (B)
5  : START   (S)
6  : MOB     (M)
7  : OBJECT  (O, I, P)
99 : UNKNOWN
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np

from ..base import (
    BaseGameHandler,
    GameSample,
    GameTag,
    TileLegend,
)
from .vglc_games.zelda import (
    ZeldaTile,
    ZeldaPreprocessor,
    ZELDA_PALETTE,
    make_legend,
)

# ── path default value ─────────────────────────────────────────────────────────────────
_DEFAULT_ZELDA_ROOT = (
    Path(__file__).parent.parent.parent / "TheVGLC" / "The Legend of Zelda"
)

# ── Room size (as laid out in the NES maps) ──────────────────────────────────────
PATCH_W = 11   # width in characters
PATCH_H = 16   # height in rows
TARGET_SIZE = 16  # output size (16x16)

# border wall remove  after  size
TRIMMED_W = PATCH_W - 2   # 11 - 2 = 9
TRIMMED_H = PATCH_H - 2   # 16 - 2 = 14


def _read_map_text(path: Path) -> List[str]:
    """Return the lines of a level file, dropping the level separators."""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    while lines and not lines[-1].strip():
        lines.pop()
    return lines


def _text_to_int_grid(
    lines: List[str],
    preprocessor: ZeldaPreprocessor,
) -> np.ndarray:
    """
    Convert a whole level map into a 2D integer array.
    Rows are padded to equal length with EMPTY, then characters are mapped to integers.
    """
    if not lines:
        return np.zeros((0, 0), dtype=np.int32)
    max_w = max(len(l) for l in lines)
    H = len(lines)
    grid = np.full((H, max_w), ZeldaTile.EMPTY, dtype=np.int32)
    for r, line in enumerate(lines):
        for c, ch in enumerate(line):
            grid[r, c] = preprocessor.char_to_int(ch)
    return grid


def _extract_rooms(
    grid: np.ndarray,
    patch_h: int = PATCH_H,
    patch_w: int = PATCH_W,
) -> List[Tuple[np.ndarray, int, int]]:
    """
    Split a level map into patch_h x patch_w rooms,
     discarding any patch that is entirely EMPTY(0).

    Returns
    -------
    list of (patch_array, row_idx, col_idx)
        patch_array : (patch_h, patch_w) int32
        row_idx     : row index of the patch (0-based)
        col_idx     : column index of the patch (0-based)
    """
    H, W = grid.shape
    rooms = []

    for iy, y in enumerate(range(0, H - patch_h + 1, patch_h)):
        for ix, x in enumerate(range(0, W - patch_w + 1, patch_w)):
            patch = grid[y : y + patch_h, x : x + patch_w]
            # Skip patches that are entirely empty (void)
            if np.all(patch == ZeldaTile.EMPTY):
                continue
            rooms.append((patch.copy(), iy, ix))

    return rooms


def _trim_outer_wall(patch: np.ndarray) -> np.ndarray:
    """
    Strip one row/column of border wall from an 11x16 patch.

    Layout (16H x 11W):
        row 0  : WWWWWWWWWWW  ← remove (wall row)
        row 1  : WWWWDDDWWWW  ← keep (wall + interior)
        ...
        row 14 : WWWWDDDWWWW  ← keep
        row 15 : WWWWWWWWWWW  ← remove (wall row)
        col 0  : W (remove)
        col 10 : W (remove)

    result: 9W × 14H
    """
    return patch[1:-1, 1:-1].copy()


def _stretch_to_square(patch: np.ndarray) -> np.ndarray:
    """
    Stretch the shorter axis with nearest-neighbour sampling to make the patch square.

    e.g. 14H x 9W → 14 x 14  (width 9 → 14)
        9H x 14W → 14 x 14  (height 9 → 14)

    Nearest-neighbour indexing is used so the integer tile ids are preserved.
    """
    h, w = patch.shape
    if h == w:
        return patch

    target = max(h, w)

    if w < h:
        # Width is the shorter axis → stretch width up to h
        col_indices = np.round(np.linspace(0, w - 1, target)).astype(int)
        return patch[:, col_indices].copy()
    else:
        # Height is the shorter axis → stretch height up to w
        row_indices = np.round(np.linspace(0, h - 1, target)).astype(int)
        return patch[row_indices, :].copy()


def _center_pad_to_16x16(patch: np.ndarray) -> np.ndarray:
    """
    Centre a patch in a 16x16 grid, padding as needed.
    The padding is WALL(1), matching the border.
    """
    h, w = patch.shape
    if h == TARGET_SIZE and w == TARGET_SIZE:
        return patch
    out = np.full((TARGET_SIZE, TARGET_SIZE), ZeldaTile.WALL, dtype=np.int32)
    y0 = (TARGET_SIZE - h) // 2
    x0 = (TARGET_SIZE - w) // 2
    out[y0 : y0 + h, x0 : x0 + w] = patch
    return out


def _rotate_90(patch: np.ndarray) -> np.ndarray:
    """Rotate the patch 90 degrees."""
    return np.rot90(patch, k=-1).copy()


def _flip_ud(patch: np.ndarray) -> np.ndarray:
    """Flip the patch horizontally."""
    return np.flipud(patch).copy()


def _is_uniform_center_12x12(padded: np.ndarray) -> bool:
    """
    Check whether the centre 12x12 region of a 16x16 map (a 2-cell margin) is a single tile.

    Parameters
    ----------
    padded : np.ndarray
        16x16 map

    Returns
    -------
    bool
        True when the centre 12x12 is uniform
    """
    center = padded[2:14, 2:14]  # center 12x12 extract
    return bool(np.all(center == center[0, 0]))


# Tiles that may be dropped in
_DROP_TILES = [ZeldaTile.MOB, ZeldaTile.OBJECT]

# Tiles that may be overwritten by a drop (FLOOR and EMPTY)
_DROPPABLE_TILES = {ZeldaTile.FLOOR, ZeldaTile.EMPTY}

# Fraction of maps the drop augmentation is applied to
DROP_AUG_RATIO = 1.0  # applied to 100% of the maps

# Number of tiles to drop
DROP_COUNT_MIN = 0
DROP_COUNT_MAX = 25


def _augment_random_drop(
    padded: np.ndarray,
    rng: np.random.Generator,
) -> Optional[np.ndarray]:
    """
    Randomly turn 0-25 of the FLOOR/EMPTY tiles of a 16x16 map into
    MOB or OBJECT tiles.

    Returns None when there is no eligible position.
    """
    droppable = np.argwhere(
        np.isin(padded, list(_DROPPABLE_TILES))
    )
    if len(droppable) == 0:
        return None

    n_drop = rng.integers(DROP_COUNT_MIN, DROP_COUNT_MAX + 1)
    n_drop = min(n_drop, len(droppable))

    aug = padded.copy()
    chosen = droppable[rng.choice(len(droppable), size=n_drop, replace=False)]
    for pos in chosen:
        tile = rng.choice(_DROP_TILES)
        aug[pos[0], pos[1]] = tile
    return aug


def _fill_missing_tiles(padded: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Fill in the missing tile types in maps that lack MOB (hazard) or OBJECT (collectable).

    - No MOB    → turn 1-DROP_COUNT_MAX FLOOR/EMPTY cells into MOB
    - No OBJECT → turn 1-DROP_COUNT_MAX of the remaining FLOOR/EMPTY cells into OBJECT
    """
    result = padded.copy()

    def _add_tile(arr: np.ndarray, tile: int) -> np.ndarray:
        droppable = np.argwhere(np.isin(arr, list(_DROPPABLE_TILES)))
        if len(droppable) == 0:
            return arr
        n = int(rng.integers(1, DROP_COUNT_MAX + 1))
        n = min(n, len(droppable))
        chosen = droppable[rng.choice(len(droppable), size=n, replace=False)]
        for pos in chosen:
            arr[pos[0], pos[1]] = tile
        return arr

    if not np.any(result == ZeldaTile.MOB):
        result = _add_tile(result, ZeldaTile.MOB)
    if not np.any(result == ZeldaTile.OBJECT):
        result = _add_tile(result, ZeldaTile.OBJECT)

    return result


# ── Integer → character mapping ───────────────────────────────────────────────────
_INT_TO_CHAR: Dict[int, str] = {
    ZeldaTile.EMPTY:   "-",
    ZeldaTile.WALL:    "W",
    ZeldaTile.FLOOR:   "F",
    ZeldaTile.DOOR:    "D",
    ZeldaTile.BLOCK:   "B",
    ZeldaTile.START:   "S",
    ZeldaTile.MOB:     "M",
    ZeldaTile.OBJECT:  "O",
    ZeldaTile.UNKNOWN: "?",
}


def _int_to_char(val: int) -> str:
    return _INT_TO_CHAR.get(val, "?")


def _array_to_char_grid(arr: np.ndarray) -> List[List[str]]:
    """Integer array → character grid."""
    return [[_int_to_char(int(val)) for val in row] for row in arr]


class ZeldaHandler(BaseGameHandler):
    """
    The Legend of Zelda (TheVGLC) handler.

    Preprocessing steps:
      1. Split each level map under Processed/ into 11x16 rooms
      2. Drop the empty patches (entirely EMPTY)
      3. Remove one border-wall row/column from each side -> 9x14
      4. Nearest-neighbour stretch along the short axis → 14x14
      5. Centre in a 16x16 grid (WALL padding)
      6. Apply 90-degree rotation augmentation -> double the data

    Parameters
    ----------
    root  : TheVGLC/The Legend of Zelda folder path
    split : subdirectory (default: "Processed")
    """

    def __init__(
        self,
        root: Path | str = _DEFAULT_ZELDA_ROOT,
        split: str = "Processed",
        handler_config: Optional[Any] = None,
    ) -> None:
        self._root = Path(root)
        self._split = split
        self._handler_config = handler_config
        self._preprocessor = ZeldaPreprocessor()
        self._legend = make_legend()
        self._samples: Optional[List[GameSample]] = None  # lazy

    @property
    def game_tag(self) -> str:
        return GameTag.ZELDA

    # ── File discovery ───────────────────────────────────────────────────────────

    def _discover_files(self) -> List[Path]:
        """Return the list of level files under Processed/."""
        processed = self._root / self._split
        if not processed.exists():
            raise FileNotFoundError(
                f"Zelda Processed directory not found: {processed}"
            )
        files = sorted(processed.glob("*.txt"))
        files = [p for p in files if not p.name.lower().startswith("readme")]
        return files

    # ── all load ────────────────────────────────────────────────────────────────

    def _load_all(self) -> List[GameSample]:
        files = self._discover_files()
        if not files:
            raise FileNotFoundError(
                f"[zelda] No level files found under: {self._root / self._split}"
            )

        rng = np.random.default_rng(seed=42)
        samples: List[GameSample] = []

        for fpath in files:
            fname = fpath.stem  # e.g. "tloz1_1"
            lines = _read_map_text(fpath)
            grid = _text_to_int_grid(lines, self._preprocessor)

            if grid.size == 0:
                continue

            rooms = _extract_rooms(grid)
            for patch, ry, rx in rooms:
                # 1) border wall remove → 14H × 9W
                trimmed = _trim_outer_wall(patch)

                # 2) Stretch the short axis → 14 x 14 square
                squared = _stretch_to_square(trimmed)

                # 3) Centre in 16x16 (one ring of WALL padding)
                padded = _center_pad_to_16x16(squared)

                # 4) Deterministically add OBJECT tiles with probability when none exist
                base_padded = self._preprocessor.postprocess_array(padded)

                source_id = f"{fname}_r{ry}_c{rx}"
                base_meta = {
                    "file": fname,
                    "room_row": ry,
                    "room_col": rx,
                    "original_size": (PATCH_H, PATCH_W),
                    "trimmed_size": (TRIMMED_H, TRIMMED_W),
                    "stretched_size": squared.shape,
                    "output_size": (TARGET_SIZE, TARGET_SIZE),
                }

                # Original — fill in missing MOB/OBJECT
                arr_orig = _fill_missing_tiles(base_padded, rng)
                samples.append(GameSample(
                    game=GameTag.ZELDA,
                    source_id=source_id,
                    array=arr_orig,
                    char_grid=_array_to_char_grid(arr_orig),
                    legend=self._legend,
                    instruction=None,
                    order=len(samples),
                    meta={**base_meta, "augmented": False},
                ))

                # 90-degree rotation augmentation — fill in missing MOB/OBJECT
                rotated = _fill_missing_tiles(_rotate_90(base_padded), rng)
                samples.append(GameSample(
                    game=GameTag.ZELDA,
                    source_id=f"{source_id}_rot90",
                    array=rotated,
                    char_grid=_array_to_char_grid(rotated),
                    legend=self._legend,
                    instruction=None,
                    order=len(samples),
                    meta={**base_meta, "augmented": True, "augmentation": "rot90"},
                ))

                # Horizontal flip augmentation — fill in missing MOB/OBJECT
                flipped = _fill_missing_tiles(_flip_ud(base_padded), rng)
                samples.append(GameSample(
                    game=GameTag.ZELDA,
                    source_id=f"{source_id}_flipud",
                    array=flipped,
                    char_grid=_array_to_char_grid(flipped),
                    legend=self._legend,
                    instruction=None,
                    order=len(samples),
                    meta={**base_meta, "augmented": True, "augmentation": "flipud"},
                ))

        # ── Filtering: drop uniform-centre maps and their augmentations ──────────
        samples_before_filter = len(samples)

        # Pass 1: collect the uniform-centre originals to remove
        uniform_source_ids = set()
        for sample in samples:
            # Check the original for uniformity (rot90 and drop share its structure)
            if not any(sample.source_id.endswith(s) for s in ("_rot90", "_flipud", "_drop")):
                padded_array = sample.array
                if _is_uniform_center_12x12(padded_array):
                    base_id = sample.source_id
                    uniform_source_ids.add(f"{base_id}_rot90")
                    uniform_source_ids.add(f"{base_id}_flipud")
                    uniform_source_ids.add(f"{base_id}_drop")

        # Pass 2: remove the augmentations of the uniform-centre maps
        filtered_samples = [s for s in samples if s.source_id not in uniform_source_ids]

        # Step 3: configure order
        for i, sample in enumerate(filtered_samples):
            sample.order = i

        return filtered_samples

    # ── BaseGameHandler interface ────────────────────────────────────────────────

    def _ensure_loaded(self) -> None:
        if self._samples is None:
            self._samples = self._load_all()

    def list_entries(self) -> List[str]:
        self._ensure_loaded()
        return [s.source_id for s in self._samples]

    def load_sample(self, source_id: str, order: Optional[int] = None) -> GameSample:
        self._ensure_loaded()
        for s in self._samples:
            if s.source_id == source_id:
                if order is not None:
                    s.order = order
                return s
        raise KeyError(f"[zelda] source_id not found: {source_id}")

    def __len__(self) -> int:
        self._ensure_loaded()
        return len(self._samples)

    def __iter__(self) -> Iterator[GameSample]:
        self._ensure_loaded()
        yield from self._samples

    def __repr__(self) -> str:
        n = len(self._samples) if self._samples is not None else "?"
        return f"ZeldaHandler(root={str(self._root)!r}, rooms={n})"
