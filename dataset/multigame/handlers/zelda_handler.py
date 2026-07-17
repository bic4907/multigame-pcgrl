"""
dataset/multigame/handlers/zelda_handler.py
===========================================
The Legend of Zelda (TheVGLC) handler.

TheVGLC of  text data  all text before  map  text of  text file to  text.
each character  tile text   of text, text(room)  11×16(W×H) size of  text to  splittext.
 before text void('-') to   text text  text text text to  removetext.

text structure (NES The Legend of Zelda)
----------------------------------
  - text text =   to  11character × text to  16text
  - text: text wall(WW…) 2text + text wall(WW) 2charactertext + internal 7×12
  - adjacent text  wall  text text (WWWW = text1 textwall + text2 textwall)
  - separatetext text  11character text('-----------') to  text

preprocessing:
  1. 11×16 text in  border wall 1text/1text remove → 9×14 (internal + wall 1text)
  2. text  text(  to  9)  nearest-neighbor stretch → 14×14 texteachtext
  3. 14×14  16×16  text sort (WALL padding)
  4. 90 also  rotate augmentation as  data 2text
  5. text map(50%) in  FLOOR/EMPTY abovetext in  MOB·OBJECT  1~5text random text (seed=42 fixed)

tile text (vglc_games/zelda.py basis)
-------------------------------------
0  : EMPTY   (-, text)
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

# ── text size (NES text text size) ─────────────────────────────────────────────────
PATCH_W = 11   #   to  (character text)
PATCH_H = 16   # text to  (text text)
TARGET_SIZE = 16  # text text size (16×16)

# border wall remove  after  size
TRIMMED_W = PATCH_W - 2   # 11 - 2 = 9
TRIMMED_H = PATCH_H - 2   # 16 - 2 = 14


def _read_map_text(path: Path) -> List[str]:
    """text file  text text list  returntext. text of  text text  remove."""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    while lines and not lines[-1].strip():
        lines.pop()
    return lines


def _text_to_int_grid(
    lines: List[str],
    preprocessor: ZeldaPreprocessor,
) -> np.ndarray:
    """
    all text map  integer 2D array to  converttext.
    text text  same text to  text(text text  EMPTY padding), character→integer convert.
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
    all map text  patch_h × patch_w text to  splittext,
     before text EMPTY(0)text text  text.

    Returns
    -------
    list of (patch_array, row_idx, col_idx)
        patch_array : (patch_h, patch_w) int32
        row_idx     : text row index (0-based)
        col_idx     : text column index (0-based)
    """
    H, W = grid.shape
    rooms = []

    for iy, y in enumerate(range(0, H - patch_h + 1, patch_h)):
        for ix, x in enumerate(range(0, W - patch_w + 1, patch_w)):
            patch = grid[y : y + patch_h, x : x + patch_w]
            #  before text empty (void) text text  text
            if np.all(patch == ZeldaTile.EMPTY):
                continue
            rooms.append((patch.copy(), iy, ix))

    return rooms


def _trim_outer_wall(patch: np.ndarray) -> np.ndarray:
    """
    11×16 text in  border wall 1text/1text remove.

    text (16H × 11W):
        row 0  : WWWWWWWWWWW  ← remove (wall row)
        row 1  : WWWWDDDWWWW  ← keep (wall+text)
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
    texteachtext text of  text  text  nearest-neighbor to  text texteachtext as  text.

    text: 14H × 9W → 14 × 14  (  to  9→14 stretch)
        9H × 14W → 14 × 14  (text to  9→14 stretch)

    integer tile ID  keeptext abovetext nearest-neighbor index text  text for text.
    """
    h, w = patch.shape
    if h == w:
        return patch

    target = max(h, w)

    if w < h:
        #   to   text →   to   h size to  text
        col_indices = np.round(np.linspace(0, w - 1, target)).astype(int)
        return patch[:, col_indices].copy()
    else:
        # text to   text → text to   w size to  text
        row_indices = np.round(np.linspace(0, h - 1, target)).astype(int)
        return patch[row_indices, :].copy()


def _center_pad_to_16x16(patch: np.ndarray) -> np.ndarray:
    """
    text of  size text  16×16 center sort to  paddingtext.
    text text  WALL(1) to  text (border text to ).
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
    """text 90 also  rotate."""
    return np.rot90(patch, k=-1).copy()


def _flip_ud(patch: np.ndarray) -> np.ndarray:
    """text flip."""
    return np.flipud(patch).copy()


def _is_uniform_center_12x12(padded: np.ndarray) -> bool:
    """
    16x16 map of  center 12x12 text(text 2text text)  text same tiletext check.

    Parameters
    ----------
    padded : np.ndarray
        16x16 map

    Returns
    -------
    bool
        center 12x12  text same text text True
    """
    center = padded[2:14, 2:14]  # center 12x12 extract
    return bool(np.all(center == center[0, 0]))


# text availabletext tile text
_DROP_TILES = [ZeldaTile.MOB, ZeldaTile.OBJECT]

# text target  text  text tile (FLOOR, EMPTY text available)
_DROPPABLE_TILES = {ZeldaTile.FLOOR, ZeldaTile.EMPTY}

# text augmentation ratio (text  during  text % in  applytext)
DROP_AUG_RATIO = 1.0  # text of  100% in  text apply

# text count range
DROP_COUNT_MIN = 0
DROP_COUNT_MAX = 25


def _augment_random_drop(
    padded: np.ndarray,
    rng: np.random.Generator,
) -> Optional[np.ndarray]:
    """
    16×16 text of  FLOOR text  EMPTY tile  during  0~25text  random as
    MOB text  OBJECT to  text.

    text availabletext abovetext  if missing None  returntext.
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
    MOB(hazard) text  OBJECT(collectable)  without map in  text tile  in-place to  text text.

    - MOB none  → FLOOR/EMPTY  during  1~DROP_COUNT_MAXtext  MOB as  text
    - OBJECT none → text  FLOOR/EMPTY  during  1~DROP_COUNT_MAXtext  OBJECT to  text
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


# ── integer → texttable character text ─────────────────────────────────────────────────────
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
    """integer array → character text."""
    return [[_int_to_char(int(val)) for val in row] for row in arr]


class ZeldaHandler(BaseGameHandler):
    """
    The Legend of Zelda (TheVGLC) handler.

    preprocessing  and text:
      1. Processed/ folder of  text map  11×16 text to  split
      2. text text( before text EMPTY) remove
      3. border wall 1text/1text remove → 9×14
      4. text  text(  to ) nearest-neighbor stretch → 14×14
      5. 16×16  text sort (WALL padding)
      6. 90 also  rotate augmentation → data 2text

    Parameters
    ----------
    root  : TheVGLC/The Legend of Zelda folder path
    split : sub foldertext (default "Processed")
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

    # ── file text ────────────────────────────────────────────────────────────────

    def _discover_files(self) -> List[Path]:
        """Processed/ folder in  text file list  returntext."""
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
                f"[zelda] level file  text  text text: {self._root / self._split}"
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

                # 2) text  text stretch → 14 × 14 texteachtext
                squared = _stretch_to_square(trimmed)

                # 3) 16×16  text sort (text 1text WALL padding)
                padded = _center_pad_to_16x16(squared)

                # 4) OBJECT  without map in  probabilitytext as  OBJECT batch (deterministic)
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

                # text — independently MOB/OBJECT text
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

                # 90 also  rotate augmentation — independently MOB/OBJECT text
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

                # text flip augmentation — independently MOB/OBJECT text
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

        # ── filtering: uniform center map of  augmentation text before text remove ─────────────────────────
        samples_before_filter = len(samples)

        # 1text: uniform center map of  augmentation text before text remove (text  keep)
        uniform_source_ids = set()
        for sample in samples:
            # text  uniformtext check (rot90, drop  text text text)
            if not any(sample.source_id.endswith(s) for s in ("_rot90", "_flipud", "_drop")):
                padded_array = sample.array
                if _is_uniform_center_12x12(padded_array):
                    base_id = sample.source_id
                    uniform_source_ids.add(f"{base_id}_rot90")
                    uniform_source_ids.add(f"{base_id}_flipud")
                    uniform_source_ids.add(f"{base_id}_drop")

        # 2text: uniform center map of  augmentation text before  remove
        filtered_samples = [s for s in samples if s.source_id not in uniform_source_ids]

        # 3text: order textconfig
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

