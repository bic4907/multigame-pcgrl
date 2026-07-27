"""
dataset/multigame/handlers/boxoban_handler.py
=============================================
Google DeepMind Boxoban dataset handler.

Source: https://github.com/google-deepmind/boxoban-levels
folder structure:
    boxoban_levels/
        hard/           000.txt … 003.txt
        medium/train/   000.txt … 449.txt
        medium/valid/   000.txt … 009.txt

Level file format
--------------
- Levels are separated by lines starting with `;`
- level size: 10×10 fixed
- Character meanings:
    '#' : wall
    ' ' : floor/empty
    '.' : target (goal square)
    '$' : box
    '@' : player
    '*' : box on target
    '+' : player on target

tile ID (remappable through tile_mapping.json)
---------------------------------------------------------
0  EMPTY  – floor/empty (' ', '.')
1  WALL   – wall ('#')
2  FLOOR  – floor (same as empty here; kept for compatibility)
3  ENEMY  – unused in Sokoban (kept for compatibility)
4  OBJECT – box ('$', '*')
5  SPAWN  – player ('@', '+')

16×16 normalize
------------
Place the 10x10 level at the top left and pad the remainder with WALL (1).

Diversity filtering
----------------
- Tile ratios (floor / wall / box / player) form the feature vector
- Greedy max-min sampling over that feature space selects a diverse subset
"""
from __future__ import annotations

import hashlib
import re
import warnings
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import numpy as np

from ..base import BaseGameHandler, GameSample, GameTag, TileLegend

# ── path default value ─────────────────────────────────────────────────────────────────
_DEFAULT_BOXOBAN_ROOT = Path(__file__).parent.parent.parent / "boxoban_levels"

# ── Tile ids ─────────────────────────────────────────────────────────────────────
class BoxobanTile:
    EMPTY  = 0   # floor / empty (' ' and '.')
    WALL   = 1   # wall  ('#')
    OBJECT = 4   # box   ('$', '*')
    SPAWN  = 5   # player('@', '+')


# character → integer tile ID
_CHAR_MAP: dict[str, int] = {
    " ": BoxobanTile.EMPTY,
    ".": BoxobanTile.EMPTY,   # target square → empty (floor in the structure layer)
    "#": BoxobanTile.WALL,
    "$": BoxobanTile.OBJECT,  # box
    "*": BoxobanTile.OBJECT,  # box on target
    "@": BoxobanTile.SPAWN,   # player
    "+": BoxobanTile.SPAWN,   # player on target
}

BOXOBAN_PALETTE: dict[int, Tuple[int, int, int]] = {
    BoxobanTile.EMPTY:  (200, 180, 120),
    BoxobanTile.WALL:   (80,  80,  80),
    BoxobanTile.OBJECT: (255, 215, 0),
    BoxobanTile.SPAWN:  (0,   200, 0),
}

# Boxoban level size
_LEVEL_SIZE = 10
_TARGET_SIZE = 16


def _make_legend() -> TileLegend:
    return TileLegend(char_to_attrs={
        " ": ["passable", "floor"],
        ".": ["passable", "floor", "target"],
        "#": ["solid", "wall"],
        "$": ["passable", "object", "box"],
        "*": ["passable", "object", "box", "target"],
        "@": ["passable", "spawn", "player"],
        "+": ["passable", "spawn", "player", "target"],
    })


# ── level parsing ────────────────────────────────────────────────────────────────────

def _parse_levels_from_file(path: Path) -> List[List[str]]:
    """
    Parse the level list out of a txt file.
    Returns: each level as a list of string rows.
    """
    levels: List[List[str]] = []
    current: List[str] = []

    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.rstrip()
        if line.startswith(";"):
            # Separator line → store the previous level
            if current:
                levels.append(current)
                current = []
        elif line == "" and current:
            # A blank line also ends a level
            levels.append(current)
            current = []
        else:
            current.append(line)

    if current:
        levels.append(current)

    return [lvl for lvl in levels if lvl]


def _lines_to_array(lines: List[str]) -> Optional[np.ndarray]:
    """
    string row list → (H, W) int32 ndarray.
    Rows shorter than the maximum are right-padded with ' ' (EMPTY).
    Levels that are not 10x10 return None.
    """
    if not lines:
        return None

    H = len(lines)
    W = max(len(l) for l in lines)

    if H != _LEVEL_SIZE or W != _LEVEL_SIZE:
        return None          # Not a standard level

    grid = np.zeros((H, W), dtype=np.int32)
    for r, line in enumerate(lines):
        padded = line.ljust(W)
        for c, ch in enumerate(padded):
            grid[r, c] = _CHAR_MAP.get(ch, BoxobanTile.WALL)

    return grid


def _strip_wall_border(arr: np.ndarray) -> np.ndarray:
    """
     Trim the outer rows/columns that are entirely WALL(1) and return the valid region.

    Example: in a 10x10 level whose first and last rows are all WALL and whose first
        and last columns are all WALL, an 8x8 array is returned.
    If there is no non-WALL cell, the array is returned unchanged.
    """
    non_wall = (arr != BoxobanTile.WALL)
    rows = np.where(non_wall.any(axis=1))[0]
    cols = np.where(non_wall.any(axis=0))[0]
    if rows.size == 0 or cols.size == 0:
        return arr  # Everything is WALL; return as-is
    r0, r1 = int(rows[0]), int(rows[-1]) + 1
    c0, c1 = int(cols[0]), int(cols[-1]) + 1
    return arr[r0:r1, c0:c1]


def _fit_to_target(arr: np.ndarray, target: int = _TARGET_SIZE) -> np.ndarray:
    """
    Resize a level to target x target.

    Steps
    -----------
    1. Structure (WALL / EMPTY) is resampled to target x target by nearest neighbour
    2. Objects (BOX / PLAYER) are re-placed by scaling their (r, c) position
       - their counts are preserved
    3. The resulting target x target array is returned
    """
    src_h, src_w = arr.shape

    # ── 1. Nearest-neighbour resampling of the structure layer ───────────────────
    structure = np.where(arr == BoxobanTile.WALL,
                         BoxobanTile.WALL, BoxobanTile.EMPTY).astype(np.int32)
    row_idx = (np.arange(target) * src_h / target).astype(np.int32)
    col_idx = (np.arange(target) * src_w / target).astype(np.int32)
    out = structure[np.ix_(row_idx, col_idx)]   # (target, target)

    # ── 2. Re-place the objects ──────────────────────────────────────────────
    object_tiles = {BoxobanTile.OBJECT, BoxobanTile.SPAWN}
    for r in range(src_h):
        for c in range(src_w):
            tid = int(arr[r, c])
            if tid not in object_tiles:
                continue
            nr = int(round((r + 0.5) / src_h * target - 0.5))
            nc = int(round((c + 0.5) / src_w * target - 0.5))
            nr = max(0, min(target - 1, nr))
            nc = max(0, min(target - 1, nc))
            out[nr, nc] = tid

    return out.astype(np.int32)

# Backward-compatible alias
_scale2x_to_16x16 = _fit_to_target


# ── Diversity filtering ─────────────────────────────────────────────────────────────

def _feature_vector(arr: np.ndarray) -> np.ndarray:
    """
    Derive the measure features from a level array (16x16 basis).

    Features:
      0  : wall ratio
      1  : empty/floor ratio
      2  : object (box) count
      3  : spawn (player) count
      4  : wall ratio in the top half
      5  : wall ratio in the bottom half
      6  : wall ratio in the left half
      7  : wall ratio in the right half
    """
    total  = arr.size
    region = arr   # the whole 16x16 grid

    wall_r  = (region == BoxobanTile.WALL).sum()  / total
    empty_r = (region == BoxobanTile.EMPTY).sum() / total
    n_box   = float((region == BoxobanTile.OBJECT).sum())
    n_player= float((region == BoxobanTile.SPAWN).sum())

    h, w    = region.shape
    mid_r, mid_c = h // 2, w // 2
    half_size    = mid_r * w

    top_wall  = (region[:mid_r, :]  == BoxobanTile.WALL).sum() / max(half_size, 1)
    bot_wall  = (region[mid_r:, :]  == BoxobanTile.WALL).sum() / max(half_size, 1)
    left_wall = (region[:, :mid_c]  == BoxobanTile.WALL).sum() / max(mid_c * h, 1)
    right_wall= (region[:, mid_c:]  == BoxobanTile.WALL).sum() / max(mid_c * h, 1)

    return np.array([
        wall_r, empty_r, n_box, n_player,
        top_wall, bot_wall, left_wall, right_wall,
    ], dtype=np.float32)



def _diversity_sample(
    arrays: List[np.ndarray],
    n: int,
    seed: int = 42,
) -> List[int]:
    """
    Greedy max-min distance sampling (farthest point sampling).
    Return the indices of n items spread as widely as possible in feature space.
    """
    if len(arrays) <= n:
        return list(range(len(arrays)))

    rng = np.random.default_rng(seed)
    features = np.stack([_feature_vector(a) for a in arrays])   # (N, D)

    # normalize
    std = features.std(axis=0) + 1e-8
    features = features / std

    chosen = [int(rng.integers(len(arrays)))]
    dists = np.full(len(arrays), np.inf)

    for _ in range(n - 1):
        last = features[chosen[-1]]
        d = np.linalg.norm(features - last, axis=1)
        dists = np.minimum(dists, d)
        chosen.append(int(np.argmax(dists)))

    return chosen


# ── Object augmentation ──────────────────────────────────────────────────────

def _is_corner(array: np.ndarray, r: int, c: int) -> bool:
    """
    Return True when (r, c) sits in a corner.
    A corner has two perpendicular neighbours that are WALL; out-of-bounds counts as WALL.
    Out-of-bounds counts as WALL.
    """
    H, W = array.shape

    def is_wall(rr, cc):
        if rr < 0 or rr >= H or cc < 0 or cc >= W:
            return True
        return array[rr, cc] == BoxobanTile.WALL

    top    = is_wall(r - 1, c)
    bottom = is_wall(r + 1, c)
    left   = is_wall(r, c - 1)
    right  = is_wall(r, c + 1)

    return (top and left) or (top and right) or (bottom and left) or (bottom and right)


def _placeable_positions(array: np.ndarray) -> np.ndarray:
    """
    Return the positions where an OBJECT may be placed.
    A position qualifies when it is an EMPTY tile that is not in a corner.
    """
    candidates = []
    for r, c in np.argwhere(array == BoxobanTile.EMPTY):
        if not _is_corner(array, r, c):
            candidates.append([r, c])
    return np.array(candidates, dtype=np.int32) if candidates else np.empty((0, 2), dtype=np.int32)


def _augment_objects(array: np.ndarray) -> np.ndarray:
    """
    Vary the number of boxes in a Sokoban map within the 1-12 range.

    A target count is drawn from 1-12 and boxes are added or removed to match it.
    The seed is the MD5 of the array contents, so the same map always yields the same result.
    """
    digest = hashlib.md5(array.tobytes()).digest()
    seed   = int.from_bytes(digest[:4], byteorder='big')
    rng    = np.random.default_rng(seed)

    # Use the full digest rather than the 4-byte seed to avoid correlating with it
    full_hash = int.from_bytes(digest, byteorder='big')
    target = (full_hash % 12) + 1  # 1~12 uniform
    result = array.copy()

    obj_positions = list(np.argwhere(result == BoxobanTile.OBJECT))
    current = len(obj_positions)

    if target < current:
        n_remove = current - target
        indices = rng.choice(len(obj_positions), size=n_remove, replace=False)
        for idx in indices:
            r, c = obj_positions[idx]
            result[r, c] = BoxobanTile.EMPTY

    elif target > current:
        n_add = target - current
        placeable = _placeable_positions(result)
        if len(placeable) > 0:
            n_add = min(n_add, len(placeable))
            chosen = rng.choice(len(placeable), size=n_add, replace=False)
            for idx in chosen:
                r, c = placeable[idx]
                result[r, c] = BoxobanTile.OBJECT

    return result


# ── handler class ────────────────────────────────────────────────────────────────

class BoxobanHandler(BaseGameHandler):
    """
    Google DeepMind Boxoban handler.

    Parameters
    ----------
    root        : boxoban_levels folder path
    difficulty  : "hard" | "medium" | "both"
    split       : medium  before  for  - "train" | "valid" | "all"
    n_sample    : how many levels diversity sampling should keep (None = all)
    seed        : diversity sampling seed

    Example
    -------
        handler = BoxobanHandler(n_sample=1000)
        for sample in handler:
            print(sample.array.shape, sample.source_id)

        samples = handler.sample(500)
    """

    game_tag = GameTag.SOKOBAN

    def __init__(
        self,
        root: Path | str = _DEFAULT_BOXOBAN_ROOT,
        difficulty: str = "both",
        split: str = "train",
        n_sample: Optional[int] = 1000,
        seed: int = 42,
    ) -> None:
        self._root = Path(root)
        self._difficulty = difficulty
        self._split = split
        self._n_sample = n_sample
        self._seed = seed
        self._samples: Optional[List[GameSample]] = None  # lazy

    @property
    def game_tag(self) -> str:
        return GameTag.SOKOBAN

    # ── File discovery ───────────────────────────────────────────────────────────

    def _collect_files(self) -> List[Path]:
        files: List[Path] = []
        d = self._difficulty

        if d in ("hard", "both"):
            hard_dir = self._root / "hard"
            if hard_dir.exists():
                files += sorted(hard_dir.glob("*.txt"))
            else:
                warnings.warn(f"[boxoban] hard folder none: {hard_dir}")

        if d in ("medium", "both"):
            splits = ["train", "valid"] if self._split == "all" else [self._split]
            for sp in splits:
                med_dir = self._root / "medium" / sp
                if med_dir.exists():
                    files += sorted(med_dir.glob("*.txt"))
                else:
                    warnings.warn(f"[boxoban] medium/{sp} folder none: {med_dir}")

        return files

    # ── all level load ────────────────────────────────────────────────────────────

    def _load_all(self) -> List[GameSample]:
        files = self._collect_files()
        if not files:
            raise FileNotFoundError(
                f"[boxoban] No level files found under: {self._root}"
            )

        legend = _make_legend()
        all_arrays: List[np.ndarray] = []
        all_ids:    List[str]        = []

        for fpath in files:
            rel = fpath.relative_to(self._root)
            level_lines = _parse_levels_from_file(fpath)
            for lvl_idx, lines in enumerate(level_lines):
                arr = _lines_to_array(lines)
                if arr is None:
                    continue
                processed = _fit_to_target(arr, _TARGET_SIZE)
                source_id = f"{rel}#{lvl_idx}"
                all_arrays.append(processed)
                all_ids.append(source_id)

        if not all_arrays:
            raise ValueError("[boxoban] No parsable levels found.")

        # ── Diversity sampling (on the pre-augmentation structure) ──────────────
        n = self._n_sample
        if n is not None and n < len(all_arrays):
            chosen_idxs = _diversity_sample(all_arrays, n, seed=self._seed)
        else:
            chosen_idxs = list(range(len(all_arrays)))

        # ── Augmentation (diversity sampling   after  apply) ───────────────────────
        samples: List[GameSample] = []
        for order, idx in enumerate(chosen_idxs):
            augmented = _augment_objects(all_arrays[idx])
            samples.append(GameSample(
                game=GameTag.SOKOBAN,
                source_id=all_ids[idx],
                array=augmented,
                legend=legend,
                order=order,
                meta={
                    "difficulty":    self._difficulty,
                    "original_size": (_LEVEL_SIZE, _LEVEL_SIZE),
                    "output_size":   (_TARGET_SIZE, _TARGET_SIZE),
                    "scale_method":  "fit_to_target_center",
                    "scale":         max(1, _TARGET_SIZE // _LEVEL_SIZE),
                },
            ))

        return samples

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
                return s
        raise KeyError(f"[boxoban] source_id not found: {source_id}")

    def __len__(self) -> int:
        self._ensure_loaded()
        return len(self._samples)

    def __iter__(self) -> Iterator[GameSample]:
        self._ensure_loaded()
        yield from self._samples

    def sample(self, n: int, seed: int = 0) -> List[GameSample]:
        """Draw n random samples from everything that was loaded."""
        self._ensure_loaded()
        rng = np.random.default_rng(seed)
        idxs = rng.choice(len(self._samples), size=min(n, len(self._samples)), replace=False)
        return [self._samples[i] for i in idxs]

    def filter_by_difficulty(self, difficulty: str) -> List[GameSample]:
        self._ensure_loaded()
        return [s for s in self._samples if s.meta.get("difficulty") == difficulty]

    def stats(self) -> dict:
        """Summary statistics over the loaded samples."""
        self._ensure_loaded()
        arrays = [s.array for s in self._samples]
        wall_ratios = [(a == BoxobanTile.WALL).mean() for a in arrays]
        return {
            "n_samples":       len(arrays),
            "difficulty":      self._difficulty,
            "scale_method":    "2x_structure_object_reposition",
            "wall_ratio_mean": float(np.mean(wall_ratios)),
            "wall_ratio_std":  float(np.std(wall_ratios)),
            "wall_ratio_min":  float(np.min(wall_ratios)),
            "wall_ratio_max":  float(np.max(wall_ratios)),
        }
