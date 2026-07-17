"""
dataset/multigame/handlers/boxoban_handler.py
=============================================
Google DeepMind Boxoban dataset handler.

text: https://github.com/google-deepmind/boxoban-levels
folder structure:
    boxoban_levels/
        hard/           000.txt … 003.txt
        medium/train/   000.txt … 449.txt
        medium/valid/   000.txt … 009.txt

level file text
--------------
- level  `;` text text to  text
- level size: 10×10 fixed
- character  of text:
    '#' : wall
    ' ' : floor/empty
    '.' : target (goal square)
    '$' : box
    '@' : player
    '*' : box on target
    '+' : player on target

tile ID (tile_mapping.json  of  text text to  text available)
---------------------------------------------------------
0  EMPTY  – floor/empty (' ', '.')
1  WALL   – wall ('#')
2  FLOOR  – floor (same empty textcolumn, text for )
3  ENEMY  – (none, text for  text text)
4  OBJECT – box ('$', '*')
5  SPAWN  – player ('@', '+')

16×16 normalize
------------
10×10 level  center(top-left) in  batchtext remaining  WALL(1) to  padding.

Diversity filtering
----------------
- tile text(floor/wall/box/player ratio)  feature vector  to  text
- all level  feature space in  greedy max-min sampling  as  text text
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

# ── tile text ────────────────────────────────────────────────────────────────────
class BoxobanTile:
    EMPTY  = 0   # floor / empty (  '.' text)
    WALL   = 1   # wall  ('#')
    OBJECT = 4   # box   ('$', '*')
    SPAWN  = 5   # player('@', '+')


# character → integer tile ID
_CHAR_MAP: dict[str, int] = {
    " ": BoxobanTile.EMPTY,
    ".": BoxobanTile.EMPTY,   # target square → empty (structure text floor)
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

# Boxoban text size
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
    txt file text in  level list  parsingtext.
    returntext: each level = text string row of  text
    """
    levels: List[List[str]] = []
    current: List[str] = []

    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.rstrip()
        if line.startswith(";"):
            # text → previous level save
            if current:
                levels.append(current)
                current = []
        elif line == "" and current:
            # text text  level text in  text  text
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
    row text   text ' '(EMPTY) to  right padding.
    10×10   text level  None return.
    """
    if not lines:
        return None

    H = len(lines)
    W = max(len(l) for l in lines)

    if H != _LEVEL_SIZE or W != _LEVEL_SIZE:
        return None          # text level text

    grid = np.zeros((H, W), dtype=np.int32)
    for r, line in enumerate(lines):
        padded = line.ljust(W)
        for c, ch in enumerate(padded):
            grid[r, c] = _CHAR_MAP.get(ch, BoxobanTile.WALL)

    return grid


def _strip_wall_border(arr: np.ndarray) -> np.ndarray:
    """
     text in  WALL(1)  to text  text row/column  removetext valid text  returntext.

    text) 10×10 level in  text/text row   before text WALL text
        text/text column   before text WALL text → 8×8 return.
    non-WALL cell  text also  if missing text as-is return.
    """
    non_wall = (arr != BoxobanTile.WALL)
    rows = np.where(non_wall.any(axis=1))[0]
    cols = np.where(non_wall.any(axis=0))[0]
    if rows.size == 0 or cols.size == 0:
        return arr  #  before text WALLtext text text return
    r0, r1 = int(rows[0]), int(rows[-1]) + 1
    c0, c1 = int(cols[0]), int(cols[-1]) + 1
    return arr[r0:r1, c0:c1]


def _fit_to_target(arr: np.ndarray, target: int = _TARGET_SIZE) -> np.ndarray:
    """
    text level  target×target size to  converttext.

    preprocessing text
    -----------
    1. structure(WALL / EMPTY) text text  target×target  as  nearest-neighbor text → text text
    2. text(BOX / PLAYER)   text (r, c) abovetext ratio  target coordinate to  text textbatch
       - count  text sametext keep (text text)
    3. result  target×target  to  text return
    """
    src_h, src_w = arr.shape

    # ── 1. structure text text nearest-neighbor text text ──────────────────────────────
    structure = np.where(arr == BoxobanTile.WALL,
                         BoxobanTile.WALL, BoxobanTile.EMPTY).astype(np.int32)
    row_idx = (np.arange(target) * src_h / target).astype(np.int32)
    col_idx = (np.arange(target) * src_w / target).astype(np.int32)
    out = structure[np.ix_(row_idx, col_idx)]   # (target, target)

    # ── 2. text textbatch ────────────────────────────────────────────────────
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

# sub text alias
_scale2x_to_16x16 = _fit_to_target


# ── Diversity filtering ─────────────────────────────────────────────────────────────

def _feature_vector(arr: np.ndarray) -> np.ndarray:
    """
    level array → text measure for  text text. (16×16 basis)

    text:
      0  : wall ratio
      1  : empty/floor ratio
      2  : object(box) text
      3  : spawn(player) text
      4  : textabove text wall ratio
      5  : sub text wall ratio
      6  : text text wall ratio
      7  : text text wall ratio
    """
    total  = arr.size
    region = arr   # all 16×16 text for

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
    text text in  text to   text text n text of  index  return.
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


# ── text augmentation ────────────────────────────────────────────────────────────────

def _is_corner(array: np.ndarray, r: int, c: int) -> bool:
    """
    (r, c)  text text.
    text = EMPTY tile text text  during  text as  adjacenttext text text  text WALLtext text.
    text outside  WALL to  text.
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
    OBJECT  text  text with abovetext  returntext.
    condition: EMPTY tile text text  text abovetext.
    """
    candidates = []
    for r, c in np.argwhere(array == BoxobanTile.EMPTY):
        if not _is_corner(array, r, c):
            candidates.append([r, c])
    return np.array(candidates, dtype=np.int32) if candidates else np.empty((0, 2), dtype=np.int32)


def _augment_objects(array: np.ndarray) -> np.ndarray:
    """
    Sokoban map of  text(box) text  1~12text range to  text.

    texttable count  1~12 in  text sampletext text, current text  text /removetext texttable in  text.
    seed  array content of  MD5 text → same text text always same result.
    """
    digest = hashlib.md5(array.tobytes()).digest()
    seed   = int.from_bytes(digest[:4], byteorder='big')
    rng    = np.random.default_rng(seed)

    # 128text all to  text to  text → 4text text text text of  text remove
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
    n_sample    : diversity sampling  as  extracttext text (None = all)
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

    # ── file list text ────────────────────────────────────────────────────────────

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
                f"[boxoban] level file  text  text text: {self._root}"
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
            raise ValueError("[boxoban] parsing availabletext level  text.")

        # ── Diversity sampling (augmentation  before  text structure basis) ───────────────
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
        """all loadtext sample  during  n text  random extract."""
        self._ensure_loaded()
        rng = np.random.default_rng(seed)
        idxs = rng.choice(len(self._samples), size=min(n, len(self._samples)), replace=False)
        return [self._samples[i] for i in idxs]

    def filter_by_difficulty(self, difficulty: str) -> List[GameSample]:
        self._ensure_loaded()
        return [s for s in self._samples if s.meta.get("difficulty") == difficulty]

    def stats(self) -> dict:
        """loadtext sample in  text text text."""
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

