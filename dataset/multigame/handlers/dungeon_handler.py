"""
dataset/multigame/handlers/dungeon_handler.py
=============================================
dungeon_level_dataset handler.

- dungeon_levels.npz + dungeon_levels_metadata.csv load
- Supports instruction, instruction_slug, level_id, and sample_id tags
- Reimplemented independently without copying DungeonLevelDataset code
  (uses only NumPy and has no external package dependency)

Tile mapping (based on the dungeon_level_dataset README)
---------------------------------------------
0  : padding / unknown
1  : floor  (original value: 1)
2  : wall   (original value: 2)
3  : enemy  (original value: 3)

preprocessing filter (cache save  before  apply, legacy annotation basis)
---------------------------------------------
1. Remove RG (region, reward_enum == 1) samples whose value is 25 or 35.
2. Remove half of the BD (bat_direction, reward_enum == 5) samples per instruction.
   (Sort keys in ascending order and keep the first half for reproducibility.)
3. Truncate the complete set to 4,000 samples.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterator, List, Optional

import numpy as np

from ..base import (
    BaseGameHandler,
    GameSample,
    GameTag,
    TileLegend,
    enforce_top_left_16x16,
)

_DEFAULT_DUNGEON_ROOT = (
    Path(__file__).parent.parent.parent / "dungeon_level_dataset"
)

# dataset/reward_annotations/legacy/dungeon_reward_annotations.csv
_LEGACY_ANNOT_PATH = (
    Path(__file__).parent.parent.parent
    / "reward_annotations" / "legacy" / "dungeon_reward_annotations.csv"
)

# ── Preprocessing constants ─────────────────────────────────────────────────────
_EXCLUDE_RG: frozenset[int] = frozenset({25, 35})
_TARGET_COUNT: int = 4_000


# ── Tile constants ──────────────────────────────────────────────────────────────
class DungeonTile:
    UNKNOWN  = 0
    FLOOR    = 1
    WALL     = 2
    ENEMY    = 3
    TREASURE = 4


DUNGEON_PALETTE: dict[int, tuple[int, int, int]] = {
    DungeonTile.UNKNOWN: (0,   0,   0),
    DungeonTile.FLOOR:   (200, 180, 120),
    DungeonTile.WALL:    (80,  80,  80),
    DungeonTile.ENEMY:   (220, 50,  50),
    DungeonTile.TREASURE: (200, 200, 0),
}


def _place_treasure(array: np.ndarray, key: str) -> np.ndarray:
    """
    Independently replace each FLOOR tile with TREASURE (4) with 10% probability.
    Convert the key (formatted as "000000") to an integer seed for reproducibility.
    Return the array unchanged when no FLOOR tiles exist.
    """
    rng = np.random.RandomState(int(key))
    floor_pos = np.argwhere(array == DungeonTile.FLOOR)
    if len(floor_pos) == 0:
        return array
    result = array.copy()
    for idx in floor_pos:
        if rng.random() < 0.1:
            result[idx[0], idx[1]] = DungeonTile.TREASURE
    return result


def _make_legend() -> TileLegend:
    return TileLegend(char_to_attrs={
        "1": ["passable", "floor"],
        "2": ["solid", "wall"],
        "3": ["enemy", "damaging"],
    })


def _apply_preprocess_filter(all_keys: List[str]) -> List[str]:
    """
    Read the legacy annotation CSV and return the keys retained after preprocessing.

    filter apply order
    --------------
    1. Remove reward_enum == 1 (RG) samples whose condition_1 is in _EXCLUDE_RG.
    2. Keep half of the reward_enum == 5 (BD) samples for each instruction.
       (Sort keys in ascending order and keep the first half for reproducibility.)
    3. Truncate to _TARGET_COUNT samples while preserving the original order.

    Return all_keys unchanged when the annotation CSV is absent.
    """
    if not _LEGACY_ANNOT_PATH.exists():
        return all_keys

    annot: dict[str, dict] = {}
    with open(_LEGACY_ANNOT_PATH, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            annot[row["key"]] = row

    keep_set: set[str] = set()
    bd_by_instruction: dict[str, List[str]] = {}

    for key in all_keys:
        row = annot.get(key)
        if row is None:
            # annotation  in  without sample  as-is keep
            keep_set.add(key)
            continue

        reward_enum = row.get("reward_enum", "")
        cond1_raw   = row.get("condition_1", "")

        if reward_enum == "1":
            # Skip excluded RG values
            try:
                rg = int(float(cond1_raw))
            except (ValueError, TypeError):
                rg = -1
            if rg in _EXCLUDE_RG:
                continue

        if reward_enum == "5":
            # Group BD samples by instruction for later processing
            instr = row.get("instruction", "")
            bd_by_instruction.setdefault(instr, []).append(key)
        else:
            keep_set.add(key)

    # BD: sort each instruction group and retain only the first half
    for instr in sorted(bd_by_instruction):
        group = sorted(bd_by_instruction[instr])   # Ascending keys ensure reproducibility
        half = max(1, len(group) // 2)
        keep_set.update(group[:half])

    # Preserve the original order, then truncate
    filtered = [k for k in all_keys if k in keep_set]
    return filtered[:_TARGET_COUNT]


# ── Lightweight metadata dataclass ──────────────────────────────────────────────
class _DungeonMeta:
    __slots__ = ("index", "key", "instruction", "instruction_slug",
                 "level_id", "sample_id")

    def __init__(self, index, key, instruction, instruction_slug,
                 level_id, sample_id):
        self.index = int(index)
        self.key = key
        self.instruction = instruction
        self.instruction_slug = instruction_slug
        self.level_id = int(level_id)
        self.sample_id = int(sample_id)


class DungeonHandler(BaseGameHandler):
    """
    dungeon_level_dataset handler.

    Parameters
    ----------
    root      : dungeon_level_dataset folder path
    npz_name  : NPZ filename (default: 'dungeon_levels.npz')
    meta_name : CSV filename (default: 'dungeon_levels_metadata.csv')

    Example
    -------
        handler = DungeonHandler()
        for sample in handler:
            print(sample.instruction, sample.shape)

        # instruction as  filter
        subset = handler.filter_by_instruction("bat swarm")
    """

    def __init__(
        self,
        root: Path | str = _DEFAULT_DUNGEON_ROOT,
        npz_name: str = "dungeon_levels.npz",
        meta_name: str = "dungeon_levels_metadata.csv",
    ) -> None:
        self._root = Path(root)
        npz_path  = self._root / npz_name
        meta_path = self._root / meta_name

        if not npz_path.exists():
            raise FileNotFoundError(f"NPZ not found: {npz_path}")
        if not meta_path.exists():
            raise FileNotFoundError(f"Metadata CSV not found: {meta_path}")

        self._archive = np.load(npz_path)
        self._legend  = _make_legend()
        self._metas: List[_DungeonMeta] = []
        self._key_to_meta: Dict[str, _DungeonMeta] = {}

        raw_metas: dict[str, _DungeonMeta] = {}
        with open(meta_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                m = _DungeonMeta(
                    index=row["index"],
                    key=row["key"],
                    instruction=row["instruction"],
                    instruction_slug=row["instruction_slug"],
                    level_id=row["level_id"],
                    sample_id=row["sample_id"],
                )
                raw_metas[m.key] = m

        # preprocessing filter apply (cache save  before )
        all_keys = [m.key for m in sorted(raw_metas.values(), key=lambda m: m.index)]

        # First pass: remove ndim != 2 maps (malformed maps that trigger RuntimeWarning)
        all_keys = [k for k in all_keys if self._archive[k].ndim == 2]

        # Second pass: apply legacy annotation filters (RG, BD, and the 4,000-sample limit)
        kept_keys = _apply_preprocess_filter(all_keys)

        for key in kept_keys:
            m = raw_metas[key]
            self._metas.append(m)
            self._key_to_meta[key] = m

    @property
    def game_tag(self) -> str:
        return GameTag.DUNGEON

    # ── BaseGameHandler ─────────────────────────────────────────────────────────
    def list_entries(self) -> List[str]:
        """npz key list return."""
        return [m.key for m in self._metas]

    def load_sample(self, source_id: str, order: Optional[int] = None) -> GameSample:
        """npz key → GameSample return."""
        m = self._key_to_meta.get(source_id)
        if m is None:
            raise KeyError(f"Key not found in dungeon dataset: {source_id!r}")
        raw = self._archive[source_id]           # (16,16) int64
        array = raw.astype(np.int32)
        array = enforce_top_left_16x16(
            array,
            game=GameTag.DUNGEON,
            source_id=source_id,
        )
        array = _place_treasure(array, source_id)
        return GameSample(
            game=GameTag.DUNGEON,
            source_id=source_id,
            array=array,
            char_grid=None,
            legend=self._legend,
            instruction=m.instruction,
            order=order if order is not None else m.index,
            meta={
                "instruction_slug": m.instruction_slug,
                "level_id":         m.level_id,
                "sample_id":        m.sample_id,
            },
        )

    # ── Extended query methods ──────────────────────────────────────────────────
    def filter_by_instruction(
        self, keyword: str, *, case_sensitive: bool = False
    ) -> List[GameSample]:
        """Return samples whose instruction contains keyword."""
        kw = keyword if case_sensitive else keyword.lower()
        result = []
        for i, m in enumerate(self._metas):
            text = m.instruction if case_sensitive else m.instruction.lower()
            if kw in text:
                result.append(self.load_sample(m.key, order=i))
        return result

    def group_by_instruction(self) -> Dict[str, List[GameSample]]:
        """Return a mapping from instruction_slug to sample lists."""
        groups: Dict[str, List[GameSample]] = {}
        for i, m in enumerate(self._metas):
            sample = self.load_sample(m.key, order=i)
            groups.setdefault(m.instruction_slug, []).append(sample)
        return groups


    def category_names(self) -> List[str]:
        """Return the unique instruction strings."""
        seen = {}
        for m in self._metas:
            seen[m.instruction_slug] = m.instruction
        return list(seen.values())

    def __repr__(self) -> str:
        return (
            f"DungeonHandler(root={str(self._root)!r}, "
            f"samples={len(self._metas)})"
        )
