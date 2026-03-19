"""
dungeon_level_dataset · dataset.py
===================================
Convenience classes for loading and querying the Dungeon Level Dataset.

Classes
-------
LevelArray   – a single 16×16 dungeon level with its metadata
Instruction  – all levels that share the same instruction text
DungeonLevelDataset – the full dataset; iterable, subscriptable, filterable

Typical usage
-------------
    from dataset import DungeonLevelDataset

    ds = DungeonLevelDataset()          # loads .npz + metadata CSV

    # index access
    level = ds[0]                       # LevelArray
    print(level.instruction)
    print(level.array)                  # numpy (16, 16) int64

    # iterate
    for level in ds:
        process(level.array)

    # filter by keyword
    bat_levels = ds.filter("bat swarm") # list[LevelArray]

    # group by instruction
    instr = ds.group("a radial bats pattern forms across the top")
    print(instr.instruction)            # str
    print(len(instr))                   # number of samples
    for level in instr:                 # iterate samples
        ...

    # all instruction categories
    for instr in ds.instructions():
        print(instr.instruction, len(instr))
"""

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, List, Optional

import numpy as np

# ── Default paths (same directory as this file) ────────────────────────────────
_HERE = Path(__file__).parent
_DEFAULT_NPZ  = _HERE / "dungeon_levels.npz"
_DEFAULT_META = _HERE / "dungeon_levels_metadata.csv"

# ── Tile constants ─────────────────────────────────────────────────────────────
TILE_FLOOR = 1
TILE_WALL  = 2
TILE_ENEMY = 3


# ── Data classes ───────────────────────────────────────────────────────────────

@dataclass
class LevelArray:
    """A single dungeon level with its associated metadata."""

    index:            int            # global 0-based index
    key:              str            # zero-padded npz key, e.g. "000042"
    instruction:      str            # human-readable instruction
    instruction_slug: str            # snake_case category name
    level_id:         int            # level identifier from the generator
    sample_id:        int            # sample index within the same level
    array:            np.ndarray     # (16, 16) int64 tile grid

    # ── Tile helpers ───────────────────────────────────────────────────────────
    @property
    def floor_mask(self) -> np.ndarray:
        """Boolean mask of floor tiles."""
        return self.array == TILE_FLOOR

    @property
    def wall_mask(self) -> np.ndarray:
        """Boolean mask of wall tiles."""
        return self.array == TILE_WALL

    @property
    def enemy_mask(self) -> np.ndarray:
        """Boolean mask of enemy tiles."""
        return self.array == TILE_ENEMY

    @property
    def n_floor(self) -> int:
        return int(self.floor_mask.sum())

    @property
    def n_wall(self) -> int:
        return int(self.wall_mask.sum())

    @property
    def n_enemy(self) -> int:
        return int(self.enemy_mask.sum())

    def __repr__(self) -> str:
        return (
            f"LevelArray(index={self.index}, "
            f"instruction='{self.instruction}', "
            f"array_shape={self.array.shape})"
        )


@dataclass
class Instruction:
    """All levels that share the same instruction text."""

    instruction:      str
    instruction_slug: str
    _levels:          List[LevelArray] = field(default_factory=list, repr=False)

    def __len__(self) -> int:
        return len(self._levels)

    def __iter__(self) -> Iterator[LevelArray]:
        return iter(self._levels)

    def __getitem__(self, idx: int) -> LevelArray:
        return self._levels[idx]

    def arrays(self) -> np.ndarray:
        """Stack all sample arrays → shape (N, 16, 16)."""
        return np.stack([lv.array for lv in self._levels])

    def __repr__(self) -> str:
        return (
            f"Instruction(slug='{self.instruction_slug}', "
            f"samples={len(self)})"
        )


# ── Main dataset class ─────────────────────────────────────────────────────────

class DungeonLevelDataset:
    """
    Full Dungeon Level Dataset.

    Parameters
    ----------
    npz_path  : path to dungeon_levels.npz
    meta_path : path to dungeon_levels_metadata.csv
    """

    def __init__(
        self,
        npz_path:  Path | str = _DEFAULT_NPZ,
        meta_path: Path | str = _DEFAULT_META,
    ) -> None:
        npz_path  = Path(npz_path)
        meta_path = Path(meta_path)

        self._archive = np.load(npz_path)

        self._levels: List[LevelArray] = []
        self._slug_map: dict[str, Instruction] = {}

        with open(meta_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                key = row["key"]
                lv  = LevelArray(
                    index            = int(row["index"]),
                    key              = key,
                    instruction      = row["instruction"],
                    instruction_slug = row["instruction_slug"],
                    level_id         = int(row["level_id"]),
                    sample_id        = int(row["sample_id"]),
                    array            = self._archive[key],
                )
                self._levels.append(lv)

                slug = row["instruction_slug"]
                if slug not in self._slug_map:
                    self._slug_map[slug] = Instruction(
                        instruction      = row["instruction"],
                        instruction_slug = slug,
                    )
                self._slug_map[slug]._levels.append(lv)

    # ── Sequence protocol ──────────────────────────────────────────────────────
    def __len__(self) -> int:
        return len(self._levels)

    def __iter__(self) -> Iterator[LevelArray]:
        return iter(self._levels)

    def __getitem__(self, idx: int) -> LevelArray:
        return self._levels[idx]

    # ── Query helpers ──────────────────────────────────────────────────────────
    def filter(self, keyword: str, *, case_sensitive: bool = False) -> List[LevelArray]:
        """Return all LevelArrays whose instruction contains *keyword*."""
        kw = keyword if case_sensitive else keyword.lower()
        return [
            lv for lv in self._levels
            if kw in (lv.instruction if case_sensitive else lv.instruction.lower())
        ]

    def group(self, instruction: str) -> Instruction:
        """
        Return the Instruction object for a given instruction string.
        Accepts both the human-readable form and the snake_case slug.
        """
        slug = instruction.replace(" ", "_")
        if slug in self._slug_map:
            return self._slug_map[slug]
        # try exact human-readable match
        for instr_obj in self._slug_map.values():
            if instr_obj.instruction == instruction:
                return instr_obj
        raise KeyError(f"Instruction not found: '{instruction}'")

    def instructions(self) -> List[Instruction]:
        """Return all Instruction objects (one per category)."""
        return list(self._slug_map.values())

    def category_names(self) -> List[str]:
        """Return all instruction strings (human-readable)."""
        return [instr.instruction for instr in self._slug_map.values()]

    # ── Repr ───────────────────────────────────────────────────────────────────
    def __repr__(self) -> str:
        return (
            f"DungeonLevelDataset("
            f"samples={len(self)}, "
            f"categories={len(self._slug_map)})"
        )

