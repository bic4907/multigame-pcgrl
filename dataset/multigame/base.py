"""
dataset/multigame/base.py
=========================
Common interfaces shared by every game handler.
Each game handler subclasses BaseGameHandler, and each
preprocessor subclasses BasePreprocessor.

No external dependencies beyond numpy.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional
import warnings

import numpy as np


# ── Shared constants ──────────────────────────────────────────────────────────
class GameTag:
    """Canonical game name constants."""
    ZELDA       = "zelda"
    MARIO       = "mario"
    LODE_RUNNER = "lode_runner"
    KID_ICARUS  = "kid_icarus"
    DOOM        = "doom"
    MEGA_MAN    = "mega_man"
    DUNGEON     = "dungeon"
    BOXOBAN     = "boxoban"
    SOKOBAN     = "sokoban"
    POKEMON     = "pokemon"

# ── common data structure ────────────────────────────────────────────────────────────

@dataclass
class TileLegend:
    """
    Mapping from tile character to its meaning.
    char_to_attrs: {'W': ['solid', 'wall'], '-': ['passable', 'empty'], ...}
    """
    char_to_attrs: Dict[str, List[str]] = field(default_factory=dict)

    def tags_for(self, char: str) -> List[str]:
        return self.char_to_attrs.get(char, [])

    def is_passable(self, char: str) -> bool:
        return "passable" in self.tags_for(char)

    def is_solid(self, char: str) -> bool:
        return "solid" in self.tags_for(char)

    def is_enemy(self, char: str) -> bool:
        return "enemy" in self.tags_for(char)


@dataclass
class GameSample:
    """
    A single level sample.

    Parameters
    ----------
    game        : GameTag constant (e.g. GameTag.ZELDA)
    source_id   : identifier of the source file / npz entry
    array       : (H, W) int32 ndarray of integer tile ids
    char_grid   : (H, W) character grid (kept as parsed from the source .txt)
    legend      : TileLegend (None available)
    instruction : natural-language instruction (dungeon only)
    order       : position (index) in the original dataset
    meta        : free-form metadata dict
    """
    game:        str
    source_id:   str
    array:       np.ndarray                    # (H, W) int32
    char_grid:   Optional[List[List[str]]] = None
    legend:      Optional[TileLegend]      = None
    instruction: Optional[str]             = None
    order:       Optional[int]             = None
    meta:        Dict[str, Any]            = field(default_factory=dict)

    @property
    def height(self) -> int:
        return self.array.shape[0]

    @property
    def width(self) -> int:
        return self.array.shape[1]

    @property
    def shape(self):
        return self.array.shape

    def __repr__(self) -> str:
        return (
            f"GameSample(game={self.game!r}, source_id={self.source_id!r}, "
            f"shape={self.shape}, instruction={self.instruction!r})"
        )


def enforce_top_left_16x16(
    array: np.ndarray,
    *,
    game: str,
    source_id: str,
) -> np.ndarray:
    """
    Normalize any 2D level array to (16, 16).

    - If shape is already (16, 16), array is returned as-is.
    - Otherwise, top-left [:16, :16] is used.
    - If the sliced region is smaller than 16x16, remaining area is zero-padded.
    """
    # Some sources can contain an extra leading axis, e.g. (1, 16, 16).
    if array.ndim > 2:
        warnings.warn(
            (
                f"[{game}] {source_id} has ndim={array.ndim}; "
                "using the first slice on leading axes before 16x16 normalization"
            ),
            RuntimeWarning,
            stacklevel=2,
        )
        # Keep only the first sample on leading axes and retain the last 2 dims.
        array = array.reshape((-1,) + array.shape[-2:])[0]

    if array.shape == (16, 16):
        return array
    warnings.warn(
        (
            f"[{game}] {source_id} has shape {array.shape}; "
            "normalizing to (16, 16) with top-left slice and zero-padding if needed"
        ),
        RuntimeWarning,
        stacklevel=2,
    )
    out = np.zeros((16, 16), dtype=array.dtype)
    h = min(array.shape[0], 16)
    w = min(array.shape[1], 16)
    out[:h, :w] = array[:h, :w]
    return out


def enforce_char_grid_top_left_16x16(
    char_grid: List[List[str]],
) -> List[List[str]]:
    """Slice char grid to top-left 16x16 for consistency with array slicing."""
    return [row[:16] for row in char_grid[:16]]


# ── Abstract handler ─────────────────────────────────────────────────────────────

class BaseGameHandler(ABC):
    """
    Base handler shared by every game / dataset loader.
    list_entries() lists all IDs, and load_sample() returns a GameSample.
    """

    @property
    @abstractmethod
    def game_tag(self) -> str:
        """Return the GameTag of this handler."""
        ...

    @abstractmethod
    def list_entries(self) -> List[str]:
        """Return available source IDs."""
        ...

    @abstractmethod
    def load_sample(self, source_id: str, order: Optional[int] = None) -> GameSample:
        """Return the GameSample corresponding to source_id."""
        ...

    def __iter__(self) -> Iterator[GameSample]:
        for i, sid in enumerate(self.list_entries()):
            yield self.load_sample(sid, order=i)

    def __len__(self) -> int:
        return len(self.list_entries())

    def all_samples(self) -> List[GameSample]:
        return list(self)


# ── Preprocessor base class ───────────────────────────────────────────────────

class BasePreprocessor(ABC):
    """
    Preprocessor converting a character grid into an integer ndarray.
    Each game provides its own subclass.
    """

    @abstractmethod
    def char_to_int(self, char: str) -> int:
        """Convert a single character to an integer tile ID."""
        ...

    def transform(self, char_grid: List[List[str]]) -> np.ndarray:
        """Convert a 2D character list to an (H, W) int32 ndarray."""
        h = len(char_grid)
        w = max(len(row) for row in char_grid) if h > 0 else 0
        arr = np.zeros((h, w), dtype=np.int32)
        for r, row in enumerate(char_grid):
            for c, ch in enumerate(row):
                arr[r, c] = self.char_to_int(ch)
        return arr

    def parse_txt(self, text: str) -> List[List[str]]:
        """Parse file contents into a 2D character grid."""
        lines = text.splitlines()
        # Drop trailing blank lines
        lines = [l for l in lines if l.strip()]
        return [list(line) for line in lines]
