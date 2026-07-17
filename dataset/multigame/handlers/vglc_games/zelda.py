"""
dataset/multigame/handlers/vglc_games/zelda.py
===============================================
The Legend of Zelda (TheVGLC) preprocessing handler.

tile text
---------
0  : empty / passable (-, text)
1  : wall   (W)
2  : floor  (F)
3  : door   (D)
4  : block  (B)
5  : start  (S)
6  : mob    (M)
7  : object (O, I, P, o)
99 : unknown
"""
from __future__ import annotations

import hashlib
from typing import List

import numpy as np

from ...base import BasePreprocessor, TileLegend


# ── integer tile ID ────────────────────────────────────────────────────────────────
class ZeldaTile:
    EMPTY   = 0
    WALL    = 1
    FLOOR   = 2
    DOOR    = 3
    BLOCK   = 4
    START   = 5
    MOB     = 6
    OBJECT  = 7
    FLOOD   = 8   # water/lava text hazard terrain (text HAZARD)
    UNKNOWN = 99


# character → integer text
_CHAR_MAP: dict[str, int] = {
    "-": ZeldaTile.EMPTY,
    " ": ZeldaTile.EMPTY,
    "W": ZeldaTile.WALL,
    "F": ZeldaTile.FLOOR,
    "D": ZeldaTile.DOOR,
    "B": ZeldaTile.BLOCK,
    "S": ZeldaTile.START,
    "M": ZeldaTile.MOB,
    "O": ZeldaTile.FLOOD,    # ELEMENT + FLOOR (LAVA/BLOCK, WATER/BLOCK)
    "I": ZeldaTile.FLOOD,    # ELEMENT + BLOCK
    "P": ZeldaTile.FLOOD,    # ELEMENT (LAVA, WATER)
    "o": ZeldaTile.OBJECT,
    "t": ZeldaTile.OBJECT,   # triforce
    "k": ZeldaTile.OBJECT,   # key
    "p": ZeldaTile.OBJECT,   # puzzle
    "b": ZeldaTile.BLOCK,    # boss room marker
    "e": ZeldaTile.MOB,      # enemy
    "s": ZeldaTile.START,
}

# rendering for  color palette (RGB)
ZELDA_PALETTE: dict[int, tuple[int, int, int]] = {
    ZeldaTile.EMPTY:   (0,   0,   0),
    ZeldaTile.WALL:    (80,  80,  80),
    ZeldaTile.FLOOR:   (200, 180, 120),
    ZeldaTile.DOOR:    (139, 90,  43),
    ZeldaTile.BLOCK:   (60,  100, 60),
    ZeldaTile.START:   (0,   200, 0),
    ZeldaTile.MOB:     (220, 50,  50),
    ZeldaTile.OBJECT:  (255, 215, 0),
    ZeldaTile.FLOOD:   (50,  120, 220),  # water/lava – text textcolumn
    ZeldaTile.UNKNOWN: (128, 0,   128),
}


def make_legend() -> TileLegend:
    attrs = {
        "-": ["passable", "empty"],
        "W": ["solid", "wall"],
        "F": ["passable", "floor"],
        "D": ["solid", "openable", "door"],
        "B": ["solid", "block"],
        "S": ["passable", "start"],
        "M": ["passable", "spawn"],
        "o": ["passable", "object"],
    }
    return TileLegend(char_to_attrs=attrs)


class ZeldaPreprocessor(BasePreprocessor):
    def char_to_int(self, char: str) -> int:
        return _CHAR_MAP.get(char, ZeldaTile.UNKNOWN)

    def postprocess_array(self, array: np.ndarray) -> np.ndarray:
        """
        OBJECT tile  without map in  text, FLOOR abovetext in  random as  OBJECT  batchtext.

        - OBJECT  1text or more text text also  text text
        - OBJECT  0text text next probability to  text  count  text:
            40% → 0text, 20% → 1text, 20% → 2text, 20% → 3text
        - batch abovetext  FLOOR tile  during  in text select
        - seed  map array content of  MD5 text → same text text always same result
        """
        if np.any(array == ZeldaTile.OBJECT):
            return array

        seed = int.from_bytes(
            hashlib.md5(array.tobytes()).digest()[:4], byteorder='big'
        )
        rng = np.random.default_rng(seed)

        # 40:20:20:20 probability to  text  count text
        n = rng.choice([0, 1, 2, 3], p=[0.4, 0.2, 0.2, 0.2])
        if n == 0:
            return array

        floor_positions = np.argwhere(array == ZeldaTile.FLOOR)
        if len(floor_positions) == 0:
            return array

        n = min(n, len(floor_positions))
        chosen_indices = rng.choice(len(floor_positions), size=n, replace=False)

        result = array.copy()
        for idx in chosen_indices:
            r, c = floor_positions[idx]
            result[r, c] = ZeldaTile.OBJECT

        return result

