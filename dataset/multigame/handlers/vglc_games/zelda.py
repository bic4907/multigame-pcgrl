"""
dataset/multigame/handlers/vglc_games/zelda.py
===============================================
The Legend of Zelda (TheVGLC) 전처리 핸들러.

타일 매핑
---------
0  : empty / passable (-, 공백)
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

from typing import List

from ...base import BasePreprocessor, TileLegend


# ── 정수 타일 ID ────────────────────────────────────────────────────────────────
class ZeldaTile:
    EMPTY   = 0
    WALL    = 1
    FLOOR   = 2
    DOOR    = 3
    BLOCK   = 4
    START   = 5
    MOB     = 6
    OBJECT  = 7
    HAZARD  = 8
    UNKNOWN = 99


# 문자 → 정수 매핑
_CHAR_MAP: dict[str, int] = {
    "-": ZeldaTile.EMPTY,
    " ": ZeldaTile.EMPTY,
    "W": ZeldaTile.WALL,
    "F": ZeldaTile.FLOOR,
    "D": ZeldaTile.DOOR,
    "B": ZeldaTile.BLOCK,
    "S": ZeldaTile.START,
    "M": ZeldaTile.MOB,
    "O": ZeldaTile.HAZARD,   # ELEMENT + FLOOR (LAVA/BLOCK, WATER/BLOCK)
    "I": ZeldaTile.HAZARD,   # ELEMENT + BLOCK
    "P": ZeldaTile.HAZARD,   # ELEMENT (LAVA, WATER)
    "o": ZeldaTile.OBJECT,
    "t": ZeldaTile.OBJECT,   # triforce
    "k": ZeldaTile.OBJECT,   # key
    "p": ZeldaTile.OBJECT,   # puzzle
    "b": ZeldaTile.BLOCK,    # boss room marker
    "e": ZeldaTile.MOB,      # enemy
    "s": ZeldaTile.START,
}

# 렌더링용 컬러 팔레트 (RGB)
ZELDA_PALETTE: dict[int, tuple[int, int, int]] = {
    ZeldaTile.EMPTY:   (0,   0,   0),
    ZeldaTile.WALL:    (80,  80,  80),
    ZeldaTile.FLOOR:   (200, 180, 120),
    ZeldaTile.DOOR:    (139, 90,  43),
    ZeldaTile.BLOCK:   (60,  100, 60),
    ZeldaTile.START:   (0,   200, 0),
    ZeldaTile.MOB:     (220, 50,  50),
    ZeldaTile.OBJECT:  (255, 215, 0),
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

