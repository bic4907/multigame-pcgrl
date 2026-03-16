"""
dataset/multigame/handlers/vglc_games/doom.py
==============================================
Doom (TheVGLC) 전처리 핸들러.

타일 매핑
---------
0  : empty   (-)
1  : wall    (X)
2  : floor   (.)
3  : hazard  (,)
4  : enemy   (E)
5  : object  (H, :)
99 : unknown
"""
from __future__ import annotations

from ...base import BasePreprocessor, TileLegend


class DoomTile:
    EMPTY   = 0
    WALL    = 1
    FLOOR   = 2
    HAZARD  = 3
    ENEMY   = 4
    OBJECT  = 5
    UNKNOWN = 99


_CHAR_MAP: dict[str, int] = {
    "-": DoomTile.EMPTY,
    " ": DoomTile.EMPTY,
    "X": DoomTile.WALL,
    ".": DoomTile.FLOOR,
    ",": DoomTile.HAZARD,
    "E": DoomTile.ENEMY,
    "H": DoomTile.OBJECT,
    ":": DoomTile.OBJECT,
    "h": DoomTile.OBJECT,
}

DOOM_PALETTE: dict[int, tuple[int, int, int]] = {
    DoomTile.EMPTY:   (20,  20,  20),
    DoomTile.WALL:    (80,  80,  80),
    DoomTile.FLOOR:   (160, 140, 120),
    DoomTile.HAZARD:  (200, 80,  20),
    DoomTile.ENEMY:   (220, 50,  50),
    DoomTile.OBJECT:  (0,   200, 200),
    DoomTile.UNKNOWN: (128, 0,   128),
}


def make_legend() -> TileLegend:
    attrs = {
        "-": ["passable", "empty"],
        "X": ["solid", "wall"],
        ".": ["passable", "floor"],
        ",": ["passable", "hazard"],
        "E": ["enemy", "damaging"],
        "H": ["passable", "object", "health"],
        ":": ["passable", "object", "item"],
    }
    return TileLegend(char_to_attrs=attrs)


class DoomPreprocessor(BasePreprocessor):
    def char_to_int(self, char: str) -> int:
        return _CHAR_MAP.get(char, DoomTile.UNKNOWN)

