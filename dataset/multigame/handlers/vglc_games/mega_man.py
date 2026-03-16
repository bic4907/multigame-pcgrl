"""
dataset/multigame/handlers/vglc_games/mega_man.py
==================================================
Mega Man (TheVGLC) 전처리 핸들러.

타일 매핑
---------
0  : null    (@)
1  : ground  (#)
2  : break   (B)
3  : hazard  (H)
4  : ladder  (클라이밍 빈칸)
5  : pickup  (L, l, W, w, +, *, U)
6  : moving  (M)
7  : enemy   (E, C - 보스 checkpoint)
99 : unknown
"""
from __future__ import annotations

from ...base import BasePreprocessor, TileLegend


class MegaManTile:
    NULL    = 0
    GROUND  = 1
    BREAK   = 2
    HAZARD  = 3
    LADDER  = 4
    PICKUP  = 5
    MOVING  = 6
    ENEMY   = 7
    UNKNOWN = 99


_CHAR_MAP: dict[str, int] = {
    "@": MegaManTile.NULL,
    " ": MegaManTile.NULL,
    "#": MegaManTile.GROUND,
    "B": MegaManTile.BREAK,
    "H": MegaManTile.HAZARD,
    "L": MegaManTile.PICKUP,
    "l": MegaManTile.PICKUP,
    "W": MegaManTile.PICKUP,
    "w": MegaManTile.PICKUP,
    "+": MegaManTile.PICKUP,
    "*": MegaManTile.PICKUP,
    "U": MegaManTile.PICKUP,
    "M": MegaManTile.MOVING,
    "E": MegaManTile.ENEMY,
    "C": MegaManTile.ENEMY,
    "-": MegaManTile.LADDER,   # rope/ladder
}

MEGA_MAN_PALETTE: dict[int, tuple[int, int, int]] = {
    MegaManTile.NULL:    (0,   0,   0),
    MegaManTile.GROUND:  (80,  80,  80),
    MegaManTile.BREAK:   (160, 120, 80),
    MegaManTile.HAZARD:  (220, 80,  20),
    MegaManTile.LADDER:  (180, 140, 0),
    MegaManTile.PICKUP:  (0,   200, 200),
    MegaManTile.MOVING:  (100, 100, 220),
    MegaManTile.ENEMY:   (220, 50,  50),
    MegaManTile.UNKNOWN: (128, 0,   128),
}


def make_legend() -> TileLegend:
    attrs = {
        "@": ["null"],
        "#": ["solid", "ground"],
        "B": ["solid", "breakable"],
        "H": ["solid", "hazard"],
        "L": ["passable", "collectable", "powerup"],
        "W": ["passable", "collectable", "powerup"],
        "+": ["passable", "collectable", "powerup"],
        "M": ["solid", "moving"],
        "E": ["enemy"],
        "C": ["enemy"],
    }
    return TileLegend(char_to_attrs=attrs)


class MegaManPreprocessor(BasePreprocessor):
    def char_to_int(self, char: str) -> int:
        return _CHAR_MAP.get(char, MegaManTile.UNKNOWN)

