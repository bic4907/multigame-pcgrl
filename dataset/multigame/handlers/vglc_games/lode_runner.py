"""
dataset/multigame/handlers/vglc_games/lode_runner.py
======================================================
Lode Runner (TheVGLC) 전처리 핸들러.

타일 매핑
---------
0  : empty     (.)
1  : solid     (B)
2  : diggable  (b)
3  : rope      (-)
4  : ladder    (#)
5  : gold      (G)
6  : enemy     (E)
7  : spawn     (M)
99 : unknown
"""
from __future__ import annotations

from ...base import BasePreprocessor, TileLegend


class LodeRunnerTile:
    EMPTY    = 0
    SOLID    = 1
    DIGGABLE = 2
    ROPE     = 3
    LADDER   = 4
    GOLD     = 5
    ENEMY    = 6
    SPAWN    = 7
    UNKNOWN  = 99


_CHAR_MAP: dict[str, int] = {
    ".": LodeRunnerTile.EMPTY,
    " ": LodeRunnerTile.EMPTY,
    "B": LodeRunnerTile.SOLID,
    "b": LodeRunnerTile.DIGGABLE,
    "-": LodeRunnerTile.ROPE,
    "#": LodeRunnerTile.LADDER,
    "G": LodeRunnerTile.GOLD,
    "E": LodeRunnerTile.ENEMY,
    "M": LodeRunnerTile.SPAWN,
}

LODE_RUNNER_PALETTE: dict[int, tuple[int, int, int]] = {
    LodeRunnerTile.EMPTY:    (0,   0,   0),
    LodeRunnerTile.SOLID:    (120, 80,  40),
    LodeRunnerTile.DIGGABLE: (160, 120, 70),
    LodeRunnerTile.ROPE:     (200, 160, 80),
    LodeRunnerTile.LADDER:   (180, 100, 0),
    LodeRunnerTile.GOLD:     (255, 215, 0),
    LodeRunnerTile.ENEMY:    (220, 50,  50),
    LodeRunnerTile.SPAWN:    (0,   200, 0),
    LodeRunnerTile.UNKNOWN:  (128, 0,   128),
}


def make_legend() -> TileLegend:
    attrs = {
        ".": ["passable", "empty"],
        "B": ["solid", "ground"],
        "b": ["solid", "diggable", "ground"],
        "-": ["passable", "climbable", "rope"],
        "#": ["passable", "climbable", "ladder"],
        "G": ["passable", "pickupable", "gold"],
        "E": ["damaging", "enemy"],
        "M": ["passable", "spawn"],
    }
    return TileLegend(char_to_attrs=attrs)


class LodeRunnerPreprocessor(BasePreprocessor):
    def char_to_int(self, char: str) -> int:
        return _CHAR_MAP.get(char, LodeRunnerTile.UNKNOWN)

