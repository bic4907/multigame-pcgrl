"""
dataset/multigame/handlers/vglc_games/kid_icarus.py
====================================================
Kid Icarus (TheVGLC) 전처리 핸들러.

타일 매핑
---------
0  : empty    (-)
1  : solid    (#)
2  : door     (D)
3  : hazard   (H)
4  : platform (T, M)
99 : unknown
"""
from __future__ import annotations

from ...base import BasePreprocessor, TileLegend


class KidIcarusTile:
    EMPTY    = 0
    SOLID    = 1
    DOOR     = 2
    HAZARD   = 3
    PLATFORM = 4
    UNKNOWN  = 99


_CHAR_MAP: dict[str, int] = {
    "-": KidIcarusTile.EMPTY,
    " ": KidIcarusTile.EMPTY,
    "#": KidIcarusTile.SOLID,
    "D": KidIcarusTile.DOOR,
    "H": KidIcarusTile.HAZARD,
    "T": KidIcarusTile.PLATFORM,
    "M": KidIcarusTile.PLATFORM,
}

KID_ICARUS_PALETTE: dict[int, tuple[int, int, int]] = {
    KidIcarusTile.EMPTY:    (100, 100, 220),
    KidIcarusTile.SOLID:    (80,  80,  80),
    KidIcarusTile.DOOR:     (139, 90,  43),
    KidIcarusTile.HAZARD:   (220, 50,  50),
    KidIcarusTile.PLATFORM: (180, 180, 60),
    KidIcarusTile.UNKNOWN:  (128, 0,   128),
}


def make_legend() -> TileLegend:
    attrs = {
        "-": ["passable", "empty"],
        "#": ["solid", "ground"],
        "D": ["solid", "openable", "door"],
        "H": ["solid", "damaging", "hazard"],
        "T": ["solidtop", "passable", "platform"],
        "M": ["solidtop", "passable", "moving", "platform"],
    }
    return TileLegend(char_to_attrs=attrs)


class KidIcarusPreprocessor(BasePreprocessor):
    def char_to_int(self, char: str) -> int:
        return _CHAR_MAP.get(char, KidIcarusTile.UNKNOWN)

