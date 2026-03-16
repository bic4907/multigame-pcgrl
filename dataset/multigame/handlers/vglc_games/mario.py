"""
dataset/multigame/handlers/vglc_games/mario.py
===============================================
Super Mario Bros (TheVGLC) preprocessor.
"""
from __future__ import annotations

from ...base import BasePreprocessor, TileLegend


class MarioTile:
    EMPTY = 0
    GROUND = 1
    BREAK = 2
    QUESTION = 3
    ENEMY = 4
    PIPE = 5
    COIN = 6
    CANNON = 7
    PLATFORM = 8
    UNKNOWN = 99


_CHAR_MAP: dict[str, int] = {
    "-": MarioTile.EMPTY,
    " ": MarioTile.EMPTY,
    "X": MarioTile.GROUND,
    "S": MarioTile.BREAK,
    "?": MarioTile.QUESTION,
    "Q": MarioTile.QUESTION,
    "E": MarioTile.ENEMY,
    "<": MarioTile.PIPE,
    ">": MarioTile.PIPE,
    "[": MarioTile.PIPE,
    "]": MarioTile.PIPE,
    "o": MarioTile.COIN,
    "B": MarioTile.CANNON,
    "b": MarioTile.CANNON,
    "*": MarioTile.PLATFORM,
}

MARIO_PALETTE: dict[int, tuple[int, int, int]] = {
    MarioTile.EMPTY: (107, 140, 255),
    MarioTile.GROUND: (150, 100, 50),
    MarioTile.BREAK: (210, 170, 100),
    MarioTile.QUESTION: (255, 200, 0),
    MarioTile.ENEMY: (220, 50, 50),
    MarioTile.PIPE: (30, 160, 30),
    MarioTile.COIN: (255, 230, 0),
    MarioTile.CANNON: (80, 80, 80),
    MarioTile.PLATFORM: (200, 200, 200),
    MarioTile.UNKNOWN: (128, 0, 128),
}


def make_legend() -> TileLegend:
    return TileLegend(
        char_to_attrs={
            "-": ["passable", "empty"],
            "X": ["solid", "ground"],
            "S": ["solid", "breakable"],
            "?": ["solid", "question block"],
            "Q": ["solid", "question block", "empty"],
            "E": ["enemy", "damaging", "hazard"],
            "<": ["solid", "pipe"],
            ">": ["solid", "pipe"],
            "[": ["solid", "pipe"],
            "]": ["solid", "pipe"],
            "o": ["passable", "coin", "collectable"],
            "B": ["solid", "cannon"],
            "b": ["solid", "cannon"],
        }
    )


class MarioPreprocessor(BasePreprocessor):
    def char_to_int(self, char: str) -> int:
        return _CHAR_MAP.get(char, MarioTile.UNKNOWN)
