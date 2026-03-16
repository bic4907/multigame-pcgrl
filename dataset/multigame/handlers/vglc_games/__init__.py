"""
dataset/multigame/handlers/vglc_games/__init__.py
==================================================
게임별 전처리기 레지스트리.
"""
from __future__ import annotations

from ...base import BasePreprocessor, TileLegend
from . import zelda, mario, lode_runner, kid_icarus, doom, mega_man

PREPROCESSORS: dict[str, type[BasePreprocessor]] = {
    "zelda":       zelda.ZeldaPreprocessor,
    "mario":       mario.MarioPreprocessor,
    "lode_runner": lode_runner.LodeRunnerPreprocessor,
    "kid_icarus":  kid_icarus.KidIcarusPreprocessor,
    "doom":        doom.DoomPreprocessor,
    "mega_man":    mega_man.MegaManPreprocessor,
}

PALETTES: dict[str, dict[int, tuple[int, int, int]]] = {
    "zelda":       zelda.ZELDA_PALETTE,
    "mario":       mario.MARIO_PALETTE,
    "lode_runner": lode_runner.LODE_RUNNER_PALETTE,
    "kid_icarus":  kid_icarus.KID_ICARUS_PALETTE,
    "doom":        doom.DOOM_PALETTE,
    "mega_man":    mega_man.MEGA_MAN_PALETTE,
}

LEGEND_FACTORIES: dict[str, callable] = {
    "zelda":       zelda.make_legend,
    "mario":       mario.make_legend,
    "lode_runner": lode_runner.make_legend,
    "kid_icarus":  kid_icarus.make_legend,
    "doom":        doom.make_legend,
    "mega_man":    mega_man.make_legend,
}

SUPPORTED_GAMES = list(PREPROCESSORS.keys())

__all__ = ["PREPROCESSORS", "PALETTES", "LEGEND_FACTORIES", "SUPPORTED_GAMES"]

