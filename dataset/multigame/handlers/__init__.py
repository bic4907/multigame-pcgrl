"""
dataset/multigame/handlers/__init__.py
"""
from .vglc_handler import VGLCHandler, VGLCGameHandler
from .dungeon_handler import DungeonHandler, DUNGEON_PALETTE
from .d2_handler import D2Handler
from .d3_handler import D3Handler
from .d5_handler import D5Handler
from .boxoban_handler import BoxobanHandler, BoxobanTile, BOXOBAN_PALETTE
from .pokemon_handler import POKEMONHandler, POKEMON_PALETTE
from .zelda_handler import ZeldaHandler, ZELDA_PALETTE

__all__ = [
    "VGLCHandler", "VGLCGameHandler",
    "DungeonHandler", "DUNGEON_PALETTE",
    "D2Handler",
    "D3Handler",
    "D5Handler",
    "BoxobanHandler", "BoxobanTile", "BOXOBAN_PALETTE",
    "POKEMONHandler", "POKEMON_PALETTE",
    "ZeldaHandler", "ZELDA_PALETTE",
]
