"""
dataset/multigame/handlers/__init__.py
"""
from .vglc_handler import VGLCHandler, VGLCGameHandler
from .dungeon_handler import DungeonHandler, DUNGEON_PALETTE
from .boxoban_handler import BoxobanHandler, BoxobanTile, BOXOBAN_PALETTE
from .pokemon_handler import POKEMONHandler, POKEMON_PALETTE

__all__ = [
    "VGLCHandler", "VGLCGameHandler",
    "DungeonHandler", "DUNGEON_PALETTE",
    "BoxobanHandler", "BoxobanTile", "BOXOBAN_PALETTE",
    "POKEMONHandler", "POKEMON_PALETTE",
]
