"""
dataset/multigame/__init__.py
"""
from .base import GameSample, GameTag, TileLegend, BaseGameHandler, BasePreprocessor
from .dataset import MultiGameDataset
from .handlers import VGLCHandler, VGLCGameHandler, DungeonHandler
from . import tags, render, tile_utils
from .tile_utils import (
    UNIFIED_CATEGORIES,
    CATEGORY_COLORS,
    NUM_CATEGORIES,
    to_unified,
    to_onehot,
    to_unified_and_onehot,
    validate_onehot,
    onehot_to_unified,
    category_name,
    category_distribution,
    render_unified_rgb,
)

__all__ = [
    "MultiGameDataset",
    "GameSample",
    "GameTag",
    "TileLegend",
    "BaseGameHandler",
    "BasePreprocessor",
    "VGLCHandler",
    "VGLCGameHandler",
    "DungeonHandler",
    "tags",
    "render",
    "tile_utils",
    # tile_utils shortcuts
    "UNIFIED_CATEGORIES",
    "CATEGORY_COLORS",
    "NUM_CATEGORIES",
    "to_unified",
    "to_onehot",
    "to_unified_and_onehot",
    "validate_onehot",
    "onehot_to_unified",
    "category_name",
    "category_distribution",
    "render_unified_rgb",
]

