"""
dataset/multigame/__init__.py
"""
import logging

# Library convention: attach a NullHandler so importing does not emit "No handlers" warnings.
# Applications opt in with logging.basicConfig(level=logging.INFO).
logging.getLogger("dataset.multigame").addHandler(logging.NullHandler())

from .base import GameSample, GameTag, TileLegend, BaseGameHandler, BasePreprocessor
from .dataset import MultiGameDataset
from .handlers import VGLCHandler, VGLCGameHandler, DungeonHandler, ZeldaHandler
from . import tags, render, tile_utils, stats
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
    game_mapping_info,
    game_mapping_rows,
)
from .stats import compute_dataset_stats, compute_game_stats, print_dataset_stats

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
    "ZeldaHandler",
    "tags",
    "render",
    "tile_utils",
    "stats",
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
    "game_mapping_info",
    "game_mapping_rows",
    # stats shortcuts
    "compute_dataset_stats",
    "compute_game_stats",
    "print_dataset_stats",
]
