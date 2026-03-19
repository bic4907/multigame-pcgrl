"""
dataset/multigame/__init__.py
"""
import logging

# 라이브러리 패키지 관례: NullHandler 등록으로 "No handlers" 경고 방지.
# 사용자가 logging.basicConfig(level=logging.INFO) 등을 설정하면 자동으로 출력된다.
logging.getLogger("dataset.multigame").addHandler(logging.NullHandler())

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
    game_mapping_info,
    game_mapping_rows,
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
    "game_mapping_info",
    "game_mapping_rows",
]
