"""Top-level package exports for project datasets."""

from .multigame import GameSample, GameTag, MultiGameDataset

__all__ = [
    "MultiGameDataset",
    "GameTag",
    "GameSample",
]

# `dungeon_level_dataset` may not exist in CI/clean clones.
try:
    from .dungeon_level_dataset.dataset import DungeonLevelDataset
except ModuleNotFoundError:
    DungeonLevelDataset = None  # type: ignore[assignment]
else:
    __all__.append("DungeonLevelDataset")
