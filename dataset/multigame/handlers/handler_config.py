"""
dataset/multigame/handlers/handler_config.py
============================================
Manage handler preprocessing settings.

Centralized settings for individual games and slicing options.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional


@dataclass
class DoomConfig:
    """Doom game config"""
    enabled: bool = True
    empty_max: int = 64
    floor_empty_max: int = 235
    event_count_min: int = 1
    rotate_90: bool = False
    max_samples: int = 1000


@dataclass
class AugmentationConfig:
    """data augmentation config"""
    enabled: bool = True  # Whether augmentation is enabled


@dataclass
class VGLCGameConfig:
    """VGLC game default config"""
    pass


@dataclass
class ZeldaConfig(VGLCGameConfig):
    """Zelda game config"""
    rotate_90: bool = False  # Augment with a 90-degree clockwise rotation
    max_samples: int = 1000


@dataclass
class MarioConfig(VGLCGameConfig):
    """Mario game config"""
    pass


@dataclass
class LodeRunnerConfig(VGLCGameConfig):
    """Lode Runner game config"""
    pass


@dataclass
class KidIcarusConfig(VGLCGameConfig):
    """Kid Icarus game config"""
    pass


@dataclass
class MegaManConfig(VGLCGameConfig):
    """MegaMan game config"""
    pass


@dataclass
class DungeonConfig:
    """Dungeon Level Dataset config"""
    rotate_90: bool = False  # Augment with a 90-degree clockwise rotation
    max_samples: int = 4000


@dataclass
class POKEMONConfig:
    """Five-Dollar-Model (POKEMON) game config"""
    rotate_90: bool = True  # Augment with a 90-degree clockwise rotation
    max_samples: int = 1000
    # filtering config
    enabled: bool = True
    min_instruction_words: int = 2  # Minimum number of words required in an instruction
    max_tile_ratio: float = 0.95  # Maximum ratio occupied by one tile (0-1); exclude at or above this value
    max_tile_count: int = 250  # Maximum count of one tile in a padded 16x16 map


@dataclass
class HandlerConfig:
    """Unified settings for all handlers."""
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    zelda: ZeldaConfig = field(default_factory=ZeldaConfig)
    mario: MarioConfig = field(default_factory=MarioConfig)
    lode_runner: LodeRunnerConfig = field(default_factory=LodeRunnerConfig)
    kid_icarus: KidIcarusConfig = field(default_factory=KidIcarusConfig)
    mega_man: MegaManConfig = field(default_factory=MegaManConfig)
    dungeon: DungeonConfig = field(default_factory=DungeonConfig)
    pokemon: POKEMONConfig = field(default_factory=POKEMONConfig)
    doom: DoomConfig = field(default_factory=DoomConfig)

    def to_dict(self) -> Dict[str, Any]:
        """config  dictionary to  convert"""
        return {
            'augmentation': asdict(self.augmentation),
            'zelda': asdict(self.zelda),
            'mario': asdict(self.mario),
            'lode_runner': asdict(self.lode_runner),
            'kid_icarus': asdict(self.kid_icarus),
            'mega_man': asdict(self.mega_man),
            'dungeon': asdict(self.dungeon),
            'pokemon': asdict(self.pokemon),
            'doom': asdict(self.doom),
        }


    def update_augmentation(
        self,
        enabled: Optional[bool] = None,
    ) -> None:
        """Update augmentation settings (enabled state only)."""
        if enabled is not None:
            self.augmentation.enabled = enabled

    def update_pokemon_filtering(
        self,
        enabled: Optional[bool] = None,
        min_instruction_words: Optional[int] = None,
        max_tile_ratio: Optional[float] = None,
        max_tile_count: Optional[int] = None,
    ) -> None:
        """POKEMON filtering config update"""
        if enabled is not None:
            self.pokemon.enabled = enabled
        if min_instruction_words is not None:
            self.pokemon.min_instruction_words = min_instruction_words
        if max_tile_ratio is not None:
            self.pokemon.max_tile_ratio = max_tile_ratio
        if max_tile_count is not None:
            self.pokemon.max_tile_count = max_tile_count

def get_default_config() -> HandlerConfig:
    """default config return"""
    return HandlerConfig()


