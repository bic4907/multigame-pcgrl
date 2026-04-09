"""
dataset/multigame/handlers/handler_config.py
============================================
핸들러 전처리 설정 관리.

각 게임 및 슬라이싱 옵션의 중앙화된 설정.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional


@dataclass
class DoomConfig:
    """Doom 게임 설정"""
    enabled: bool = True
    empty_max: int = 64
    floor_empty_max: int = 235
    event_count_min: int = 1
    rotate_90: bool = False
    max_samples: int = 1000


@dataclass
class AugmentationConfig:
    """데이터 증강 설정"""
    enabled: bool = True  # 증강 활성화 여부


@dataclass
class VGLCGameConfig:
    """VGLC 게임 기본 설정"""
    pass


@dataclass
class ZeldaConfig(VGLCGameConfig):
    """Zelda 게임 설정"""
    rotate_90: bool = False  # 시계방향 90도 회전 증강
    max_samples: int = 1000


@dataclass
class MarioConfig(VGLCGameConfig):
    """Mario 게임 설정"""
    pass


@dataclass
class LodeRunnerConfig(VGLCGameConfig):
    """Lode Runner 게임 설정"""
    pass


@dataclass
class KidIcarusConfig(VGLCGameConfig):
    """Kid Icarus 게임 설정"""
    pass


@dataclass
class MegaManConfig(VGLCGameConfig):
    """MegaMan 게임 설정"""
    pass


@dataclass
class DungeonConfig:
    """Dungeon Level Dataset 설정"""
    rotate_90: bool = False  # 시계방향 90도 회전 증강
    max_samples: int = 10000


@dataclass
class D2Config:
    """Dungeon Legacy (d2) 설정"""
    rotate_90: bool = False  # 시계방향 90도 회전 증강
    max_samples: int = 10000


@dataclass
class POKEMONConfig:
    """Five-Dollar-Model (POKEMON) 게임 설정"""
    rotate_90: bool = True  # 시계방향 90도 회전 증강
    max_samples: int = 1000
    # 필터링 설정
    enabled: bool = True
    min_instruction_words: int = 2  # instruction이 이 이상의 단어 수를 가져야 함
    max_tile_ratio: float = 0.95  # 한 타일이 차지하는 최대 비율 (0~1). 이상이면 제외. 예: 0.95 = 100개 중 95개 이상
    max_tile_count: int = 250  # 패딩 후 16x16에서 한 타일이 차지할 수 있는 최대 개수


@dataclass
class HandlerConfig:
    """모든 핸들러의 통합 설정"""
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    zelda: ZeldaConfig = field(default_factory=ZeldaConfig)
    mario: MarioConfig = field(default_factory=MarioConfig)
    lode_runner: LodeRunnerConfig = field(default_factory=LodeRunnerConfig)
    kid_icarus: KidIcarusConfig = field(default_factory=KidIcarusConfig)
    mega_man: MegaManConfig = field(default_factory=MegaManConfig)
    dungeon: DungeonConfig = field(default_factory=DungeonConfig)
    d2: D2Config = field(default_factory=D2Config)
    pokemon: POKEMONConfig = field(default_factory=POKEMONConfig)
    doom: DoomConfig = field(default_factory=DoomConfig)
    
    def to_dict(self) -> Dict[str, Any]:
        """설정을 딕셔너리로 변환"""
        return {
            'augmentation': asdict(self.augmentation),
            'zelda': asdict(self.zelda),
            'mario': asdict(self.mario),
            'lode_runner': asdict(self.lode_runner),
            'kid_icarus': asdict(self.kid_icarus),
            'mega_man': asdict(self.mega_man),
            'dungeon': asdict(self.dungeon),
            'd2': asdict(self.d2),
            'pokemon': asdict(self.pokemon),
            'doom': asdict(self.doom),
        }


    def update_augmentation(
        self,
        enabled: Optional[bool] = None,
    ) -> None:
        """증강 설정 업데이트 (활성화 여부만)"""
        if enabled is not None:
            self.augmentation.enabled = enabled

    def update_pokemon_filtering(
        self,
        enabled: Optional[bool] = None,
        min_instruction_words: Optional[int] = None,
        max_tile_ratio: Optional[float] = None,
        max_tile_count: Optional[int] = None,
    ) -> None:
        """POKEMON 필터링 설정 업데이트"""
        if enabled is not None:
            self.pokemon.enabled = enabled
        if min_instruction_words is not None:
            self.pokemon.min_instruction_words = min_instruction_words
        if max_tile_ratio is not None:
            self.pokemon.max_tile_ratio = max_tile_ratio
        if max_tile_count is not None:
            self.pokemon.max_tile_count = max_tile_count

def get_default_config() -> HandlerConfig:
    """기본 설정 반환"""
    return HandlerConfig()



