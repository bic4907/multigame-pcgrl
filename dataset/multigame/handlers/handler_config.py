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
class DoomSlicingConfig:
    """Doom 게임 슬라이싱 설정"""
    enabled: bool = True
    empty_max: int = 144
    floor_empty_max: int = 239


@dataclass
class FilteringConfig:
    """데이터셋 필터링 설정"""
    enabled: bool = True
    min_instruction_words: int = 2  # instruction이 이 이상의 단어 수를 가져야 함
    max_tile_ratio: float = 0.95  # 한 타일이 차지하는 최대 비율 (0~1). 이상이면 제외. 예: 0.95 = 100개 중 95개 이상


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
    pass


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


@dataclass
class POKEMONConfig:
    """Five-Dollar-Model (POKEMON) 게임 설정"""
    rotate_90: bool =True  # 시계방향 90도 회전 증강


@dataclass
class HandlerConfig:
    """모든 핸들러의 통합 설정"""
    doom_slicing: DoomSlicingConfig = field(default_factory=DoomSlicingConfig)
    filtering: FilteringConfig = field(default_factory=FilteringConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    zelda: ZeldaConfig = field(default_factory=ZeldaConfig)
    mario: MarioConfig = field(default_factory=MarioConfig)
    lode_runner: LodeRunnerConfig = field(default_factory=LodeRunnerConfig)
    kid_icarus: KidIcarusConfig = field(default_factory=KidIcarusConfig)
    mega_man: MegaManConfig = field(default_factory=MegaManConfig)
    dungeon: DungeonConfig = field(default_factory=DungeonConfig)
    pokemon: POKEMONConfig = field(default_factory=POKEMONConfig)
    
    def to_dict(self) -> Dict[str, Any]:
        """설정을 딕셔너리로 변환"""
        return {
            'doom_slicing': asdict(self.doom_slicing),
            'filtering': asdict(self.filtering),
            'augmentation': asdict(self.augmentation),
            'zelda': asdict(self.zelda),
            'mario': asdict(self.mario),
            'lode_runner': asdict(self.lode_runner),
            'kid_icarus': asdict(self.kid_icarus),
            'mega_man': asdict(self.mega_man),
            'dungeon': asdict(self.dungeon),
            'pokemon': asdict(self.pokemon),
        }

    def update_doom_slicing(
        self,
        enabled: Optional[bool] = None,
        empty_max: Optional[int] = None,
        floor_empty_max: Optional[int] = None,
    ) -> None:
        """Doom 슬라이싱 설정 업데이트"""
        if enabled is not None:
            self.doom_slicing.enabled = enabled
        if empty_max is not None:
            self.doom_slicing.empty_max = empty_max
        if floor_empty_max is not None:
            self.doom_slicing.floor_empty_max = floor_empty_max

    def update_filtering(
        self,
        enabled: Optional[bool] = None,
        min_instruction_words: Optional[int] = None,
        max_tile_ratio: Optional[float] = None,
    ) -> None:
        """필터링 설정 업데이트"""
        if enabled is not None:
            self.filtering.enabled = enabled
        if min_instruction_words is not None:
            self.filtering.min_instruction_words = min_instruction_words
        if max_tile_ratio is not None:
            self.filtering.max_tile_ratio = max_tile_ratio

    def update_augmentation(
        self,
        enabled: Optional[bool] = None,
    ) -> None:
        """증강 설정 업데이트 (활성화 여부만)"""
        if enabled is not None:
            self.augmentation.enabled = enabled

def get_default_config() -> HandlerConfig:
    """기본 설정 반환"""
    return HandlerConfig()



