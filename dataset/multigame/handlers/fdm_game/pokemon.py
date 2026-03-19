"""
dataset/multigame/handlers/fdm_game/pokemon.py
==============================================
POKEMON 게임 맵 전처리 핸들러.
"""
from __future__ import annotations

from typing import List, Dict, Any
from pathlib import Path
import numpy as np

from ...base import BasePreprocessor, TileLegend, enforce_top_left_16x16, GameSample


class POKEMONTile:
    """POKEMON 타일 ID (DoomTile과 동일한 매핑 사용)"""
    EMPTY   = 0
    WALL    = 1
    FLOOR   = 2
    ENEMY   = 3
    OBJECT  = 4
    SPAWN   = 5
    HAZARD  = 6
    UNKNOWN = 99


POKEMON_TILESET_MAPPING = {
    0: POKEMONTile.FLOOR,
    1: POKEMONTile.FLOOR,
    2: POKEMONTile.FLOOR,
    3: POKEMONTile.FLOOR,
    4: POKEMONTile.FLOOR,
    5: POKEMONTile.FLOOR,
    6: POKEMONTile.FLOOR,
    7: POKEMONTile.OBJECT,
    8: POKEMONTile.OBJECT,
    9: POKEMONTile.SPAWN,
    10: POKEMONTile.WALL,
    11: POKEMONTile.HAZARD,
    12: POKEMONTile.WALL,
    13: POKEMONTile.SPAWN,
    14: POKEMONTile.WALL,
    15: POKEMONTile.OBJECT,
}


def make_legend() -> TileLegend:
    """POKEMON 타일 범례 생성."""
    attrs = {
        "0": ["empty", "out of bounds"],
        "1": ["solid", "wall"],
        "2": ["floor", "walkable"],
        "3": ["enemy", "hazard"],
        "4": ["object", "collectible"],
        "5": ["spawn", "interactive"],
        "6": ["hazard", "damaging"],
    }
    return TileLegend(char_to_attrs=attrs)


class POKEMONPreprocessor(BasePreprocessor):
    """
    POKEMON 맵 데이터 전처리기.
    
    10x10 one-hot 인코딩 -> 16x16 정수 인코딩
    (테두리 3칙 edge 패딩)
    """

    def char_to_int(self, char: str) -> int:
        """POKEMON은 one-hot 인코딩이므로 이 메서드는 사용 안 함."""
        return POKEMONTile.UNKNOWN

    def parse_txt(self, text: str) -> List[List[str]]:
        """POKEMON은 텍스트 형식이 아니므로 이 메서드는 사용 안 함."""
        return []

    def is_valid_pokemon_map(
        self,
        onehot_map: np.ndarray,
        max_tile_ratio: float = 0.95,
    ) -> bool:
        """POKEMON 맵의 유효성을 검사한다. (패딩 전 10x10 기반)"""
        total_tiles = 10 * 10
        
        tile_counts = {}
        for i in range(10):
            for j in range(10):
                channel_idx = np.argmax(onehot_map[i, j, :])
                tile_counts[channel_idx] = tile_counts.get(channel_idx, 0) + 1
        
        max_count = max(tile_counts.values()) if tile_counts else 0
        max_ratio = max_count / total_tiles
        
        if max_ratio >= max_tile_ratio:
            return False
        
        return True

    def transform_pokemon_onehot(self, onehot_map: np.ndarray) -> np.ndarray:
        """One-hot 인코딩된 10x10 맵 -> 정수 인코딩."""
        h, w, c = onehot_map.shape
        result = np.zeros((h, w), dtype=np.int32)
        
        for i in range(h):
            for j in range(w):
                channel_idx = np.argmax(onehot_map[i, j, :])
                result[i, j] = POKEMON_TILESET_MAPPING.get(channel_idx, POKEMONTile.UNKNOWN)
        
        return result

    def pad_to_16x16(self, map_10x10: np.ndarray) -> np.ndarray:
        """10x10 맵을 16x16으로 확장."""
        padded = np.pad(
            map_10x10,
            pad_width=((3, 3), (3, 3)),
            mode='edge'
        )
        return padded

    def process_pokemon_sample(
        self,
        onehot_map: np.ndarray,
        source_id: str,
        instruction: str,
    ) -> GameSample:
        """POKEMON one-hot 맵 -> GameSample 변환."""
        map_10x10 = self.transform_pokemon_onehot(onehot_map)
        array = self.pad_to_16x16(map_10x10)
        
        array = enforce_top_left_16x16(
            array,
            game="pokemon",
            source_id=source_id
        )
        
        return GameSample(
            game="pokemon",
            source_id=source_id,
            array=array,
            char_grid=None,
            legend=make_legend(),
            instruction=instruction,
            order=None,
            meta={"source": "five_dollar_model"}
        )

