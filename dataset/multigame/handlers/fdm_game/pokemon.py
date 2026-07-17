"""
dataset/multigame/handlers/fdm_game/pokemon.py
==============================================
POKEMON game map preprocessing handler.
"""
from __future__ import annotations

import hashlib
from typing import List, Dict, Any
from pathlib import Path
import numpy as np

from ...base import BasePreprocessor, TileLegend, enforce_top_left_16x16, GameSample


POKEMON_PALETTE: dict[int, tuple[int, int, int]] = {
    0:  (20,  20,  20),
    #1:  (80,  80,  80),
    2:  (200, 180, 120),
    3:  (220, 50,  50),
    4:  (255, 215, 0),
    5:  (80,  80,  80),
    6:  (100, 100, 255),
    7:  (150, 75,  0),
    8:  (34,  139, 34),
    9:  (200, 150, 150),
    10: (170, 150,  90),
    99: (255, 0,   255),
}


class POKEMONTile:
    EMPTY   = 0
    WALL    = 1
    FLOOR   = 2
    ENEMY   = 3
    OBJECT  = 4
    SPAWN   = 5
    WATER   = 6
    FENCE   = 7
    TREE    = 8
    HOUSE   = 9
    GRASS  = 10
    UNKNOWN = 99


POKEMON_TILESET_MAPPING = {
    0: POKEMONTile.FLOOR,
    1: POKEMONTile.GRASS,
    2: POKEMONTile.GRASS,
    3: POKEMONTile.GRASS,
    4: POKEMONTile.GRASS,
    5: POKEMONTile.GRASS,
    6: POKEMONTile.GRASS,
    7: POKEMONTile.OBJECT,
    8: POKEMONTile.OBJECT,
    9: POKEMONTile.FENCE,
    10: POKEMONTile.TREE,
    11: POKEMONTile.WATER,
    12: POKEMONTile.HOUSE,
    13: POKEMONTile.SPAWN,
    14: POKEMONTile.HOUSE,
    15: POKEMONTile.OBJECT,
}




def make_legend() -> TileLegend:
    """POKEMON tile legend create."""
    attrs = {
        "0": ["empty", "out of bounds"],
        "1": ["solid", "wall"],
        "2": ["floor", "walkable"],
        "3": ["enemy", "hazard"],
        "4": ["object", "collectible"],
        "5": ["spawn", "interactive"],
        "6": ["hazard", "damaging"],
        "7": ["hazard", "blocked"],
        "8": ["solid", "Tree"],
        "9": ["solid", "House"],
        "10": ["grass", "walkable"],
    }
    return TileLegend(char_to_attrs=attrs)


class POKEMONPreprocessor(BasePreprocessor):
    """
    POKEMON map data preprocessingtext.

    10x10 one-hot text -> 16x16 integer text
    (text 3text edge padding)
    """

    def char_to_int(self, char: str) -> int:
        """POKEMON  one-hot text text to    text  text for  text text."""
        return POKEMONTile.UNKNOWN

    def parse_txt(self, text: str) -> List[List[str]]:
        """POKEMON  text text  text to    text  text for  text text."""
        return []

    def is_valid_pokemon_map(
        self,
        onehot_map: np.ndarray,
        max_tile_ratio: float = 0.95,
    ) -> bool:
        """POKEMON map of  validtext  validatetext. (padding  before  10x10 based)"""
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
        """One-hot text 10x10 map -> integer text."""
        h, w, c = onehot_map.shape
        result = np.zeros((h, w), dtype=np.int32)

        for i in range(h):
            for j in range(w):
                channel_idx = np.argmax(onehot_map[i, j, :])
                result[i, j] = POKEMON_TILESET_MAPPING.get(channel_idx, POKEMONTile.UNKNOWN)

        return result

    def pad_to_16x16(self, map_10x10: np.ndarray) -> np.ndarray:
        """
        10x10 map  16x16 as  expand.

        padding text:
        - text map of  tile  during  empty(0), floor(2), water(6), tree(8)  keep
        - remaining  text floor(2) to  convert
        """
        # text edge padding as  padding
        padded = np.pad(
            map_10x10,
            pad_width=((3, 3), (3, 3)),
            mode='edge'
        )

        # keeptext tile text of
        keep_tiles = {0, 2, 6, 8, 10}  # empty, floor, water, tree, grass

        # paddingtext text in  keep_tiles in  without tile  floor(2) to  convert
        # paddingtext text:
        # - top: padded[0:3, :]
        # - bottom: padded[13:16, :]
        # - text: padded[:, 0:3]
        # - text: padded[:, 13:16]

        # top padding (3row)
        for i in range(3):
            for j in range(16):
                if padded[i, j] not in keep_tiles:
                    padded[i, j] = POKEMONTile.FLOOR

        # bottom padding (3row)
        for i in range(13, 16):
            for j in range(16):
                if padded[i, j] not in keep_tiles:
                    padded[i, j] = POKEMONTile.FLOOR

        # text padding (3column)
        for i in range(16):
            for j in range(3):
                if padded[i, j] not in keep_tiles:
                    padded[i, j] = POKEMONTile.FLOOR

        # text padding (3column)
        for i in range(16):
            for j in range(13, 16):
                if padded[i, j] not in keep_tiles:
                    padded[i, j] = POKEMONTile.FLOOR

        return padded

    def apply_grass_to_monster(self, array: np.ndarray) -> np.ndarray:
        """
        GRASS tile text  ENEMY(monster) tile to  converttext.

        - each GRASS tiletext independently 1/5 probability to  ENEMY to  text
        - seed  map array content of  MD5 text in  text → same text text always same result
        """
        seed = int.from_bytes(
            hashlib.md5(array.tobytes()).digest()[:4], byteorder='big'
        )
        rng = np.random.default_rng(seed)

        result = array.copy()
        grass_mask = result == POKEMONTile.GRASS
        grass_positions = np.argwhere(grass_mask)

        for pos in grass_positions:
            result[pos[0], pos[1]] = POKEMONTile.ENEMY

        return result

    def process_pokemon_sample(
        self,
        onehot_map: np.ndarray,
        source_id: str,
        instruction: str,
    ) -> GameSample:
        """POKEMON one-hot map -> GameSample convert."""
        map_10x10 = self.transform_pokemon_onehot(onehot_map)
        array = self.pad_to_16x16(map_10x10)
        array = self.apply_grass_to_monster(array)

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

