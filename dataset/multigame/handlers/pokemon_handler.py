"""
dataset/multigame/handlers/pokemon_handler.py
==============================================
POKEMON 데이터셋 핸들러.

POKEMON은 싱글 NPY 파일에 모든 맵과 레이블이 저장되어 있다.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from ..base import BaseGameHandler, GameSample, GameTag, TileLegend
from .fdm_game.pokemon import POKEMONPreprocessor, make_legend

_DEFAULT_POKEMON_ROOT = Path(__file__).parent.parent.parent / "five-dollar-model"

# ── POKEMON 팔레트 ─────────────────────────────────────────────────────────────
POKEMON_PALETTE: Dict[int, tuple[int, int, int]] = {
    0:  (20,  20,  20),   # empty   – 어두운 회색
    1:  (80,  80,  80),   # wall    – 중간 회색
    2:  (200, 180, 120),  # floor   – 밝은 베이지
    3:  (220, 50,  50),   # enemy   – 빨강
    4:  (255, 215, 0),    # object  – 금색
    5:  (0,   200, 0),    # spawn   – 초록색
    6:  (220, 100, 20),   # hazard  – 주황색
    99: (255, 0,   255),  # unknown – 분홍색 (오류)
}


class POKEMONHandler(BaseGameHandler):
    """
    POKEMON 핸들러.
    
    Parameters
    ----------
    root : POKEMON 데이터셋 루트 경로 (기본: dataset/five-dollar-model)
    npy_name : NPY 파일명 (기본: datasets/maps_noaug.npy)
    """

    def __init__(
        self,
        root: Path | str = _DEFAULT_POKEMON_ROOT,
        npy_name: str = "datasets/maps_noaug.npy",
    ) -> None:
        self._root = Path(root)
        npy_path = self._root / npy_name

        if not npy_path.exists():
            raise FileNotFoundError(f"POKEMON NPY not found: {npy_path}")

        # NPY 파일 로드
        data = np.load(npy_path, allow_pickle=True)
        if data.ndim == 0:
            data = data.item()

        if not isinstance(data, dict):
            raise ValueError(f"Expected dict in NPY, got {type(data)}")

        self._images: List[np.ndarray] = data.get("images", [])
        self._labels: List[str] = data.get("labels", [])
        self._preprocessor = POKEMONPreprocessor()
        self._legend: TileLegend = make_legend()

        if len(self._images) != len(self._labels):
            raise ValueError(
                f"Mismatch: {len(self._images)} images, {len(self._labels)} labels"
            )

    @property
    def game_tag(self) -> str:
        return GameTag.POKEMON

    @property
    def game_dir(self) -> Path:
        return self._root

    def list_entries(self) -> List[str]:
        """NPY 인덱스를 source_id로 반환."""
        return [f"pokemon_{i:04d}" for i in range(len(self._images))]

    def load_sample(self, source_id: str, order: Optional[int] = None) -> GameSample:
        """
        source_id (예: "pokemon_0000") -> GameSample 반환.
        """
        # source_id에서 인덱스 추출
        try:
            idx = int(source_id.split("_")[1])
        except (ValueError, IndexError):
            raise KeyError(f"Invalid source_id format: {source_id!r}")

        if idx < 0 or idx >= len(self._images):
            raise KeyError(f"Index out of range: {idx}")

        onehot_map = self._images[idx]
        instruction = self._labels[idx]

        sample = self._preprocessor.process_pokemon_sample(
            onehot_map=onehot_map,
            source_id=source_id,
            instruction=instruction,
        )

        if order is not None:
            sample.order = order

        return sample

    def list_entries_with_filtering(self, max_tile_ratio: float = 0.95, max_tile_count: int = 250) -> tuple[List[str], int, int]:
        """
        필터링을 적용하여 유효한 엔트리만 반환.
        
        Parameters
        ----------
        max_tile_ratio : float
            한 타일이 차지할 수 있는 최대 비율 (패딩 전 10x10)
        max_tile_count : int
            패딩 후 16x16에서 한 타일이 차지할 수 있는 최대 개수
        
        Returns
        -------
        tuple[List[str], int, int]
            (유효한 source_id 목록, max_tile_ratio로 제거된 개수, max_tile_count로 제거된 개수)
        """
        valid_ids = []
        filtered_by_ratio = 0
        filtered_by_count = 0
        
        # "house on the beach" 중복 제거: 마지막 7개 제외 (874-880 인덱스)
        excluded_duplicates = set(range(874, 881))
        
        for i in range(len(self._images)):
            if i in excluded_duplicates:
                continue
            
            onehot_map = self._images[i]
            
            # 1단계: max_tile_ratio 필터링 (패딩 전 10x10 기반)
            if not self._preprocessor.is_valid_pokemon_map(onehot_map, max_tile_ratio):
                filtered_by_ratio += 1
                continue
            
            # 2단계: 패딩 후 tileset 필터링 (16x16 기반)
            map_10x10 = self._preprocessor.transform_pokemon_onehot(onehot_map)
            padded_map = self._preprocessor.pad_to_16x16(map_10x10)
            
            # 패딩된 맵에서 타일 개수 확인
            tile_counts = {}
            for val in padded_map.flatten():
                tile_counts[val] = tile_counts.get(val, 0) + 1
            
            max_count = max(tile_counts.values()) if tile_counts else 0
            if max_count >= max_tile_count:
                filtered_by_count += 1
                continue
            
            valid_ids.append(f"pokemon_{i:04d}")
        
        return valid_ids, filtered_by_ratio, filtered_by_count

    def __iter__(self):
        """모든 샘플 반복."""
        for i, source_id in enumerate(self.list_entries()):
            yield self.load_sample(source_id, order=i)

    def __len__(self) -> int:
        return len(self._images)

    def __repr__(self) -> str:
        return f"POKEMONHandler(samples={len(self._images)})"

