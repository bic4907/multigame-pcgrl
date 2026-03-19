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

    def list_entries_with_filtering(self, max_tile_ratio: float = 0.95) -> tuple[List[str], int]:
        """
        필터링을 적용하여 유효한 엔트리만 반환.
        
        Parameters
        ----------
        max_tile_ratio : float
            한 타일이 차지할 수 있는 최대 비율
        
        Returns
        -------
        tuple[List[str], int]
            (유효한 source_id 목록, 제외된 샘플 수)
        """
        valid_ids = []
        filtered_count = 0
        
        # "house on the beach" 중복 제거: 마지막 7개 제외 (874-880 인덱스)
        # 첫 번째 항목만 유지 (873: "a house on the beach")
        excluded_duplicates = set(range(874, 881))  # indices 874-880
        
        for i in range(len(self._images)):
            # 중복 필터링 (heuristic)
            if i in excluded_duplicates:
                filtered_count += 1
                continue
            
            onehot_map = self._images[i]
            # 패딩 전에 유효성 검사
            if self._preprocessor.is_valid_pokemon_map(onehot_map, max_tile_ratio):
                valid_ids.append(f"pokemon_{i:04d}")
            else:
                filtered_count += 1
        
        return valid_ids, filtered_count

    def __iter__(self):
        """모든 샘플 반복."""
        for i, source_id in enumerate(self.list_entries()):
            yield self.load_sample(source_id, order=i)

    def __len__(self) -> int:
        return len(self._images)

    def __repr__(self) -> str:
        return f"POKEMONHandler(samples={len(self._images)})"

