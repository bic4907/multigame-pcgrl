from __future__ import annotations

import json as _json
from pathlib import Path
from typing import Any, Dict, List

from ..base import GameTag, GameSample
from ..tile_utils import CATEGORY_COLORS, UNIFIED_CATEGORIES, to_unified
from ..handlers.dungeon_handler import DUNGEON_PALETTE, _DEFAULT_DUNGEON_ROOT
from ..handlers.boxoban_handler import BOXOBAN_PALETTE, _DEFAULT_BOXOBAN_ROOT
from ..handlers.pokemon_handler import POKEMON_PALETTE, _DEFAULT_POKEMON_ROOT
from ..handlers.doom_handler import DOOM_PALETTE_DICT, _DEFAULT_DOOM_ROOT, _DEFAULT_DOOM2_ROOT
from .. import MultiGameDataset

# ── unified 카테고리 메타 ───────────────────────────────────────────────────────
_UNIFIED_PALETTE: Dict[str, Any] = {
    str(k): list(v) for k, v in CATEGORY_COLORS.items()
}
_UNIFIED_NAMES: Dict[str, str] = {
    str(k): v for k, v in UNIFIED_CATEGORIES.items()
}

_MAPPING_FILE = Path(__file__).parent.parent / "tile_mapping.json"

def _load_tile_mapping() -> Dict[str, Any]:
    with _MAPPING_FILE.open("r", encoding="utf-8") as fh:
        return _json.load(fh)

_TILE_MAPPING_RAW: Dict[str, Any] = _load_tile_mapping()


def _raw_tile_names(game: str, mapping: Dict[str, Any] | None = None) -> Dict[str, str]:
    m = mapping if mapping is not None else _TILE_MAPPING_RAW
    entry = m.get(game, {})
    names = entry.get("_tile_names", {})
    return {str(k): str(v) for k, v in names.items()}


def _raw_tile_images(game: str, mapping: Dict[str, Any] | None = None) -> Dict[str, str]:
    m = mapping if mapping is not None else _TILE_MAPPING_RAW
    entry = m.get(game, {})
    images = entry.get("_tile_images", {})
    return {str(k): str(v) for k, v in images.items()}


def _unified_tile_images(mapping: Dict[str, Any] | None = None) -> Dict[str, str]:
    m = mapping if mapping is not None else _TILE_MAPPING_RAW
    images = m.get("_category_tile_images", {})
    return {str(k): str(v) for k, v in images.items() if str(k).lstrip("-").isdigit()}


class DatasetViewerBackend:
    """Provides counts and random-access samples for browser viewer."""

    def __init__(
        self,
        *,
        dungeon_root: Path | str = _DEFAULT_DUNGEON_ROOT,
        boxoban_root: Path | str = _DEFAULT_BOXOBAN_ROOT,
        pokemon_root: Path | str = _DEFAULT_POKEMON_ROOT,
        doom_root: Path | str = _DEFAULT_DOOM_ROOT,
        doom2_root: Path | str = _DEFAULT_DOOM2_ROOT,
    ) -> None:
        print("[DatasetViewerBackend] 초기화 중...", flush=True)
        
        self._games: List[str] = []
        self._counts: Dict[str, int] = {}
        self._samples_by_game: Dict[str, List[GameSample]] = {}  # 게임별 샘플 캐시
        
        # MultiGameDataset 로드 (기본값 사용 - 캐시 자동 활용)
        try:
            self._dataset = MultiGameDataset(use_cache=True)
            print(f"[DatasetViewerBackend] ✅ 로드 완료", flush=True)
        except Exception as e:
            print(f"[DatasetViewerBackend] ❌ 로드 실패: {e}", flush=True)
            import traceback
            traceback.print_exc()
            self._dataset = None
            return
        
        self._games: List[str] = self._dataset.available_games()
        self._counts: Dict[str, int] = self._dataset.count_by_game()
        
        # 게임별 샘플을 미리 필터링해서 캐시 (초기화 시점에 한 번만)
        for game in self._games:
            self._samples_by_game[game] = self._dataset.by_game(game)
        
        print(f"[DatasetViewerBackend] 게임: {self._games}", flush=True)
        print(f"[DatasetViewerBackend] 샘플: {self._counts}", flush=True)

    def games_with_counts(self) -> List[Dict[str, Any]]:
        return [
            {
                "game": game,
                "count": self._counts.get(game, 0),
                "has_palette": bool(self._palette_for_game(game)),
            }
            for game in self._games
        ]

    def count(self, game: str) -> int:
        if game not in self._counts:
            raise KeyError(f"unknown game: {game}")
        return self._counts[game]

    def get_sample(self, game: str, index: int) -> Dict[str, Any]:
        sample = self._load_sample(game, index)
        raw_palette = self._palette_for_game(game)
        unified_arr = to_unified(sample.array, game, warn_unmapped=False)
        return {
            "game": game,
            "index": index,
            "count": self.count(game),
            "source_id": sample.source_id,
            "shape": [int(sample.array.shape[0]), int(sample.array.shape[1])],
            "array": sample.array.astype(int).tolist(),
            "palette": {str(k): list(v) for k, v in raw_palette.items()},
            "tile_names": _raw_tile_names(game),
            "unified_array": unified_arr.astype(int).tolist(),
            "unified_palette": _UNIFIED_PALETTE,
            "unified_names": _UNIFIED_NAMES,
            "instruction": sample.instruction,
            "meta": sample.meta,
        }

    def _load_sample(self, game: str, index: int) -> GameSample:
        if game not in self._samples_by_game:
            raise KeyError(f"Game {game} not found")
        
        samples = self._samples_by_game[game]
        n = len(samples)
        
        if index < 0 or index >= n:
            index = index % n
        
        return samples[index]


    def _palette_for_game(self, game: str) -> Dict[int, tuple[int, int, int]]:
        if game == GameTag.BOXOBAN or game == GameTag.SOKOBAN:
            return BOXOBAN_PALETTE
        if game == GameTag.DUNGEON:
            return DUNGEON_PALETTE
        if game == GameTag.DOOM:
            return DOOM_PALETTE_DICT
        if game == GameTag.POKEMON:
            return POKEMON_PALETTE
        return {}

    def reload(self) -> Dict[str, Any]:
        """tile_mapping.json을 프로세스 재시작 없이 다시 로드한다."""
        global _TILE_MAPPING_RAW
        
        try:
            _TILE_MAPPING_RAW = _load_tile_mapping()
        except Exception as exc:
            raise RuntimeError(f"tile_mapping.json 재파싱 실패: {exc}") from exc
        
        return {
            "status": "ok",
            "games": self.games_with_counts(),
        }

    def get_game_mapping(self, game: str) -> Dict[str, Any]:
        if game not in self._games:
            raise KeyError(f"unknown game: {game}")

        # reload() 이후에도 최신 파일 내용을 반영
        mapping_raw = _TILE_MAPPING_RAW
        entry = mapping_raw.get(game, {})
        raw_mapping = entry.get("mapping", {})
        mapping = {str(k): int(v) for k, v in raw_mapping.items()}

        return {
            "game": game,
            "mapping": mapping,
            "tile_names": _raw_tile_names(game, mapping_raw),
            "raw_tile_images": _raw_tile_images(game, mapping_raw),
            "unified_names": _UNIFIED_NAMES,
            "unified_palette": _UNIFIED_PALETTE,
            "unified_tile_images": _unified_tile_images(mapping_raw),
            "tile_image_base_url": "/tile_ims/",
        }
