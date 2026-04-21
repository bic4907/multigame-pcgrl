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
from ..handlers.zelda_handler import ZELDA_PALETTE
from ..cache_utils import find_game_cache_key, load_game_annotations_from_cache
from .. import MultiGameDataset

_ENUM_TO_COND_COL = {0: "condition_0", 1: "condition_1", 2: "condition_2",
                     3: "condition_3", 4: "condition_4"}

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
        print("[DatasetViewerBackend] Initializing...", flush=True)

        self._games: List[str] = []
        self._counts: Dict[str, int] = {}
        self._raw_samples_by_game: Dict[str, List[GameSample]] = {}  # per-game raw sample cache

        # Load MultiGameDataset with use_tile_mapping=False to keep raw arrays
        try:
            self._dataset = MultiGameDataset(use_cache=True, use_tile_mapping=False)
            print(f"[DatasetViewerBackend] ✅ Loaded successfully (raw array)", flush=True)
        except Exception as e:
            print(f"[DatasetViewerBackend] ❌ Failed to load: {e}", flush=True)
            import traceback
            traceback.print_exc()
            self._dataset = None
            return
        
        self._games: List[str] = self._dataset.available_games()

        # Pre-filter and cache raw samples per game (once at init)
        # 뷰어에서는 맵이 한 번씩만 표시되도록 source_id 기준 중복 제거
        for game in self._games:
            all_samples = self._dataset.by_game(game)
            seen: set = set()
            deduped: List[GameSample] = []
            for s in all_samples:
                if s.source_id not in seen:
                    seen.add(s.source_id)
                    deduped.append(s)
            self._raw_samples_by_game[game] = deduped

        self._counts: Dict[str, int] = {
            game: len(samples)
            for game, samples in self._raw_samples_by_game.items()
        }

        # ann.json 로드: game → {ann_key → annotation_row}
        cache_dir = self._dataset._cache_dir
        self._ann_lookup: Dict[str, Dict[str, Any]] = {}
        for game in self._games:
            cache_key = find_game_cache_key(cache_dir, game)
            if cache_key:
                ann_data = load_game_annotations_from_cache(cache_dir, game, cache_key)
                if ann_data:
                    self._ann_lookup[game] = {
                        row["key"]: row for row in ann_data.get("annotations", [])
                    }
                    print(f"[DatasetViewerBackend] ann.json loaded: {game} ({len(self._ann_lookup[game])} rows)", flush=True)

        print(f"[DatasetViewerBackend] Games: {self._games}", flush=True)
        print(f"[DatasetViewerBackend] Samples: {self._counts}", flush=True)

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
        # raw_array: 변형되지 않은 원본 데이터
        raw_array = sample.array
        # unified_array: mapping에 따라 변환된 unified category
        unified_array = to_unified(raw_array, game, warn_unmapped=False)
        
        return {
            "game": game,
            "index": index,
            "count": self.count(game),
            "source_id": sample.source_id,
            "shape": [int(sample.array.shape[0]), int(sample.array.shape[1])],
            "array": raw_array.astype(int).tolist(),
            "palette": {str(k): list(v) for k, v in raw_palette.items()},
            "tile_names": _raw_tile_names(game),
            "unified_array": unified_array.astype(int).tolist(),
            "unified_palette": _UNIFIED_PALETTE,
            "unified_names": _UNIFIED_NAMES,
            "instruction": sample.instruction,
            "annotations": self._get_annotations(game, sample),
            "meta": sample.meta,
        }

    def _get_annotations(self, game: str, sample: GameSample) -> List[Dict[str, Any]]:
        """샘플의 모든 reward_enum annotation을 반환한다."""
        lookup = self._ann_lookup.get(game, {})
        if not lookup:
            return []

        ann_keys: List[str] = sample.meta.get("ann_keys", [])
        if ann_keys:
            rows = [lookup[k] for k in ann_keys if k in lookup]
        else:
            return []

        result = []
        for row in rows:
            enum = int(row["reward_enum"])
            cond_col = _ENUM_TO_COND_COL.get(enum)
            cond_val = row.get(cond_col) if cond_col else None
            raw  = row.get("instruction_raw")
            uni  = row.get("instruction_uni")
            result.append({
                "reward_enum":     enum,
                "feature_name":    row["feature_name"],
                "sub_condition":   row.get("sub_condition", ""),
                "condition":       cond_val,
                "instruction_raw": str(raw) if raw and str(raw) != "None" else None,
                "instruction_uni": str(uni) if uni and str(uni) != "None" else None,
            })
        return result

    def _load_sample(self, game: str, index: int) -> GameSample:
        if game not in self._raw_samples_by_game:
            raise KeyError(f"Game {game} not found")
        
        samples = self._raw_samples_by_game[game]
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
        if game == GameTag.ZELDA:
            return ZELDA_PALETTE
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
