from __future__ import annotations

import json as _json
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..base import GameTag
from ..handlers.boxoban_handler import BOXOBAN_PALETTE, BoxobanHandler
from ..handlers.dungeon_handler import DUNGEON_PALETTE, DungeonHandler, _DEFAULT_DUNGEON_ROOT
from ..handlers.pokemon_handler import POKEMONHandler, _DEFAULT_POKEMON_ROOT
from ..handlers.doom_handler import DoomHandler, DOOM_PALETTE_DICT, _DEFAULT_DOOM_ROOT
from ..tile_utils import CATEGORY_COLORS, UNIFIED_CATEGORIES, to_unified

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


_DEFAULT_BOXOBAN_ROOT = Path(__file__).parent.parent.parent / "boxoban_levels"


class DatasetViewerBackend:
    """Provides counts and random-access samples for browser viewer."""

    def __init__(
        self,
        *,
        dungeon_root: Path | str = _DEFAULT_DUNGEON_ROOT,
        boxoban_root: Path | str = _DEFAULT_BOXOBAN_ROOT,
        pokemon_root: Path | str = _DEFAULT_POKEMON_ROOT,
        doom_root: Path | str = _DEFAULT_DOOM_ROOT,
        boxoban_n_sample: int = 1000,
    ) -> None:
        self._dungeon_root = Path(dungeon_root)
        self._boxoban_root = Path(boxoban_root)
        self._pokemon_root = Path(pokemon_root)
        self._doom_root = Path(doom_root)
        self._boxoban_n_sample = int(boxoban_n_sample)

        self._games: List[str] = []
        self._counts: Dict[str, int] = {}

        self._dungeon_handler: Optional[DungeonHandler] = None
        self._boxoban_handler: Optional[BoxobanHandler] = None
        self._pokemon_handler: Optional[POKEMONHandler] = None

        self._init_sources()

    def _init_sources(self) -> None:
        # Viewer policy: expose dungeon, boxoban, pokemon, and doom datasets.
        if self._dungeon_root.exists():
            self._dungeon_handler = DungeonHandler(root=self._dungeon_root)
            n = len(self._dungeon_handler)
            if n > 0:
                self._games.append(GameTag.DUNGEON)
                self._counts[GameTag.DUNGEON] = n

        if self._boxoban_root.exists():
            # Use handler final dataset (diversity sampled) — hard only.
            self._boxoban_handler = BoxobanHandler(
                root=self._boxoban_root,
                difficulty="hard",
                n_sample=self._boxoban_n_sample,
            )
            n = len(self._boxoban_handler)
            if n > 0:
                self._games.append(GameTag.BOXOBAN)
                self._counts[GameTag.BOXOBAN] = n

        if self._pokemon_root.exists():
            try:
                self._pokemon_handler = POKEMONHandler(root=self._pokemon_root)
                n = len(self._pokemon_handler)
                if n > 0:
                    self._games.append(GameTag.POKEMON)
                    self._counts[GameTag.POKEMON] = n
            except (FileNotFoundError, ValueError):
                pass  # Pokemon dataset not available

        if self._doom_root.exists():
            try:
                self._doom_handler = DoomHandler(root=self._doom_root)
                n = len(self._doom_handler)
                if n > 0:
                    self._games.append(GameTag.DOOM)
                    self._counts[GameTag.DOOM] = n
            except (FileNotFoundError, ValueError):
                pass  # Doom dataset not available

    def games_with_counts(self) -> List[Dict[str, Any]]:
        return [
            {
                "game": game,
                "count": self._counts[game],
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

    def _load_sample(self, game: str, index: int):
        n = self.count(game)
        if index < 0 or index >= n:
            raise IndexError(f"index out of range for {game}: {index} (size={n})")

        if game == GameTag.DUNGEON:
            if self._dungeon_handler is None:
                raise RuntimeError("Dungeon handler is not initialized")
            source_ids = self._dungeon_handler.list_entries()
            return self._dungeon_handler.load_sample(source_ids[index], order=index)

        if game == GameTag.BOXOBAN:
            if self._boxoban_handler is None:
                raise RuntimeError("Boxoban handler is not initialized")
            source_ids = self._boxoban_handler.list_entries()
            return self._boxoban_handler.load_sample(source_ids[index], order=index)

        if game == GameTag.POKEMON:
            if self._pokemon_handler is None:
                raise RuntimeError("Pokemon handler is not initialized")
            source_ids = self._pokemon_handler.list_entries()
            return self._pokemon_handler.load_sample(source_ids[index], order=index)

        if game == GameTag.DOOM:
            if self._doom_handler is None:
                raise RuntimeError("Doom handler is not initialized")
            source_ids = self._doom_handler.list_entries()
            return self._doom_handler.load_sample(source_ids[index], order=index)

        raise KeyError(f"unsupported game: {game}")

    def _palette_for_game(self, game: str) -> Dict[int, tuple[int, int, int]]:
        if game == GameTag.BOXOBAN:
            return BOXOBAN_PALETTE
        if game == GameTag.DUNGEON:
            return DUNGEON_PALETTE
        if game == GameTag.DOOM:
            return DOOM_PALETTE_DICT
        return {}

    def reload(self) -> Dict[str, Any]:
        """
        데이터셋과 tile_mapping.json을 프로세스 재시작 없이 다시 로드한다.
        - 핸들러 모듈 내 캐시(lazy _samples)를 초기화
        - tile_mapping.json 파일을 다시 파싱
        - 핸들러를 새로 생성
        반환값: 리로드 후의 games_with_counts()
        """
        global _TILE_MAPPING_RAW

        # tile_mapping.json 재파싱
        try:
            _TILE_MAPPING_RAW = _load_tile_mapping()
        except Exception as exc:
            raise RuntimeError(f"tile_mapping.json 재파싱 실패: {exc}") from exc

        # 핸들러·카운트 초기화
        self._games = []
        self._counts = {}
        self._dungeon_handler = None
        self._boxoban_handler = None
        self._pokemon_handler = None
        self._doom_handler = None

        # 핸들러 재초기화
        self._init_sources()

        return {
            "status": "ok",
            "games": self.games_with_counts(),
        }

    def get_game_mapping(self, game: str) -> Dict[str, Any]:
        if game not in self._counts:
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
