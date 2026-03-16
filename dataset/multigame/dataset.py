"""
dataset/multigame/dataset.py
============================
MultiGameDataset: VGLC + Dungeon 통합 데이터셋 클래스.

외부 의존: numpy (Pillow는 렌더링 시에만 필요).

Example
-------
    from dataset.multigame import MultiGameDataset

    ds = MultiGameDataset(
        vglc_games=["zelda", "mario"],
        include_dungeon=True,
    )
    print(ds)          # MultiGameDataset(total=N, games=[...])

    sample = ds[0]     # GameSample
    print(sample.game, sample.instruction, sample.order)

    # 필터
    zelda_samples = ds.by_game("zelda")
    bat_levels    = ds.by_instruction("bat swarm")

    # 렌더링
    ds.render(sample, save_path="out.png")
    ds.render_grid(zelda_samples[:8], save_path="grid.png")
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import numpy as np

from .base import GameSample, GameTag
from .handlers.vglc_handler import VGLCHandler, _DEFAULT_VGLC_ROOT
from .handlers.dungeon_handler import DungeonHandler, _DEFAULT_DUNGEON_ROOT
from .handlers.vglc_games import SUPPORTED_GAMES
from . import tags as tag_utils
from .cache_utils import (
    build_cache_key,
    load_samples_from_cache,
    save_samples_to_cache,
)

_HERE = Path(__file__).parent


class MultiGameDataset:
    """
    VGLC + Dungeon Level Dataset 통합 클래스.

    Parameters
    ----------
    vglc_root      : TheVGLC 루트 경로 (기본: dataset/TheVGLC)
    dungeon_root   : dungeon_level_dataset 루트 경로
    vglc_games     : 로드할 VGLC 게임 태그 리스트 (None이면 전체)
    include_dungeon: Dungeon 데이터셋 포함 여부
    vglc_split     : VGLC 하위 폴더 (기본 "Processed")
    """

    def __init__(
        self,
        vglc_root:       Path | str = _DEFAULT_VGLC_ROOT,
        dungeon_root:    Path | str = _DEFAULT_DUNGEON_ROOT,
        vglc_games:      Optional[List[str]] = None,
        include_dungeon: bool = True,
        vglc_split:      str = "Processed",
        use_cache:       bool = True,
        cache_dir:       Path | str | None = None,
    ) -> None:
        self._samples: List[GameSample] = []
        self._vglc_handler: Optional[VGLCHandler] = None
        self._dungeon_handler: Optional[DungeonHandler] = None

        if cache_dir is None:
            cache_dir = _HERE / "cache" / "artifacts"
        cache_dir = Path(cache_dir)

        args_for_key = {
            "vglc_root": str(vglc_root),
            "dungeon_root": str(dungeon_root),
            "vglc_games": vglc_games,
            "include_dungeon": include_dungeon,
            "vglc_split": vglc_split,
        }
        cache_key = build_cache_key(args_for_key, code_root=_HERE)

        if use_cache:
            cached = load_samples_from_cache(cache_dir, cache_key)
            if cached is not None:
                self._samples = cached
                return

        # ── VGLC 로드 ───────────────────────────────────────────────────────────
        if vglc_games is not None or Path(vglc_root).exists():
            if Path(vglc_root).exists():
                self._vglc_handler = VGLCHandler(
                    vglc_root=vglc_root,
                    selected_games=vglc_games,
                    split=vglc_split,
                )
                for i, sample in enumerate(self._vglc_handler):
                    sample.order = len(self._samples)
                    self._samples.append(sample)

        # ── Dungeon 로드 ────────────────────────────────────────────────────────
        if include_dungeon and Path(dungeon_root).exists():
            self._dungeon_handler = DungeonHandler(root=dungeon_root)
            for i, sample in enumerate(self._dungeon_handler):
                sample.order = len(self._samples)
                self._samples.append(sample)

        if use_cache:
            save_samples_to_cache(cache_dir, cache_key, self._samples)

    # ── Sequence protocol ───────────────────────────────────────────────────────
    def __len__(self) -> int:
        return len(self._samples)

    def __iter__(self) -> Iterator[GameSample]:
        return iter(self._samples)

    def __getitem__(self, idx: int) -> GameSample:
        return self._samples[idx]

    # ── 태그 기반 필터 ──────────────────────────────────────────────────────────
    def by_game(self, game: str) -> List[GameSample]:
        """특정 게임 샘플만 반환."""
        return tag_utils.extract_by_game(self._samples, game)

    def by_games(self, games: List[str]) -> List[GameSample]:
        """복수 게임 샘플 반환."""
        return tag_utils.extract_by_games(self._samples, games)

    def by_instruction(
        self, keyword: str, *, case_sensitive: bool = False
    ) -> List[GameSample]:
        """instruction 키워드 필터."""
        return tag_utils.extract_by_instruction(
            self._samples, keyword, case_sensitive=case_sensitive
        )

    def with_instruction(self) -> List[GameSample]:
        """instruction이 있는 샘플만."""
        return tag_utils.extract_with_instruction(self._samples)

    def without_instruction(self) -> List[GameSample]:
        """instruction이 없는 샘플만."""
        return tag_utils.extract_without_instruction(self._samples)

    def by_order(self, start: int, end: int) -> List[GameSample]:
        """order 범위 [start, end) 샘플."""
        return tag_utils.extract_by_order(self._samples, start, end)

    def by_meta(self, key: str, value: Any) -> List[GameSample]:
        """meta 속성 필터."""
        return tag_utils.extract_by_meta(self._samples, key, value)

    def filter(
        self, fn
    ) -> List[GameSample]:
        """임의 조건 함수로 필터링."""
        return tag_utils.extract_by_predicate(self._samples, fn)

    # ── 집계 ────────────────────────────────────────────────────────────────────
    def group_by_game(self) -> Dict[str, List[GameSample]]:
        return tag_utils.group_by_game(self._samples)

    def group_by_instruction(self) -> Dict[str, List[GameSample]]:
        return tag_utils.group_by_instruction(self._samples)

    def count_by_game(self) -> Dict[str, int]:
        return tag_utils.count_by_game(self._samples)

    def summary(self) -> Dict[str, Any]:
        return tag_utils.summary(self._samples)

    # ── 렌더링 (Pillow 필요) ────────────────────────────────────────────────────
    def render(
        self,
        sample: GameSample,
        tile_size: int = 16,
        save_path: Optional[Path | str] = None,
    ):
        """단일 샘플 렌더링. save_path 지정 시 PNG 저장, 없으면 PIL Image 반환."""
        from .render import render_sample_pil, save_rendered
        if save_path:
            return save_rendered(sample, save_path, tile_size=tile_size)
        return render_sample_pil(sample, tile_size=tile_size)

    def render_grid(
        self,
        samples: List[GameSample],
        cols: int = 4,
        tile_size: int = 16,
        save_path: Optional[Path | str] = None,
    ):
        """여러 샘플 격자 렌더링. save_path 지정 시 PNG 저장, 없으면 PIL Image 반환."""
        from .render import render_grid as _rg, save_grid
        from PIL import Image
        if save_path:
            return save_grid(samples, save_path, cols=cols, tile_size=tile_size)
        canvas = _rg(samples, cols=cols, tile_size=tile_size)
        return Image.fromarray(canvas, mode="RGB")

    # ── 유틸 ────────────────────────────────────────────────────────────────────
    def get_tags(self, idx: int) -> Dict[str, Any]:
        """인덱스 기준 태그 dict 반환."""
        return tag_utils.build_tags(self._samples[idx])

    def all_tags(self) -> List[Dict[str, Any]]:
        """전체 샘플 태그 리스트."""
        return [tag_utils.build_tags(s) for s in self._samples]

    def available_games(self) -> List[str]:
        """로드된 게임 목록."""
        return list(self.count_by_game().keys())

    def sample(
        self,
        n: int,
        game: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> List[GameSample]:
        """
        랜덤 샘플링.

        Parameters
        ----------
        n    : 샘플 수
        game : 특정 게임만 (None이면 전체)
        seed : 랜덤 시드
        """
        rng = np.random.default_rng(seed)
        pool = self.by_game(game) if game else self._samples
        n = min(n, len(pool))
        indices = rng.choice(len(pool), size=n, replace=False)
        return [pool[i] for i in indices]

    # ── repr ────────────────────────────────────────────────────────────────────
    def __repr__(self) -> str:
        counts = self.count_by_game()
        games  = list(counts.keys())
        return (
            f"MultiGameDataset(total={len(self)}, "
            f"games={games}, counts={counts})"
        )
