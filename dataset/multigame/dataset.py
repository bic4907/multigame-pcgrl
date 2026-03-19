"""
dataset/multigame/dataset.py
============================
MultiGameDataset: Dungeon + POKEMON + Sokoban + DOOM 통합 데이터셋 클래스.

외부 의존: numpy (Pillow는 렌더링 시에만 필요).

Example
-------
    from dataset.multigame import MultiGameDataset

    # use_tile_mapping=True (기본값): 모든 샘플 array가 unified 7-category로 변환되어 반환
    ds = MultiGameDataset(include_dungeon=True, include_pokemon=True, include_doom=True)
    sample = ds[0]
    # sample.array 값 범위: [0, 6]  (unified category index)

    # use_tile_mapping=False: 원본 게임별 tile_id 그대로 반환
    ds_raw = MultiGameDataset(use_tile_mapping=False)
    sample_raw = ds_raw[0]
    # sample_raw.array 값: 게임 원본 정수 tile_id

    # 런타임 토글 (데이터셋 재로드 없이 전환)
    ds.use_tile_mapping = False   # 이후 __getitem__ / __iter__ 원본 반환
    ds.use_tile_mapping = True    # 다시 unified 반환

    # 필터
    dungeon_samples = ds.by_game("dungeon")
    pokemon_samples = ds.by_game("pokemon")
    doom_samples = ds.by_game("doom")

    # 렌더링 (use_tile_mapping 설정 자동 반영)
    ds.render(sample, save_path="out.png")
    ds.render_grid(dungeon_samples[:8], save_path="grid.png")
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import numpy as np

from .base import GameSample, GameTag
from .handlers.dungeon_handler import DungeonHandler, _DEFAULT_DUNGEON_ROOT
from .handlers.boxoban_handler import BoxobanHandler, _DEFAULT_BOXOBAN_ROOT
from .handlers.pokemon_handler import POKEMONHandler, _DEFAULT_POKEMON_ROOT
from .handlers.doom_handler import DoomHandler, _DEFAULT_DOOM_ROOT, _DEFAULT_DOOM2_ROOT
from .handlers.fdm_game.augmentation import create_rotated_sample
from .handlers.handler_config import HandlerConfig, get_default_config
from . import tags as tag_utils
from .cache_utils import (
    build_cache_key,
    load_samples_from_cache,
    save_samples_to_cache,
)
from .tile_utils import to_unified, render_unified_rgb, game_mapping_rows

_HERE = Path(__file__).parent


class MultiGameDataset:
    """
    Dungeon + Sokoban(Boxoban) + POKEMON + DOOM 통합 데이터셋 클래스.

    Parameters
    ----------
    dungeon_root     : dungeon_level_dataset 루트 경로
    pokemon_root     : Five-Dollar-Model 루트 경로
    sokoban_root     : boxoban_levels 루트 경로
    doom_root        : doom_levels 루트 경로
    include_dungeon  : Dungeon 데이터셋 포함 여부
    include_pokemon  : POKEMON 데이터셋 포함 여부
    include_sokoban  : Sokoban 데이터셋 포함 여부
    include_doom     : DOOM 데이터셋 포함 여부
    use_tile_mapping : True(기본)면 array를 unified 7-category로 변환해서 반환.
                       False면 원본 tile_id 그대로 반환.
                       로드 이후에도 속성으로 언제든 토글 가능.
    handler_config   : HandlerConfig 객체. None이면 기본값 사용.
                       (게임별 전처리 설정 포함, augmentation 설정도 포함)
    """

    def __init__(
        self,
        dungeon_root:     Path | str = _DEFAULT_DUNGEON_ROOT,
        pokemon_root:     Path | str = _DEFAULT_POKEMON_ROOT,
        sokoban_root:     Path | str = _DEFAULT_BOXOBAN_ROOT,
        doom_root:        Path | str = _DEFAULT_DOOM_ROOT,
        doom2_root:       Path | str = _DEFAULT_DOOM2_ROOT,
        include_dungeon:  bool = True,
        include_pokemon:  bool = True,
        include_sokoban:  bool = True,
        include_doom:     bool = True,
        include_doom2:    bool = True,
        use_cache:        bool = True,
        cache_dir:        Path | str | None = None,
        use_tile_mapping: bool = True,
        handler_config:   Optional[HandlerConfig] = None,
        # 하위 호환: 구 파라미터명 지원
        boxoban_root:     Path | str | None = None,
        include_boxoban:  bool | None = None,
    ) -> None:
        self.use_tile_mapping: bool = use_tile_mapping

        # 하위 호환 처리
        if boxoban_root is not None:
            sokoban_root = boxoban_root
        if include_boxoban is not None:
            include_sokoban = include_boxoban

        if handler_config is None:
            handler_config = get_default_config()
        self._handler_config = handler_config

        self._samples: List[GameSample] = []
        self._dungeon_handler: Optional[DungeonHandler] = None
        self._pokemon_handler: Optional[POKEMONHandler] = None
        self._sokoban_handler: Optional[BoxobanHandler] = None
        self._doom_handler: Optional[DoomHandler] = None

        if cache_dir is None:
            cache_dir = _HERE / "cache" / "artifacts"
        cache_dir = Path(cache_dir)

        args_for_key = {
            "dungeon_root": str(dungeon_root),
            "pokemon_root": str(pokemon_root),
            "sokoban_root": str(sokoban_root),
            "doom_root": str(doom_root),
            "doom2_root": str(doom2_root),
            "include_dungeon": include_dungeon,
            "include_pokemon": include_pokemon,
            "handler_config": handler_config.to_dict(),
            "include_sokoban": include_sokoban,
            "include_doom": include_doom,
            "include_doom2": include_doom2,
        }
        cache_key = build_cache_key(args_for_key, code_root=_HERE)

        if use_cache:
            cached = load_samples_from_cache(cache_dir, cache_key)
            if cached is not None:
                self._samples = cached
                return

        # ── Dungeon 로드 ────────────────────────────────────────────────────────
        if include_dungeon and Path(dungeon_root).exists():
            self._dungeon_handler = DungeonHandler(root=dungeon_root)
            for i, sample in enumerate(self._dungeon_handler):
                sample.order = len(self._samples)
                self._samples.append(sample)

        # ── Sokoban 로드 ────────────────────────────────────────────────────────
        if include_sokoban and Path(sokoban_root).exists():
            self._sokoban_handler = BoxobanHandler(root=sokoban_root)
            for sample in self._sokoban_handler:
                sample.order = len(self._samples)
                self._samples.append(sample)

        # ── POKEMON 로드 ────────────────────────────────────────────────────────────
        if include_pokemon and Path(pokemon_root).exists():
            try:
                self._pokemon_handler = POKEMONHandler(root=pokemon_root)
                # POKEMON은 로드 전에 필터링 적용 (패딩 전 10x10 기반)
                valid_ids, filtered_count = self._pokemon_handler.list_entries_with_filtering(
                    max_tile_ratio=self._handler_config.filtering.max_tile_ratio
                )
                for i, source_id in enumerate(valid_ids):
                    sample = self._pokemon_handler.load_sample(source_id)
                    sample.order = len(self._samples)
                    self._samples.append(sample)

                if filtered_count > 0:
                    total_pokemon = len(valid_ids) + filtered_count
                    print(f"[MultiGameDataset] POKEMON: Filtered {total_pokemon} → {len(valid_ids)} samples "
                          f"({filtered_count} removed, max_tile_ratio={self._handler_config.filtering.max_tile_ratio})")
            except (FileNotFoundError, ValueError) as e:
                print(f"Warning: Could not load FDM dataset: {e}")

        # ── DOOM 로드 ────────────────────────────────────────────────────────
        if include_doom:
            if Path(doom_root).exists():
                try:
                    self._doom_handler = DoomHandler(root=doom_root, handler_config=self._handler_config)
                    for sample in self._doom_handler:
                        sample.order = len(self._samples)
                        self._samples.append(sample)
                    
                    n_doom = len([s for s in self._samples if s.game == GameTag.DOOM])
                    if n_doom > 0:
                        print(f"[MultiGameDataset] DOOM: Loaded {n_doom} samples")
                except (FileNotFoundError, ValueError) as e:
                    print(f"Warning: Could not load DOOM dataset: {e}")
            else:
                print(f"[MultiGameDataset] DOOM dataset directory not found: {doom_root}")

        # ── DOOM2 로드 ────────────────────────────────────────────────────────
        if include_doom2:
            if Path(doom2_root).exists():
                try:
                    doom2_handler = DoomHandler(root=doom2_root, handler_config=self._handler_config)
                    for sample in doom2_handler:
                        sample.order = len(self._samples)
                        self._samples.append(sample)
                    
                    n_doom2 = len([s for s in self._samples if s.game == GameTag.DOOM and s.meta.get("file", "").startswith("MAP")])
                    if n_doom2 > 0:
                        print(f"[MultiGameDataset] DOOM 2: Loaded {n_doom2} samples")
                except (FileNotFoundError, ValueError) as e:
                    print(f"Warning: Could not load DOOM 2 dataset: {e}")
            else:
                print(f"[MultiGameDataset] DOOM 2 dataset directory not found: {doom2_root}")


        # ── POKEMON 패딩 후 필터링 (타일셋 기준: 256개 중 250개 이상) ──────────────────
        if self._handler_config.filtering.enabled:
            self._apply_pokemon_tileset_filtering()




        # ── instruction 단어 수 기반 필터링 (패딩 후) ────────────────────────
        if self._handler_config.filtering.enabled:
            self._apply_instruction_filtering()

        # ── 데이터 증강: 시계방향 90도 회전 (게임별 설정) ────────────────────────
        if self._handler_config.augmentation.enabled:
            self._augment_with_rotations_per_game()

        if use_cache:
            save_samples_to_cache(cache_dir, cache_key, self._samples)

    def _apply_floor_filtering(self, samples: List[GameSample], floor_empty_max: int) -> List[GameSample]:
        """
        Floor + empty 개수가 floor_empty_max 이하인 샘플만 필터링
        """
        filtered = []
        for sample in samples:
            if sample.game == GameTag.DOOM:
                floor_count = sample.meta.get('floor_count', 0)
                empty_count = sample.meta.get('empty_count', 0)
                if floor_count + empty_count <= floor_empty_max:
                    filtered.append(sample)
            else:
                filtered.append(sample)
        return filtered

    def _is_valid_sample(self, sample: GameSample) -> bool:
        """
        [Deprecated] 이 메서드는 더 이상 사용되지 않습니다.
        필터링은 각 게임별 로드 시점에 수행됩니다.
        """
        pass

    def _apply_instruction_filtering(self) -> None:
        """
        instruction 단어 수 기반 필터링을 적용한다.
        (타일 비율 필터링은 각 게임별 로드 시 수행됨)

        필터링 기준:
        - instruction 단어 수 < min_instruction_words인 샘플 제외
        """
        original_count = len(self._samples)

        # instruction이 있는 샘플만 필터링 (instruction이 없는 샘플은 유지)
        self._samples = [
            s for s in self._samples
            if s.instruction is None or len(s.instruction.split()) >= self._handler_config.filtering.min_instruction_words
        ]

        filtered_count = original_count - len(self._samples)
        if filtered_count > 0:
            print(f"[MultiGameDataset] Instruction filtering: {original_count} → {len(self._samples)} samples "
                  f"({filtered_count} removed, min_words={self._handler_config.filtering.min_instruction_words})")

    def _apply_pokemon_tileset_filtering(self) -> None:
        """
        POKEMON 샘플만 타일셋 기준으로 필터링.
        (패딩 후 16x16 그리드에서 한 타일이 250개 이상이면 제외)

        필터링 기준:
        - POKEMON 게임만 대상
        - 한 타일 종류가 256개 중 250개 이상이면 제외 (모노톤한 맵)
        """
        pokemon_indices = [i for i, s in enumerate(self._samples) if s.game == "pokemon"]

        if not pokemon_indices:
            return

        original_pokemon_count = len(pokemon_indices)
        filtered_samples = []

        for i, sample in enumerate(self._samples):
            if sample.game == "pokemon":
                # POKEMON 샘플: 타일셋 기준 필터링
                flat = sample.array.ravel()
                tile_counts = np.bincount(flat.astype(int))
                max_tile_count = int(np.max(tile_counts)) if len(tile_counts) > 0 else 0

                # 256개 타일 중 250개 이상이 같은 타일이 아니면 유지
                if max_tile_count < 250:
                    filtered_samples.append(sample)
            else:
                # 다른 게임: 그대로 유지
                filtered_samples.append(sample)

        self._samples = filtered_samples
        pokemon_filtered_count = original_pokemon_count - len([s for s in self._samples if s.game == "pokemon"])
        if pokemon_filtered_count > 0:
            print(f"[MultiGameDataset] POKEMON tileset filtering: {original_pokemon_count} → {len([s for s in self._samples if s.game == 'pokemon'])} samples "
                  f"({pokemon_filtered_count} removed, max_tile_count_threshold=250)")

    def _augment_with_rotations_per_game(self) -> None:
        """
        게임별 설정에 따라 각 게임의 샘플을 회전시켜 증강.

        각 게임의 config에 rotate_90 설정이 있으면 해당 게임만 회전 증강을 수행한다.
        예: config.pokemon.rotate_90 = True면 POKEMON 게임만 회전 증강
        """
        original_count = len(self._samples)
        rotated_samples = []

        for sample in self._samples:
            # 게임별 config에서 rotate_90 설정 확인
            should_augment = False
            if sample.game == "pokemon" and self._handler_config.pokemon.rotate_90:
                should_augment = True
            elif sample.game == "dungeon" and self._handler_config.dungeon.rotate_90:
                should_augment = True
            elif sample.game == GameTag.DOOM and self._handler_config.doom.rotate_90:
                should_augment = True

            if should_augment:
                rotated = create_rotated_sample(sample)
                rotated_samples.append(rotated)

        # 원본 다음에 회전 샘플 추가
        self._samples.extend(rotated_samples)

        # order 재지정
        for i, sample in enumerate(self._samples):
            sample.order = i

        if len(rotated_samples) > 0:
            print(f"[MultiGameDataset] Data augmentation: {original_count} → {len(self._samples)} samples "
                  f"(added {len(rotated_samples)} rotated versions)")

    def apply_filtering(self, apply_filter: bool = True) -> None:
        """
        필터링 조건을 적용하여 _samples를 재필터링한다.

        Note: 이 메서드는 instruction 단어 수 필터링만 수행합니다.
        타일 비율 필터링은 각 게임별 로드 시점에 적용됩니다.

        필터링 기준:
        - instruction 단어 수 >= min_instruction_words

        Parameters
        ----------
        apply_filter : bool
            True이면 필터링 적용, False이면 원본 유지
        """
        if not apply_filter or not self._handler_config.filtering.enabled:
            return

        self._apply_instruction_filtering()

    def _apply_mapping(self, sample: GameSample) -> GameSample:
        """
        use_tile_mapping 설정에 따라 array를 변환한 새 GameSample을 반환.
        원본 _samples 리스트는 항상 raw tile_id를 유지한다.
        """
        if not self.use_tile_mapping:
            return sample
        import dataclasses
        unified_array = to_unified(sample.array, sample.game, warn_unmapped=False)
        return dataclasses.replace(sample, array=unified_array)

    def _find_raw_sample(self, sample: GameSample) -> GameSample:
        """source_id/game 기준으로 내부 raw 샘플을 찾아 반환한다."""
        for s in self._samples:
            if s.game == sample.game and s.source_id == sample.source_id:
                return s
        return sample

    # ── Sequence protocol ───────────────────────────────────────────────────────
    def __len__(self) -> int:
        return len(self._samples)

    def __iter__(self) -> Iterator[GameSample]:
        for s in self._samples:
            yield self._apply_mapping(s)

    def __getitem__(self, idx: int) -> GameSample:
        return self._apply_mapping(self._samples[idx])

    # ── 태그 기반 필터 ──────────────────────────────────────────────────────────
    def by_game(self, game: str) -> List[GameSample]:
        """특정 게임 샘플만 반환."""
        return [self._apply_mapping(s)
                for s in tag_utils.extract_by_game(self._samples, game)]

    def by_games(self, games: List[str]) -> List[GameSample]:
        """복수 게임 샘플 반환."""
        return [self._apply_mapping(s)
                for s in tag_utils.extract_by_games(self._samples, games)]

    def by_instruction(
        self, keyword: str, *, case_sensitive: bool = False
    ) -> List[GameSample]:
        """instruction 키워드 필터."""
        return [self._apply_mapping(s)
                for s in tag_utils.extract_by_instruction(
                    self._samples, keyword, case_sensitive=case_sensitive)]

    def with_instruction(self) -> List[GameSample]:
        """instruction이 있는 샘플만."""
        return [self._apply_mapping(s)
                for s in tag_utils.extract_with_instruction(self._samples)]

    def without_instruction(self) -> List[GameSample]:
        """instruction이 없는 샘플만."""
        return [self._apply_mapping(s)
                for s in tag_utils.extract_without_instruction(self._samples)]

    def by_order(self, start: int, end: int) -> List[GameSample]:
        """order 범위 [start, end) 샘플."""
        return [self._apply_mapping(s)
                for s in tag_utils.extract_by_order(self._samples, start, end)]

    def by_meta(self, key: str, value: Any) -> List[GameSample]:
        """meta 속성 필터."""
        return [self._apply_mapping(s)
                for s in tag_utils.extract_by_meta(self._samples, key, value)]

    def filter(self, fn) -> List[GameSample]:
        """임의 조건 함수로 필터링."""
        return [self._apply_mapping(s)
                for s in tag_utils.extract_by_predicate(self._samples, fn)]

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
        """
        단일 샘플 렌더링.
        use_tile_mapping=True 이면 unified 스프라이트로, False 이면 원본 팔레트로 렌더링.
        save_path 지정 시 PNG 저장, 없으면 PIL Image 반환.
        """
        from .render import render_sample_pil, save_rendered
        from .tile_utils import render_unified_rgb
        from PIL import Image

        if self.use_tile_mapping:
            # array가 이미 unified로 변환된 sample을 받을 수도 있고
            # 원본 raw sample을 받을 수도 있으므로 항상 mapping 적용
            mapped = self._apply_mapping(sample)
            rgb = render_unified_rgb(mapped.array, tile_size=tile_size)
            img = Image.fromarray(rgb, "RGB")
            if save_path:
                out = Path(save_path)
                out.parent.mkdir(parents=True, exist_ok=True)
                img.save(str(out))
                return out
            return img
        else:
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
        """
        여러 샘플 격자 렌더링.
        use_tile_mapping 설정 자동 반영.
        save_path 지정 시 PNG 저장, 없으면 PIL Image 반환.
        """
        from .render import render_grid as _rg, save_grid
        from PIL import Image

        # 모든 샘플에 mapping 적용
        mapped_samples = [self._apply_mapping(s) for s in samples]

        if save_path:
            return save_grid(mapped_samples, save_path, cols=cols, tile_size=tile_size)
        canvas = _rg(mapped_samples, cols=cols, tile_size=tile_size)
        return Image.fromarray(canvas, mode="RGB")

    def render_before_after(
        self,
        sample: GameSample,
        tile_size: int = 16,
        gap: int = 8,
        save_path: Optional[Path | str] = None,
    ):
        """
        원본(raw)과 7-category mapped 이미지를 좌우로 붙여 렌더링한다.

        Left  : raw palette
        Right : unified palette
        """
        from .render import render_sample
        from PIL import Image

        raw_sample = self._find_raw_sample(sample)
        raw_rgb = render_sample(raw_sample, tile_size=tile_size)

        unified = to_unified(raw_sample.array, raw_sample.game, warn_unmapped=False)
        mapped_rgb = render_unified_rgb(unified, tile_size=tile_size)

        h = max(raw_rgb.shape[0], mapped_rgb.shape[0])
        w = raw_rgb.shape[1] + gap + mapped_rgb.shape[1]
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        canvas[:, :] = (30, 30, 30)
        canvas[:raw_rgb.shape[0], :raw_rgb.shape[1]] = raw_rgb
        x2 = raw_rgb.shape[1] + gap
        canvas[:mapped_rgb.shape[0], x2:x2 + mapped_rgb.shape[1]] = mapped_rgb

        img = Image.fromarray(canvas, mode="RGB")
        if save_path:
            out = Path(save_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            img.save(str(out))
            return out
        return img

    def mapping_rows(self, game: str):
        """tile_mapping.json 기준 원본 타일 -> unified 매핑 row 목록."""
        return game_mapping_rows(game)

    # ── 유틸 ────────────────────────────────────────────────────────────────────
    def get_tags(self, idx: int) -> Dict[str, Any]:
        """인덱스 기준 태그 dict 반환."""
        return tag_utils.build_tags(self._samples[idx])

    def all_tags(self) -> List[Dict[str, Any]]:
        """전체 샘플 태그 리스트."""
        return [tag_utils.build_tags(s) for s in self._samples]

    def available_games(self) -> List[str]:
        """등록된 게임 목록(현재: dungeon, sokoban) 반환."""
        return [GameTag.DUNGEON, GameTag.SOKOBAN, GameTag.DOOM, GameTag.POKEMON]

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
        rng  = np.random.default_rng(seed)
        pool = (tag_utils.extract_by_game(self._samples, game)
                if game else self._samples)
        n    = min(n, len(pool))
        idxs = rng.choice(len(pool), size=n, replace=False)
        return [self._apply_mapping(pool[i]) for i in idxs]

    # ── repr ────────────────────────────────────────────────────────────────────
    def __repr__(self) -> str:
        counts  = self.count_by_game()
        games   = list(counts.keys())
        mapping = "unified" if self.use_tile_mapping else "raw"
        return (
            f"MultiGameDataset(total={len(self)}, "
            f"games={games}, counts={counts}, mapping={mapping!r})"
        )
