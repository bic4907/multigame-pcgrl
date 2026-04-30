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

import csv
import logging
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import numpy as np

from .base import GameSample, GameTag
from .handlers.dungeon_handler import DungeonHandler, _DEFAULT_DUNGEON_ROOT
from .handlers.boxoban_handler import BoxobanHandler, _DEFAULT_BOXOBAN_ROOT
from .handlers.pokemon_handler import POKEMONHandler, _DEFAULT_POKEMON_ROOT
from .handlers.doom_handler import DoomHandler, _DEFAULT_DOOM_ROOT, _DEFAULT_DOOM2_ROOT
from .handlers.zelda_handler import ZeldaHandler, _DEFAULT_ZELDA_ROOT
from .handlers.fdm_game.augmentation import create_rotated_sample
from .handlers.handler_config import HandlerConfig, get_default_config
from . import tags as tag_utils
from .cache_utils import (
    build_per_game_cache_key,
    build_combined_doom_cache_key,
    load_game_samples_from_cache,
    save_game_samples_to_cache,
    load_any_game_cache,
    save_game_annotations_to_cache,
    load_game_annotations_from_cache,
    find_game_cache_key,
    update_ann_batch_id,
    update_json_with_ann_keys,
    # legacy (하위 호환)
    build_cache_key,
    load_samples_from_cache,
    save_samples_to_cache,
)
from .tile_utils import to_unified, render_unified_rgb, game_mapping_rows

_HERE = Path(__file__).parent

logger = logging.getLogger(__name__)


class _WarningConditionsDict(dict):
    """
    placeholder conditions dict.
    아직 per-sample annotation이 없는 게임에서 conditions 값에 접근하면
    WARNING 로그를 출력한다.  (게임별 1회만)
    """
    _warned_games: set = set()  # 클래스 변수: 이미 경고한 게임 이름

    def __init__(self, data: dict, game: str, logger) -> None:
        super().__init__(data)
        self._game = game
        self._logger = logger

    def __getitem__(self, key):
        self._warn()
        return super().__getitem__(key)

    def get(self, key, default=None):
        self._warn()
        return super().get(key, default)

    def __iter__(self):
        self._warn()
        return super().__iter__()

    def items(self):
        self._warn()
        return super().items()

    def values(self):
        self._warn()
        return super().values()

    def _warn(self):
        if self._game not in _WarningConditionsDict._warned_games:
            self._logger.warning(
                "[%s] conditions accessed: this game does not have per-sample reward annotations yet. "
                "Placeholder values will be returned.",
                self._game,
            )
            _WarningConditionsDict._warned_games.add(self._game)


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
        zelda_root:       Path | str = _DEFAULT_ZELDA_ROOT,
        N:                    int = 0,
        include_dungeon:      bool = True,
        include_pokemon:      bool = True,
        include_sokoban:      bool = True,
        include_doom:         bool = True,
        include_doom2:        bool = True,
        include_zelda:        bool = True,
        use_cache:            bool = True,
        cache_dir:            Path | str | None = None,
        use_tile_mapping:     bool = True,
        handler_config:       Optional[HandlerConfig] = None,
        reward_annotations_dir: Path | str | None = None,  # deprecated: ignored, ann.json used
        max_samples_per_game: int = 0,
        max_samples_seed:     int = 42,
        # 하위 호환: 구 파라미터명 지원
        boxoban_root:         Path | str | None = None,
        include_boxoban:      bool | None = None,
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
        self._zelda_handler: Optional[ZeldaHandler] = None

        if cache_dir is None:
            cache_dir = _HERE / "cache" / "artifacts"
        cache_dir = Path(cache_dir)
        self._cache_dir = cache_dir
        self._use_cache = use_cache

        # 게임별 캐시 키 추적 (annotation 로드에 사용)
        self._game_cache_keys: Dict[str, str] = {}

        hc = handler_config.to_dict()

        # ── 게임별 로드 설정 (game, include, root, handler_config_sub) ────────
        _game_specs = []
        if include_dungeon:
            _game_specs.append(("dungeon", str(dungeon_root), hc.get("dungeon", {})))
        if include_sokoban:
            _game_specs.append(("sokoban", str(sokoban_root), hc.get("sokoban", {})))
        if include_zelda:
            _game_specs.append(("zelda", str(zelda_root), hc.get("zelda", {})))
        if include_pokemon:
            _game_specs.append(("pokemon", str(pokemon_root), hc.get("pokemon", {})))
        # doom/doom2는 통합 처리 (루프 후 별도 블록에서)

        # ── 게임별 로드 루프 ─────────────────────────────────────────────────
        for game, game_root, game_hc in _game_specs:
            cache_key = build_per_game_cache_key(game, game_root, game_hc)
            logger.debug("[%s] cache key: %s", game, cache_key[:12])
            self._game_cache_keys[game] = cache_key
            # (1) per-game 캐시 히트 시도
            if use_cache:
                cached = load_game_samples_from_cache(cache_dir, game, cache_key)
                if cached is not None:
                    for s in cached:
                        s.order = len(self._samples)
                        self._samples.append(s)
                    continue

            # (2) 원본 데이터셋에서 로드
            game_samples = self._load_game_from_source(
                game, game_root, handler_config
            )

            if game_samples is not None:
                # 캐시 저장 전 max_samples 적용
                max_s = game_hc.get("max_samples") if isinstance(game_hc, dict) else getattr(game_hc, "max_samples", None)
                if max_s is not None and len(game_samples) > max_s:
                    game_samples = game_samples[:max_s]
                # 캐시 저장 전 필터링 + 증강 적용 (viewer/annotate 모두 동일한 수를 보도록)
                game_samples = self._postprocess_game_samples(game, game_samples, handler_config)
                for s in game_samples:
                    s.order = len(self._samples)
                    self._samples.append(s)
                # 캐시에 저장
                if use_cache:
                    save_game_samples_to_cache(
                        cache_dir, game, cache_key, game_samples
                    )
                continue

            # (3) artifact-only fallback: 키 불일치지만 해당 게임 캐시가 있으면 로드
            if use_cache:
                fallback = load_any_game_cache(cache_dir, game)
                if fallback is not None:
                    logger.info("%s: artifact-only fallback (%d samples from existing cache)",
                                game, len(fallback))
                    for s in fallback:
                        s.order = len(self._samples)
                        self._samples.append(s)
                    # fallback 시 실제 로드된 파일의 키로 갱신
                    actual_key = find_game_cache_key(cache_dir, game)
                    if actual_key:
                        self._game_cache_keys[game] = actual_key
                    continue

            # (4) 아무것도 없으면 경고만 출력
            logger.warning("%s: no source data and no cache — skipped", game)

        # ── doom + doom2 통합 로드 (합산 max_samples=1000 적용 후 캐시 저장) ──
        if include_doom or include_doom2:
            doom_hc = hc.get("doom", {})
            doom_cache_key = build_combined_doom_cache_key(
                str(doom_root), str(doom2_root),
                include_doom, include_doom2,
                doom_hc,
            )
            logger.debug("[doom] cache key: %s", doom_cache_key[:12])
            self._game_cache_keys["doom"] = doom_cache_key
            doom_cached = load_game_samples_from_cache(cache_dir, "doom", doom_cache_key) if use_cache else None
            if doom_cached is not None:
                for s in doom_cached:
                    s.order = len(self._samples)
                    self._samples.append(s)
            else:
                doom_combined: List[GameSample] = []
                if include_doom:
                    raw = self._load_game_from_source("doom", str(doom_root), handler_config)
                    if raw:
                        doom_combined.extend(raw)
                if include_doom2:
                    raw = self._load_game_from_source("doom2", str(doom2_root), handler_config)
                    if raw:
                        doom_combined.extend(raw)
                if doom_combined:
                    max_s = doom_hc.get("max_samples") if isinstance(doom_hc, dict) else getattr(doom_hc, "max_samples", None)
                    if max_s is not None and len(doom_combined) > max_s:
                        doom_combined = doom_combined[:max_s]
                    doom_combined = self._postprocess_game_samples("doom", doom_combined, handler_config)
                    for s in doom_combined:
                        s.order = len(self._samples)
                        self._samples.append(s)
                    if use_cache:
                        save_game_samples_to_cache(cache_dir, "doom", doom_cache_key, doom_combined)
                else:
                    fallback = load_any_game_cache(cache_dir, "doom") if use_cache else None
                    if fallback is not None:
                        logger.info("doom: artifact-only fallback (%d samples from existing cache)", len(fallback))
                        for s in fallback:
                            s.order = len(self._samples)
                            self._samples.append(s)
                        # fallback 시 실제 로드된 파일의 키로 갱신
                        actual_key = find_game_cache_key(cache_dir, "doom")
                        if actual_key:
                            self._game_cache_keys["doom"] = actual_key

        # ── annotation 자동 로드 (ann.json → 샘플 복제) ─────────────────────
        if use_cache and self._game_cache_keys:
            self._ensure_and_load_all_annotations()

        # ── raw counts 기록 (max_samples_per_game 적용 전, (game, reward_enum) 기준) ──
        self._raw_game_re_counts: dict = {}
        for s in self._samples:
            re = s.meta.get("reward_enum")
            if re is not None:
                self._raw_game_re_counts[(s.game, re)] = self._raw_game_re_counts.get((s.game, re), 0) + 1

        # ── 게임별 베이스 샘플 수 제한 (source_id 기준, annotation 복제 이후) ──
        # source_id 단위로 선택하므로 모든 reward_enum 복제본이 함께 유지됨
        if max_samples_per_game >= 1:
            import random as _random
            _rng = _random.Random(max_samples_seed)
            _sid_buckets: dict = {}  # game → {source_id → [index]}
            for i, s in enumerate(self._samples):
                _sid_buckets.setdefault(s.game, {}).setdefault(s.source_id, []).append(i)
            _keep: set = set()
            for _game, _sid_map in sorted(_sid_buckets.items()):
                source_ids = sorted(_sid_map.keys())
                if len(source_ids) > max_samples_per_game:
                    chosen = _rng.sample(source_ids, max_samples_per_game)
                    logger.info("max_samples_per_game=%d [%s]: %d → %d unique samples (seed=%d)",
                                max_samples_per_game, _game, len(source_ids), max_samples_per_game, max_samples_seed)
                    for sid in chosen:
                        _keep.update(_sid_map[sid])
                else:
                    for idxs in _sid_map.values():
                        _keep.update(idxs)
            _before = len(self._samples)
            self._samples = [s for i, s in enumerate(self._samples) if i in _keep]
            if len(self._samples) < _before:
                logger.info("max_samples_per_game=%d: total %d → %d samples",
                            max_samples_per_game, _before, len(self._samples))

        # ── N 샘플 서브샘플링 (게임별, 마스크 기반) ─────────────────────────
        if N >= 1:
            import random as _random
            _total = len(self._samples)
            _rng = _random.Random(42)
            _mask = [False] * _total
            # 게임별 인덱스를 삽입 순서 유지로 수집
            _game_buckets: dict = {}
            for i, s in enumerate(self._samples):
                _game_buckets.setdefault(s.game, []).append(i)
            for _game, _idxs in _game_buckets.items():
                if len(_idxs) > N:
                    _chosen = _rng.sample(_idxs, N)
                    logger.info("N=%d per-game subsampling [%s]: %d → %d", N, _game, len(_idxs), N)
                else:
                    _chosen = _idxs
                for i in _chosen:
                    _mask[i] = True
            self._samples = [s for s, m in zip(self._samples, _mask) if m]
            if len(self._samples) < _total:
                logger.info("N=%d per-game subsampling total: %d → %d samples", N, _total, len(self._samples))

    def _postprocess_game_samples(
        self, game: str, samples: List[GameSample], handler_config: HandlerConfig
    ) -> List[GameSample]:
        """
        캐시 저장 전에 필터링과 증강을 적용한다.

        적용 순서:
        1. Pokemon 타일셋 필터링 (max_tile_count 초과 샘플 제거)
        2. Instruction 단어 수 필터링 (min_instruction_words 미만 제거)
        3. 회전 증강 (rotate_90 설정 시 90도 회전 사본 추가)
        4. 증강 후 max_samples 재적용
        """
        # (1) Pokemon 타일셋 필터링
        if game == "pokemon" and handler_config.pokemon.enabled:
            max_tile_count = handler_config.pokemon.max_tile_count
            before = len(samples)
            samples = [
                s for s in samples
                if int(np.max(np.bincount(s.array.ravel().astype(int)))) < max_tile_count
            ]
            removed = before - len(samples)
            if removed > 0:
                logger.info("POKEMON tileset filtering: %d → %d (%d removed, max_tile_count=%d)",
                            before, len(samples), removed, max_tile_count)

        # (2) Instruction 단어 수 필터링
        if handler_config.pokemon.enabled:
            min_words = handler_config.pokemon.min_instruction_words
            before = len(samples)
            samples = [
                s for s in samples
                if s.instruction is None or len(s.instruction.split()) >= min_words
            ]
            removed = before - len(samples)
            if removed > 0:
                logger.info("%s instruction filtering: %d → %d (%d removed, min_words=%d)",
                            game, before, len(samples), removed, min_words)

        # (3) 회전 증강
        if handler_config.augmentation.enabled:
            should_augment = (
                (game == "pokemon" and handler_config.pokemon.rotate_90) or
                (game == "dungeon" and handler_config.dungeon.rotate_90) or
                (game in ("doom", "doom2") and handler_config.doom.rotate_90) or
                (game == "zelda" and handler_config.zelda.rotate_90)
            )
            if should_augment:
                rotated = [create_rotated_sample(s) for s in samples]
                samples = samples + rotated
                logger.info("%s augmentation: %d rotated samples added → %d total",
                            game, len(rotated), len(samples))

        # (4) 증강 후 max_samples 재적용
        max_s: Optional[int] = None
        if game == "pokemon":
            max_s = handler_config.pokemon.max_samples
        elif game in ("doom", "doom2"):
            max_s = handler_config.doom.max_samples
        elif game == "zelda":
            max_s = handler_config.zelda.max_samples
        elif game == "dungeon":
            max_s = handler_config.dungeon.max_samples
        if max_s is not None and len(samples) > max_s:
            logger.info("%s post-augmentation limit: %d → %d", game, len(samples), max_s)
            samples = samples[:max_s]

        return samples

    def _load_game_from_source(
        self, game: str, game_root: str, handler_config: HandlerConfig
    ) -> Optional[List[GameSample]]:
        """원본 데이터셋에서 게임 샘플을 로드한다. 실패 시 None 반환."""
        root = Path(game_root)
        if not root.exists():
            return None

        samples: List[GameSample] = []
        try:
            if game == "dungeon":
                self._dungeon_handler = DungeonHandler(root=game_root)
                for sample in self._dungeon_handler:
                    samples.append(sample)

            elif game == "sokoban":
                self._sokoban_handler = BoxobanHandler(root=game_root)
                for sample in self._sokoban_handler:
                    samples.append(sample)

            elif game == "zelda":
                self._zelda_handler = ZeldaHandler(root=game_root, handler_config=handler_config)
                for sample in self._zelda_handler:
                    samples.append(sample)
                if samples:
                    logger.info("Zelda: Loaded %d rooms", len(samples))

            elif game == "pokemon":
                self._pokemon_handler = POKEMONHandler(root=game_root, handler_config=handler_config)
                valid_ids, filtered_ratio, filtered_count = \
                    self._pokemon_handler.list_entries_with_filtering(
                        max_tile_ratio=handler_config.pokemon.max_tile_ratio,
                        max_tile_count=handler_config.pokemon.max_tile_count,
                    )
                for source_id in valid_ids:
                    sample = self._pokemon_handler.load_sample(source_id)
                    samples.append(sample)
                total_filtered = filtered_ratio + filtered_count
                if total_filtered > 0:
                    total_pokemon = len(valid_ids) + total_filtered
                    logger.info("POKEMON: Filtered %d → %d samples (%d removed)",
                                total_pokemon, len(valid_ids), total_filtered)

            elif game in ("doom", "doom2"):
                handler = DoomHandler(root=game_root, handler_config=handler_config)
                if game == "doom":
                    self._doom_handler = handler
                for sample in handler:
                    samples.append(sample)
                if samples:
                    logger.info("%s: Loaded %d samples", game.upper(), len(samples))

            else:
                logger.warning("Unknown game: %s", game)
                return None

        except (FileNotFoundError, ValueError) as e:
            logger.warning("Could not load %s dataset: %s", game, e)
            return None

        return samples if samples else None

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


        filtered_count = original_count - len(self._samples)
        if filtered_count > 0:
            logger.info("Instruction filtering: %d → %d samples (%d removed, min_words=%d)",
                        original_count, len(self._samples), filtered_count,
                        self._handler_config.pokemon.min_instruction_words)

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
            logger.info("POKEMON tileset filtering: %d → %d samples (%d removed, max_tile_count_threshold=250)",
                        original_pokemon_count,
                        len([s for s in self._samples if s.game == 'pokemon']),
                        pokemon_filtered_count)

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
            elif sample.game == GameTag.ZELDA and self._handler_config.zelda.rotate_90:
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
            logger.info("Data augmentation: %d → %d samples (added %d rotated versions)",
                        original_count, len(self._samples), len(rotated_samples))

        # ── 증강 후 각 게임별 제한 (handler_config의 max_samples 참조) ────────────
        game_sample_counts = {}
        filtered_samples = []

        for sample in self._samples:
            game = sample.game
            if game not in game_sample_counts:
                game_sample_counts[game] = 0

            # 각 게임의 handler_config에서 max_samples 가져오기
            max_samples = None
            if game == "pokemon":
                max_samples = self._handler_config.pokemon.max_samples
            elif game == "doom":
                max_samples = self._handler_config.doom.max_samples
            elif game == "zelda":
                max_samples = self._handler_config.zelda.max_samples
            elif game == "dungeon":
                max_samples = self._handler_config.dungeon.max_samples
            # sokoban은 handler_config에 설정이 없으므로 제한하지 않음

            # max_samples 제한 확인
            if max_samples is None or game_sample_counts[game] < max_samples:
                filtered_samples.append(sample)
                game_sample_counts[game] += 1

        # 필터링된 샘플이 있으면 적용
        if len(filtered_samples) < len(self._samples):
            self._samples = filtered_samples

            # order 재지정
            for i, sample in enumerate(self._samples):
                sample.order = i

            logger.info("Game-wise limit (per config): %d → %d samples",
                        original_count, len(self._samples))
            for game, count in sorted(game_sample_counts.items()):
                limited_count = count
                max_samples = None
                if game == "pokemon":
                    max_samples = self._handler_config.pokemon.max_samples
                elif game == "doom":
                    max_samples = self._handler_config.doom.max_samples
                elif game == "zelda":
                    max_samples = self._handler_config.zelda.max_samples
                elif game == "dungeon":
                    max_samples = self._handler_config.dungeon.max_samples

                if max_samples is not None:
                    limited_count = min(count, max_samples)
                    if limited_count < count:
                        logger.info("  %s: %d → %d (max_samples=%d)", game, count, limited_count, max_samples)
                    else:
                        logger.info("  %s: %d (max_samples=%d)", game, count, max_samples)
                else:
                    logger.info("  %s: %d (no limit)", game, count)

    # ── ann.json 기반 annotation 자동 로드 ─────────────────────────────────────

    def _ensure_and_load_all_annotations(self) -> None:
        """모든 게임의 ann.json을 확인·생성하고 샘플에 부착한다.

        ann.json이 없으면 compute_game_annotations()로 자동 계산 후 저장.
        이미 있으면 그대로 로드.
        """
        import time as _time

        games = list(self._game_cache_keys.items())
        logger.debug("[Annotation] Starting: %d game(s) to process (%s)",
                    len(games), ", ".join(g for g, _ in games))

        total_attached = 0
        for game, cache_key in games:
            existing = load_game_annotations_from_cache(self._cache_dir, game, cache_key)
            if existing is None:
                # ann.json 없음: 자동 계산
                game_samples = [s for s in self._samples if s.game == game]
                if not game_samples:
                    logger.info("[Annotation][%s] No samples — skipping", game)
                    continue
                logger.info("[Annotation][%s] ann.json not found — computing measures (%d samples)",
                            game, len(game_samples))
                t0 = _time.perf_counter()
                try:
                    # JAX 의존 모듈 lazy import
                    from dataset.reward_annotations.annotate import compute_game_annotations
                    rows = compute_game_annotations(game_samples, game)
                except Exception as exc:
                    logger.warning("[Annotation][%s] Computation failed: %s — skipping", game, exc)
                    continue
                elapsed = _time.perf_counter() - t0
                logger.info("[Annotation][%s] Computation done: %d rows  [%.1fs]",
                            game, len(rows), elapsed)
                save_game_annotations_to_cache(
                    self._cache_dir, game, cache_key, rows,
                    has_instructions=False,
                    n_samples=len(game_samples),
                )
                existing = load_game_annotations_from_cache(self._cache_dir, game, cache_key)
                if existing is None:
                    logger.warning("[Annotation][%s] Failed to reload after save — skipping", game)
                    continue
                # 신규 생성 시 .json에 ann_keys 기록
                update_json_with_ann_keys(self._cache_dir, game, cache_key, existing)
            else:
                n_rows = len(existing.get("annotations", []))
                has_instr = existing.get("has_instructions", False)
                logger.debug("[Annotation][%s] ann.json cache hit: %d rows, has_instructions=%s",
                            game, n_rows, has_instr)
                # ann_keys가 .json에 없으면 기록 (기존 캐시 호환)
                meta_path = self._cache_dir / game / f"{cache_key}.json"
                if meta_path.exists():
                    import json as _json
                    first = _json.loads(meta_path.read_text())[0] if meta_path.stat().st_size > 2 else {}
                    if "ann_keys" not in first:
                        update_json_with_ann_keys(self._cache_dir, game, cache_key, existing)
                if not has_instr:
                    self._try_submit_instruction_batch(game, cache_key, existing)

            before = len(self._samples)
            self._attach_annotations_from_cache(game, existing)
            added = len(self._samples) - before
            total_attached += added

        logger.debug("[Annotation] Done: total samples %d (replicas added %d)",
                    len(self._samples), total_attached)

    def _try_submit_instruction_batch(
        self, game: str, cache_key: str, ann_data: Dict[str, Any]
    ) -> None:
        """instruction이 없는 게임의 배치를 OpenAI Batch API에 제출한다.

        - ann.json에 batch_id가 이미 있으면 제출 건너뜀 (완료 대기 중).
        - OPENAI_API_KEY 환경 변수 없으면 건너뜀.
        - 제출 성공 시 batch_id를 ann.json에 기록.
        """
        import os

        # 이미 배치 제출됨 → 상태 확인 후 완료 시 자동 수령
        existing_batch_id = ann_data.get("batch_id")
        if existing_batch_id:
            try:
                from dataset.reward_annotations.generate_instructions import (
                    check_batch_status,
                    retrieve_batch_results,
                    update_caches,
                )
                status_info = check_batch_status(existing_batch_id)
                status = status_info["status"]
                counts = status_info["request_counts"]
                logger.info(
                    "[Instruction][%s] Checking batch status: batch_id=%s  status=%s  "
                    "(%d/%d completed)",
                    game, existing_batch_id, status,
                    counts["completed"], counts["total"],
                )
                if status == "completed":
                    logger.info("[Instruction][%s] Batch completed — retrieving results...", game)
                    results = retrieve_batch_results(existing_batch_id)
                    n = update_caches(results, self._cache_dir, [game])
                    logger.info("[Instruction][%s] %d instructions applied to ann.json", game, n)
                    # ann.json 재로드하여 existing 갱신 (부착 시 최신 데이터 사용)
                    updated = load_game_annotations_from_cache(self._cache_dir, game, cache_key)
                    if updated is not None:
                        ann_data.clear()
                        ann_data.update(updated)
                elif status in ("failed", "expired", "cancelled"):
                    logger.warning(
                        "[Instruction][%s] Batch %s — re-submission required (batch_id=%s)",
                        game, status, existing_batch_id,
                    )
                else:
                    logger.info("[Instruction][%s] Batch in progress (status=%s) — will retry next run",
                                game, status)
            except Exception as exc:
                logger.warning("[Instruction][%s] Failed to check batch status: %s", game, exc)
            return

        # API 키 없으면 건너뜀
        if not os.environ.get("OPENAI_API_KEY"):
            logger.warning(
                "[Instruction][%s] OPENAI_API_KEY not set — skipping instruction generation "
                "(run generate_instructions.py --submit --games %s after setting the key)",
                game, game,
            )
            return

        logger.info("[Instruction][%s] No instructions found — submitting batch", game)
        try:
            from dataset.reward_annotations.generate_instructions import (
                fill_none_instructions,
                build_jsonl,
                submit_batch,
                load_system_prompt,
            )
            from dataset.reward_annotations.annotate import _shorten_source_id

            enums = list(range(5))
            cache_dir = self._cache_dir

            # threshold=None 행 미리 채우기
            fill_none_instructions([game], enums, cache_dir)

            # source_id → array 맵 구성 (shortened key 사용)
            cache_by_game: Dict[str, Dict[str, Any]] = {}
            for s in self._samples:
                if s.game == game:
                    sid = _shorten_source_id(s.source_id, game)
                    cache_by_game.setdefault(game, {})[sid] = s.array

            system_prompt = load_system_prompt()
            jsonl_path = build_jsonl(
                [game], enums, cache_dir, cache_by_game, system_prompt
            )
            if jsonl_path is None:
                logger.info("[Instruction][%s] No pending requests (all already filled)", game)
                return

            n_requests = sum(1 for _ in jsonl_path.read_text(encoding="utf-8").splitlines() if _.strip())
            batch_id = submit_batch(jsonl_path, [game], enums, n_requests)
            logger.info("[Instruction][%s] Batch submitted: batch_id=%s (%d requests)",
                        game, batch_id, n_requests)

            # ann.json에 batch_id 기록
            update_ann_batch_id(cache_dir, game, cache_key, batch_id)

        except Exception as exc:
            logger.warning("[Instruction][%s] Batch submission failed: %s", game, exc)

    def _attach_annotations_from_cache(self, game: str, ann_data: Dict[str, Any]) -> None:
        """ann.json 데이터를 게임 샘플에 reward_enum별로 복제·부착한다.

        ann_keys 기반 매핑 (샘플 meta["ann_keys"] → ann.json 행 직접 조회).
        ann_keys 없는 구 포맷은 index 산술로 fallback.
        """
        import dataclasses
        import time as _time

        all_rows: List[Dict[str, Any]] = ann_data.get("annotations", [])
        if not all_rows:
            logger.warning("[Annotation][%s] No annotations in ann.json — skipping", game)
            return

        # key → ann row 딕셔너리 (빠른 조회)
        ann_by_key: Dict[str, Dict[str, Any]] = {r["key"]: r for r in all_rows}

        game_samples = [s for s in self._samples if s.game == game]
        n_samples = len(game_samples)
        if n_samples == 0:
            logger.warning("[Annotation][%s] No loaded samples — skipping", game)
            return

        # fallback: index 산술용 정렬 행
        sorted_rows = sorted(all_rows, key=lambda r: r["key"])
        n_rewards = len(sorted_rows) // n_samples if n_samples else 0
        if n_rewards == 0:
            logger.warning("[Annotation][%s] rows(%d) < samples(%d) — skipping",
                           game, len(all_rows), n_samples)
            return

        t0 = _time.perf_counter()
        attached = 0
        instr_count = 0
        new_samples: List[GameSample] = []

        for i, sample in enumerate(game_samples):
            # ann_keys 기반 (신규 포맷)
            ann_keys: Optional[List[str]] = sample.meta.get("ann_keys")
            if ann_keys:
                ann_list = [ann_by_key[k] for k in ann_keys if k in ann_by_key]
            else:
                # 구 포맷 fallback: index 산술
                ann_list = [sorted_rows[r * n_samples + i]
                            for r in range(n_rewards)
                            if r * n_samples + i < len(sorted_rows)]

            for r, ann in enumerate(ann_list):
                if r == 0:
                    target = sample
                else:
                    target = dataclasses.replace(sample, meta=dict(sample.meta))
                    new_samples.append(target)
                target.meta["key"]           = ann["key"]
                target.meta["reward_enum"]   = int(ann["reward_enum"])
                target.meta["feature_name"]  = ann["feature_name"]
                target.meta["sub_condition"] = ann.get("sub_condition", "")
                conditions: Dict[int, float] = {}
                for ci in range(5):
                    val = ann.get(f"condition_{ci}")
                    if val is not None:
                        conditions[ci] = float(val)
                target.meta["conditions"] = conditions
                # instruction_raw / instruction_uni 분리 저장
                raw = ann.get("instruction_raw")
                uni = ann.get("instruction_uni")
                target.meta["instruction_raw"] = str(raw) if raw and str(raw) != "None" else None
                target.meta["instruction_uni"] = str(uni) if uni and str(uni) != "None" else None
                # instruction 필드: instruction_uni 우선, 없으면 instruction_raw
                instr = target.meta["instruction_uni"] or target.meta["instruction_raw"]
                target.instruction = instr if instr else None
                if instr:
                    instr_count += 1
                attached += 1

        if new_samples:
            self._samples.extend(new_samples)
        elapsed = _time.perf_counter() - t0

        logger.debug(
            "[Annotation][%s] Attached: %d samples x %d enums = %d rows "
            "(original %d + replicas %d) | instruction=%d/%d  [%.3fs]",
            game, n_samples, n_rewards, attached,
            n_samples, len(new_samples),
            instr_count, attached, elapsed,
        )

    def _load_reward_annotations(self, annotations_dir: Path) -> None:
        """
        reward_annotations 폴더에서 CSV 파일을 읽어 해당 게임 샘플의 meta에
        reward annotation 정보를 부착한다.
        - {game}_reward_annotations.csv         : per-sample 실제 annotation
            → 각 샘플을 reward 수만큼 복제하여 reward_enum별 샘플 생성
        - {game}_reward_annotations_placeholder.csv : 게임 단위 더미 annotation
            → conditions 접근 시 WARNING 로그 출력
        """
        import dataclasses

        # ── per-sample CSV가 있는 게임: key 순서 기반으로 샘플을 reward 수만큼 복제 ──
        # CSV 구조: key 순서로 정렬 시 [reward0: sample0..N-1, reward1: sample0..N-1, ...]
        for csv_path in sorted(annotations_dir.glob("*_reward_annotations.csv")):
            game_name = csv_path.name.replace("_reward_annotations.csv", "")

            # key 순서대로 모든 행 로드
            all_rows: List[Dict[str, Any]] = []
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    all_rows.append(row)
            all_rows.sort(key=lambda r: r["key"])

            # 이 게임의 로드된 샘플 목록 (순서 유지)
            game_samples = [s for s in self._samples if s.game == game_name]
            n_samples = len(game_samples)
            if n_samples == 0 or len(all_rows) == 0:
                continue

            # CSV 행 수 / 샘플 수 = reward 수
            n_rewards = len(all_rows) // n_samples
            if n_rewards == 0:
                logger.warning("Reward annotations [%s]: CSV rows (%d) < samples (%d), skipped",
                               game_name, len(all_rows), n_samples)
                continue

            # sample_index i, reward_block r → CSV row: all_rows[r * n_samples + i]
            attached = 0
            new_samples: List[GameSample] = []
            for i, sample in enumerate(game_samples):
                for r in range(n_rewards):
                    row_idx = r * n_samples + i
                    if row_idx >= len(all_rows):
                        break
                    ann = all_rows[row_idx]
                    if r == 0:
                        target = sample
                    else:
                        target = dataclasses.replace(sample, meta=dict(sample.meta))
                        new_samples.append(target)
                    target.meta["key"] = ann["key"]
                    target.meta["reward_enum"] = int(ann["reward_enum"])
                    target.meta["feature_name"] = ann["feature_name"]
                    target.meta["sub_condition"] = ann["sub_condition"]
                    conditions: Dict[int, float] = {}
                    for ci in range(0, 5):
                        val = ann.get(f"condition_{ci}", "")
                        if val != "":
                            conditions[ci] = float(val)
                    target.meta["conditions"] = conditions
                    # instruction_uni를 직접 읽음 ("None" 또는 빈 값이면 None 처리)
                    instr = ann.get("instruction_uni", "").strip()
                    target.instruction = instr if instr and instr != "None" else None
                    attached += 1

            if new_samples:
                self._samples.extend(new_samples)
            if attached > 0:
                logger.info("Reward annotations [%s]: %d samples × %d rewards = %d attached "
                            "(%d original + %d duplicated)",
                            game_name, n_samples, n_rewards, attached,
                            n_samples, len(new_samples))

        # ── placeholder CSV: per-sample CSV가 없는 게임에만 적용 ──────────
        for ph_csv in sorted(annotations_dir.glob("*_reward_annotations_placeholder.csv")):
            game_name = ph_csv.name.replace("_reward_annotations_placeholder.csv", "")
            # per-sample CSV가 이미 있으면 스킵
            if (annotations_dir / f"{game_name}_reward_annotations.csv").exists():
                continue
            ph_features: list[Dict[str, Any]] = []
            with open(ph_csv, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    reward_enum = int(row["reward_enum"])
                    conditions: Dict[int, float] = {}
                    for i in range(1, 6):
                        val = row.get(f"condition_{i}", "")
                        if val != "":
                            conditions[i] = float(val)
                    ph_features.append({
                        "reward_enum":  reward_enum,
                        "feature_name": row["feature_name"],
                        "sub_condition": row["sub_condition"],
                        "conditions":   conditions,
                    })
            if not ph_features:
                continue
            all_conditions: Dict[int, float] = {}
            for feat in ph_features:
                all_conditions.update(feat["conditions"])
            game_attached = 0
            for sample in self._samples:
                if sample.game != game_name:
                    continue
                sample.meta["reward_enum"]  = ph_features[0]["reward_enum"]
                sample.meta["feature_name"] = ph_features[0]["feature_name"]
                sample.meta["sub_condition"] = ph_features[0]["sub_condition"]
                sample.meta["conditions"] = _WarningConditionsDict(
                    all_conditions,
                    game=game_name,
                    logger=logger,
                )
                game_attached += 1
            if game_attached > 0:
                logger.info("Reward annotations (placeholder): attached to %d %s samples",
                            game_attached, game_name)


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

    # ── reward annotation 기반 필터 ──────────────────────────────────────────
    def by_reward_enum(self, reward_enum: int) -> List[GameSample]:
        """reward_enum 값으로 필터링 (1=region, 2=path_length, 3=block, 4=bat_amount, 5=bat_direction)."""
        return [self._apply_mapping(s)
                for s in self._samples
                if s.meta.get("reward_enum") == reward_enum]

    def by_feature_name(self, feature_name: str) -> List[GameSample]:
        """feature_name으로 필터링 (region, path_length, block, bat_amount, bat_direction)."""
        return [self._apply_mapping(s)
                for s in self._samples
                if s.meta.get("feature_name") == feature_name]

    def with_reward_annotation(self) -> List[GameSample]:
        """reward annotation이 있는 샘플만 반환."""
        return [self._apply_mapping(s)
                for s in self._samples
                if "reward_enum" in s.meta]

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
        """등록된 게임 목록 반환."""
        return [GameTag.DUNGEON, GameTag.SOKOBAN, GameTag.DOOM, GameTag.POKEMON, GameTag.ZELDA]

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
