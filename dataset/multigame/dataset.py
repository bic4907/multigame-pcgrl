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
    load_game_samples_from_cache,
    save_game_samples_to_cache,
    load_any_game_cache,
    # legacy (하위 호환)
    build_cache_key,
    load_samples_from_cache,
    save_samples_to_cache,
)
from .tile_utils import to_unified, render_unified_rgb, game_mapping_rows

_HERE = Path(__file__).parent
_DEFAULT_REWARD_ANNOTATIONS_DIR = _HERE.parent / "reward_annotations"

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
        include_dungeon:  bool = True,
        include_pokemon:  bool = True,
        include_sokoban:  bool = False,
        include_doom:     bool = False,
        include_doom2:    bool = False,
        include_zelda:    bool = False,
        use_cache:        bool = True,
        cache_dir:        Path | str | None = None,
        use_tile_mapping: bool = True,
        handler_config:   Optional[HandlerConfig] = None,
        reward_annotations_dir: Path | str | None = _DEFAULT_REWARD_ANNOTATIONS_DIR,
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
        self._zelda_handler: Optional[ZeldaHandler] = None
        self._zelda_handler: Optional[ZeldaHandler] = None

        if cache_dir is None:
            cache_dir = _HERE / "cache" / "artifacts"
        cache_dir = Path(cache_dir)
        self._cache_dir = cache_dir
        self._use_cache = use_cache

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
        if include_doom:
            _game_specs.append(("doom", str(doom_root), hc.get("doom", {})))
        if include_doom2:
            _game_specs.append(("doom2", str(doom2_root), hc.get("doom", {})))

        # ── 게임별 로드 루프 ─────────────────────────────────────────────────
        for game, game_root, game_hc in _game_specs:
            cache_key = build_per_game_cache_key(game, game_root, game_hc)

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
                    continue

            # (4) 아무것도 없으면 경고만 출력
            logger.warning("%s: no source data and no cache — skipped", game)

        # ── 글로벌 후처리 (캐시에 저장하지 않고 매번 런타임 적용) ──────────────
        if self._handler_config.pokemon.enabled:
            self._apply_pokemon_tileset_filtering()

        if self._handler_config.pokemon.enabled:
            self._apply_instruction_filtering()

        if self._handler_config.augmentation.enabled:
            self._augment_with_rotations_per_game()

        if reward_annotations_dir is not None:
            self._load_reward_annotations(Path(reward_annotations_dir))


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
            if s.instruction is None or len(s.instruction.split()) >= self._handler_config.pokemon.min_instruction_words
        ]

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

    def _load_reward_annotations(self, annotations_dir: Path) -> None:
        """
        reward_annotations 폴더에서 CSV 파일을 읽어 해당 게임 샘플의 meta에
        reward annotation 정보를 부착한다.
        - {game}_reward_annotations.csv         : per-sample 실제 annotation
        - {game}_reward_annotations_placeholder.csv : 게임 단위 더미 annotation
            → conditions 접근 시 WARNING 로그 출력
        reward_enum은 모든 게임 통일 1~5:
          1=region / 2=path_length / 3=block / 4=bat_amount / 5=bat_direction
        """
        # ── dungeon: per-sample CSV ───────────────────────────────────────
        dungeon_csv = annotations_dir / "dungeon_reward_annotations.csv"
        if dungeon_csv.exists():
            annotation_map: Dict[str, Dict[str, Any]] = {}
            with open(dungeon_csv, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    annotation_map[row["key"]] = row
            attached = 0
            for sample in self._samples:
                if sample.game != GameTag.DUNGEON:
                    continue
                ann = annotation_map.get(sample.source_id)
                if ann is None:
                    continue
                sample.meta["reward_enum"] = int(ann["reward_enum"])
                sample.meta["feature_name"] = ann["feature_name"]
                sample.meta["sub_condition"] = ann["sub_condition"]
                conditions: Dict[int, float] = {}
                for i in range(1, 6):
                    val = ann.get(f"condition_{i}", "")
                    if val != "":
                        conditions[i] = float(val)
                sample.meta["conditions"] = conditions
                attached += 1
            if attached > 0:
                logger.info("Reward annotations: attached to %d dungeon samples", attached)
        # ── 나머지 게임: *_placeholder.csv 읽어서 game-level 적용 ─────────
        # 파일명에 _placeholder가 포함된 CSV를 자동 탐지
        for ph_csv in sorted(annotations_dir.glob("*_reward_annotations_placeholder.csv")):
            game_name = ph_csv.name.replace("_reward_annotations_placeholder.csv", "")
            # CSV에서 feature 목록 파싱
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
            # 모든 조건을 합친 dict (placeholder 전체를 한 번에)
            all_conditions: Dict[int, float] = {}
            for feat in ph_features:
                all_conditions.update(feat["conditions"])
            game_attached = 0
            for sample in self._samples:
                if sample.game != game_name:
                    continue
                # 기본 reward_enum = 첫 번째 feature
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


    def _load_reward_annotations(self, annotations_dir: Path) -> None:
        """
        reward_annotations 폴더에서 CSV 파일을 읽어 해당 게임 샘플의 meta에
        reward annotation 정보를 부착한다.
        - {game}_reward_annotations.csv         : per-sample 실제 annotation
        - {game}_reward_annotations_placeholder.csv : 게임 단위 더미 annotation
            → conditions 접근 시 WARNING 로그 출력
        reward_enum은 모든 게임 통일 1~5:
          1=region / 2=path_length / 3=block / 4=bat_amount / 5=bat_direction
        """
        # ── dungeon: per-sample CSV ───────────────────────────────────────
        dungeon_csv = annotations_dir / "dungeon_reward_annotations.csv"
        if dungeon_csv.exists():
            annotation_map: Dict[str, Dict[str, Any]] = {}
            with open(dungeon_csv, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    annotation_map[row["key"]] = row
            attached = 0
            for sample in self._samples:
                if sample.game != GameTag.DUNGEON:
                    continue
                ann = annotation_map.get(sample.source_id)
                if ann is None:
                    continue
                sample.meta["reward_enum"] = int(ann["reward_enum"])
                sample.meta["feature_name"] = ann["feature_name"]
                sample.meta["sub_condition"] = ann["sub_condition"]
                conditions: Dict[int, float] = {}
                for i in range(1, 6):
                    val = ann.get(f"condition_{i}", "")
                    if val != "":
                        conditions[i] = float(val)
                sample.meta["conditions"] = conditions
                attached += 1
            if attached > 0:
                logger.info("Reward annotations: attached to %d dungeon samples", attached)
        # ── 나머지 게임: *_placeholder.csv 읽어서 game-level 적용 ─────────
        # 파일명에 _placeholder가 포함된 CSV를 자동 탐지
        for ph_csv in sorted(annotations_dir.glob("*_reward_annotations_placeholder.csv")):
            game_name = ph_csv.name.replace("_reward_annotations_placeholder.csv", "")
            # CSV에서 feature 목록 파싱
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
            # 모든 조건을 합친 dict (placeholder 전체를 한 번에)
            all_conditions: Dict[int, float] = {}
            for feat in ph_features:
                all_conditions.update(feat["conditions"])
            game_attached = 0
            for sample in self._samples:
                if sample.game != game_name:
                    continue
                # 기본 reward_enum = 첫 번째 feature
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
        if not apply_filter or not self._handler_config.pokemon.enabled:
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
