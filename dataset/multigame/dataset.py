"""
dataset/multigame/dataset.py
============================
MultiGameDataset: Dungeon + POKEMON + Sokoban + DOOM text dataset class.

text  of text: numpy (Pillow  rendering text in text text).

Example
-------
    from dataset.multigame import MultiGameDataset

    # use_tile_mapping=True (default value): text sample array  unified 7-category to  converttext return
    ds = MultiGameDataset(include_dungeon=True, include_pokemon=True, include_doom=True)
    sample = ds[0]
    # sample.array text range: [0, 6]  (unified category index)

    # use_tile_mapping=False: text gametext tile_id as-is return
    ds_raw = MultiGameDataset(use_tile_mapping=False)
    sample_raw = ds_raw[0]
    # sample_raw.array text: game text integer tile_id

    # text text (dataset textload text   before text)
    ds.use_tile_mapping = False   #   after  __getitem__ / __iter__ text return
    ds.use_tile_mapping = True    # text unified return

    # filter
    dungeon_samples = ds.by_game("dungeon")
    pokemon_samples = ds.by_game("pokemon")
    doom_samples = ds.by_game("doom")

    # rendering (use_tile_mapping config automatic apply)
    ds.render(sample, save_path="out.png")
    ds.render_grid(dungeon_samples[:8], save_path="grid.png")
"""
from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union

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
    # legacy (sub text)
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
    text per-sample annotation  without game in  conditions text in  text
    WARNING  to text  text.  (gametext 1text)
    """
    _warned_games: set = set()  # class text:  text warningtext game name

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
    Dungeon + Sokoban(Boxoban) + POKEMON + DOOM text dataset class.

    Parameters
    ----------
    dungeon_root     : dungeon_level_dataset text path
    pokemon_root     : Five-Dollar-Model text path
    sokoban_root     : boxoban_levels text path
    doom_root        : doom_levels text path
    include_dungeon  : Dungeon dataset text text
    include_pokemon  : POKEMON dataset text text
    include_sokoban  : Sokoban dataset text text
    include_doom     : DOOM dataset text text
    use_tile_mapping : True(default)text array  unified 7-category to  converttext return.
                       Falsetext text tile_id as-is return.
                       load   after  in  also  text as  text text available.
    handler_config   : HandlerConfig text. None text default value text for .
                       (gametext preprocessing config text, augmentation config also  text)
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
        instruction_field:    str = "uni",   # "uni" = instruction_uni, "raw" = instruction_raw
        # sub text: text parametertext text
        boxoban_root:         Path | str | None = None,
        include_boxoban:      bool | None = None,
    ) -> None:
        self.use_tile_mapping: bool = use_tile_mapping
        self._instruction_field: str = instruction_field  # "uni" or "raw"

        # sub text process
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

        # gametext cache text text (annotation load in  text for )
        self._game_cache_keys: Dict[str, str] = {}

        hc = handler_config.to_dict()

        # ── gametext load config (game, include, root, handler_config_sub) ────────
        _game_specs = []
        if include_dungeon:
            _game_specs.append(("dungeon", str(dungeon_root), hc.get("dungeon", {})))
        if include_sokoban:
            _game_specs.append(("sokoban", str(sokoban_root), hc.get("sokoban", {})))
        if include_zelda:
            _game_specs.append(("zelda", str(zelda_root), hc.get("zelda", {})))
        if include_pokemon:
            _game_specs.append(("pokemon", str(pokemon_root), hc.get("pokemon", {})))
        # doom/doom2  text process (text  after  separate text in )

        # ── gametext load text ─────────────────────────────────────────────────
        for game, game_root, game_hc in _game_specs:
            cache_key = build_per_game_cache_key(game, game_root, game_hc)
            logger.debug("[%s] cache key: %s", game, cache_key[:12])
            self._game_cache_keys[game] = cache_key
            # (1) per-game cache text text also
            if use_cache:
                cached = load_game_samples_from_cache(cache_dir, game, cache_key)
                if cached is not None:
                    for s in cached:
                        s.order = len(self._samples)
                        self._samples.append(s)
                    continue

            # (2) text dataset in  load
            game_samples = self._load_game_from_source(
                game, game_root, handler_config
            )

            if game_samples is not None:
                # cache save  before  max_samples apply
                max_s = game_hc.get("max_samples") if isinstance(game_hc, dict) else getattr(game_hc, "max_samples", None)
                if max_s is not None and len(game_samples) > max_s:
                    game_samples = game_samples[:max_s]
                # cache save  before  filtering + augmentation apply (viewer/annotate text sametext text  text also text)
                game_samples = self._postprocess_game_samples(game, game_samples, handler_config)
                for s in game_samples:
                    s.order = len(self._samples)
                    self._samples.append(s)
                # cache in  save
                if use_cache:
                    save_game_samples_to_cache(
                        cache_dir, game, cache_key, game_samples
                    )
                continue

            # (3) artifact-only fallback: text text text game cache  text load
            if use_cache:
                fallback = load_any_game_cache(cache_dir, game)
                if fallback is not None:
                    logger.info("%s: artifact-only fallback (%d samples from existing cache)",
                                game, len(fallback))
                    for s in fallback:
                        s.order = len(self._samples)
                        self._samples.append(s)
                    # fallback text text loadtext file of  text to  text
                    actual_key = find_game_cache_key(cache_dir, game)
                    if actual_key:
                        self._game_cache_keys[game] = actual_key
                    continue

            # (4) text also  if missing warningtext text
            logger.warning("%s: no source data and no cache — skipped", game)

        # ── doom + doom2 text load (sum max_samples=1000 apply  after  cache save) ──
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
                        # fallback text text loadtext file of  text to  text
                        actual_key = find_game_cache_key(cache_dir, "doom")
                        if actual_key:
                            self._game_cache_keys["doom"] = actual_key

        # ── annotation automatic load (ann.json → sample text) ─────────────────────
        if use_cache and self._game_cache_keys:
            self._ensure_and_load_all_annotations()

        # ── raw counts write (max_samples_per_game apply  before , (game, reward_enum) basis) ──
        self._raw_game_re_counts: dict = {}
        for s in self._samples:
            re = s.meta.get("reward_enum")
            if re is not None:
                self._raw_game_re_counts[(s.game, re)] = self._raw_game_re_counts.get((s.game, re), 0) + 1

        # ── gametext text text sample text text (source_id basis, annotation text   after ) ──
        # source_id textabove to  selecttext to  text reward_enum text  text keeptext
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

        # ── N sample textsampletext (gametext, text based) ─────────────────────────
        if N >= 1:
            import random as _random
            _total = len(self._samples)
            _rng = _random.Random(42)
            _mask = [False] * _total
            # gametext index  text order keep to  text
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
        cache save  before  in  filtering and  augmentation  applytext.

        apply order:
        1. Pokemon tiletext filtering (max_tile_count exceed sample remove)
        2. Instruction text text filtering (min_instruction_words less than remove)
        3. rotate augmentation (rotate_90 config text 90 also  rotate text text )
        4. augmentation  after  max_samples textapply
        """
        # (1) Pokemon tiletext filtering
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

        # (2) Instruction text text filtering
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

        # (3) rotate augmentation
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

        # (4) augmentation  after  max_samples textapply
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
        """text dataset in  game sample  loadtext. failure text None return."""
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
        Floor + empty count  floor_empty_max  text sampletext filtering
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
        POKEMON sampletext tiletext basis as  filtering.
        (padding  after  16x16 text in  text tile  250text or more text text)

        filtering basis:
        - POKEMON gametext target
        - text tile text  256text  during  250text or more text text (text map)
        """
        pokemon_indices = [i for i, s in enumerate(self._samples) if s.game == "pokemon"]

        if not pokemon_indices:
            return

        original_pokemon_count = len(pokemon_indices)
        filtered_samples = []

        for i, sample in enumerate(self._samples):
            if sample.game == "pokemon":
                # POKEMON sample: tiletext basis filtering
                flat = sample.array.ravel()
                tile_counts = np.bincount(flat.astype(int))
                max_tile_count = int(np.max(tile_counts)) if len(tile_counts) > 0 else 0

                # 256text tile  during  250text or more  same tile  text keep
                if max_tile_count < 250:
                    filtered_samples.append(sample)
            else:
                # different game: as-is keep
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
        gametext config in  text each game of  sample  rotatetext augmentation.

        each game of  config in  rotate_90 config  text text gametext rotate augmentation  textrowtext.
        text: config.pokemon.rotate_90 = Truetext POKEMON gametext rotate augmentation
        """
        original_count = len(self._samples)
        rotated_samples = []

        for sample in self._samples:
            # gametext config in  rotate_90 config check
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

        # text next in  rotate sample text
        self._samples.extend(rotated_samples)

        # order text
        for i, sample in enumerate(self._samples):
            sample.order = i

        if len(rotated_samples) > 0:
            logger.info("Data augmentation: %d → %d samples (added %d rotated versions)",
                        original_count, len(self._samples), len(rotated_samples))

        # ── augmentation  after  each gametext text (handler_config of  max_samples text) ────────────
        game_sample_counts = {}
        filtered_samples = []

        for sample in self._samples:
            game = sample.game
            if game not in game_sample_counts:
                game_sample_counts[game] = 0

            # each game of  handler_config in  max_samples  text
            max_samples = None
            if game == "pokemon":
                max_samples = self._handler_config.pokemon.max_samples
            elif game == "doom":
                max_samples = self._handler_config.doom.max_samples
            elif game == "zelda":
                max_samples = self._handler_config.zelda.max_samples
            elif game == "dungeon":
                max_samples = self._handler_config.dungeon.max_samples
            # sokoban  handler_config in  config  text to  text text

            # max_samples text check
            if max_samples is None or game_sample_counts[game] < max_samples:
                filtered_samples.append(sample)
                game_sample_counts[game] += 1

        # filteringtext sample  text apply
        if len(filtered_samples) < len(self._samples):
            self._samples = filtered_samples

            # order text
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

    # ── ann.json based annotation automatic load ─────────────────────────────────────

    def _ensure_and_load_all_annotations(self) -> None:
        """text game of  ann.json  check·createtext sample in  text.

        ann.json  if missing compute_game_annotations() to  automatic compute  after  save.
         text text as-is load.
        """
        import time as _time

        games = list(self._game_cache_keys.items())
        logger.debug("[Annotation] Starting: %d game(s) to process (%s)",
                    len(games), ", ".join(g for g, _ in games))

        total_attached = 0
        for game, cache_key in games:
            existing = load_game_annotations_from_cache(self._cache_dir, game, cache_key)
            if existing is None:
                # ann.json none: automatic compute
                game_samples = [s for s in self._samples if s.game == game]
                if not game_samples:
                    logger.info("[Annotation][%s] No samples — skipping", game)
                    continue
                logger.info("[Annotation][%s] ann.json not found — computing measures (%d samples)",
                            game, len(game_samples))
                t0 = _time.perf_counter()
                try:
                    # JAX  of text text lazy import
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
                # text create text .json in  ann_keys write
                update_json_with_ann_keys(self._cache_dir, game, cache_key, existing)
            else:
                n_rows = len(existing.get("annotations", []))
                has_instr = existing.get("has_instructions", False)
                logger.debug("[Annotation][%s] ann.json cache hit: %d rows, has_instructions=%s",
                            game, n_rows, has_instr)
                # ann_keys  .json in  if missing write (existing cache text)
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
        """instruction  without game of  batch  OpenAI Batch API in  text.

        - ann.json in  batch_id   text text text text (finish text  during ).
        - OPENAI_API_KEY text text if missing text.
        - text success text batch_id  ann.json in  write.
        """
        import os

        #  text batch text → text check  after  finish text automatic text
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
                    # ann.json textloadtext existing text (text text latest data text for )
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

        # API text if missing text
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

            # threshold=None row text text
            fill_none_instructions([game], enums, cache_dir)

            # source_id → array map text (shortened key text for )
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

            # ann.json in  batch_id write
            update_ann_batch_id(cache_dir, game, cache_key, batch_id)

        except Exception as exc:
            logger.warning("[Instruction][%s] Batch submission failed: %s", game, exc)

    def _attach_annotations_from_cache(self, game: str, ann_data: Dict[str, Any]) -> None:
        """ann.json data  game sample in  reward_enumby text·text.

        ann_keys based text (sample meta["ann_keys"] → ann.json row direct text).
        ann_keys without text text  index text to  fallback.
        """
        import dataclasses
        import time as _time

        all_rows: List[Dict[str, Any]] = ann_data.get("annotations", [])
        if not all_rows:
            logger.warning("[Annotation][%s] No annotations in ann.json — skipping", game)
            return

        # key → ann row dictionary (text text)
        ann_by_key: Dict[str, Dict[str, Any]] = {r["key"]: r for r in all_rows}

        game_samples = [s for s in self._samples if s.game == game]
        n_samples = len(game_samples)
        if n_samples == 0:
            logger.warning("[Annotation][%s] No loaded samples — skipping", game)
            return

        # fallback: index text for  sort row
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
            # ann_keys based (text text)
            ann_keys: Optional[List[str]] = sample.meta.get("ann_keys")
            if ann_keys:
                ann_list = [ann_by_key[k] for k in ann_keys if k in ann_by_key]
            else:
                # text text fallback: index text
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
                # instruction_raw / instruction_uni separate save
                raw = ann.get("instruction_raw")
                uni = ann.get("instruction_uni")
                target.meta["instruction_raw"] = str(raw) if raw and str(raw) != "None" else None
                target.meta["instruction_uni"] = str(uni) if uni and str(uni) != "None" else None
                # instruction text: instruction_field config in  text select
                if self._instruction_field == "raw":
                    instr = target.meta["instruction_raw"] or target.meta["instruction_uni"]
                else:
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
        reward_annotations folder in  CSV file  text text game sample of  meta in
        reward annotation info  text.
        - {game}_reward_annotations.csv         : per-sample text annotation
            → each sample  reward text text reward_enumtext sample create
        - {game}_reward_annotations_placeholder.csv : game textabove text annotation
            → conditions text text WARNING  to text text
        """
        import dataclasses

        # ── per-sample CSV  with game: key order based as  sample  reward text text ──
        # CSV structure: key order to  sort text [reward0: sample0..N-1, reward1: sample0..N-1, ...]
        for csv_path in sorted(annotations_dir.glob("*_reward_annotations.csv")):
            game_name = csv_path.name.replace("_reward_annotations.csv", "")

            # key ordertext to  text row load
            all_rows: List[Dict[str, Any]] = []
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    all_rows.append(row)
            all_rows.sort(key=lambda r: r["key"])

            #   game of  loadtext sample list (order keep)
            game_samples = [s for s in self._samples if s.game == game_name]
            n_samples = len(game_samples)
            if n_samples == 0 or len(all_rows) == 0:
                continue

            # CSV row text / sample text = reward text
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
                    # instruction_field config in  text raw text  uni select
                    if self._instruction_field == "raw":
                        raw_val = ann.get("instruction_raw", "").strip()
                        uni_val = ann.get("instruction_uni", "").strip()
                        instr = raw_val if raw_val and raw_val != "None" else uni_val
                    else:
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

        # ── placeholder CSV: per-sample CSV  without game in text apply ──────────
        for ph_csv in sorted(annotations_dir.glob("*_reward_annotations_placeholder.csv")):
            game_name = ph_csv.name.replace("_reward_annotations_placeholder.csv", "")
            # per-sample CSV   text text text
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
        use_tile_mapping config in  text array  converttext text GameSample  return.
        text _samples text  always raw tile_id  keeptext.
        """
        if not self.use_tile_mapping:
            return sample
        import dataclasses
        unified_array = to_unified(sample.array, sample.game, warn_unmapped=False)
        return dataclasses.replace(sample, array=unified_array)

    def _find_raw_sample(self, sample: GameSample) -> GameSample:
        """source_id/game basis as  internal raw sample  text returntext."""
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

    # ── text based filter ──────────────────────────────────────────────────────────
    def by_game(self, game: str) -> List[GameSample]:
        """text game sampletext return."""
        return [self._apply_mapping(s)
                for s in tag_utils.extract_by_game(self._samples, game)]

    def by_games(self, games: List[str]) -> List[GameSample]:
        """text game sample return."""
        return [self._apply_mapping(s)
                for s in tag_utils.extract_by_games(self._samples, games)]

    def by_instruction(
        self, keyword: str, *, case_sensitive: bool = False
    ) -> List[GameSample]:
        """instruction text filter."""
        return [self._apply_mapping(s)
                for s in tag_utils.extract_by_instruction(
                    self._samples, keyword, case_sensitive=case_sensitive)]

    def with_instruction(self) -> List[GameSample]:
        """instruction  with sampletext."""
        return [self._apply_mapping(s)
                for s in tag_utils.extract_with_instruction(self._samples)]

    def without_instruction(self) -> List[GameSample]:
        """instruction  without sampletext."""
        return [self._apply_mapping(s)
                for s in tag_utils.extract_without_instruction(self._samples)]

    def by_order(self, start: int, end: int) -> List[GameSample]:
        """order range [start, end) sample."""
        return [self._apply_mapping(s)
                for s in tag_utils.extract_by_order(self._samples, start, end)]

    def by_meta(self, key: str, value: Any) -> List[GameSample]:
        """meta text filter."""
        return [self._apply_mapping(s)
                for s in tag_utils.extract_by_meta(self._samples, key, value)]

    def filter(self, fn) -> List[GameSample]:
        """text of  condition function to  filtering."""
        return [self._apply_mapping(s)
                for s in tag_utils.extract_by_predicate(self._samples, fn)]

    # ── reward annotation based filter ──────────────────────────────────────────
    def by_reward_enum(self, reward_enum: int) -> List[GameSample]:
        """reward_enum text as  filtering (1=region, 2=path_length, 3=block, 4=bat_amount, 5=bat_direction)."""
        return [self._apply_mapping(s)
                for s in self._samples
                if s.meta.get("reward_enum") == reward_enum]

    def by_feature_name(self, feature_name: str) -> List[GameSample]:
        """feature_name as  filtering (region, path_length, block, bat_amount, bat_direction)."""
        return [self._apply_mapping(s)
                for s in self._samples
                if s.meta.get("feature_name") == feature_name]

    def with_reward_annotation(self) -> List[GameSample]:
        """reward annotation  with sampletext return."""
        return [self._apply_mapping(s)
                for s in self._samples
                if "reward_enum" in s.meta]

    # ── text ────────────────────────────────────────────────────────────────────
    def group_by_game(self) -> Dict[str, List[GameSample]]:
        return tag_utils.group_by_game(self._samples)

    def group_by_instruction(self) -> Dict[str, List[GameSample]]:
        return tag_utils.group_by_instruction(self._samples)

    def count_by_game(self) -> Dict[str, int]:
        return tag_utils.count_by_game(self._samples)

    def summary(self) -> Dict[str, Any]:
        return tag_utils.summary(self._samples)

    # ── rendering (Pillow text) ────────────────────────────────────────────────────
    def render(
        self,
        sample: GameSample,
        tile_size: int = 16,
        save_path: Optional[Path | str] = None,
    ):
        """
        text sample rendering.
        use_tile_mapping=True  text unified text text to , False  text text palette to  rendering.
        save_path text text PNG save, if missing PIL Image return.
        """
        from .render import render_sample_pil, save_rendered
        from .tile_utils import render_unified_rgb
        from PIL import Image

        if self.use_tile_mapping:
            # array   text unified to  converttext sample  text  text also  text
            # text raw sample  text  text also  text to  always mapping apply
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
        text sample text rendering.
        use_tile_mapping config automatic apply.
        save_path text text PNG save, if missing PIL Image return.
        """
        from .render import render_grid as _rg, save_grid
        from PIL import Image

        # text sample in  mapping apply
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
        text(raw) and  7-category mapped image  text to  text renderingtext.

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

    def render_sample(
        self,
        sample: GameSample,
        tile_size: int = 16,
        save_path: Optional[Union[str, Path]] = None,
        show_tile_numbers: bool = False
    ) -> "Image.Image":
        """
        sample  tile image to  renderingtext.

        Parameters
        ----------
        sample : GameSample text
        tile_size : tile size (textcell)
        save_path : save path
        show_tile_numbers : tile text tabletext text

        Returns
        -------
        PIL.Image.Image : renderingtext image

        Examples
        --------
        >>> ds = MultiGameDataset()
        >>> sample = ds[0]
        >>> img = ds.render_sample(sample, tile_size=20, show_tile_numbers=True)
        >>> img.save("level.png")
        """
        from .render import GameLevelRenderer
        renderer = GameLevelRenderer()
        return renderer.render(
            game=sample.game,
            level=sample.array,
            tile_size=tile_size,
            save_path=save_path,
            show_tile_numbers=show_tile_numbers
        )

    def render_level(
        self,
        game: str,
        level: np.ndarray,
        tile_size: int = 16,
        save_path: Optional[Union[str, Path]] = None,
        show_tile_numbers: bool = False
    ) -> "Image.Image":
        """
        game level  tile image to  direct renderingtext.

        Parameters
        ----------
        game : game name (dungeon, doom, pokemon, sokoban, zelda)
        level : 2D numpy array
        tile_size : tile size (textcell)
        save_path : save path
        show_tile_numbers : tile text tabletext text

        Returns
        -------
        PIL.Image.Image : renderingtext image

        Examples
        --------
        >>> ds = MultiGameDataset()
        >>> level = np.random.randint(1, 5, (16, 16))
        >>> img = ds.render_level("dungeon", level, tile_size=20)
        """
        from .render import GameLevelRenderer
        renderer = GameLevelRenderer()
        return renderer.render(
            game=game,
            level=level,
            tile_size=tile_size,
            save_path=save_path,
            show_tile_numbers=show_tile_numbers
        )

    def mapping_rows(self, game: str):
        """tile_mapping.json basis text tile -> unified text row list."""
        return game_mapping_rows(game)

    # ── utility ────────────────────────────────────────────────────────────────────
    def get_tags(self, idx: int) -> Dict[str, Any]:
        """index basis text dict return."""
        return tag_utils.build_tags(self._samples[idx])

    def all_tags(self) -> List[Dict[str, Any]]:
        """all sample text text."""
        return [tag_utils.build_tags(s) for s in self._samples]

    def available_games(self) -> List[str]:
        """text game list return."""
        return [GameTag.DUNGEON, GameTag.SOKOBAN, GameTag.DOOM, GameTag.POKEMON, GameTag.ZELDA]

    def sample(
        self,
        n: int,
        game: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> List[GameSample]:
        """
        random sampletext.

        Parameters
        ----------
        n    : sample text
        game : text gametext (None text all)
        seed : random seed
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
