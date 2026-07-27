"""
dataset/multigame/dataset.py
============================
MultiGameDataset: a unified dataset class for Dungeon, POKEMON, Sokoban, and DOOM.

External dependencies: numpy (Pillow is needed only for rendering).

Example
-------
    from dataset.multigame import MultiGameDataset

    # use_tile_mapping=True (default): sample arrays are converted to the unified categories
    ds = MultiGameDataset(include_dungeon=True, include_pokemon=True, include_doom=True)
    sample = ds[0]
    # sample.array values lie in [0, NUM_CATEGORIES-1] (unified category index)

    # use_tile_mapping=False: the raw per-game tile ids are returned unchanged
    ds_raw = MultiGameDataset(use_tile_mapping=False)
    sample_raw = ds_raw[0]
    # sample_raw.array holds the game-specific integer tile ids

    # The mode can be switched at any time without reloading the dataset.
    ds.use_tile_mapping = False   # subsequent __getitem__ / __iter__ return raw ids
    ds.use_tile_mapping = True    # and unified categories again

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
    # legacy (kept for backward compatibility)
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
    Logs a WARNING the first time `conditions` is accessed for a game that has no
    per-sample annotation (once per game).
    """
    _warned_games: set = set()  # class-level: games already warned about

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
    Unified dataset class for Dungeon, Sokoban (Boxoban), POKEMON, and DOOM.

    Parameters
    ----------
    dungeon_root     : path to dungeon_level_dataset
    pokemon_root     : path to Five-Dollar-Model
    sokoban_root     : path to boxoban_levels
    doom_root        : path to doom_levels
    include_dungeon  : whether to load the Dungeon dataset
    include_pokemon  : whether to load the POKEMON dataset
    include_sokoban  : whether to load the Sokoban dataset
    include_doom     : whether to load the DOOM dataset
    use_tile_mapping : if True (default), arrays are converted to unified categories;
                       if False, raw tile ids are returned unchanged. The mode can be
                       switched at any time after loading.
    handler_config   : HandlerConfig instance; None uses the defaults
                       (per-game preprocessing and augmentation options)
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
        # Deprecated aliases, kept for backward compatibility
        boxoban_root:         Path | str | None = None,
        include_boxoban:      bool | None = None,
    ) -> None:
        self.use_tile_mapping: bool = use_tile_mapping
        self._instruction_field: str = instruction_field  # "uni" or "raw"

        # Backward compatibility
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

        # Per-game cache keys, reused when loading annotations
        self._game_cache_keys: Dict[str, str] = {}

        hc = handler_config.to_dict()

        # ── Per-game load spec: (game, root, handler_config subsection) ──────────
        _game_specs = []
        if include_dungeon:
            _game_specs.append(("dungeon", str(dungeon_root), hc.get("dungeon", {})))
        if include_sokoban:
            _game_specs.append(("sokoban", str(sokoban_root), hc.get("sokoban", {})))
        if include_zelda:
            _game_specs.append(("zelda", str(zelda_root), hc.get("zelda", {})))
        if include_pokemon:
            _game_specs.append(("pokemon", str(pokemon_root), hc.get("pokemon", {})))
        # doom/doom2 are handled separately below (they share one cache entry)

        # ── Load each game ──────────────────────────────────────────────────────
        for game, game_root, game_hc in _game_specs:
            cache_key = build_per_game_cache_key(game, game_root, game_hc)
            logger.debug("[%s] cache key: %s", game, cache_key[:12])
            self._game_cache_keys[game] = cache_key
            # (1) Try the per-game cache first
            if use_cache:
                cached = load_game_samples_from_cache(cache_dir, game, cache_key)
                if cached is not None:
                    for s in cached:
                        s.order = len(self._samples)
                        self._samples.append(s)
                    continue

            # (2) Load from the source dataset
            game_samples = self._load_game_from_source(
                game, game_root, handler_config
            )

            if game_samples is not None:
                # Apply max_samples before caching
                max_s = game_hc.get("max_samples") if isinstance(game_hc, dict) else getattr(game_hc, "max_samples", None)
                if max_s is not None and len(game_samples) > max_s:
                    game_samples = game_samples[:max_s]
                # Filter and augment before caching, so viewer/annotate see the same data
                game_samples = self._postprocess_game_samples(game, game_samples, handler_config)
                for s in game_samples:
                    s.order = len(self._samples)
                    self._samples.append(s)
                # Store in the cache
                if use_cache:
                    save_game_samples_to_cache(
                        cache_dir, game, cache_key, game_samples
                    )
                continue

            # (3) artifact-only fallback: load whatever cache exists for this game
            if use_cache:
                fallback = load_any_game_cache(cache_dir, game)
                if fallback is not None:
                    logger.info("%s: artifact-only fallback (%d samples from existing cache)",
                                game, len(fallback))
                    for s in fallback:
                        s.order = len(self._samples)
                        self._samples.append(s)
                    # Record the key of the file the fallback actually loaded
                    actual_key = find_game_cache_key(cache_dir, game)
                    if actual_key:
                        self._game_cache_keys[game] = actual_key
                    continue

            # (4) Neither source nor cache is available
            logger.warning("%s: no source data and no cache — skipped", game)

        # ── Load doom + doom2 together (max_samples applied to the combined set) ──
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
                        # Record the key of the file the fallback actually loaded
                        actual_key = find_game_cache_key(cache_dir, "doom")
                        if actual_key:
                            self._game_cache_keys["doom"] = actual_key

        # ── Load annotations automatically (ann.json -> samples) ─────────────────
        if use_cache and self._game_cache_keys:
            self._ensure_and_load_all_annotations()

        # ── Record raw counts per (game, reward_enum), before max_samples_per_game ──
        self._raw_game_re_counts: dict = {}
        for s in self._samples:
            re = s.meta.get("reward_enum")
            if re is not None:
                self._raw_game_re_counts[(s.game, re)] = self._raw_game_re_counts.get((s.game, re), 0) + 1

        # ── Cap the per-game sample count (by source_id, after annotations load) ──
        # Selecting whole source_ids keeps every reward_enum of a level together.
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

        # ── Subsample down to N per game (uniformly at random) ────────────────────
        if N >= 1:
            import random as _random
            _total = len(self._samples)
            _rng = _random.Random(42)
            _mask = [False] * _total
            # Bucket indices per game, preserving their original order
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
        Apply filtering and augmentation before the samples are cached.

        Order:
        1. Pokemon tile filtering (drop samples exceeding max_tile_count)
        2. Instruction length filtering (drop samples below min_instruction_words)
        3. Rotation augmentation (90-degree rotations when rotate_90 is enabled)
        4. max_samples is applied after augmentation
        """
        # (1) Pokemon tile filtering
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

        # (2) Instruction length filtering
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

        # (4) Apply max_samples after augmentation
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
        """Load a game's samples from its source dataset. Returns None on failure."""
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
        Keep only samples whose floor + empty count is at most floor_empty_max.
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
        Filter POKEMON samples by how dominant a single tile type is.

        A padded 16x16 map has 256 cells; if one tile accounts for 250 or more of them the
        level is essentially empty, so it is dropped. Only POKEMON samples are affected.
        """
        pokemon_indices = [i for i, s in enumerate(self._samples) if s.game == "pokemon"]

        if not pokemon_indices:
            return

        original_pokemon_count = len(pokemon_indices)
        filtered_samples = []

        for i, sample in enumerate(self._samples):
            if sample.game == "pokemon":
                # POKEMON sample: filter on tile counts
                flat = sample.array.ravel()
                tile_counts = np.bincount(flat.astype(int))
                max_tile_count = int(np.max(tile_counts)) if len(tile_counts) > 0 else 0

                # Drop levels where one tile covers 250 or more of the 256 cells
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
        Augment samples with rotated copies, per game.

        A game is augmented only when rotate_90 is enabled in its handler config.
        For example, config.pokemon.rotate_90 = True augments only POKEMON samples.
        """
        original_count = len(self._samples)
        rotated_samples = []

        for sample in self._samples:
            # Check rotate_90 in this game's handler config
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

        # Append the rotated samples
        self._samples.extend(rotated_samples)

        # Renumber
        for i, sample in enumerate(self._samples):
            sample.order = i

        if len(rotated_samples) > 0:
            logger.info("Data augmentation: %d → %d samples (added %d rotated versions)",
                        original_count, len(self._samples), len(rotated_samples))

        # ── Cap each game after augmentation, using max_samples from handler_config ──
        game_sample_counts = {}
        filtered_samples = []

        for sample in self._samples:
            game = sample.game
            if game not in game_sample_counts:
                game_sample_counts[game] = 0

            # Read max_samples from this game's handler config
            max_samples = None
            if game == "pokemon":
                max_samples = self._handler_config.pokemon.max_samples
            elif game == "doom":
                max_samples = self._handler_config.doom.max_samples
            elif game == "zelda":
                max_samples = self._handler_config.zelda.max_samples
            elif game == "dungeon":
                max_samples = self._handler_config.dungeon.max_samples
            # sokoban has no such option in handler_config, so it is left uncapped

            # Keep the sample if the cap has not been reached
            if max_samples is None or game_sample_counts[game] < max_samples:
                filtered_samples.append(sample)
                game_sample_counts[game] += 1

        # Apply the filtered list
        if len(filtered_samples) < len(self._samples):
            self._samples = filtered_samples

            # Renumber
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
        """Ensure every game has an ann.json and attach it to the samples.

        When ann.json is missing it is computed via compute_game_annotations() and saved;
        otherwise the existing file is loaded as-is.
        """
        import time as _time

        games = list(self._game_cache_keys.items())
        logger.debug("[Annotation] Starting: %d game(s) to process (%s)",
                    len(games), ", ".join(g for g, _ in games))

        total_attached = 0
        for game, cache_key in games:
            existing = load_game_annotations_from_cache(self._cache_dir, game, cache_key)
            if existing is None:
                # No ann.json: compute it
                game_samples = [s for s in self._samples if s.game == game]
                if not game_samples:
                    logger.info("[Annotation][%s] No samples — skipping", game)
                    continue
                logger.info("[Annotation][%s] ann.json not found — computing measures (%d samples)",
                            game, len(game_samples))
                t0 = _time.perf_counter()
                try:
                    # Imported lazily to avoid pulling in JAX at module load
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
                # Write the ann_keys into the freshly created .json
                update_json_with_ann_keys(self._cache_dir, game, cache_key, existing)
            else:
                n_rows = len(existing.get("annotations", []))
                has_instr = existing.get("has_instructions", False)
                logger.debug("[Annotation][%s] ann.json cache hit: %d rows, has_instructions=%s",
                            game, n_rows, has_instr)
                # Write ann_keys into .json if absent (older caches lack them)
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
        """Submit instruction generation for a game to the OpenAI Batch API.

        - If ann.json already records a batch_id, the existing batch is reused
          (a finished one is retrieved, a running one is left alone).
        - Skipped when OPENAI_API_KEY is not set.
        - On success the new batch_id is written back to ann.json.
        """
        import os

        # An existing batch: check its status and retrieve it if it has finished
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
                    # Reload ann.json so the caller sees the freshly written data
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

        # No API key available
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

            # Fill in rows whose threshold is None
            fill_none_instructions([game], enums, cache_dir)

            # source_id -> array index map (used to resolve the shortened keys)
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
        """Attach ann.json rows to a game's samples, one per reward_enum.

        Rows are matched through sample.meta["ann_keys"]; samples without ann_keys fall
        back to matching by index.
        """
        import dataclasses
        import time as _time

        all_rows: List[Dict[str, Any]] = ann_data.get("annotations", [])
        if not all_rows:
            logger.warning("[Annotation][%s] No annotations in ann.json — skipping", game)
            return

        # key -> ann row lookup
        ann_by_key: Dict[str, Dict[str, Any]] = {r["key"]: r for r in all_rows}

        game_samples = [s for s in self._samples if s.game == game]
        n_samples = len(game_samples)
        if n_samples == 0:
            logger.warning("[Annotation][%s] No loaded samples — skipping", game)
            return

        # fallback: rows sorted for index-based matching
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
            # Preferred path: match through ann_keys
            ann_keys: Optional[List[str]] = sample.meta.get("ann_keys")
            if ann_keys:
                ann_list = [ann_by_key[k] for k in ann_keys if k in ann_by_key]
            else:
                # Legacy fallback: match by index
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
                # Select the instruction field according to instruction_field
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
        Read the CSV files in the reward_annotations folder and attach the annotations to
        each game's samples via sample.meta.

        - {game}_reward_annotations.csv             : per-sample annotations
            -> one sample is created per reward_enum
        - {game}_reward_annotations_placeholder.csv : game-level placeholder annotations
            -> accessing `conditions` logs a WARNING
        """
        import dataclasses

        # ── Games with a per-sample CSV: match samples to rewards by key order ──
        # CSV layout when sorted by key: [reward0: sample0..N-1, reward1: sample0..N-1, ...]
        for csv_path in sorted(annotations_dir.glob("*_reward_annotations.csv")):
            game_name = csv_path.name.replace("_reward_annotations.csv", "")

            # Load every row, sorted by key
            all_rows: List[Dict[str, Any]] = []
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    all_rows.append(row)
            all_rows.sort(key=lambda r: r["key"])

            # Samples already loaded for this game, in order
            game_samples = [s for s in self._samples if s.game == game_name]
            n_samples = len(game_samples)
            if n_samples == 0 or len(all_rows) == 0:
                continue

            # rows / samples = number of reward types
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
                    # Select raw or uni according to instruction_field
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

        # ── placeholder CSV: only used for games without a per-sample CSV ─────────
        for ph_csv in sorted(annotations_dir.glob("*_reward_annotations_placeholder.csv")):
            game_name = ph_csv.name.replace("_reward_annotations_placeholder.csv", "")
            # Skip when a per-sample CSV already exists
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
        Return a new GameSample whose array honours the use_tile_mapping setting.
        The internal _samples list always keeps the raw tile ids.
        """
        if not self.use_tile_mapping:
            return sample
        import dataclasses
        unified_array = to_unified(sample.array, sample.game, warn_unmapped=False)
        return dataclasses.replace(sample, array=unified_array)

    def _find_raw_sample(self, sample: GameSample) -> GameSample:
        """Look up the internal raw sample by (game, source_id)."""
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

    # ── Filters ────────────────────────────────────────────────────────────────────
    def by_game(self, game: str) -> List[GameSample]:
        """Return every sample of the given game."""
        return [self._apply_mapping(s)
                for s in tag_utils.extract_by_game(self._samples, game)]

    def by_games(self, games: List[str]) -> List[GameSample]:
        """Return the samples of the given games."""
        return [self._apply_mapping(s)
                for s in tag_utils.extract_by_games(self._samples, games)]

    def by_instruction(
        self, keyword: str, *, case_sensitive: bool = False
    ) -> List[GameSample]:
        """Filter by an instruction keyword."""
        return [self._apply_mapping(s)
                for s in tag_utils.extract_by_instruction(
                    self._samples, keyword, case_sensitive=case_sensitive)]

    def with_instruction(self) -> List[GameSample]:
        """Return the samples that have an instruction."""
        return [self._apply_mapping(s)
                for s in tag_utils.extract_with_instruction(self._samples)]

    def without_instruction(self) -> List[GameSample]:
        """Return the samples that have no instruction."""
        return [self._apply_mapping(s)
                for s in tag_utils.extract_without_instruction(self._samples)]

    def by_order(self, start: int, end: int) -> List[GameSample]:
        """order range [start, end) sample."""
        return [self._apply_mapping(s)
                for s in tag_utils.extract_by_order(self._samples, start, end)]

    def by_meta(self, key: str, value: Any) -> List[GameSample]:
        """Filter by a metadata attribute."""
        return [self._apply_mapping(s)
                for s in tag_utils.extract_by_meta(self._samples, key, value)]

    def filter(self, fn) -> List[GameSample]:
        """Filter with a user-supplied predicate over the conditions."""
        return [self._apply_mapping(s)
                for s in tag_utils.extract_by_predicate(self._samples, fn)]

    # ── reward annotation based filter ──────────────────────────────────────────
    def by_reward_enum(self, reward_enum: int) -> List[GameSample]:
        """Filter by reward_enum (1=region, 2=path_length, 3=block, 4=bat_amount, 5=bat_direction)."""
        return [self._apply_mapping(s)
                for s in self._samples
                if s.meta.get("reward_enum") == reward_enum]

    def by_feature_name(self, feature_name: str) -> List[GameSample]:
        """feature_name as  filtering (region, path_length, block, bat_amount, bat_direction)."""
        return [self._apply_mapping(s)
                for s in self._samples
                if s.meta.get("feature_name") == feature_name]

    def with_reward_annotation(self) -> List[GameSample]:
        """Return the samples that carry a reward annotation."""
        return [self._apply_mapping(s)
                for s in self._samples
                if "reward_enum" in s.meta]

    # ── Aggregation ─────────────────────────────────────────────────────────────
    def group_by_game(self) -> Dict[str, List[GameSample]]:
        return tag_utils.group_by_game(self._samples)

    def group_by_instruction(self) -> Dict[str, List[GameSample]]:
        return tag_utils.group_by_instruction(self._samples)

    def count_by_game(self) -> Dict[str, int]:
        return tag_utils.count_by_game(self._samples)

    def summary(self) -> Dict[str, Any]:
        return tag_utils.summary(self._samples)

    # ── Rendering (requires Pillow) ───────────────────────────────────────────────
    def render(
        self,
        sample: GameSample,
        tile_size: int = 16,
        save_path: Optional[Path | str] = None,
    ):
        """
        Render a single sample.
        With use_tile_mapping=True the unified palette is used, otherwise the game's own
        palette. Saves a PNG when save_path is given, else returns a PIL Image.
        """
        from .render import render_sample_pil, save_rendered
        from .tile_utils import render_unified_rgb
        from PIL import Image

        if self.use_tile_mapping:
            # The sample handed in may already be unified or still raw, so the mapping is
            # applied unconditionally (it is a no-op on already-mapped arrays).
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
        Render several samples in a grid.
        use_tile_mapping config automatic apply.
        Saves a PNG when save_path is given, else returns a PIL Image.
        """
        from .render import render_grid as _rg, save_grid
        from PIL import Image

        # Apply the mapping to each sample
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
        Render the raw and the unified-category image side by side.

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
        Render a sample using its tile images.

        Parameters
        ----------
        sample : GameSample to render
        tile_size : tile size in pixels
        save_path : save path
        show_tile_numbers : overlay the tile id on each cell

        Returns
        -------
        PIL.Image.Image : the rendered image

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
        Render a raw level array directly using its tile images.

        Parameters
        ----------
        game : game name (dungeon, doom, pokemon, sokoban, zelda)
        level : 2D numpy array
        tile_size : tile size in pixels
        save_path : save path
        show_tile_numbers : overlay the tile id on each cell

        Returns
        -------
        PIL.Image.Image : the rendered image

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
        """Rows of the raw tile -> unified category mapping, from tile_mapping.json."""
        return game_mapping_rows(game)

    # ── utility ────────────────────────────────────────────────────────────────────
    def get_tags(self, idx: int) -> Dict[str, Any]:
        """Return the tag dictionary for an index."""
        return tag_utils.build_tags(self._samples[idx])

    def all_tags(self) -> List[Dict[str, Any]]:
        """Total number of samples."""
        return [tag_utils.build_tags(s) for s in self._samples]

    def available_games(self) -> List[str]:
        """List of the loaded games."""
        return [GameTag.DUNGEON, GameTag.SOKOBAN, GameTag.DOOM, GameTag.POKEMON, GameTag.ZELDA]

    def sample(
        self,
        n: int,
        game: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> List[GameSample]:
        """
        Draw random samples.

        Parameters
        ----------
        n    : number of samples
        game : restrict to this game (None = all games)
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
