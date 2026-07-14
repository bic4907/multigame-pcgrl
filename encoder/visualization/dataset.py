from __future__ import annotations

import logging
import os
from os.path import basename
from typing import Iterable

import jax
import numpy as np
from transformers import CLIPProcessor

from conf.config import CLIPDecoderTrainConfig
from dataset.multigame import MultiGameDataset
from encoder.data.clip_batch import CLIPDatasetBuilder
from encoder.visualization.common import canonical_game_name, normalize_games


log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logger = logging.getLogger(basename(__file__))
logger.setLevel(getattr(logging, log_level, logging.INFO))


def build_dataset(
    config: CLIPDecoderTrainConfig,
    games: Iterable[str],
    seed: int,
    max_samples_per_game: int,
):
    include = {canonical_game_name(game) for game in games}
    dataset = MultiGameDataset(
        include_dungeon="dungeon" in include,
        include_pokemon="pokemon" in include,
        include_sokoban="sokoban" in include,
        include_doom="doom" in include,
        include_doom2="doom" in include,
        include_zelda="zelda" in include,
        use_tile_mapping=True,
        max_samples_per_game=max_samples_per_game,
        max_samples_seed=getattr(config, "max_samples_seed", 42),
        instruction_field=getattr(config, "instruction_field", "raw"),
    )
    dataset._game_str = ",".join(games)

    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    builder = CLIPDatasetBuilder(
        processor=processor,
        paired_data=dataset,
        rng_key=jax.random.PRNGKey(seed),
        max_len=config.encoder.token_max_len,
        train_ratio=1.0,
        max_samples=None,
        instruction_prefix=config.instruction_prefix,
        longtail_cut=config.longtail_cut,
        tile_offset=getattr(config.encoder, "tile_offset", 0),
    )
    dataset = builder.get_dataset()
    language_inst = np.array(builder.preprocessed_dataset_dict["language_inst"], dtype=object)
    return dataset, language_inst


def sample_indices_by_game(
    full_dataset,
    games: list[str],
    samples_per_game: int,
    seed: int,
    allow_replacement: bool,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    games = normalize_games(games)
    all_game_names = np.array([canonical_game_name(rc["game_name"]) for rc in full_dataset.reward_cond])
    selected = []

    for game in games:
        game_indices = np.where(all_game_names == game)[0]
        if len(game_indices) == 0:
            raise ValueError(f"No usable samples found for game={game!r}")

        if samples_per_game <= 0:
            selected.append(game_indices)
            logger.info("Selected all %s samples: %d", game, len(game_indices))
            continue

        replace = len(game_indices) < samples_per_game
        if replace and not allow_replacement:
            raise ValueError(
                f"Game {game!r} has only {len(game_indices)} usable samples; "
                f"requested {samples_per_game}. Set +tsne.allow_replacement=true "
                f"or lower +tsne.samples_per_game."
            )

        chosen = rng.choice(game_indices, size=samples_per_game, replace=replace)
        selected.append(chosen)
        mode = "with replacement" if replace else "without replacement"
        logger.info("Sampled %s: %d / %d (%s)", game, len(chosen), len(game_indices), mode)

    return np.concatenate(selected)
