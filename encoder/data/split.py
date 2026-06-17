from __future__ import annotations

import logging
import os
from os.path import basename
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from conf.game_utils import GAME_ABBR
from encoder.data.clip_batch import CLIPDataset

log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logger = logging.getLogger(basename(__file__))
logger.setLevel(getattr(logging, log_level, logging.INFO))


def parse_unseen_game_names(unseen_str: Optional[str]) -> Set[str]:
    """Convert a two-character abbreviation string to full game names.

    Return an empty set for None or an empty string.

    Examples
    --------
    parse_unseen_game_names(None)   -> set()
    parse_unseen_game_names("")     -> set()
    parse_unseen_game_names("zd")   -> {'zelda'}
    parse_unseen_game_names("pkzd") -> {'pokemon', 'zelda'}
    """
    if not unseen_str:
        return set()
    names: Set[str] = set()
    for i in range(0, len(unseen_str), 2):
        abbr = unseen_str[i : i + 2]
        names.update(GAME_ABBR.get(abbr, []))
    return names


def subset_clip_dataset(dataset: CLIPDataset, indices: np.ndarray) -> CLIPDataset:
    """Return a subset of CLIPDataset for the given indices."""
    idx = np.asarray(indices, dtype=int)
    return CLIPDataset(
        class_ids=dataset.class_ids[idx],
        reward_cond=dataset.reward_cond[idx],
        input_ids=dataset.input_ids[idx],
        attention_masks=dataset.attention_masks[idx],
        pixel_values=dataset.pixel_values[idx],
        is_train=dataset.is_train[idx],
        reward_enum_targets=dataset.reward_enum_targets[idx],
        condition_targets=dataset.condition_targets[idx],
        quantized_condition_targets=dataset.quantized_condition_targets[idx],
    )


def split_dataset_by_game(
    full_dataset: CLIPDataset,
    unseen_game_names: Set[str],
    test_ratio: float,
    test_seed: int,
) -> Tuple[
    Dict[str, np.ndarray],  # game → train pool indices
    Dict[str, np.ndarray],  # game → test indices
    np.ndarray,             # all game names (per sample)
]:
    """Split the full dataset into per-game train pools and test sets.

    - Split ``test_ratio`` from every game (seen and unseen) into the test set.
    - ``test_seed`` makes the split deterministic.
    - The sweep ratio controls how much unseen-game data is used from each train pool.

    Returns
    -------
    game_train_pool : {game_name: ndarray of indices}
    game_test       : {game_name: ndarray of indices}
    all_game_names  : ndarray of str  (length = len(full_dataset.class_ids))
    """
    all_game_names = np.array(
        [rc["game_name"] for rc in full_dataset.reward_cond]
    )
    unique_games = sorted(set(all_game_names))

    rng = np.random.RandomState(test_seed)

    game_train_pool: Dict[str, np.ndarray] = {}
    game_test: Dict[str, np.ndarray] = {}

    for game in unique_games:
        game_indices = np.where(all_game_names == game)[0]
        perm = rng.permutation(game_indices)
        n_test = max(1, int(len(perm) * test_ratio))
        game_test[game] = perm[:n_test]
        game_train_pool[game] = perm[n_test:]  # Fixed order; ratio subsets take prefixes.
        tag = "(unseen)" if game in unseen_game_names else "(seen)"
        logger.debug(
            "split_dataset_by_game [%s] %s: total=%d, train_pool=%d, test=%d",
            game, tag, len(game_indices), len(game_train_pool[game]), len(game_test[game]),
        )

    return game_train_pool, game_test, all_game_names


def build_train_indices_for_ratio(
    game_train_pool: Dict[str, np.ndarray],
    unseen_game_names: Set[str],
    ratio: float,
    seen_ratio: float = 1.0,
) -> np.ndarray:
    """Build training indices for a given few-shot ``ratio``.

    - Seen games: use a ``seen_ratio`` prefix from the train pool.
    - Unseen games: use a ``ratio`` prefix from the train pool.
    - ``ratio=0.0`` means no unseen-game training samples.
    - ``seen_ratio=0.0`` means no seen-game training samples.
    """
    train_indices: List[np.ndarray] = []
    for game, pool in sorted(game_train_pool.items()):
        if game in unseen_game_names:
            n_use = int(len(pool) * ratio)
            if n_use > 0:
                train_indices.append(pool[:n_use])
        else:
            n_use = int(len(pool) * seen_ratio)
            if n_use > 0:
                train_indices.append(pool[:n_use])
    if train_indices:
        return np.concatenate(train_indices)
    return np.array([], dtype=int)


