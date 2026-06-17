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
    """2글자 약어 문자열 → full game name set.

    None 또는 빈 문자열이면 빈 set을 반환한다.

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
    """CLIPDataset에서 주어진 인덱스의 서브셋을 추출한다."""
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
    """전체 데이터셋을 게임별로 train pool / test 로 분할한다.

    - 모든 게임(seen + unseen)에서 ``test_ratio`` 만큼 테스트 세트로 분리
    - 분할은 ``test_seed`` 로 결정 → 동일한 시드에서 항상 같은 테스트셋
    - train pool 내 unseen 게임 데이터의 실제 사용량은 sweep ratio 에 의해 결정

    Returns
    -------
    game_train_pool : {game_name: ndarray of indices}
    game_test       : {game_name: ndarray of indices}
    all_game_names  : ndarray of str  (길이 = len(full_dataset.class_ids))
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
        game_train_pool[game] = perm[n_test:]  # 고정 순서 (ratio 서브셋은 prefix)
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
    """주어진 few-shot ``ratio`` 에 대해 학습 인덱스를 구성한다.

    - Seen 게임: train pool 중 seen_ratio 비율만큼 (prefix) 사용
    - Unseen 게임: train pool 중 ratio 비율만큼 (prefix) 사용
    - ratio=0.0 이면 unseen 게임의 학습 데이터 = 0
    - seen_ratio=0.0 이면 seen 게임의 학습 데이터 = 0
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


