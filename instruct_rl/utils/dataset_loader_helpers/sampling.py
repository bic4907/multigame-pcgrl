from __future__ import annotations

import hashlib
import random
from collections import defaultdict


def _subsample_per_group(samples, n_per_group: int, seed: int = 0):
    """(game, re) 그룹별로 최대 n_per_group 개를 서브샘플링한다.

    가용 샘플이 n_per_group 보다 적은 그룹은 전부 사용.

    Returns
    -------
    subsampled : list
    sampled_counts : dict  game -> sampled count  (re가 1개인 경우 game 기준)
    """
    by_group: dict = defaultdict(list)
    for sample in samples:
        reward_enum = sample.meta.get("reward_enum", None)
        by_group[(sample.game, reward_enum)].append(sample)

    for key in by_group:
        by_group[key].sort(key=lambda s: str(getattr(s, "source_id", s)))

    result = []
    sampled_counts: dict = {}

    for (game, reward_enum) in sorted(by_group.keys()):
        key_bytes = f"{game}_{reward_enum}".encode()
        key_hash = int(hashlib.md5(key_bytes).hexdigest(), 16) & 0xFFFFFFFF
        group_seed = seed ^ key_hash
        group_rng = random.Random(group_seed)
        pool = by_group[(game, reward_enum)][:]
        group_rng.shuffle(pool)
        chosen = pool[:n_per_group]
        result.extend(chosen)
        sampled_counts[game] = sampled_counts.get(game, 0) + len(chosen)

    random.Random(seed).shuffle(result)
    return result, sampled_counts

