from __future__ import annotations

import hashlib
import random
from collections import defaultdict


def _subsample_per_group(samples, n_per_group: int, seed: int = 0):
    """(game, re) textby maximum n_per_group text  textsampletext.

      for  sample  n_per_group text text  text   before text text for .

    Returns
    -------
    subsampled : list
    sampled_counts : dict  game -> sampled count  (re  1text text game basis)
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

