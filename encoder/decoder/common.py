from __future__ import annotations

import logging
import os
from os.path import basename

from dataset.multigame import MultiGameDataset

log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logger = logging.getLogger(basename(__file__))
logger.setLevel(getattr(logging, log_level, logging.INFO))


# reward_enum 이름 매핑 (0-based: CSV reward_enum은 0-indexed)
_REWARD_ENUM_NAMES = {
    0: "region",
    1: "path_length",
    2: "interactable_count",
    3: "hazard_count",
    4: "collectable_count",
}


def _log_reward_condition_summary(dataset: MultiGameDataset):
    """학습 시작 전에 reward_enum별 condition 범위를 출력한다 (게임 구분 없이 enum 기준)."""
    from collections import defaultdict

    # reward_enum → [(game, condition_value)]
    enum_stats: dict = defaultdict(list)
    # game → reward_enum → [condition_values]  (게임별 분해용)
    game_enum_stats: dict = defaultdict(lambda: defaultdict(list))

    for s in dataset._samples:
        reward_enum = s.meta.get("reward_enum")
        if reward_enum is None:
            continue
        game = s.game
        conditions = s.meta.get("conditions", {})
        cond_val = list(conditions.values())[0] if conditions else None
        re_id = int(reward_enum)
        enum_stats[re_id].append(cond_val)
        game_enum_stats[game][re_id].append(cond_val)

    logger.info("=" * 80)
    logger.info("  Reward Enum & Condition Range Summary  (raw, before normalization)")
    logger.info("=" * 80)

    # ── reward_enum별 전체 통계 ──
    logger.info(f"  {'enum':>5}  {'name':<22} {'count':>6}  {'min':>10}  {'max':>10}  {'mean':>10}  {'std':>10}")
    logger.info(f"  {'-'*5}  {'-'*22} {'-'*6}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}")

    for re_id in sorted(enum_stats.keys()):
        vals = enum_stats[re_id]
        valid_vals = [v for v in vals if v is not None]
        name = _REWARD_ENUM_NAMES.get(re_id, f"unknown_{re_id}")
        count = len(vals)

        if valid_vals:
            v_min = min(valid_vals)
            v_max = max(valid_vals)
            v_mean = sum(valid_vals) / len(valid_vals)
            v_std = (sum((v - v_mean) ** 2 for v in valid_vals) / len(valid_vals)) ** 0.5
            logger.info(f"  {re_id:>5}  {name:<22} {count:>6}  {v_min:>10.2f}  {v_max:>10.2f}  {v_mean:>10.2f}  {v_std:>10.2f}")
        else:
            logger.info(f"  {re_id:>5}  {name:<22} {count:>6}  {'N/A':>10}  {'N/A':>10}  {'N/A':>10}  {'N/A':>10}")

    logger.info("")

    # ── 게임별 분해 ──
    for game in sorted(game_enum_stats.keys()):
        enum_dict = game_enum_stats[game]
        n_total = sum(len(v) for v in enum_dict.values())
        logger.info(f"  [{game}]  ({n_total} samples)")
        for re_id in sorted(enum_dict.keys()):
            vals = enum_dict[re_id]
            valid_vals = [v for v in vals if v is not None]
            name = _REWARD_ENUM_NAMES.get(re_id, f"unknown_{re_id}")
            if valid_vals:
                logger.info(f"    enum {re_id} ({name}): "
                            f"n={len(vals)}, "
                            f"range=[{min(valid_vals):.2f}, {max(valid_vals):.2f}], "
                            f"mean={sum(valid_vals)/len(valid_vals):.2f}")
            else:
                logger.info(f"    enum {re_id} ({name}): n={len(vals)}, range=N/A")
    logger.info("")

    # 전체 요약
    all_enums = set(enum_stats.keys())
    logger.info(f"  Total games: {len(game_enum_stats)},  "
                f"Unique reward_enums (0-based): {sorted(all_enums)},  "
                f"num_reward_classes should be >= {max(all_enums) + 1 if all_enums else 0}")
    logger.info("=" * 80)
    logger.info("  ※ Condition values will be min-max normalized per reward_enum to [0, 1]")
    logger.info("=" * 80)
