from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np

from instruct_rl.utils.log_utils import get_logger

from .constants import REWARD_ENUM_NAMES
from .filters import _parse_dataset_reward_enum_filter

logger = get_logger(__file__)


def _log_dataset_table(ds, all_samples, config, *, sampled_counts: dict = None, re_filter_list=None):
    """(game, re) 조합별 처리 통계를 표로 출력한다 (줄마다 별도 logger.info)."""
    has_sampled = bool(sampled_counts)
    col_sampled = "Sampled"

    cell: dict = defaultdict(lambda: {"n": 0, "instr": 0, "cond_vals": []})

    for sample in ds:
        reward_enum = sample.meta.get("reward_enum")
        if reward_enum is None or "conditions" not in sample.meta:
            continue
        key = (sample.game, reward_enum)
        cell[key]["n"] += 1
        if sample.instruction:
            cell[key]["instr"] += 1
        conds = sample.meta.get("conditions", {})
        val = conds.get(reward_enum, conds.get(str(reward_enum), None))
        if val is not None:
            cell[key]["cond_vals"].append(float(val))

    cache_keys: dict = getattr(ds, "_game_cache_keys", {})

    games = sorted({k[0] for k in cell})
    re_vals = sorted({k[1] for k in cell})
    if re_filter_list:
        re_filter_set = set(re_filter_list)
        re_filter = None
    else:
        re_filter = _parse_dataset_reward_enum_filter(
            getattr(config, "dataset_reward_enum", None),
            field_name="dataset_reward_enum",
        )
        re_filter_set = set(re_filter) if re_filter is not None else None

    col_game = "Game"
    col_re = "re(name)"
    col_hash = "Hash"
    col_n = "Samples"
    col_instr = "w/ Instr"
    col_cmin = "cond_min"
    col_cmax = "cond_max"

    rows_data = []
    for game in games:
        for reward_enum in re_vals:
            if (game, reward_enum) not in cell:
                continue
            c = cell[(game, reward_enum)]
            re_label = f"{reward_enum}({REWARD_ENUM_NAMES.get(reward_enum, '?')})"
            highlight = reward_enum in re_filter_set if re_filter_set is not None else True
            arr = np.array(c["cond_vals"]) if c["cond_vals"] else None
            rows_data.append(
                {
                    "game": game,
                    "re": re_label,
                    "hash": cache_keys.get(game, "")[:12],
                    "n": c["n"],
                    "instr": c["instr"],
                    "cmin": f"{arr.min():.1f}" if arr is not None else "-",
                    "cmax": f"{arr.max():.1f}" if arr is not None else "-",
                    "selected": highlight,
                }
            )

    selected = [row for row in rows_data if row["selected"]]
    if not selected:
        logger.info("No samples found for re=%s", re_filter)
        return

    w0 = max(len(col_game), max(len(row["game"]) for row in selected))
    w1 = max(len(col_re), max(len(row["re"]) for row in selected))
    w2 = max(len(col_hash), max(len(row["hash"]) for row in selected))
    w3 = max(len(col_n), max(len(str(row["n"])) for row in selected))
    w4 = max(len(col_instr), max(len(str(row["instr"])) for row in selected))
    w5 = max(len(col_cmin), max(len(row["cmin"]) for row in selected))
    w6 = max(len(col_cmax), max(len(row["cmax"]) for row in selected))
    if has_sampled:
        w7 = max(len(col_sampled), max(len(str(sampled_counts.get(row["game"], "-"))) for row in selected))

    if has_sampled:
        sep = (
            f"+{'-' * (w0 + 2)}+{'-' * (w1 + 2)}+{'-' * (w2 + 2)}"
            f"+{'-' * (w3 + 2)}+{'-' * (w4 + 2)}+{'-' * (w7 + 2)}+{'-' * (w5 + 2)}+{'-' * (w6 + 2)}+"
        )
        header = (
            f"| {col_game:<{w0}} | {col_re:<{w1}} | {col_hash:<{w2}} "
            f"| {col_n:>{w3}} | {col_instr:>{w4}} "
            f"| {col_sampled:>{w7}} | {col_cmin:>{w5}} | {col_cmax:>{w6}} |"
        )
    else:
        sep = (
            f"+{'-' * (w0 + 2)}+{'-' * (w1 + 2)}+{'-' * (w2 + 2)}"
            f"+{'-' * (w3 + 2)}+{'-' * (w4 + 2)}+{'-' * (w5 + 2)}+{'-' * (w6 + 2)}+"
        )
        header = (
            f"| {col_game:<{w0}} | {col_re:<{w1}} | {col_hash:<{w2}} "
            f"| {col_n:>{w3}} | {col_instr:>{w4}} "
            f"| {col_cmin:>{w5}} | {col_cmax:>{w6}} |"
        )

    tot_n = sum(row["n"] for row in selected)
    tot_instr = sum(row["instr"] for row in selected)
    tot_sampled = sum(sampled_counts.values()) if has_sampled else None
    if re_filter_set is not None:
        re_label_str = ",".join(str(r) for r in sorted(re_filter_set))
        re_name_str = ",".join(REWARD_ENUM_NAMES.get(r, "?") for r in sorted(re_filter_set))
    elif re_filter is None:
        re_label_str, re_name_str = "all", "all"
    else:
        re_label_str = str(re_filter)
        re_name_str = REWARD_ENUM_NAMES.get(re_filter, "?")
    if has_sampled:
        total_row = (
            f"| {'TOTAL (re=' + re_label_str + ')':<{w0}} | {'':<{w1}} | {'':<{w2}} "
            f"| {tot_n:>{w3}} | {tot_instr:>{w4}} | {tot_sampled:>{w7}} | {'':{w5}} | {'':{w6}} |"
        )
    else:
        total_row = (
            f"| {'TOTAL (re=' + re_label_str + ')':<{w0}} | {'':<{w1}} | {'':<{w2}} "
            f"| {tot_n:>{w3}} | {tot_instr:>{w4}} | {'':{w5}} | {'':{w6}} |"
        )

    logger.info(
        "Dataset Summary  "
        f"(game={config.dataset_game}, re={re_label_str}/{re_name_str}, "
        f"train_ratio={config.dataset_train_ratio})"
    )
    logger.info(sep)
    logger.info(header)
    logger.info(sep)
    prev_game = None
    for row_data in rows_data:
        if not row_data["selected"]:
            continue
        if prev_game and prev_game != row_data["game"]:
            logger.info(sep)
        if has_sampled:
            sampled_count = sampled_counts.get(row_data["game"], "-")
            row = (
                f"| {row_data['game']:<{w0}} | {row_data['re']:<{w1}} | {row_data['hash']:<{w2}} "
                f"| {row_data['n']:>{w3}} | {row_data['instr']:>{w4}} "
                f"| {sampled_count:>{w7}} | {row_data['cmin']:>{w5}} | {row_data['cmax']:>{w6}} |"
            )
        else:
            row = (
                f"| {row_data['game']:<{w0}} | {row_data['re']:<{w1}} | {row_data['hash']:<{w2}} "
                f"| {row_data['n']:>{w3}} | {row_data['instr']:>{w4}} "
                f"| {row_data['cmin']:>{w5}} | {row_data['cmax']:>{w6}} |"
            )
        logger.info(row)
        prev_game = row_data["game"]
    logger.info(sep)
    logger.info(total_row)
    logger.info(sep)


def _log_dataset_summary(config, samples):
    """reward_enum 필터 후 최종 샘플 요약 (condition 통계 포함)."""
    re_counter = Counter(s.meta["reward_enum"] for s in samples)
    logger.info("Filtered samples: %d  (reward_enum breakdown below)", len(samples))
    for re_val in sorted(re_counter.keys()):
        re_samples = [s for s in samples if s.meta["reward_enum"] == re_val]
        cond_vals = []
        for sample in re_samples:
            conds = sample.meta.get("conditions", {})
            val = conds.get(re_val, conds.get(str(re_val), None))
            if val is not None:
                cond_vals.append(float(val))
        if cond_vals:
            arr = np.array(cond_vals)
            logger.info(
                "  re=%d (%s): %d samples  cond[min=%.1f, max=%.1f, mean=%.2f, std=%.2f]",
                re_val,
                REWARD_ENUM_NAMES.get(re_val, "?"),
                len(re_samples),
                arr.min(),
                arr.max(),
                arr.mean(),
                arr.std(),
            )
        else:
            logger.info(
                "  re=%d (%s): %d samples",
                re_val,
                REWARD_ENUM_NAMES.get(re_val, "?"),
                len(re_samples),
            )


def _log_split_summary(train_samples, test_samples, train_inst, *, sampled_counts: dict = None):
    """Train/Test 분할 결과 요약. sampled_counts가 있으면 Sampled 컬럼을 추가한다."""
    train_game = Counter(s.game for s in train_samples)
    test_game = Counter(s.game for s in test_samples)
    games = sorted(set(train_game) | set(test_game))
    has_sampled = bool(sampled_counts)

    col_game = "Game"
    col_train = "Train"
    col_test = "Test"
    col_sampled = "Sampled"

    w0 = max(len(col_game), max(len(g) for g in games))
    w1 = max(len(col_train), max(len(str(train_game[g])) for g in games))
    w2 = max(len(col_test), max(len(str(test_game[g])) for g in games))
    if has_sampled:
        w3 = max(len(col_sampled), max(len(str(sampled_counts.get(g, 0))) for g in games))

    if has_sampled:
        sep = f"+{'-' * (w0 + 2)}+{'-' * (w1 + 2)}+{'-' * (w2 + 2)}+{'-' * (w3 + 2)}+"
        header = f"| {col_game:<{w0}} | {col_train:>{w1}} | {col_test:>{w2}} | {col_sampled:>{w3}} |"
        total_sampled = sum(sampled_counts.values())
        total_row = (
            f"| {'TOTAL':<{w0}} | {len(train_samples):>{w1}} | "
            f"{len(test_samples):>{w2}} | {total_sampled:>{w3}} |"
        )
    else:
        sep = f"+{'-' * (w0 + 2)}+{'-' * (w1 + 2)}+{'-' * (w2 + 2)}+"
        header = f"| {col_game:<{w0}} | {col_train:>{w1}} | {col_test:>{w2}} |"
        total_row = f"| {'TOTAL':<{w0}} | {len(train_samples):>{w1}} | {len(test_samples):>{w2}} |"

    logger.debug(
        "Train/Test Split  "
        f"(total=%d, train=%d, test=%d)",
        len(train_samples) + len(test_samples),
        len(train_samples),
        len(test_samples),
    )
    logger.debug(sep)
    logger.debug(header)
    logger.debug(sep)
    for game in games:
        if has_sampled:
            row = (
                f"| {game:<{w0}} | {train_game[game]:>{w1}} | "
                f"{test_game[game]:>{w2}} | {sampled_counts.get(game, 0):>{w3}} |"
            )
        else:
            row = f"| {game:<{w0}} | {train_game[game]:>{w1}} | {test_game[game]:>{w2}} |"
        logger.debug(row)
    logger.debug(sep)
    logger.debug(total_row)
    logger.debug(sep)

