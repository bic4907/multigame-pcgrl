from __future__ import annotations

import hashlib
from collections import Counter, defaultdict

import numpy as np

from instruct_rl.utils.log_utils import get_logger

from .constants import REWARD_ENUM_NAMES
from .filters import _parse_dataset_reward_enum_filter

logger = get_logger(__file__)


def _compute_sample_set_hash(samples) -> str:
    """(game, source_id, reward_enum) 정렬 기준 MD5 해시 앞 8자리."""
    h = hashlib.md5()
    for s in sorted(samples, key=lambda x: (x.game, str(x.source_id), x.meta.get("reward_enum", -1))):
        h.update(f"{s.game}:{s.source_id}:{s.meta.get('reward_enum', -1)}".encode())
    return h.hexdigest()[:8]


def _log_dataset_table(ds, all_samples, config, *, sampled_counts: dict = None, re_filter_list=None):
    """(game, re) 조합별 처리 통계를 표로 출력한다 (줄마다 별도 logger.info).

    Raw  열: MultiGameDataset 로딩 직후 (max_samples_per_game 적용, 필터 전)
    Final열: 공통 전처리 후 최종 샘플 수 (instruction 필터 + longtail cut)
    Hash 열: Final 샘플셋의 MD5 앞 8자리 — 환경 간 데이터 일치 검증용
    """
    has_sampled = bool(sampled_counts)
    col_sampled = "Sampled"

    # ── Raw: max_samples_per_game 적용 전 원본 카운트 ──
    raw_cell: dict = getattr(ds, "_raw_game_re_counts", {})
    if not raw_cell:
        # fallback: ds 순회 (max_samples_per_game 이후)
        raw_cell = defaultdict(int)
        for sample in ds:
            re = sample.meta.get("reward_enum")
            if re is not None and "conditions" in sample.meta:
                raw_cell[(sample.game, re)] += 1

    # ── Final: all_samples 순회 (공통 전처리 후) ──
    final_cell: dict = defaultdict(lambda: {"n": 0, "cond_vals": []})
    group_samples: dict = defaultdict(list)
    for sample in all_samples:
        reward_enum = sample.meta.get("reward_enum")
        if reward_enum is None:
            continue
        key = (sample.game, reward_enum)
        final_cell[key]["n"] += 1
        conds = sample.meta.get("conditions", {})
        val = conds.get(reward_enum, conds.get(str(reward_enum), None))
        if val is not None:
            final_cell[key]["cond_vals"].append(float(val))
        group_samples[key].append(sample)

    all_keys = set(raw_cell) | set(final_cell)
    games = sorted({k[0] for k in all_keys})
    re_vals = sorted({k[1] for k in all_keys})

    if re_filter_list:
        re_filter_set = set(re_filter_list)
        re_filter = None
    else:
        re_filter = _parse_dataset_reward_enum_filter(
            getattr(config, "dataset_reward_enum", None),
            field_name="dataset_reward_enum",
        )
        re_filter_set = set(re_filter) if re_filter is not None else None

    col_game  = "Game"
    col_re    = "re(name)"
    col_hash  = "Hash"
    col_raw   = "Raw"
    col_final = "Final"
    col_cmin  = "cond_min"
    col_cmax  = "cond_max"

    rows_data = []
    for game in games:
        for reward_enum in re_vals:
            key = (game, reward_enum)
            if key not in raw_cell and key not in final_cell:
                continue
            c = final_cell.get(key, {"n": 0, "cond_vals": []})
            re_label = f"{reward_enum}({REWARD_ENUM_NAMES.get(reward_enum, '?')})"
            highlight = reward_enum in re_filter_set if re_filter_set is not None else True
            arr = np.array(c["cond_vals"]) if c["cond_vals"] else None
            sample_hash = _compute_sample_set_hash(group_samples[key]) if group_samples[key] else "-"
            rows_data.append({
                "game":     game,
                "re":       re_label,
                "hash":     sample_hash,
                "raw":      raw_cell.get(key, 0),
                "final":    c["n"],
                "cmin":     f"{arr.min():.1f}" if arr is not None else "-",
                "cmax":     f"{arr.max():.1f}" if arr is not None else "-",
                "selected": highlight,
            })

    selected = [row for row in rows_data if row["selected"]]
    if not selected:
        logger.info("No samples found for re=%s", re_filter)
        return

    w0 = max(len(col_game),  max(len(row["game"])        for row in selected))
    w1 = max(len(col_re),    max(len(row["re"])          for row in selected))
    w2 = max(len(col_hash),  max(len(row["hash"])        for row in selected))
    w3 = max(len(col_raw),   max(len(str(row["raw"]))    for row in selected))
    w4 = max(len(col_final), max(len(str(row["final"]))  for row in selected))
    w5 = max(len(col_cmin),  max(len(row["cmin"])        for row in selected))
    w6 = max(len(col_cmax),  max(len(row["cmax"])        for row in selected))
    if has_sampled:
        w7 = max(len(col_sampled), max(len(str(sampled_counts.get(row["game"], "-"))) for row in selected))

    def _sep():
        base = (
            f"+{'-' * (w0+2)}+{'-' * (w1+2)}+{'-' * (w2+2)}"
            f"+{'-' * (w3+2)}+{'-' * (w4+2)}"
        )
        if has_sampled:
            base += f"+{'-' * (w7+2)}"
        base += f"+{'-' * (w5+2)}+{'-' * (w6+2)}+"
        return base

    def _header():
        base = (
            f"| {col_game:<{w0}} | {col_re:<{w1}} | {col_hash:<{w2}} "
            f"| {col_raw:>{w3}} | {col_final:>{w4}} "
        )
        if has_sampled:
            base += f"| {col_sampled:>{w7}} "
        base += f"| {col_cmin:>{w5}} | {col_cmax:>{w6}} |"
        return base

    def _row(r):
        base = (
            f"| {r['game']:<{w0}} | {r['re']:<{w1}} | {r['hash']:<{w2}} "
            f"| {r['raw']:>{w3}} | {r['final']:>{w4}} "
        )
        if has_sampled:
            sc = sampled_counts.get(r["game"], "-")
            base += f"| {sc:>{w7}} "
        base += f"| {r['cmin']:>{w5}} | {r['cmax']:>{w6}} |"
        return base

    tot_raw   = sum(row["raw"]   for row in selected)
    tot_final = sum(row["final"] for row in selected)
    global_hash = _compute_sample_set_hash(all_samples)

    if re_filter_set is not None:
        re_label_str = ",".join(str(r) for r in sorted(re_filter_set))
        re_name_str  = ",".join(REWARD_ENUM_NAMES.get(r, "?") for r in sorted(re_filter_set))
    elif re_filter is None:
        re_label_str, re_name_str = "all", "all"
    else:
        re_label_str = str(re_filter)
        re_name_str  = REWARD_ENUM_NAMES.get(re_filter, "?")

    def _total_row():
        label = f"TOTAL (re={re_label_str})"
        base = (
            f"| {label:<{w0}} | {'':<{w1}} | {global_hash:<{w2}} "
            f"| {tot_raw:>{w3}} | {tot_final:>{w4}} "
        )
        if has_sampled:
            tot_sampled = sum(sampled_counts.values())
            base += f"| {tot_sampled:>{w7}} "
        base += f"| {'':{w5}} | {'':{w6}} |"
        return base

    sep = _sep()
    logger.info(
        "Dataset Summary  "
        f"(game={getattr(config, 'dataset_game', '?')}, re={re_label_str}/{re_name_str}, "
        f"train_ratio={getattr(config, 'dataset_train_ratio', '?')})"
    )
    logger.info(sep)
    logger.info(_header())
    logger.info(sep)
    prev_game = None
    for row in rows_data:
        if not row["selected"]:
            continue
        if prev_game and prev_game != row["game"]:
            logger.info(sep)
        logger.info(_row(row))
        prev_game = row["game"]
    logger.info(sep)
    logger.info(_total_row())
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
