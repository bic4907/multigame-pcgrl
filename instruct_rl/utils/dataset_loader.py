"""
instruct_rl/utils/dataset_loader.py
====================================
MultiGameDataset 기반 Instruct 빌더.
jax.jit 바깥에서 호출하여 데이터셋을 로드하고 Instruct 객체를 빌드한다.
"""
from __future__ import annotations

from dataset.multigame import MultiGameDataset
from instruct_rl.utils.log_utils import get_logger
from instruct_rl.utils.dataset_loader_helpers.constants import REWARD_ENUM_NAMES
from instruct_rl.utils.dataset_loader_helpers.embeddings import (
    _build_clip_embedding_cache_path,
    _build_decoder_reward_cache_path,
    _build_instruct,
    _build_instruct_embedding,
    _build_reward_and_condition,
    _build_reward_and_condition_with_decoder,
    _checkpoint_signature_for_cache,
    _compute_bert_embeddings,
    _compute_clip_embeddings,
    _compute_tree_signature_hash,
    _encode_texts_batched,
    _format_num_bytes,
    _load_clip_encoder_module,
    _load_shared_clip_module_and_ckpt,
    _log_checkpoint_signature_hash,
    _postprocess_embeddings,
    _restore_encoder_checkpoint,
    _tokenize_texts,
    denorm_condition,
    load_norm_stats,
)
from instruct_rl.utils.dataset_loader_helpers.filters import (
    _ConditionFilter,
    _apply_condition_filters,
    _parse_condition_filters,
    _parse_dataset_reward_enum_filter,
    _parse_reward_enum_list,
)
from instruct_rl.utils.dataset_loader_helpers.reporting import (
    _log_dataset_summary,
    _log_dataset_table,
    _log_split_summary,
)
from instruct_rl.utils.dataset_loader_helpers.sampling import _subsample_per_group

logger = get_logger(__file__)


def load_dataset_instruct(config):
    """MultiGameDataset에서 Instruct 객체를 빌드한다."""
    eval_games_str = getattr(config, "eval_games", None)
    load_game = eval_games_str if eval_games_str is not None else config.dataset_game

    eval_re_raw = getattr(config, "eval_dataset_reward_enums", None)
    eval_re_list = _parse_reward_enum_list(eval_re_raw, field_name="eval_dataset_reward_enums")
    dataset_re_filter_list = _parse_dataset_reward_enum_filter(
        getattr(config, "dataset_reward_enum", None),
        field_name="dataset_reward_enum",
    )
    effective_re = eval_re_list if eval_re_list is not None else dataset_re_filter_list

    logger.info(
        f"Loading MultiGameDataset (game={load_game}, reward_enum={effective_re})"
        + (f"  [eval_games override: {eval_games_str}]" if eval_games_str else "")
    )

    from conf.game_utils import ALL_GAMES, GAME_ABBR, parse_game_str

    if load_game == "all":
        game_names = ALL_GAMES
    elif load_game in GAME_ABBR:
        game_names = GAME_ABBR[load_game]
    elif len(load_game) % 2 == 0 and all(load_game[i : i + 2] in GAME_ABBR for i in range(0, len(load_game), 2)):
        includes = parse_game_str(load_game)
        game_names = [name for name in ALL_GAMES if includes.get(f"include_{name}", False)]
    else:
        game_names = [load_game]

    ds = MultiGameDataset(
        include_dungeon=("dungeon" in game_names),
        include_pokemon=("pokemon" in game_names),
        include_sokoban=("sokoban" in game_names),
        include_doom=("doom" in game_names),
        include_doom2=("doom2" in game_names),
        include_zelda=("zelda" in game_names),
        use_tile_mapping=True,
        max_samples_per_game=getattr(config, "max_samples_per_game", 0),
        instruction_field=getattr(config, "instruction_field", "uni"),
    )

    samples = list(ds) if load_game == "all" else ds.by_games(game_names)

    from instruct_rl.utils.dataset_loader_helpers.preprocessing import preprocess_samples, apply_tile_offset
    samples = preprocess_samples(
        samples,
        longtail_cut=getattr(config, "longtail_cut", True),
    )
    samples = apply_tile_offset(samples, getattr(config, "rl_tile_offset", 0))

    if eval_re_list is not None:
        re_set = set(eval_re_list)
        samples = [s for s in samples if s.meta.get("reward_enum") in re_set]
        logger.info("eval_dataset_reward_enums=%s: %d samples", eval_re_list, len(samples))
    elif dataset_re_filter_list is not None:
        re_set = set(dataset_re_filter_list)
        samples = [s for s in samples if s.meta.get("reward_enum") in re_set]
        logger.info("dataset_reward_enum=%s: %d samples", dataset_re_filter_list, len(samples))

    samples = [s for s in samples if "reward_enum" in s.meta and "conditions" in s.meta]

    cond_filter = getattr(config, "dataset_condition_filter", None)
    if cond_filter:
        filters = _parse_condition_filters(cond_filter)
        before = len(samples)
        samples = _apply_condition_filters(samples, filters)
        logger.info("Condition filter '%s': %d -> %d samples", cond_filter, before, len(samples))

    assert len(samples) > 0, (
        f"No samples found for game={load_game}, "
        f"reward_enum={getattr(config, 'dataset_reward_enum', None)}. "
        f"Check that reward annotations exist."
    )

    # ── per-game ratio 필터링 ──────────────────────────────────────────────────
    # dataset_unseen_ratio 가 명시된 경우: seen/unseen 게임별 다른 비율 적용
    #   seen  게임 → dataset_seen_ratio
    #   unseen 게임 → dataset_unseen_ratio  (0.0 이면 해당 게임 제외)
    # dataset_unseen_ratio 가 None(미설정)인 경우: 기존 동작 (dataset_seen_ratio 만 사용)
    dataset_seen_ratio = getattr(config, "dataset_seen_ratio", 1.0)
    dataset_unseen_ratio = getattr(config, "dataset_unseen_ratio", None)

    if dataset_unseen_ratio is not None:
        from collections import defaultdict
        from conf.game_utils import compute_seen_unseen_split as _cu_split
        _reward_seen_raw = getattr(config, "reward_seen_games", None) or []
        _seen_set, _ = _cu_split(_reward_seen_raw)
        reward_seen_set: set = set(_seen_set)
        # doom / doom2 alias: encoder 가 한쪽을 seen 으로 기록하면 양쪽 모두 seen 으로 처리
        if "doom" in reward_seen_set or "doom2" in reward_seen_set:
            reward_seen_set.update({"doom", "doom2"})

        game_buckets: dict = defaultdict(list)
        for s in samples:
            game_buckets[s.game].append(s)
        filtered: list = []
        for game, bucket in sorted(game_buckets.items()):
            is_seen = game in reward_seen_set
            ratio = dataset_seen_ratio if is_seen else dataset_unseen_ratio
            if ratio <= 0.0:
                logger.info(
                    "per-game ratio: game=%s (%s) ratio=0 → skipped",
                    game, "seen" if is_seen else "unseen",
                )
                continue
            n_use = max(1, int(len(bucket) * ratio)) if ratio < 1.0 else len(bucket)
            filtered.extend(bucket[:n_use])
            logger.info(
                "per-game ratio: game=%s (%s) %.4f  %d → %d samples",
                game, "seen" if is_seen else "unseen", ratio, len(bucket), n_use,
            )
        samples = filtered
        logger.info(
            "per-game ratio filtering done: total %d samples", len(samples),
        )
    elif dataset_seen_ratio < 1.0:
        # 기존 동작: 모든 게임에 동일한 seen_ratio 적용
        from collections import defaultdict
        game_buckets = defaultdict(list)
        for s in samples:
            game_buckets[s.game].append(s)
        filtered = []
        for game, bucket in sorted(game_buckets.items()):
            n_use = max(1, int(len(bucket) * dataset_seen_ratio))
            filtered.extend(bucket[:n_use])
            logger.info(
                "dataset_seen_ratio=%.4f: game=%s  %d → %d samples",
                dataset_seen_ratio, game, len(bucket), n_use,
            )
        samples = filtered
        logger.info(
            "dataset_seen_ratio=%.4f: total %d samples after per-game ratio filtering",
            dataset_seen_ratio, len(samples),
        )

    eval_samples_per_group = getattr(config, "eval_samples_per_group", None)
    sampled_counts: dict = {}
    if eval_samples_per_group is not None:
        subsample_seed = getattr(config, "eval_seed", None)
        if subsample_seed is None:
            subsample_seed = config.seed
        samples, sampled_counts = _subsample_per_group(
            samples,
            eval_samples_per_group,
            seed=subsample_seed,
        )
        logger.info(
            "[eval_samples_per_group=%s, seed=%s] subsampled: %d samples",
            eval_samples_per_group,
            subsample_seed,
            len(samples),
        )

    all_inst = _build_instruct(samples, config)
    _log_dataset_table(
        ds,
        samples,
        config,
        sampled_counts=sampled_counts,
        re_filter_list=effective_re,
    )
    return all_inst, all_inst, samples

