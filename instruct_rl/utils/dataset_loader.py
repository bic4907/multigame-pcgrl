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
        use_tile_mapping=False,
    )

    samples = list(ds) if load_game == "all" else ds.by_games(game_names)

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

