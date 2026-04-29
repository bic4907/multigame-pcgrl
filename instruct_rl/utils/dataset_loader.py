"""
instruct_rl/utils/dataset_loader.py
====================================
MultiGameDataset 기반 Instruct 빌더.
jax.jit 바깥에서 호출하여 데이터셋을 로드하고 Instruct 객체를 빌드한다.
"""
from __future__ import annotations

from collections import Counter

import jax
import jax.numpy as jnp
import hashlib
import numpy as np
import os
from pathlib import Path
from glob import glob
from os.path import basename, join

from instruct_rl.dataclass import Instruct
from instruct_rl.utils.log_utils import get_logger

from dataset.multigame import MultiGameDataset

from flax.training import checkpoints as flax_ckpts

logger = get_logger(__file__)
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_EVAL_CACHE_ROOT = Path(
    os.environ.get(
        "EVAL_CACHE_DIR",
        os.path.join(os.path.dirname(__file__), "..", "..", ".eval_cache"),
    )
).resolve()
_CLIP_EMBED_CACHE_DIR = _EVAL_CACHE_ROOT / "clip_latent_embeddings"
_DECODER_REWARD_CACHE_DIR = _EVAL_CACHE_ROOT / "decoder_reward_predictions"

# reward_enum → 사람이 읽을 수 있는 이름
REWARD_ENUM_NAMES = {
    0: "region",
    1: "path_length",
    2: "interactable",
    3: "hazard",
    4: "collectable",
}


def _parse_dataset_reward_enum_filter(raw_value, *, field_name: str = "dataset_reward_enum"):
    """dataset_reward_enum 설정을 정규화한다.

    Returns
    -------
    list[int] | None
        None 이면 필터 비활성화(=전체 reward_enum 허용).
        예: "01" -> [0, 1], "0,1" -> [0, 1], 2 -> [2]
    """
    if raw_value is None:
        return None

    parsed = None
    if isinstance(raw_value, str):
        v = raw_value.strip().lower()
        if v in ("", "none", "all"):
            return None
        try:
            if "," in v:
                parsed = [int(x.strip()) for x in v.split(",") if x.strip()]
            else:
                parsed = [int(c) for c in v]
        except ValueError as e:
            raise ValueError(
                f"Invalid {field_name}='{raw_value}'. "
                f"Use digits like '01', comma list like '0,1', or 'all'."
            ) from e
    elif isinstance(raw_value, int):
        parsed = [int(c) for c in str(raw_value)]
    else:
        try:
            parsed = [int(x) for x in raw_value]
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"Invalid {field_name}={raw_value!r}. "
                f"Use int/list, digits-string, comma-list, or 'all'."
            ) from e

    valid = set(REWARD_ENUM_NAMES.keys())
    normalized = []
    seen = set()
    for re in parsed:
        if re not in valid:
            raise ValueError(
                f"Invalid {field_name} value: {re}. "
                f"Valid enums are {sorted(valid)}."
            )
        if re not in seen:
            normalized.append(re)
            seen.add(re)
    return normalized if normalized else None


def _parse_reward_enum_list(raw_value, *, field_name: str = "eval_dataset_reward_enums"):
    """복수 reward_enum 설정을 int 리스트로 정규화한다.

    None/'none'/'' -> None
    'all'          -> [0,1,2,3,4]
    '012'          -> [0,1,2]   (기존 동작 유지)
    '0,2,4'        -> [0,2,4]
    """
    if raw_value is None:
        return None

    if isinstance(raw_value, str):
        v = raw_value.strip().lower()
        if v in ("", "none"):
            return None
        if v == "all":
            return sorted(REWARD_ENUM_NAMES.keys())
        try:
            if "," in v:
                return [int(x.strip()) for x in v.split(",") if x.strip()]
            return [int(c) for c in v]
        except ValueError as e:
            raise ValueError(
                f"Invalid {field_name}='{raw_value}'. "
                f"Use digits like '012', comma list like '0,1,2', or 'all'."
            ) from e

    if isinstance(raw_value, int):
        return [int(c) for c in str(raw_value)]

    try:
        return [int(x) for x in raw_value]
    except (TypeError, ValueError) as e:
        raise ValueError(
            f"Invalid {field_name}={raw_value!r}. "
            f"Use iterable of ints, digits-string, comma-list, or 'all'."
        ) from e


def load_dataset_instruct(config):
    """MultiGameDataset에서 Instruct 객체를 빌드한다.

    Parameters
    ----------
    config : TrainConfig
        dataset_game, dataset_reward_enum, dataset_train_ratio, seed, nlp_input_dim 필요.

    Returns
    -------
    (train_inst, test_inst) : tuple[Instruct, Instruct]
    """
    # eval_games가 지정된 경우 평가 데이터 로딩에 우선 사용 (체크포인트 경로는 game 기준 유지)
    _eval_games_str = getattr(config, 'eval_games', None)
    _load_game = _eval_games_str if _eval_games_str is not None else config.dataset_game

    # reward_enum 파싱 (로딩 로그·필터링·테이블 표시에서 공통 사용)
    _eval_re_raw = getattr(config, 'eval_dataset_reward_enums', None)
    _eval_re_list = _parse_reward_enum_list(_eval_re_raw, field_name="eval_dataset_reward_enums")
    _dataset_re_filter_list = _parse_dataset_reward_enum_filter(
        getattr(config, "dataset_reward_enum", None),
        field_name="dataset_reward_enum",
    )
    _effective_re = _eval_re_list if _eval_re_list is not None else (
        _dataset_re_filter_list
    )

    logger.info(
        f"Loading MultiGameDataset (game={_load_game}, reward_enum={_effective_re})"
        + (f"  [eval_games override: {_eval_games_str}]" if _eval_games_str else "")
    )

    # 'all'이면 전체 게임 로드, 약어면 역매핑으로 full name 리스트 획득
    from conf.game_utils import GAME_ABBR, ALL_GAMES, parse_game_str
    _dg = _load_game
    if _dg == 'all':
        _game_names = ALL_GAMES  # ['dungeon', 'pokemon', 'sokoban', 'doom', 'doom2', 'zelda']
    elif _dg in GAME_ABBR:
        _game_names = GAME_ABBR[_dg]  # 단일 약어 → full name 리스트
    elif len(_dg) % 2 == 0 and all(_dg[i:i+2] in GAME_ABBR for i in range(0, len(_dg), 2)):
        # 복합 약어 (예: "dgpk") → parse_game_str로 파싱
        includes = parse_game_str(_dg)
        _game_names = [name for name in ALL_GAMES if includes.get(f"include_{name}", False)]
    else:
        _game_names = [_dg]  # 이미 full name

    ds = MultiGameDataset(
        include_dungeon=('dungeon' in _game_names),
        include_pokemon=('pokemon' in _game_names),
        include_sokoban=('sokoban' in _game_names),
        include_doom=('doom' in _game_names),
        include_doom2=('doom2' in _game_names),
        include_zelda=('zelda' in _game_names),
        use_tile_mapping=False,
    )

    # 게임별 필터링 ('all'이면 전체 사용)
    if _dg == 'all':
        samples = list(ds)
    else:
        samples = ds.by_games(_game_names)

    # ── 게임별 처리 통계 표 출력 (분할/샘플링 후 최종 정보 포함) ──────────────

    # reward_enum 필터링 — 위에서 파싱한 _eval_re_list / _effective_re 사용
    if _eval_re_list is not None:
        _re_set = set(_eval_re_list)
        samples = [s for s in samples if s.meta.get("reward_enum") in _re_set]
        logger.info(f"eval_dataset_reward_enums={_eval_re_list}: {len(samples)} samples")
    elif _dataset_re_filter_list is not None:
        _re_set = set(_dataset_re_filter_list)
        samples = [s for s in samples if s.meta.get("reward_enum") in _re_set]
        logger.info(f"dataset_reward_enum={_dataset_re_filter_list}: {len(samples)} samples")

    # reward annotation이 있는 샘플만
    samples = [s for s in samples if "reward_enum" in s.meta and "conditions" in s.meta]

    # condition 값 기반 필터링
    cond_filter = getattr(config, "dataset_condition_filter", None)
    if cond_filter:
        filters = _parse_condition_filters(cond_filter)
        before = len(samples)
        samples = _apply_condition_filters(samples, filters)
        logger.info(f"Condition filter '{cond_filter}': {before} → {len(samples)} samples")

    assert len(samples) > 0, (
        f"No samples found for game={_load_game}, "
        f"reward_enum={getattr(config, 'dataset_reward_enum', None)}. "
        f"Check that reward annotations exist."
    )

    # ── eval 모드: (game, re) 그룹별 고정 수 서브샘플링 ──────────────────
    eval_samples_per_group = getattr(config, 'eval_samples_per_group', None)
    sampled_counts: dict = {}  # game → sampled count (테이블용)
    if eval_samples_per_group is not None:
        # eval_seed 가 있으면 우선 사용 (없으면 config.seed fallback)
        _subsample_seed = getattr(config, 'eval_seed', None)
        if _subsample_seed is None:
            _subsample_seed = config.seed
        samples, sampled_counts = _subsample_per_group(
            samples, eval_samples_per_group, seed=_subsample_seed
        )
        logger.info(
            f"[eval_samples_per_group={eval_samples_per_group}, seed={_subsample_seed}]"
            f" subsampled: {len(samples)} samples"
        )

    all_inst = _build_instruct(samples, config)

    _log_dataset_table(ds, samples, config, sampled_counts=sampled_counts,
                       re_filter_list=_effective_re)

    return all_inst, all_inst, samples


# ── 내부 헬퍼 ──────────────────────────────────────────────────────────────────


def _subsample_per_group(samples, n_per_group: int, seed: int = 0):
    """(game, re) 그룹별로 최대 n_per_group 개를 서브샘플링한다.

    가용 샘플이 n_per_group 보다 적은 그룹은 전부 사용.

    Returns
    -------
    subsampled : list
    sampled_counts : dict  game → sampled count  (re가 1개인 경우 game 기준)
    """
    import random as _random
    from collections import defaultdict

    # (game, re) 그룹핑
    by_group: dict = defaultdict(list)
    for s in samples:
        re = s.meta.get("reward_enum", None)
        by_group[(s.game, re)].append(s)

    # 그룹 내 순서 고정 (결정론적 보장)
    for key in by_group:
        by_group[key].sort(key=lambda s: str(getattr(s, 'source_id', s)))

    result = []
    sampled_counts: dict = {}  # game → count (re 단일 가정, 복수면 합산)

    for (game, re) in sorted(by_group.keys()):
        # 그룹별로 독립 시드 사용 → eval_games가 달라도 같은 게임은 같은 샘플이 뽑힘
        # NOTE: Python built-in hash() is randomized per-process (PYTHONHASHSEED).
        #       hashlib 기반 결정론적 해시 사용.
        _key_bytes = f"{game}_{re}".encode()
        _key_hash = int(hashlib.md5(_key_bytes).hexdigest(), 16) & 0xFFFFFFFF
        group_seed = seed ^ _key_hash
        group_rng = _random.Random(group_seed)
        pool = by_group[(game, re)][:]
        group_rng.shuffle(pool)
        chosen = pool[:n_per_group]
        result.extend(chosen)
        sampled_counts[game] = sampled_counts.get(game, 0) + len(chosen)

    # 최종 결과 셔플도 재현 가능하게 고정 시드 사용
    _random.Random(seed).shuffle(result)
    return result, sampled_counts

def _build_instruct(sample_list, config):
    """샘플 리스트에서 Instruct 객체를 빌드한다."""
    logger.info(
        "Building Instruct: samples=%d, use_clip=%s, use_nlp=%s, use_decoder=%s",
        len(sample_list),
        getattr(config, "use_clip", False),
        getattr(config, "use_nlp", False),
        hasattr(config, "decoder"),
    )

    # ── CLIP 모듈/체크포인트는 임베딩 계산과 디코더 리워드 예측에서 공유될 수 있다.
    # 두 경로가 모두 필요할 때 ckpt 를 한 번만 로드해 재사용한다.
    use_clip = getattr(config, "use_clip", False)
    needs_clip_embed   = use_clip and config.nlp_input_dim > 0
    needs_decoder_pred = use_clip and hasattr(config, "decoder")

    shared_module, shared_variables = None, None
    if needs_clip_embed or needs_decoder_pred:
        logger.info(
            "Loading shared CLIP module + checkpoint once "
            "(needs_embed=%s, needs_decoder=%s)",
            needs_clip_embed, needs_decoder_pred,
        )
        shared_module, shared_variables = _load_shared_clip_module_and_ckpt(config)


    embedding = _build_instruct_embedding(
        sample_list, config,
        shared_module=shared_module, shared_variables=shared_variables,
    )
    reward_i, condition = _build_reward_and_condition(
        sample_list, config,
        text_embeddings=embedding,
        shared_module=shared_module, shared_variables=shared_variables,
    )
    logger.info(
        "Built Instruct tensors: embedding=%s, reward_i=%s, condition=%s",
        embedding.shape,
        reward_i.shape,
        condition.shape,
    )
    condition_id = jnp.arange(len(sample_list), dtype=jnp.int32).reshape(-1, 1)

    if shared_module is not None or shared_variables is not None:
        del shared_module
        del shared_variables

    return Instruct(
        reward_i=reward_i,
        condition=condition,
        embedding=embedding,
        condition_id=condition_id,
    )

def _load_shared_clip_module_and_ckpt(config):
    """CLIP 인코더 모듈을 init 하고 encoder.ckpt_path 의 체크포인트를 1회 복원한다.

    임베딩 계산(`_compute_clip_embeddings`)과 디코더 리워드 예측
    (`_build_reward_and_condition_with_decoder`) 양쪽에서 동일한 모듈/파라미터를
    재사용하기 위한 헬퍼.

    Returns
    -------
    (module, variables)
    """
    encoder_config = config.encoder
    module, variables = _load_clip_encoder_module(config, encoder_config)
    variables = _restore_encoder_checkpoint(encoder_config, variables)
    return module, variables


def _build_instruct_embedding(sample_list, config, *,
                              shared_module=None, shared_variables=None):
    """설정값에 따라 텍스트 임베딩을 계산한다."""
    n = len(sample_list)
    if getattr(config, "use_clip", False) and config.nlp_input_dim > 0:
        embedding = _compute_clip_embeddings(
            sample_list, config,
            module=shared_module, variables=shared_variables,
        )
        logger.info("Embedding done: shape=%s", embedding.shape)
        return embedding
    if getattr(config, "use_nlp", False) and config.nlp_input_dim > 0:
        embedding = _compute_bert_embeddings(sample_list, config.nlp_input_dim)
        logger.info("Embedding done: shape=%s", embedding.shape)
        return embedding

    fallback_shape = (n, max(1, config.nlp_input_dim))
    logger.info("Embedding disabled: using zeros %s", fallback_shape)
    return jnp.zeros(fallback_shape, dtype=jnp.float32)


def _build_reward_and_condition(sample_list, config, *,
                                text_embeddings=None,
                                shared_module=None, shared_variables=None):
    """reward_i 및 condition 벡터를 생성한다."""
    if getattr(config, "use_clip", False) and hasattr(config, "decoder"):
        logger.info("Reward/Condition mode: CLIP decoder prediction path")
        return _build_reward_and_condition_with_decoder(
            sample_list, config,
            text_embeddings=text_embeddings,
            module=shared_module, variables=shared_variables,
        )

    logger.info("Reward/Condition mode: metadata fallback path")
    reward_i_list = []
    for s in sample_list:
        reward_i_list.append([s.meta["reward_enum"]])
    reward_i = jnp.array(reward_i_list, dtype=jnp.int32)

    condition_list = []
    for s in sample_list:
        conds = s.meta.get("conditions", {})
        row = []
        for i in range(0, 5):
            val = conds.get(i, conds.get(str(i), -1))
            row.append(float(val))
        condition_list.append(row)
    condition = jnp.array(condition_list, dtype=jnp.float32)
    return reward_i, condition


def _build_reward_and_condition_with_decoder(sample_list, config, *,
                                              text_embeddings=None,
                                              module=None, variables=None):
    """CLIP decoder 추론으로 reward_i/condition을 생성한다.

    Parameters
    ----------
    module, variables : optional
        `_load_shared_clip_module_and_ckpt` 에서 미리 로드된 모듈과 파라미터.
        제공되면 ckpt 를 다시 읽지 않고 그대로 재사용한다.
    """
    from conf.config import DecoderConfig
    from tqdm import tqdm

    n = len(sample_list)
    num_classes = config.decoder.num_reward_classes

    logger.info(
        "Decoder reward prediction: n=%d, num_classes=%d, ckpt=%s, reuse_ckpt=%s",
        n,
        num_classes,
        config.encoder.ckpt_path,
        module is not None and variables is not None,
    )

    # ── 모듈/파라미터 준비 ──
    # 외부에서 공유 인코더가 주어지면 그대로 사용하고, 그렇지 않으면
    # `load_decoder` 로 단독 로드한다 (이전 동작과 호환).
    if module is not None and variables is not None:
        decoder_apply_fn = module.apply
        decoder_vars = variables
    else:
        from encoder.utils.decoder_reward import load_decoder
        # NOTE: ckpt 와 동일한 구조로 디코더 모듈을 초기화해야 한다.
        # cnn_reward_enum_onehot 을 빠뜨리면 학습 시 (예: True → +5 channel) 와
        # 입력 채널 수가 달라져 Conv_0 에서 ScopeParamShapeError 가 발생한다.
        decoder_cfg = DecoderConfig(
            num_reward_classes=num_classes,
            hidden_dim=getattr(config.decoder, "hidden_dim", DecoderConfig.hidden_dim),
            num_layers=getattr(config.decoder, "num_layers", DecoderConfig.num_layers),
            output_dim=getattr(config.decoder, "output_dim", DecoderConfig.output_dim),
            cnn_reward_enum_onehot=getattr(
                config.decoder, "cnn_reward_enum_onehot",
                DecoderConfig.cnn_reward_enum_onehot,
            ),
        )
        decoder_apply_fn, decoder_vars = load_decoder(
            ckpt_dir=config.encoder.ckpt_path,
            encoder_config=config.encoder,
            decoder_config=decoder_cfg,
        )

    ckpt_signature = _checkpoint_signature_for_cache(decoder_vars)
    cache_key, cache_path = _build_decoder_reward_cache_path(
        sample_list, ckpt_signature=ckpt_signature
    )
    if cache_path.exists():
        try:
            cached = np.load(cache_path)
            reward_i_cached = cached["reward_i"]
            condition_cached = cached["condition"]
            expected_reward_shape = (n, 1)
            expected_condition_shape = (n, num_classes)
            if (
                reward_i_cached.shape == expected_reward_shape
                and condition_cached.shape == expected_condition_shape
            ):
                logger.info(
                    "Decoder reward cache HIT (hash=%s) — loaded from %s",
                    cache_key[:12],
                    cache_path,
                )
                logger.info(
                    "Decoder reward prediction done: reward_i=%s, condition=%s",
                    reward_i_cached.shape,
                    condition_cached.shape,
                )
                return (
                    jnp.array(reward_i_cached, dtype=jnp.int32),
                    jnp.array(condition_cached, dtype=jnp.float32),
                )
            logger.warning(
                "Decoder reward cache invalid shape at %s: reward_i=%s (expected=%s), condition=%s (expected=%s). Recomputing.",
                cache_path,
                reward_i_cached.shape,
                expected_reward_shape,
                condition_cached.shape,
                expected_condition_shape,
            )
        except Exception as e:
            logger.warning(
                "Failed to load decoder reward cache from %s: %s. Recomputing.",
                cache_path,
                e,
            )

    if text_embeddings is None:
        raise ValueError(
            "Decoder reward prediction requires precomputed text embeddings "
            "from Instruct. Got text_embeddings=None."
        )
    text_embeddings = jnp.array(text_embeddings, dtype=jnp.float32)
    if text_embeddings.shape[0] != n:
        raise ValueError(
            f"text_embeddings batch mismatch: got={text_embeddings.shape[0]}, expected={n}"
        )

    @jax.jit
    def _decode_batch(text_embed_batch):
        # reward branch는 text embedding을 사용한다.
        # ContrastiveDecoderModule.decoder 를 직접 호출해
        # reward logits / condition_pred_raw 를 얻는다.
        reward_logits, _, condition_pred_raw = decoder_apply_fn(
            decoder_vars,
            text_embed_batch,
            training=False,
            method=lambda m, embed, training=False: m.decoder(embed, training=training),
        )

        reward_enum_i = jnp.argmax(reward_logits, axis=-1).astype(jnp.int32)
        pred_cond = condition_pred_raw[
            jnp.arange(condition_pred_raw.shape[0]), reward_enum_i
        ]
        return reward_enum_i, pred_cond

    batch_size = 256
    total_batches = (n + batch_size - 1) // batch_size if n > 0 else 0
    logger.info("Decoder batching: batch_size=%d, total_batches=%d", batch_size, total_batches)
    reward_i_list = []
    cond_list = []
    pbar = tqdm(
        range(0, n, batch_size),
        total=total_batches,
        desc="Decoder reward",
        unit="batch",
    )
    for start_idx in pbar:
        end_idx = min(start_idx + batch_size, n)
        embed_batch = text_embeddings[start_idx:end_idx]
        reward_enum_i, pred_cond = _decode_batch(embed_batch)
        reward_i_list.append(np.array(reward_enum_i))
        cond_list.append(np.array(pred_cond))

    reward_i_flat = np.concatenate(reward_i_list, axis=0)
    pred_cond_raw = np.concatenate(cond_list, axis=0)

    reward_i = jnp.array(reward_i_flat, dtype=jnp.int32).reshape(-1, 1)

    # reward_enum_i 컬럼만 condition 값을 채우고, 나머지는 -1 sentinel 유지
    condition = jnp.full((n, num_classes), -1.0, dtype=jnp.float32)
    row_idx = jnp.arange(n)
    condition = condition.at[row_idx, reward_i_flat].set(jnp.array(pred_cond_raw))
    logger.info(
        "Decoder reward prediction done: reward_i=%s, condition=%s",
        reward_i.shape,
        condition.shape,
    )
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_path,
            reward_i=np.asarray(reward_i, dtype=np.int32),
            condition=np.asarray(condition, dtype=np.float32),
        )
        logger.info(
            "Decoder reward cache saved (hash=%s): %s",
            cache_key[:12],
            cache_path,
        )
    except Exception as e:
        logger.warning("Failed to save decoder reward cache to %s: %s", cache_path, e)

    return reward_i, condition


def _tokenize_texts(sample_list, encoder_config):
    """샘플 리스트에서 CLIP 토크나이저로 input_ids / attention_mask 를 반환한다.

    Returns
    -------
    input_ids      : jnp.ndarray  (N, token_max_len)
    attention_mask : jnp.ndarray  (N, token_max_len)
    has_text       : list[bool]   instruction 유무 플래그
    """
    from transformers import CLIPProcessor

    model_name = "openai/clip-vit-base-patch32"
    processor = CLIPProcessor.from_pretrained(model_name)

    texts, has_text = [], []
    for s in sample_list:
        if s.instruction is not None and len(s.instruction.strip()) > 0:
            texts.append(s.instruction)
            has_text.append(True)
        else:
            texts.append("")  # placeholder
            has_text.append(False)

    num_valid = sum(1 for flag in has_text if flag)
    logger.debug(
        "Tokenize CLIP texts: total=%d, non_empty=%d, max_len=%d",
        len(sample_list),
        num_valid,
        encoder_config.token_max_len,
    )

    inputs = processor(
        text=texts, return_tensors="jax",
        padding="max_length", truncation=True, max_length=encoder_config.token_max_len,
    )
    return (
        jnp.array(inputs["input_ids"]),
        jnp.array(inputs["attention_mask"]),
        has_text,
    )


def _load_clip_encoder_module(config, encoder_config):
    """config에 따라 ContrastiveModule 또는 ContrastiveDecoderModule을 초기화한다.

    Returns
    -------
    module    : flax.linen.Module
    variables : dict  (random initialized params)
    """
    from encoder.clip_model import get_cnnclip_encoder, get_cnnclip_decoder_encoder

    _use_decoder = hasattr(config, "decoder")
    if _use_decoder:
        from conf.config import DecoderConfig
        decoder_cfg = DecoderConfig(
            num_reward_classes=config.decoder.num_reward_classes,
            cnn_reward_enum_onehot=config.decoder.cnn_reward_enum_onehot
        )
        module, _ = get_cnnclip_decoder_encoder(
            encoder_config, decoder_config=decoder_cfg, RL_training=True,
        )
    else:
        module, _ = get_cnnclip_encoder(encoder_config, RL_training=True)

    # dummy forward 로 파라미터 형상 초기화
    rng = jax.random.PRNGKey(0)
    dummy_ids  = jnp.ones((1, encoder_config.token_max_len), dtype=jnp.int32)
    dummy_mask = jnp.ones((1, encoder_config.token_max_len), dtype=jnp.int32)
    dummy_pix  = jnp.ones((1, 16, 16, 6), dtype=jnp.float32)
    mode = "text_state" if encoder_config.state else "text"
    init_kwargs = dict(mode=mode, training=False)
    if _use_decoder:
        init_kwargs["reward_enum"] = jnp.zeros((1,), dtype=jnp.int32)

    variables = module.init(rng, dummy_ids, dummy_mask, dummy_pix, **init_kwargs)
    return module, variables


def _format_num_bytes(num_bytes: int) -> str:
    """바이트 수를 사람이 읽기 쉬운 문자열로 변환한다."""
    if num_bytes < 1024:
        return f"{num_bytes} B"
    if num_bytes < 1024 ** 2:
        return f"{num_bytes / 1024:.1f} KB"
    return f"{num_bytes / 1024 ** 2:.2f} MB"


def _compute_tree_signature_hash(tree, *, algo: str = "sha256"):
    """중첩 dict/PyTree의 결정론적 signature hash를 계산한다.

    Returns
    -------
    (signature_hex, leaf_count, total_bytes)
    """
    from flax.core import FrozenDict
    from flax.traverse_util import flatten_dict

    if isinstance(tree, FrozenDict):
        tree = tree.unfreeze()

    if isinstance(tree, dict):
        flat = flatten_dict(tree)
    else:
        flat = {("root",): tree}

    hasher = hashlib.new(algo)
    leaf_count = 0
    total_bytes = 0

    for path in sorted(flat.keys(), key=lambda p: tuple(str(k) for k in p)):
        leaf = flat[path]
        path_str = "/".join(str(k) for k in path)
        hasher.update(path_str.encode("utf-8"))
        hasher.update(b"\0")

        if leaf is None:
            hasher.update(b"NONE")
            leaf_count += 1
            continue

        try:
            arr = np.asarray(leaf)
            if arr.dtype == np.dtype("O"):
                raise TypeError("object dtype")
            arr = np.ascontiguousarray(arr)
            hasher.update(str(arr.shape).encode("utf-8"))
            hasher.update(b"|")
            hasher.update(str(arr.dtype).encode("utf-8"))
            hasher.update(b"|")
            hasher.update(memoryview(arr.reshape(-1).view(np.uint8)))
            leaf_count += 1
            total_bytes += int(arr.nbytes)
            continue
        except Exception:
            pass

        hasher.update(type(leaf).__name__.encode("utf-8"))
        hasher.update(b"|")
        hasher.update(repr(leaf).encode("utf-8"))
        leaf_count += 1

    return hasher.hexdigest(), leaf_count, total_bytes


def _checkpoint_signature_for_cache(variables) -> str:
    """캐시 키 생성을 위한 체크포인트 시그니처(hex)를 계산한다."""
    try:
        signature_hex, _, _ = _compute_tree_signature_hash(variables)
        return signature_hex
    except Exception:
        return "unknown"


def _build_clip_embedding_cache_path(sample_list, *, ckpt_signature: str) -> tuple[str, Path]:
    """CLIP 임베딩 캐시 키와 저장 경로를 생성한다."""
    hasher = hashlib.sha256()
    hasher.update(b"clip-latent-embedding-cache-v2")
    hasher.update(f"|ckpt_signature={ckpt_signature}".encode("utf-8"))

    for s in sample_list:
        hasher.update(f"|game={getattr(s, 'game', '')}".encode("utf-8"))
        hasher.update(f"|source_id={getattr(s, 'source_id', '')}".encode("utf-8"))
        instr = s.instruction.strip() if (s.instruction and s.instruction.strip()) else ""
        hasher.update(f"|instruction={instr}".encode("utf-8"))

    cache_key = hasher.hexdigest()
    return cache_key, _CLIP_EMBED_CACHE_DIR / f"{cache_key}.npy"


def _build_decoder_reward_cache_path(sample_list, *, ckpt_signature: str) -> tuple[str, Path]:
    """디코더 reward/condition 예측 캐시 키와 저장 경로를 생성한다."""
    hasher = hashlib.sha256()
    hasher.update(b"decoder-reward-cache-v4-text-embed")
    hasher.update(f"|ckpt_signature={ckpt_signature}".encode("utf-8"))

    for s in sample_list:
        hasher.update(f"|game={getattr(s, 'game', '')}".encode("utf-8"))
        hasher.update(f"|source_id={getattr(s, 'source_id', '')}".encode("utf-8"))
        instr = s.instruction.strip() if (s.instruction and s.instruction.strip()) else ""
        hasher.update(f"|instruction={instr}".encode("utf-8"))
        arr = np.asarray(s.array, dtype=np.int32)
        hasher.update(f"|array_shape={arr.shape}".encode("utf-8"))
        hasher.update(memoryview(np.ascontiguousarray(arr).reshape(-1).view(np.uint8)))

    cache_key = hasher.hexdigest()
    return cache_key, _DECODER_REWARD_CACHE_DIR / f"{cache_key}.npz"


def _log_checkpoint_signature_hash(state, *, ckpt_dir: str, step: int, fmt: str):
    """복원된 체크포인트 state의 signature hash를 로그로 출력한다."""
    try:
        signature_hex, leaf_count, total_bytes = _compute_tree_signature_hash(state)
        logger.info(
            "Encoder checkpoint signature hash: %s (algo=sha256, step=%d, format=%s, leaves=%d, bytes=%s)",
            signature_hex,
            step,
            fmt,
            leaf_count,
            _format_num_bytes(total_bytes),
        )
    except Exception as e:
        logger.warning(
            "Failed to compute checkpoint signature hash (step=%s, dir=%s): %s",
            step,
            ckpt_dir,
            e,
        )


def _restore_encoder_checkpoint(encoder_config, variables):
    """encoder_config.ckpt_path 에서 가장 최신 체크포인트를 복원한다.

    체크포인트가 없거나 로드 실패 시 초기 variables 를 그대로 반환한다.

    Returns
    -------
    variables : dict  (restored or original params)
    """
    ckpt_path = encoder_config.ckpt_path
    if ckpt_path is None:
        logger.warning("encoder.ckpt_path is None — using randomly initialized encoder for text projection")
        return variables

    ckpt_subdirs = glob(join(ckpt_path, '*'))
    ckpt_steps = sorted(
        [int(basename(d)) for d in ckpt_subdirs if basename(d).isdigit()],
        reverse=True,
    )
    if not ckpt_steps:
        logger.warning(f"No checkpoint steps found in {ckpt_path}")
        return variables

    ckpt_dir = join(ckpt_path, str(ckpt_steps[0]))
    enc_state = flax_ckpts.restore_checkpoint(ckpt_dir, target=None, prefix="")
    if enc_state is None:
        logger.warning(f"Checkpoint restore returned None from {ckpt_dir}")
        return variables

    # TrainState 형태: {"step": ..., "params": {"params": {...}}, "opt_state": ...}
    # module.apply 에는 {"params": {...}} 를 넘겨야 한다.
    if "params" in enc_state and "params" in enc_state["params"]:
        logger.info(f"Encoder checkpoint loaded (TrainState format) from {ckpt_dir} (step {ckpt_steps[0]})")
        restored_state = enc_state["params"]
        _log_checkpoint_signature_hash(
            restored_state,
            ckpt_dir=ckpt_dir,
            step=ckpt_steps[0],
            fmt="TrainState.params",
        )
        return restored_state

    logger.info(f"Encoder checkpoint loaded from {ckpt_dir} (step {ckpt_steps[0]})")
    _log_checkpoint_signature_hash(
        enc_state,
        ckpt_dir=ckpt_dir,
        step=ckpt_steps[0],
        fmt="raw",
    )
    return enc_state


def _encode_texts_batched(module, variables, input_ids, attention_mask, batch_size=256):
    """module.apply 를 배치 단위로 호출해 text_embed 를 수집한다.

    Returns
    -------
    embeddings : np.ndarray  (N, output_dim)
    """
    from tqdm import tqdm

    @jax.jit
    def _encode_batch(variables, ids, mask):
        return module.apply(
            variables, ids, mask, None,
            mode="text", training=False,
        )

    n = input_ids.shape[0]
    total_batches = (n + batch_size - 1) // batch_size if n > 0 else 0
    all_embeddings = []
    pbar = tqdm(
        range(0, n, batch_size),
        total=total_batches,
        desc="CLIP text embed",
        unit="batch",
    )
    for start in pbar:
        end = min(start + batch_size, n)
        out = _encode_batch(variables, input_ids[start:end], attention_mask[start:end])
        all_embeddings.append(np.array(out["text_embed"]))  # (batch, output_dim)

    return np.concatenate(all_embeddings, axis=0)  # (N, output_dim)


def _postprocess_embeddings(embeddings, has_text, nlp_input_dim):
    """instruction 없는 행을 zeros 로 채우고, nlp_input_dim 에 맞게 패딩/절삭한다.

    Returns
    -------
    embeddings : jnp.ndarray  (N, nlp_input_dim)
    """
    for i, ht in enumerate(has_text):
        if not ht:
            embeddings[i] = 0.0

    embed_dim = embeddings.shape[1]
    if nlp_input_dim > embed_dim:
        pad_width = ((0, 0), (0, nlp_input_dim - embed_dim))
        embeddings = np.pad(embeddings, pad_width, mode="constant")
    elif nlp_input_dim < embed_dim:
        embeddings = embeddings[:, :nlp_input_dim]

    return jnp.array(embeddings, dtype=jnp.float32)


def _compute_clip_embeddings(sample_list, config, *, module=None, variables=None):
    """사전학습된 CLIP encoder를 통해 instruction 텍스트 → latent embedding을 계산한다.

    1) openai/clip-vit-base-patch32 로 토크나이즈
    2) 사전학습된 ContrastiveModule 의 encode_text 를 사용하여
       512-dim raw CLIP → output_dim (e.g. 64) latent space 로 projection

    Parameters
    ----------
    module, variables : optional
        `_load_shared_clip_module_and_ckpt` 에서 미리 로드된 모듈/파라미터.
        제공되면 ckpt 를 다시 읽지 않고 그대로 재사용한다.
    """
    nlp_input_dim  = config.nlp_input_dim
    encoder_config = config.encoder

    # 모듈/체크포인트: 외부 공유본이 있으면 그대로, 없으면 단독 로드
    if module is None or variables is None:
        module, variables = _load_clip_encoder_module(config, encoder_config)
        variables = _restore_encoder_checkpoint(encoder_config, variables)

    ckpt_signature = _checkpoint_signature_for_cache(variables)
    cache_key, cache_path = _build_clip_embedding_cache_path(
        sample_list, ckpt_signature=ckpt_signature
    )
    if cache_path.exists():
        try:
            cached = np.load(cache_path)
            expected_shape = (len(sample_list), nlp_input_dim)
            if cached.shape == expected_shape:
                logger.info(
                    "CLIP latent embedding cache HIT (hash=%s) — loaded from %s",
                    cache_key[:12],
                    cache_path,
                )
                logger.info(f"CLIP latent embeddings shape: {cached.shape}")
                return jnp.array(cached, dtype=jnp.float32)
            logger.warning(
                "CLIP latent embedding cache invalid shape at %s: got=%s, expected=%s. Recomputing.",
                cache_path,
                cached.shape,
                expected_shape,
            )
        except Exception as e:
            logger.warning(
                "Failed to load CLIP latent embedding cache from %s: %s. Recomputing.",
                cache_path,
                e,
            )

    try:
        from transformers import CLIPProcessor  # noqa: F401 (import 가능 여부 확인)
    except ImportError as e:
        raise ImportError(
            "use_clip=True requires the `transformers` package. "
            "Install it with: pip install transformers"
        ) from e

    logger.info(
        "Computing CLIP latent embeddings (output_dim=%d) for %d samples (reuse_ckpt=%s)",
        encoder_config.output_dim, len(sample_list),
        module is not None and variables is not None,
    )

    # 1) 토크나이즈
    input_ids, attention_mask, has_text = _tokenize_texts(sample_list, encoder_config)

    # 4) 배치 인코딩
    clip_embeddings = _encode_texts_batched(module, variables, input_ids, attention_mask)

    # 5) 후처리 (zeros for no-text, pad/truncate to nlp_input_dim)
    result = _postprocess_embeddings(clip_embeddings, has_text, nlp_input_dim)

    logger.info(f"CLIP latent embeddings shape: {result.shape}")
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, np.asarray(result, dtype=np.float32))
        logger.info(
            "CLIP latent embedding cache saved (hash=%s): %s",
            cache_key[:12],
            cache_path,
        )
    except Exception as e:
        logger.warning("Failed to save CLIP latent embedding cache to %s: %s", cache_path, e)

    return result


def _compute_bert_embeddings(sample_list, nlp_input_dim):
    """BERT를 사용하여 instruction 텍스트에서 임베딩을 계산한다.

    instruction이 None인 샘플은 zeros 임베딩으로 대체된다.
    """
    import numpy as np

    try:
        from transformers import AutoTokenizer, FlaxAutoModel
    except ImportError as e:
        raise ImportError(
            "use_nlp=True requires the `transformers` package. "
            "Install it with: pip install transformers"
        ) from e

    model_name = "bert-base-uncased"  # 768-dim
    logger.info(f"Computing BERT embeddings with {model_name} for {len(sample_list)} samples")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = FlaxAutoModel.from_pretrained(model_name)

    texts = []
    has_text = []
    for s in sample_list:
        if s.instruction is not None and len(s.instruction.strip()) > 0:
            texts.append(s.instruction)
            has_text.append(True)
        else:
            texts.append("")  # placeholder
            has_text.append(False)

    # 배치 단위로 BERT forward → OOM 방지
    from tqdm import tqdm
    bert_batch_size = 64
    all_cls = []
    batches = range(0, len(texts), bert_batch_size)
    for i in tqdm(batches, desc="BERT embeddings", unit="batch"):
        batch_texts = texts[i: i + bert_batch_size]
        tokens = tokenizer(
            batch_texts, return_tensors="jax",
            padding=True, truncation=True, max_length=128,
        )
        outputs = model(**tokens)
        all_cls.append(np.array(outputs.last_hidden_state[:, 0, :]))

    cls_embeddings = np.concatenate(all_cls, axis=0)  # (N, 768)

    # instruction 없는 샘플은 zeros
    for i, ht in enumerate(has_text):
        if not ht:
            cls_embeddings[i] = 0.0

    # nlp_input_dim에 맞게 패딩 또는 절삭
    bert_dim = cls_embeddings.shape[1]
    if nlp_input_dim > bert_dim:
        pad_width = ((0, 0), (0, nlp_input_dim - bert_dim))
        cls_embeddings = np.pad(cls_embeddings, pad_width, mode="constant")
    elif nlp_input_dim < bert_dim:
        cls_embeddings = cls_embeddings[:, :nlp_input_dim]

    logger.info(f"BERT embeddings shape: {cls_embeddings.shape}")
    return jnp.array(cls_embeddings, dtype=jnp.float32)


def _log_dataset_table(ds, all_samples, config, *, sampled_counts: dict = None,
                       re_filter_list=None):
    """(game, re) 조합별 처리 통계를 표로 출력한다 (줄마다 별도 logger.info)."""
    import numpy as _np
    from collections import defaultdict

    has_sampled = bool(sampled_counts)
    col_sampled = "Sampled"

    # (game, re) → {n, instr, cond_vals}
    cell: dict = defaultdict(lambda: {"n": 0, "instr": 0, "cond_vals": []})

    for s in ds:
        re = s.meta.get("reward_enum")
        if re is None or "conditions" not in s.meta:
            continue
        key = (s.game, re)
        cell[key]["n"] += 1
        if s.instruction:
            cell[key]["instr"] += 1
        conds = s.meta.get("conditions", {})
        val = conds.get(re, conds.get(str(re), None))
        if val is not None:
            cell[key]["cond_vals"].append(float(val))

    # 캐시 해시 (앞 12자)
    cache_keys: dict = getattr(ds, "_game_cache_keys", {})

    games   = sorted({k[0] for k in cell})
    re_vals = sorted({k[1] for k in cell})
    # re_filter_list 가 있으면 그 목록 기준, 없으면 단일 dataset_reward_enum 기준
    if re_filter_list:
        _re_filter_set = set(re_filter_list)
        re_filter = None  # highlight 판단에 set 사용
    else:
        re_filter = _parse_dataset_reward_enum_filter(
            getattr(config, "dataset_reward_enum", None),
            field_name="dataset_reward_enum",
        )
        _re_filter_set = set(re_filter) if re_filter is not None else None

    col_game  = "Game"
    col_re    = "re(name)"
    col_hash  = "Hash"
    col_n     = "Samples"
    col_instr = "w/ Instr"
    col_cmin  = "cond_min"
    col_cmax  = "cond_max"

    rows_data = []
    for g in games:
        for re in re_vals:
            if (g, re) not in cell:
                continue
            c = cell[(g, re)]
            re_label = f"{re}({REWARD_ENUM_NAMES.get(re, '?')})"
            if _re_filter_set is not None:
                highlight = re in _re_filter_set
            else:
                highlight = True
            arr = _np.array(c["cond_vals"]) if c["cond_vals"] else None
            rows_data.append({
                "game":     g,
                "re":       re_label,
                "hash":     cache_keys.get(g, "")[:12],
                "n":        c["n"],
                "instr":    c["instr"],
                "cmin":     f"{arr.min():.1f}" if arr is not None else "-",
                "cmax":     f"{arr.max():.1f}" if arr is not None else "-",
                "selected": highlight,
            })

    selected = [r for r in rows_data if r["selected"]]
    if not selected:
        logger.info("No samples found for re=%s", re_filter)
        return

    w0 = max(len(col_game),  max(len(r["game"]) for r in selected))
    w1 = max(len(col_re),    max(len(r["re"])   for r in selected))
    w2 = max(len(col_hash),  max(len(r["hash"]) for r in selected))
    w3 = max(len(col_n),     max(len(str(r["n"]))     for r in selected))
    w4 = max(len(col_instr), max(len(str(r["instr"])) for r in selected))
    w5 = max(len(col_cmin),  max(len(r["cmin"]) for r in selected))
    w6 = max(len(col_cmax),  max(len(r["cmax"]) for r in selected))
    if has_sampled:
        w7 = max(len(col_sampled), max(len(str(sampled_counts.get(r["game"], "-"))) for r in selected))

    if has_sampled:
        sep = (f"+{'-'*(w0+2)}+{'-'*(w1+2)}+{'-'*(w2+2)}"
               f"+{'-'*(w3+2)}+{'-'*(w4+2)}+{'-'*(w7+2)}+{'-'*(w5+2)}+{'-'*(w6+2)}+")
        header = (f"| {col_game:<{w0}} | {col_re:<{w1}} | {col_hash:<{w2}} "
                  f"| {col_n:>{w3}} | {col_instr:>{w4}} "
                  f"| {col_sampled:>{w7}} | {col_cmin:>{w5}} | {col_cmax:>{w6}} |")
    else:
        sep = (f"+{'-'*(w0+2)}+{'-'*(w1+2)}+{'-'*(w2+2)}"
               f"+{'-'*(w3+2)}+{'-'*(w4+2)}+{'-'*(w5+2)}+{'-'*(w6+2)}+")
        header = (f"| {col_game:<{w0}} | {col_re:<{w1}} | {col_hash:<{w2}} "
                  f"| {col_n:>{w3}} | {col_instr:>{w4}} "
                  f"| {col_cmin:>{w5}} | {col_cmax:>{w6}} |")

    tot_n     = sum(r["n"]     for r in selected)
    tot_instr = sum(r["instr"] for r in selected)
    tot_sampled = sum(sampled_counts.values()) if has_sampled else None
    if _re_filter_set is not None:
        re_label_str = ",".join(str(r) for r in sorted(_re_filter_set))
        re_name_str  = ",".join(REWARD_ENUM_NAMES.get(r, "?") for r in sorted(_re_filter_set))
    elif re_filter is None:
        re_label_str, re_name_str = "all", "all"
    else:
        re_label_str = str(re_filter)
        re_name_str  = REWARD_ENUM_NAMES.get(re_filter, "?")
    if has_sampled:
        total_row = (f"| {'TOTAL (re='+re_label_str+')':<{w0}} | {'':<{w1}} | {'':<{w2}} "
                     f"| {tot_n:>{w3}} | {tot_instr:>{w4}} | {tot_sampled:>{w7}} | {'':{w5}} | {'':{w6}} |")
    else:
        total_row = (f"| {'TOTAL (re='+re_label_str+')':<{w0}} | {'':<{w1}} | {'':<{w2}} "
                     f"| {tot_n:>{w3}} | {tot_instr:>{w4}} | {'':{w5}} | {'':{w6}} |")

    logger.info("Dataset Summary  "
                f"(game={config.dataset_game}, re={re_label_str}/{re_name_str}, "
                f"train_ratio={config.dataset_train_ratio})")
    logger.info(sep)
    logger.info(header)
    logger.info(sep)
    prev_game = None
    for r in rows_data:
        if not r["selected"]:
            continue
        if prev_game and prev_game != r["game"]:
            logger.info(sep)
        if has_sampled:
            sc = sampled_counts.get(r["game"], "-")
            row = (f"| {r['game']:<{w0}} | {r['re']:<{w1}} | {r['hash']:<{w2}} "
                   f"| {r['n']:>{w3}} | {r['instr']:>{w4}} "
                   f"| {sc:>{w7}} | {r['cmin']:>{w5}} | {r['cmax']:>{w6}} |")
        else:
            row = (f"| {r['game']:<{w0}} | {r['re']:<{w1}} | {r['hash']:<{w2}} "
                   f"| {r['n']:>{w3}} | {r['instr']:>{w4}} "
                   f"| {r['cmin']:>{w5}} | {r['cmax']:>{w6}} |")
        logger.info(row)
        prev_game = r["game"]
    logger.info(sep)
    logger.info(total_row)
    logger.info(sep)


def _log_dataset_summary(config, samples):
    """reward_enum 필터 후 최종 샘플 요약 (condition 통계 포함)."""
    import numpy as _np

    re_counter = Counter(s.meta["reward_enum"] for s in samples)
    logger.info("Filtered samples: %d  (reward_enum breakdown below)", len(samples))
    for re_val in sorted(re_counter.keys()):
        re_samples = [s for s in samples if s.meta["reward_enum"] == re_val]
        cond_vals = []
        for s in re_samples:
            conds = s.meta.get("conditions", {})
            val = conds.get(re_val, conds.get(str(re_val), None))
            if val is not None:
                cond_vals.append(float(val))
        if cond_vals:
            arr = _np.array(cond_vals)
            logger.info(
                "  re=%d (%s): %d samples  cond[min=%.1f, max=%.1f, mean=%.2f, std=%.2f]",
                re_val, REWARD_ENUM_NAMES.get(re_val, "?"), len(re_samples),
                arr.min(), arr.max(), arr.mean(), arr.std(),
            )
        else:
            logger.info("  re=%d (%s): %d samples",
                        re_val, REWARD_ENUM_NAMES.get(re_val, "?"), len(re_samples))


def _log_split_summary(train_samples, test_samples, train_inst, *, sampled_counts: dict = None):
    """Train/Test 분할 결과 요약. sampled_counts가 있으면 Sampled 컬럼을 추가한다."""
    train_game = Counter(s.game for s in train_samples)
    test_game  = Counter(s.game for s in test_samples)
    games = sorted(set(train_game) | set(test_game))

    has_sampled = bool(sampled_counts)

    col_game    = "Game"
    col_train   = "Train"
    col_test    = "Test"
    col_sampled = "Sampled"

    w0 = max(len(col_game),  max(len(g) for g in games))
    w1 = max(len(col_train), max(len(str(train_game[g])) for g in games))
    w2 = max(len(col_test),  max(len(str(test_game[g]))  for g in games))
    if has_sampled:
        w3 = max(len(col_sampled), max(len(str(sampled_counts.get(g, 0))) for g in games))

    if has_sampled:
        sep    = f"+{'-'*(w0+2)}+{'-'*(w1+2)}+{'-'*(w2+2)}+{'-'*(w3+2)}+"
        header = f"| {col_game:<{w0}} | {col_train:>{w1}} | {col_test:>{w2}} | {col_sampled:>{w3}} |"
        total_sampled = sum(sampled_counts.values())
        total_row = (f"| {'TOTAL':<{w0}} | {len(train_samples):>{w1}} | "
                     f"{len(test_samples):>{w2}} | {total_sampled:>{w3}} |")
    else:
        sep    = f"+{'-'*(w0+2)}+{'-'*(w1+2)}+{'-'*(w2+2)}+"
        header = f"| {col_game:<{w0}} | {col_train:>{w1}} | {col_test:>{w2}} |"
        total_row = (f"| {'TOTAL':<{w0}} | {len(train_samples):>{w1}} | "
                     f"{len(test_samples):>{w2}} |")

    logger.debug("Train/Test Split  "
                f"(total=%d, train=%d, test=%d)",
                len(train_samples) + len(test_samples),
                len(train_samples), len(test_samples))
    logger.debug(sep)
    logger.debug(header)
    logger.debug(sep)
    for g in games:
        if has_sampled:
            row = f"| {g:<{w0}} | {train_game[g]:>{w1}} | {test_game[g]:>{w2}} | {sampled_counts.get(g, 0):>{w3}} |"
        else:
            row = f"| {g:<{w0}} | {train_game[g]:>{w1}} | {test_game[g]:>{w2}} |"
        logger.debug(row)
    logger.debug(sep)
    logger.debug(total_row)
    logger.debug(sep)


# ── Condition 필터 유틸 ────────────────────────────────────────────────────────

import re as _re
from dataclasses import dataclass as _dataclass
from typing import Optional as _Optional, List as _List


@_dataclass
class _ConditionFilter:
    """단일 condition 필터 조건."""
    enum_idx: int                   # condition index (0~4)
    min_val: _Optional[float] = None  # inclusive lower bound
    max_val: _Optional[float] = None  # inclusive upper bound


def _parse_condition_filters(filter_str: str) -> _List[_ConditionFilter]:
    """필터 문자열을 파싱한다.

    포맷 (쉼표로 여러 개 구분):
        enum_{i}_min_{lo}_max_{hi}   — lo ≤ condition[i] ≤ hi
        enum_{i}_min_{lo}            — lo ≤ condition[i]
        enum_{i}_max_{hi}            — condition[i] ≤ hi

    Examples
    --------
        "enum_0_min_3_max_10"        → condition[0] in [3, 10]
        "enum_0_min_3_max_10,enum_2_max_50"  → 두 필터 AND
    """
    pattern = _re.compile(
        r"enum_(\d+)"
        r"(?:_min_([\d.]+))?"
        r"(?:_max_([\d.]+))?"
    )
    filters = []
    for token in filter_str.split(","):
        token = token.strip()
        if not token:
            continue
        m = pattern.fullmatch(token)
        if m is None:
            raise ValueError(
                f"Invalid condition filter: '{token}'. "
                f"Expected format: enum_{{i}}_min_{{lo}}_max_{{hi}} "
                f"(min/max are each optional but at least one required)"
            )
        idx = int(m.group(1))
        min_val = float(m.group(2)) if m.group(2) is not None else None
        max_val = float(m.group(3)) if m.group(3) is not None else None
        if min_val is None and max_val is None:
            raise ValueError(
                f"Condition filter 'enum_{idx}' has neither min nor max — "
                f"at least one bound is required."
            )
        filters.append(_ConditionFilter(enum_idx=idx, min_val=min_val, max_val=max_val))
    return filters


def _apply_condition_filters(samples, filters: _List[_ConditionFilter]):
    """필터 리스트를 AND 조합으로 적용하여 샘플을 필터링한다."""
    for f in filters:
        def _keep(s, _f=f):
            conds = s.meta.get("conditions", {})
            val = conds.get(_f.enum_idx, conds.get(str(_f.enum_idx), None))
            if val is None:
                return False
            val = float(val)
            if _f.min_val is not None and val < _f.min_val:
                return False
            if _f.max_val is not None and val > _f.max_val:
                return False
            return True
        samples = [s for s in samples if _keep(s)]
    return samples
