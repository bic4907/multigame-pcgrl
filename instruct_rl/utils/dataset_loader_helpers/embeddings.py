from __future__ import annotations

import hashlib
import json
import os
from glob import glob
from os.path import basename, join
from pathlib import Path
from typing import Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from flax.training import checkpoints as flax_ckpts

from instruct_rl.dataclass import Instruct
from instruct_rl.utils.log_utils import get_logger

from .constants import CLIP_EMBED_CACHE_DIR, DECODER_REWARD_CACHE_DIR

logger = get_logger(__file__)


# ═══════════════════════════════════════════════════════════════════════════════
#  Norm stats file I/O
# ═══════════════════════════════════════════════════════════════════════════════

NORM_STATS_FILENAME = "norm_stats.json"


def load_norm_stats(ckpt_dir: str) -> Optional[Tuple[Dict[int, float], Dict[int, float]]]:
    """Load ``norm_stats.json`` from near ckpt_dir.

    Reads the file saved by ``save_norm_stats`` during training and returns a
    ``(cond_norm_min, cond_norm_max)`` dict pair.
    Keys are integer reward_enum (0-based), values are float.

    Returns ``None`` if the file is missing or cannot be parsed.
    """
    abs_ckpt = os.path.abspath(ckpt_dir)
    candidates = [
        # exp_dir level (sibling of ckpts/) — primary location
        os.path.join(os.path.dirname(abs_ckpt), NORM_STATS_FILENAME),
        # inside ckpts/ — legacy / fallback
        os.path.join(abs_ckpt, NORM_STATS_FILENAME),
        # parent of parent (in case ckpt_dir is a step sub-directory)
        os.path.join(os.path.dirname(os.path.dirname(abs_ckpt)), NORM_STATS_FILENAME),
    ]

    for path in candidates:
        if os.path.isfile(path):
            try:
                with open(path, "r") as f:
                    data = json.load(f)
                cond_norm_min = {int(k): float(v) for k, v in data["cond_norm_min"].items()}
                cond_norm_max = {int(k): float(v) for k, v in data["cond_norm_max"].items()}
                logger.info("Norm stats loaded from %s", path)
                return cond_norm_min, cond_norm_max
            except Exception as e:
                logger.warning("Failed to load norm stats from %s: %s", path, e)

    logger.warning("norm_stats.json not found near ckpt_dir=%s.", ckpt_dir)
    return None


def denorm_condition(
    cond_norm: np.ndarray,
    reward_enum_indices: np.ndarray,
    cond_norm_min: Dict[int, float],
    cond_norm_max: Dict[int, float],
    num_classes: int,
) -> np.ndarray:
    """Denormalize condition values from [0,1] log-space back to the original linear scale.

    Inverse of the training-time preprocessing:
        log_val = cond_norm * (norm_max - norm_min) + norm_min
        raw     = expm1(max(log_val, 0))

    Parameters
    ----------
    cond_norm : (N,) ndarray
        Normalized condition values selected per reward_enum.
    reward_enum_indices : (N,) int ndarray
        Predicted reward_enum (0-based) for each sample.
    cond_norm_min, cond_norm_max : dict[int, float]
        Per-reward_enum log-space min/max values.
    num_classes : int
        Number of reward classes.

    Returns
    -------
    (N,) ndarray — condition values in the original linear scale.
    """
    min_arr = np.array([cond_norm_min.get(i, 0.0) for i in range(num_classes)], dtype=np.float32)
    max_arr = np.array([cond_norm_max.get(i, 1.0) for i in range(num_classes)], dtype=np.float32)

    r_min = min_arr[reward_enum_indices]   # (N,)
    r_max = max_arr[reward_enum_indices]   # (N,)

    log_val = cond_norm * (r_max - r_min) + r_min
    raw = np.expm1(np.maximum(log_val, 0.0))
    return raw


def _build_instruct(sample_list, config):
    """샘플 리스트에서 Instruct 객체를 빌드한다."""
    logger.info(
        "Building Instruct: samples=%d, use_clip=%s, use_nlp=%s, use_decoder=%s",
        len(sample_list),
        getattr(config, "use_clip", False),
        getattr(config, "use_nlp", False),
        hasattr(config, "decoder"),
    )

    use_clip = getattr(config, "use_clip", False)
    needs_clip_embed = use_clip and config.nlp_input_dim > 0
    needs_decoder_pred = use_clip and hasattr(config, "decoder")

    shared_module, shared_variables = None, None
    if needs_clip_embed or needs_decoder_pred:
        logger.info(
            "Loading shared CLIP module + checkpoint once "
            "(needs_embed=%s, needs_decoder=%s)",
            needs_clip_embed,
            needs_decoder_pred,
        )
        shared_module, shared_variables = _load_shared_clip_module_and_ckpt(config)

    embedding = _build_instruct_embedding(
        sample_list,
        config,
        shared_module=shared_module,
        shared_variables=shared_variables,
    )
    reward_i, condition = _build_reward_and_condition(
        sample_list,
        config,
        text_embeddings=embedding,
        shared_module=shared_module,
        shared_variables=shared_variables,
    )
    logger.info(
        "Built Instruct tensors: embedding=%s, reward_i=%s, condition=%s",
        embedding.shape,
        reward_i.shape,
        condition.shape,
    )
    condition_id = jnp.arange(len(sample_list), dtype=jnp.int32).reshape(-1, 1)

    if hasattr(config, 'coef_human_sim') and config.coef_human_sim:
        level_array = jnp.array(
            np.stack([s.array for s in sample_list], axis=0).astype(np.int32)
        )  # (N, H, W)
    else:
        level_array = None

    if shared_module is not None or shared_variables is not None:
        del shared_module
        del shared_variables

    return Instruct(
        reward_i=reward_i,
        condition=condition,
        embedding=embedding,
        condition_id=condition_id,
        level=level_array,
    )


def _load_shared_clip_module_and_ckpt(config):
    """CLIP 인코더 모듈을 init 하고 encoder.ckpt_path 의 체크포인트를 1회 복원한다."""
    encoder_config = config.encoder
    module, variables = _load_clip_encoder_module(config, encoder_config)
    variables = _restore_encoder_checkpoint(encoder_config, variables)
    return module, variables


def _build_instruct_embedding(sample_list, config, *, shared_module=None, shared_variables=None):
    """설정값에 따라 텍스트 임베딩을 계산한다."""
    n = len(sample_list)
    if getattr(config, "use_clip", False) and config.nlp_input_dim > 0:
        embedding = _compute_clip_embeddings(
            sample_list,
            config,
            module=shared_module,
            variables=shared_variables,
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


def _build_reward_and_condition(
    sample_list,
    config,
    *,
    text_embeddings=None,
    shared_module=None,
    shared_variables=None,
):
    """reward_i 및 condition 벡터를 생성한다.

    동작 순서:
    1. 항상 데이터셋 메타데이터로 전체 배열을 초기화한다.
    2. reward_decoder_mode에 따라 decoder로 덮어쓸 범위를 결정한다:
       - "noop"  : 초기화한 메타데이터를 그대로 반환 (decoder 미사용)
       - "all"   : 전체 게임을 decoder 예측값으로 덮어쓴다
       - "unseen": unseen 게임 샘플만 decoder 예측값으로 덮어쓴다
    """
    use_decoder = getattr(config, "use_clip", False) and hasattr(config, "decoder")
    num_classes = config.decoder.num_reward_classes if use_decoder else 5
    reward_decoder_mode = getattr(config, "reward_decoder_mode", "unseen")
    n = len(sample_list)

    # ── 1. 메타데이터로 전체 배열 초기화 (항상) ───────────────────────────────
    reward_i_arr = np.zeros((n, 1), dtype=np.int32)
    condition_arr = np.full((n, num_classes), -1.0, dtype=np.float32)
    for idx, sample in enumerate(sample_list):
        reward_i_arr[idx, 0] = sample.meta["reward_enum"]
        conds = sample.meta.get("conditions", {})
        for j in range(num_classes):
            val = conds.get(j, conds.get(str(j), -1))
            condition_arr[idx, j] = float(val)
    logger.info("Reward/Condition: metadata loaded for all %d samples", n)

    # ── 2. "noop" 모드 또는 decoder 미사용: 메타데이터 그대로 반환 ─────────────
    if reward_decoder_mode == "noop" or not use_decoder:
        logger.info("Reward/Condition mode: noop (metadata only, decoder not applied)")
        return jnp.array(reward_i_arr, dtype=jnp.int32), jnp.array(condition_arr, dtype=jnp.float32)

    # ── 3. decoder로 덮어쓸 target_indices 결정 ───────────────────────────────
    if reward_decoder_mode == "unseen":
        reward_seen_games = set(getattr(config, "reward_seen_games", None) or [])
        # doom / doom2 alias
        if "doom" in reward_seen_games or "doom2" in reward_seen_games:
            reward_seen_games.update({"doom", "doom2"})

        # reward_seen_games 가 비어있으면 (encoder 가 seen 게임 없이 unseen 게임만
        # 학습한 경우 — source_target 실험의 source 없는 baseline) 모든 게임이
        # unseen 이다. 이때도 아래 로직이 그대로 성립하므로 별도 분기하지 않는다.
        # 특히 reward_unseen_ratio > 0 이면 few-shot 분할이 유지되어야 한다:
        # encoder 가 학습한 앞쪽 비율은 baseline 에서도 metadata 로 남아야
        # source 가 있는 run 과 조건이 맞는다.
        if not reward_seen_games:
            logger.info(
                "reward_decoder_mode=unseen with empty reward_seen_games — "
                "all games treated as unseen"
            )

        reward_unseen_ratio = float(getattr(config, "reward_unseen_ratio", 0.0))

        if reward_unseen_ratio > 0.0:
            # ── few-shot 분할: 각 unseen 게임 내에서 샘플을 순서 기준으로 분할 ──
            # 앞쪽 (reward_unseen_ratio 비율) → metadata 유지 (encoder 학습분)
            # 나머지 (1 - reward_unseen_ratio) → decoder 적용
            from collections import defaultdict
            unseen_game_indices: dict = defaultdict(list)
            for i, s in enumerate(sample_list):
                if s.game not in reward_seen_games:
                    unseen_game_indices[s.game].append(i)

            decoder_set: set = set()
            for game, indices in unseen_game_indices.items():
                n_meta = int(len(indices) * reward_unseen_ratio)
                # 앞쪽 n_meta 개 → metadata, 나머지 → decoder
                for idx in indices[n_meta:]:
                    decoder_set.add(idx)

            target_indices = sorted(decoder_set)
            n_meta_total = n - len(target_indices)
            logger.info(
                "Reward/Condition mode: unseen→split (reward_unseen_ratio=%.4f) "
                "metadata=%d samples, decoder=%d samples",
                reward_unseen_ratio, n_meta_total, len(target_indices),
            )
        else:
            # ── zero-shot: 모든 unseen 게임 샘플에 decoder 적용 ──────────
            target_indices = [i for i, s in enumerate(sample_list) if s.game not in reward_seen_games]
            logger.info(
                "Reward/Condition mode: unseen→decoder (zero-shot) "
                "(seen=%d samples metadata, unseen=%d samples → decoder)",
                n - len(target_indices),
                len(target_indices),
            )
    else:
        # "all": 전체를 decoder로 덮어쓴다
        target_indices = list(range(n))
        logger.info("Reward/Condition mode: all (%d samples → decoder)", n)

    if not target_indices:
        logger.info("No decoder target samples — returning metadata as-is")
        return jnp.array(reward_i_arr, dtype=jnp.int32), jnp.array(condition_arr, dtype=jnp.float32)

    # ── 4. 대상 samples에 대해 decoder 예측 후 덮어쓰기 ──────────────────────
    target_samples = [sample_list[i] for i in target_indices]
    target_embeddings = (
        np.asarray(text_embeddings)[target_indices]
        if text_embeddings is not None
        else None
    )

    reward_i_dec, cond_dec = _build_reward_and_condition_with_decoder(
        target_samples,
        config,
        text_embeddings=target_embeddings,
        module=shared_module,
        variables=shared_variables,
    )
    reward_i_dec_np = np.asarray(reward_i_dec, dtype=np.int32)
    cond_dec_np = np.asarray(cond_dec, dtype=np.float32)

    for local_i, orig_i in enumerate(target_indices):
        reward_i_arr[orig_i] = reward_i_dec_np[local_i]
        condition_arr[orig_i] = cond_dec_np[local_i]

    logger.info(
        "Reward/Condition: decoder overwrite done — reward_i=%s, condition=%s",
        reward_i_arr.shape,
        condition_arr.shape,
    )
    return jnp.array(reward_i_arr, dtype=jnp.int32), jnp.array(condition_arr, dtype=jnp.float32)


def _build_reward_and_condition_with_decoder(
    sample_list,
    config,
    *,
    text_embeddings=None,
    module=None,
    variables=None,
):
    """CLIP decoder 추론으로 reward_i/condition을 생성한다."""
    from conf.config import DecoderConfig

    n = len(sample_list)
    num_classes = config.decoder.num_reward_classes

    logger.info(
        "Decoder reward prediction: n=%d, num_classes=%d, ckpt=%s, reuse_ckpt=%s",
        n,
        num_classes,
        config.encoder.ckpt_path,
        module is not None and variables is not None,
    )

    if module is not None and variables is not None:
        decoder_apply_fn = module.apply
        decoder_vars = variables
    else:
        from encoder.utils.decoder_reward import load_decoder

        decoder_cfg = DecoderConfig(
            num_reward_classes=num_classes,
            hidden_dim=getattr(config.decoder, "hidden_dim", DecoderConfig.hidden_dim),
            num_layers=getattr(config.decoder, "num_layers", DecoderConfig.num_layers),
            output_dim=getattr(config.decoder, "output_dim", DecoderConfig.output_dim),
            cnn_reward_enum_onehot=getattr(
                config.decoder,
                "cnn_reward_enum_onehot",
                DecoderConfig.cnn_reward_enum_onehot,
            ),
        )
        decoder_apply_fn, decoder_vars = load_decoder(
            ckpt_dir=config.encoder.ckpt_path,
            encoder_config=config.encoder,
            decoder_config=decoder_cfg,
        )

    ckpt_signature = _checkpoint_signature_for_cache(decoder_vars)
    _instr_prefix = _resolve_instruction_prefix(config)
    cache_key, cache_path = _build_decoder_reward_cache_path(
        sample_list,
        ckpt_signature=ckpt_signature,
        instruction_prefix=_instr_prefix,
    )

    # ── Validate norm stats exist before cache check (required for denorm) ──
    # Load and validate early so that missing stats are caught regardless of cache state.
    _loaded_norm_stats = load_norm_stats(config.encoder.ckpt_path)
    if _loaded_norm_stats is None:
        raise FileNotFoundError(
            f"norm_stats.json not found (ckpt_path={config.encoder.ckpt_path}). "
            "It is saved automatically when training with train_clip_decoder.py. "
            "Check the checkpoint path or retrain the model."
        )
    _ext_norm_min, _ext_norm_max = _loaded_norm_stats
    logger.info(
        "Norm stats ready for denorm: enums=%s",
        sorted(_ext_norm_min.keys()),
    )

    _use_cache = getattr(config, "use_embedding_cache", True)
    if _use_cache and cache_path.exists():
        try:
            cached = np.load(cache_path)
            reward_i_cached = cached["reward_i"]
            condition_cached = cached["condition"]
            expected_reward_shape = (n, 1)
            expected_condition_shape = (n, num_classes)
            if reward_i_cached.shape == expected_reward_shape and condition_cached.shape == expected_condition_shape:
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
    _text_embed_arr = jnp.array(text_embeddings, dtype=jnp.float32)
    if _text_embed_arr.shape[0] != n:
        raise ValueError(
            f"text_embeddings batch mismatch: got={_text_embed_arr.shape[0]}, expected={n}"
        )

    # ── 배치 추론 + denorm: reward_decode에 위임 ──
    from encoder.utils.decoder_reward import reward_decode

    reward_i, condition = reward_decode(
        text_embeddings=_text_embed_arr,
        apply_fn=decoder_apply_fn,
        variables=decoder_vars,
        cond_norm_min=_ext_norm_min,
        cond_norm_max=_ext_norm_max,
        num_classes=num_classes,
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


def _tokenize_texts(sample_list, encoder_config, instruction_prefix: str = "none"):
    """샘플 리스트에서 CLIP 토크나이저로 input_ids / attention_mask 를 반환한다."""
    from transformers import CLIPProcessor
    import random as _random

    model_name = "openai/clip-vit-base-patch32"
    processor = CLIPProcessor.from_pretrained(model_name)

    # instruction_prefix dispatcher 를 인코더 학습 코드와 공유
    from encoder.data.clip_batch import apply_instruction_prefix

    # 재현성을 위해 고정 시드로 rng 생성
    _rng = _random.Random(42)

    texts, has_text = [], []
    for sample in sample_list:
        if sample.instruction is not None and len(sample.instruction.strip()) > 0:
            inst = sample.instruction
            if getattr(sample, "game", None):
                inst = apply_instruction_prefix(inst, sample.game, _rng, instruction_prefix)
            texts.append(inst)
            has_text.append(True)
        else:
            texts.append("")
            has_text.append(False)

    num_valid = sum(1 for flag in has_text if flag)
    _example = next((t for t in texts if t), "")
    logger.info(
        "Prepended instruction_prefix='%s' to %d instructions "
        "(total=%d, max_len=%d, example: '%s')",
        instruction_prefix,
        num_valid,
        len(sample_list),
        encoder_config.token_max_len,
        _example[:120],
    )

    inputs = processor(
        text=texts,
        return_tensors="jax",
        padding="max_length",
        truncation=True,
        max_length=encoder_config.token_max_len,
    )
    return (
        jnp.array(inputs["input_ids"]),
        jnp.array(inputs["attention_mask"]),
        has_text,
    )


def _load_clip_encoder_module(config, encoder_config):
    """config에 따라 ContrastiveModule 또는 ContrastiveDecoderModule을 초기화한다.

    분기:
      - model == 'pretrained_clip'  → encoder.pretrained_clip_model.ContrastiveModule
        (HF CLIP wrapper, `final_text_projection` 없음 — pretrained CLIP ckpt 와 일치)
      - model == 'finetuned_clip'   → encoder.finetuned_clip_model 의 trainable 변종
        (구조는 pretrained 과 동일, stop_gradient 만 제거)
      - 그 외                       → 기존 cnnclip / cnnclip+decoder 경로
        (`encoder.clip_model.PretrainedTextEncoder` 의 `final_text_projection` 분기 사용)
    """
    _model = getattr(config, "model", None)

    # ── pretrained / finetuned CLIP 분기 ──────────────────────────────────
    if _model in ("pretrained_clip", "finetuned_clip"):
        if _model == "finetuned_clip":
            from encoder.finetuned_clip_model import (
                get_finetuned_clip_encoder as _get_clip,
            )
        else:
            from encoder.pretrained_clip_model import (
                get_pretrained_clip_encoder as _get_clip,
            )
        module, _ = _get_clip(encoder_config)

        rng = jax.random.PRNGKey(0)
        dummy_ids = jnp.ones((1, encoder_config.token_max_len), dtype=jnp.int32)
        dummy_mask = jnp.ones((1, encoder_config.token_max_len), dtype=jnp.int32)
        # HF pretrained CLIP vision tower 는 항상 224×224×3 RGB 입력
        dummy_pix = jnp.ones((1, 224, 224, 3), dtype=jnp.float32)
        mode = "text_state" if encoder_config.state else "text"
        variables = module.init(
            rng, input_ids=dummy_ids, attention_mask=dummy_mask,
            pixel_values=dummy_pix, mode=mode,
        )
        return module, variables

    # ── 기존 cnnclip 경로 ─────────────────────────────────────────────────
    from encoder.clip_model import get_cnnclip_decoder_encoder, get_cnnclip_encoder

    use_decoder = hasattr(config, "decoder")
    if use_decoder:
        from conf.config import DecoderConfig

        decoder_cfg = DecoderConfig(
            num_reward_classes=config.decoder.num_reward_classes,
            cnn_reward_enum_onehot=config.decoder.cnn_reward_enum_onehot,
        )
        module, _ = get_cnnclip_decoder_encoder(
            encoder_config,
            decoder_config=decoder_cfg,
            RL_training=True,
        )
    else:
        module, _ = get_cnnclip_encoder(encoder_config, RL_training=True)

    rng = jax.random.PRNGKey(0)
    dummy_ids = jnp.ones((1, encoder_config.token_max_len), dtype=jnp.int32)
    dummy_mask = jnp.ones((1, encoder_config.token_max_len), dtype=jnp.int32)
    dummy_pix = jnp.ones((1, 16, 16, 6), dtype=jnp.float32)
    mode = "text_state" if encoder_config.state else "text"
    init_kwargs = dict(mode=mode, training=False)
    if use_decoder:
        init_kwargs["reward_enum"] = jnp.zeros((1,), dtype=jnp.int32)

    variables = module.init(rng, dummy_ids, dummy_mask, dummy_pix, **init_kwargs)
    return module, variables


def _format_num_bytes(num_bytes: int) -> str:
    """바이트 수를 사람이 읽기 쉬운 문자열로 변환한다."""
    if num_bytes < 1024:
        return f"{num_bytes} B"
    if num_bytes < 1024**2:
        return f"{num_bytes / 1024:.1f} KB"
    return f"{num_bytes / 1024**2:.2f} MB"


def _compute_tree_signature_hash(tree, *, algo: str = "sha256"):
    """중첩 dict/PyTree의 결정론적 signature hash를 계산한다."""
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


def _resolve_instruction_prefix(config) -> str:
    """config에서 instruction_prefix 모드 문자열을 읽어 normalize한다.

    Returns "name" / "desc" / "none" 중 하나.
    """
    from encoder.data.clip_batch import _normalize_instruction_prefix_mode
    return _normalize_instruction_prefix_mode(getattr(config, "instruction_prefix", "none"))


def _build_clip_embedding_cache_path(
    sample_list, *, ckpt_signature: str, instruction_prefix: str = "none"
) -> tuple[str, Path]:
    """CLIP 임베딩 캐시 키와 저장 경로를 생성한다."""
    hasher = hashlib.sha256()
    hasher.update(b"clip-latent-embedding-cache-v2")
    hasher.update(f"|ckpt_signature={ckpt_signature}".encode("utf-8"))
    hasher.update(f"|instruction_prefix={instruction_prefix}".encode("utf-8"))

    for sample in sample_list:
        hasher.update(f"|game={getattr(sample, 'game', '')}".encode("utf-8"))
        hasher.update(f"|source_id={getattr(sample, 'source_id', '')}".encode("utf-8"))
        instr = sample.instruction.strip() if (sample.instruction and sample.instruction.strip()) else ""
        hasher.update(f"|instruction={instr}".encode("utf-8"))

    cache_key = hasher.hexdigest()
    return cache_key, CLIP_EMBED_CACHE_DIR / f"{cache_key}.npy"


def _build_decoder_reward_cache_path(
    sample_list, *, ckpt_signature: str, instruction_prefix: str = "none"
) -> tuple[str, Path]:
    """디코더 reward/condition 예측 캐시 키와 저장 경로를 생성한다."""
    hasher = hashlib.sha256()
    hasher.update(b"decoder-reward-cache-v4-text-embed")
    hasher.update(f"|ckpt_signature={ckpt_signature}".encode("utf-8"))
    # 디코더 입력은 CLIP text embedding이므로 instruction_prefix 가 바뀌면
    # text embedding도 바뀐다 → 캐시 키에 반드시 포함해야 한다.
    hasher.update(f"|instruction_prefix={instruction_prefix}".encode("utf-8"))

    for sample in sample_list:
        hasher.update(f"|game={getattr(sample, 'game', '')}".encode("utf-8"))
        hasher.update(f"|source_id={getattr(sample, 'source_id', '')}".encode("utf-8"))
        instr = sample.instruction.strip() if (sample.instruction and sample.instruction.strip()) else ""
        hasher.update(f"|instruction={instr}".encode("utf-8"))
        arr = np.asarray(sample.array, dtype=np.int32)
        hasher.update(f"|array_shape={arr.shape}".encode("utf-8"))
        hasher.update(memoryview(np.ascontiguousarray(arr).reshape(-1).view(np.uint8)))

    cache_key = hasher.hexdigest()
    return cache_key, DECODER_REWARD_CACHE_DIR / f"{cache_key}.npz"


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
    """encoder_config.ckpt_path 에서 가장 최신 체크포인트를 복원한다."""
    ckpt_path = encoder_config.ckpt_path
    if ckpt_path is None:
        logger.warning(
            "encoder.ckpt_path is None — using randomly initialized encoder for text projection"
        )
        return variables

    ckpt_subdirs = glob(join(ckpt_path, "*"))
    ckpt_steps = sorted(
        [int(basename(d)) for d in ckpt_subdirs if basename(d).isdigit()],
        reverse=True,
    )
    if not ckpt_steps:
        logger.warning("No checkpoint steps found in %s", ckpt_path)
        return variables

    ckpt_dir = join(ckpt_path, str(ckpt_steps[0]))
    enc_state = flax_ckpts.restore_checkpoint(ckpt_dir, target=None, prefix="")
    if enc_state is None:
        logger.warning("Checkpoint restore returned None from %s", ckpt_dir)
        return variables

    if "params" in enc_state and "params" in enc_state["params"]:
        logger.info(
            "Encoder checkpoint loaded (TrainState format) from %s (step %s)",
            ckpt_dir,
            ckpt_steps[0],
        )
        restored_state = enc_state["params"]
        _log_checkpoint_signature_hash(
            restored_state,
            ckpt_dir=ckpt_dir,
            step=ckpt_steps[0],
            fmt="TrainState.params",
        )
        return restored_state

    logger.info("Encoder checkpoint loaded from %s (step %s)", ckpt_dir, ckpt_steps[0])
    _log_checkpoint_signature_hash(
        enc_state,
        ckpt_dir=ckpt_dir,
        step=ckpt_steps[0],
        fmt="raw",
    )
    return enc_state


def _encode_texts_batched(module, variables, input_ids, attention_mask, batch_size=256):
    """module.apply 를 배치 단위로 호출해 text_embed 를 수집한다."""
    from tqdm import tqdm

    @jax.jit
    def _encode_batch(variables, ids, mask):
        return module.apply(variables, ids, mask, None, mode="text", training=False)

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
        all_embeddings.append(np.array(out["text_embed"]))

    return np.concatenate(all_embeddings, axis=0)


def _postprocess_embeddings(embeddings, has_text, nlp_input_dim):
    """instruction 없는 행을 zeros 로 채우고, nlp_input_dim 에 맞게 패딩/절삭한다."""
    for i, has_text_flag in enumerate(has_text):
        if not has_text_flag:
            embeddings[i] = 0.0

    embed_dim = embeddings.shape[1]
    if nlp_input_dim > embed_dim:
        pad_width = ((0, 0), (0, nlp_input_dim - embed_dim))
        embeddings = np.pad(embeddings, pad_width, mode="constant")
    elif nlp_input_dim < embed_dim:
        embeddings = embeddings[:, :nlp_input_dim]

    return jnp.array(embeddings, dtype=jnp.float32)


def _compute_clip_embeddings(sample_list, config, *, module=None, variables=None):
    """사전학습된 CLIP encoder를 통해 instruction 텍스트 -> latent embedding을 계산한다."""
    nlp_input_dim = config.nlp_input_dim
    encoder_config = config.encoder

    if module is None or variables is None:
        module, variables = _load_clip_encoder_module(config, encoder_config)
        variables = _restore_encoder_checkpoint(encoder_config, variables)

    ckpt_signature = _checkpoint_signature_for_cache(variables)
    _instr_prefix = _resolve_instruction_prefix(config)
    cache_key, cache_path = _build_clip_embedding_cache_path(
        sample_list,
        ckpt_signature=ckpt_signature,
        instruction_prefix=_instr_prefix,
    )
    _use_cache = getattr(config, "use_embedding_cache", True)
    if _use_cache and cache_path.exists():
        try:
            cached = np.load(cache_path)
            expected_shape = (len(sample_list), nlp_input_dim)
            if cached.shape == expected_shape:
                logger.info(
                    "CLIP latent embedding cache HIT (hash=%s) — loaded from %s",
                    cache_key[:12],
                    cache_path,
                )
                logger.info("CLIP latent embeddings shape: %s", cached.shape)
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
        from transformers import CLIPProcessor  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "use_clip=True requires the `transformers` package. "
            "Install it with: pip install transformers"
        ) from e

    logger.info(
        "Computing CLIP latent embeddings (output_dim=%d) for %d samples (reuse_ckpt=%s)",
        encoder_config.output_dim,
        len(sample_list),
        module is not None and variables is not None,
    )

    _instr_prefix = _resolve_instruction_prefix(config)
    input_ids, attention_mask, has_text = _tokenize_texts(
        sample_list, encoder_config, instruction_prefix=_instr_prefix
    )
    clip_embeddings = _encode_texts_batched(module, variables, input_ids, attention_mask)
    result = _postprocess_embeddings(clip_embeddings, has_text, nlp_input_dim)

    logger.info("CLIP latent embeddings shape: %s", result.shape)
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
    """BERT를 사용하여 instruction 텍스트에서 임베딩을 계산한다."""
    try:
        from transformers import AutoTokenizer, FlaxAutoModel
    except ImportError as e:
        raise ImportError(
            "use_nlp=True requires the `transformers` package. "
            "Install it with: pip install transformers"
        ) from e

    model_name = "bert-base-uncased"
    logger.info("Computing BERT embeddings with %s for %d samples", model_name, len(sample_list))

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = FlaxAutoModel.from_pretrained(model_name)

    texts = []
    has_text = []
    for sample in sample_list:
        if sample.instruction is not None and len(sample.instruction.strip()) > 0:
            texts.append(sample.instruction)
            has_text.append(True)
        else:
            texts.append("")
            has_text.append(False)

    from tqdm import tqdm

    bert_batch_size = 256
    all_cls = []
    batches = range(0, len(texts), bert_batch_size)
    for i in tqdm(batches, desc="BERT embeddings", unit="batch"):
        batch_texts = texts[i : i + bert_batch_size]
        tokens = tokenizer(
            batch_texts,
            return_tensors="jax",
            padding=True,
            truncation=True,
            max_length=128,
        )
        outputs = model(**tokens)
        all_cls.append(np.array(outputs.last_hidden_state[:, 0, :]))

    cls_embeddings = np.concatenate(all_cls, axis=0)

    for i, has_text_flag in enumerate(has_text):
        if not has_text_flag:
            cls_embeddings[i] = 0.0

    bert_dim = cls_embeddings.shape[1]
    if nlp_input_dim > bert_dim:
        pad_width = ((0, 0), (0, nlp_input_dim - bert_dim))
        cls_embeddings = np.pad(cls_embeddings, pad_width, mode="constant")
    elif nlp_input_dim < bert_dim:
        cls_embeddings = cls_embeddings[:, :nlp_input_dim]

    logger.info("BERT embeddings shape: %s", cls_embeddings.shape)
    return jnp.array(cls_embeddings, dtype=jnp.float32)

