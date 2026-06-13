"""
encoder/utils/decoder_reward.py
================================
학습된 CLIP Decoder 체크포인트를 로드하여,
instruction(text) embedding → (reward_enum, condition) 예측을 수행하는 유틸리티.

RL 학습 루프에서 `get_reward_batch` 대신 디코더가 예측한
reward_enum / condition 으로 보상을 계산할 때 사용한다.
"""

from __future__ import annotations

import logging
import os
from typing import Callable, Dict, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from flax.training import checkpoints

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
#  Checkpoint 로드
# ═══════════════════════════════════════════════════════════════════════════════

def load_decoder(
    ckpt_dir: str | None,
    encoder_config,
    decoder_config,
    cond_norm_min: jnp.ndarray | None = None,
    cond_norm_max: jnp.ndarray | None = None,
) -> Tuple:
    """CLIP Decoder 체크포인트를 로드한다.

    Returns
    -------
    (apply_fn, variables)
        apply_fn : ContrastiveDecoderModule.apply
        variables : {"params": ..., "norm_stats": ...}
    """
    from encoder.clip_model import get_cnnclip_decoder_encoder

    # 체크포인트는 RL_training=False 로 학습되었으므로 동일한 모드로 module 생성
    # → params tree 구조가 체크포인트와 일치
    module, _ = get_cnnclip_decoder_encoder(
        encoder_config,
        decoder_config=decoder_config,
        cond_norm_min=cond_norm_min,
        cond_norm_max=cond_norm_max,
        RL_training=False,
    )

    # Dummy init to get pytree structure
    rng = jax.random.PRNGKey(0)
    dummy_ids = jnp.ones((1, encoder_config.token_max_len), dtype=jnp.int32)
    dummy_mask = jnp.ones((1, encoder_config.token_max_len), dtype=jnp.int32)
    from dataset.multigame.tile_utils import NUM_CATEGORIES
    dummy_pix = jnp.ones((1, 16, 16, NUM_CATEGORIES + 2), dtype=jnp.float32)
    dummy_reward_enum = jnp.zeros((1,), dtype=jnp.int32)

    mode = "text_state" if encoder_config.state else "text"
    variables_template = module.init(
        rng, dummy_ids, dummy_mask, dummy_pix,
        prev_pixel_values=dummy_pix,
        curr_pixel_values=dummy_pix,
        reward_enum=dummy_reward_enum,
        mode=mode, training=False,
    )

    ckpt_dir = os.path.abspath(ckpt_dir)
    if not os.path.exists(ckpt_dir):
        logger.warning(f"Decoder checkpoint path not found: {ckpt_dir}. Falling back to initialized decoder.")
        return module.apply, variables_template

    # 최신 step 디렉토리 탐색 (ckpt_dir 이 ckpts/ 레벨일 수 있음)
    from glob import glob
    from os.path import basename, join
    step_dirs = [d for d in glob(join(ckpt_dir, '*')) if basename(d).isdigit()]
    if step_dirs:
        latest_step = max(step_dirs, key=lambda d: int(basename(d)))
        restore_dir = latest_step
    else:
        restore_dir = ckpt_dir

    # ── target 기반 복원 ──
    # train_clip_decoder 는 TrainState.create(params=variables) 로 저장하므로
    # 체크포인트 = {"step": ..., "params": <variables>, "opt_state": ...}
    # <variables> = {"params": {...}, "norm_stats": {...}}
    # flax restore_checkpoint 에 target={"params": template} 을 넘기면
    # 체크포인트의 "params" 필드를 template 구조로 매핑해 복원한다.
    from flax.training.train_state import TrainState
    import optax

    dummy_state = TrainState.create(
        apply_fn=module.apply,
        params=variables_template,
        tx=optax.identity(),  # optimizer는 불필요 (추론 전용)
    )

    restored_state = checkpoints.restore_checkpoint(
        restore_dir, target=dummy_state, prefix=""
    )
    if restored_state is None:
        logger.warning(f"Checkpoint restore returned None from {restore_dir}. Using initialized decoder.")
        return module.apply, variables_template

    variables = restored_state.params  # {"params": {...}, "norm_stats": {...}}
    logger.info(f"Decoder checkpoint loaded from {restore_dir}")
    logger.info(f"  variables keys: {list(variables.keys())}")

    return module.apply, variables


# ═══════════════════════════════════════════════════════════════════════════════
#  배치 추론: text_embeddings → (reward_i, condition)
# ═══════════════════════════════════════════════════════════════════════════════

def reward_decode(
    text_embeddings: jnp.ndarray,
    apply_fn,
    variables: dict,
    cond_norm_min: Dict[int, float],
    cond_norm_max: Dict[int, float],
    num_classes: int,
    batch_size: int = 256,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """CLIP 디코더로 텍스트 임베딩에서 reward_i / condition을 배치 예측한다.

    Parameters
    ----------
    text_embeddings : (N, D) jnp.ndarray
        CLIP encoder를 통과한 latent embedding.
    apply_fn : Callable
        ContrastiveDecoderModule.apply (또는 동일 시그니처의 함수).
    variables : dict
        디코더 파라미터 변수 딕셔너리 (``{"params": ..., "norm_stats": ...}``).
    cond_norm_min, cond_norm_max : Dict[int, float]
        reward_enum별 log-space 정규화 min/max 값.
    num_classes : int
        reward_enum 클래스 수.
    batch_size : int
        배치 크기 (기본값 256).

    Returns
    -------
    reward_i : (N, 1) jnp.int32
        예측된 reward_enum (0-based).
    condition : (N, num_classes) jnp.float32
        예측된 condition 벡터. 선택된 enum 슬롯만 값이 채워지고 나머지는 -1.
    """
    raise NotImplementedError(
        "Enum/condition reward decoding has been replaced by the transition "
        "reward model. Use predict_transition_reward or "
        "build_transition_reward_inject_fn instead."
    )
    from tqdm import tqdm

    n = text_embeddings.shape[0]
    text_embeddings = jnp.array(text_embeddings, dtype=jnp.float32)

    @jax.jit
    def _decode_batch(embed_batch):
        reward_logits, condition_pred, _ = apply_fn(
            variables,
            embed_batch,
            training=False,
            method=lambda m, embed, training=False: m.decoder(embed, training=training),
        )
        reward_enum_i = jnp.argmax(reward_logits, axis=-1).astype(jnp.int32)
        pred_cond_norm = condition_pred[jnp.arange(condition_pred.shape[0]), reward_enum_i]
        return reward_enum_i, pred_cond_norm

    total_batches = (n + batch_size - 1) // batch_size if n > 0 else 0
    logger.info("Decoder batching: batch_size=%d, total_batches=%d", batch_size, total_batches)

    reward_i_list: list = []
    cond_norm_list: list = []
    pbar = tqdm(range(0, n, batch_size), total=total_batches, desc="Decoder reward", unit="batch")
    for start_idx in pbar:
        end_idx = min(start_idx + batch_size, n)
        reward_enum_i, pred_cond_norm = _decode_batch(text_embeddings[start_idx:end_idx])
        reward_i_list.append(np.array(reward_enum_i))
        cond_norm_list.append(np.array(pred_cond_norm))

    reward_i_flat = np.concatenate(reward_i_list, axis=0)       # (N,)
    pred_cond_norm_all = np.concatenate(cond_norm_list, axis=0)  # (N,)

    # ── Denorm: [0,1] log-space → original linear scale ──
    min_arr = np.array([cond_norm_min.get(i, 0.0) for i in range(num_classes)], dtype=np.float32)
    max_arr = np.array([cond_norm_max.get(i, 1.0) for i in range(num_classes)], dtype=np.float32)
    r_min = min_arr[reward_i_flat]
    r_max = max_arr[reward_i_flat]
    log_val = pred_cond_norm_all * (r_max - r_min) + r_min
    pred_cond_raw = np.expm1(np.maximum(log_val, 0.0))
    logger.info(
        "Denorm applied: raw range=[%.4f, %.4f]",
        float(pred_cond_raw.min()),
        float(pred_cond_raw.max()),
    )

    # ── 최종 텐서 조립 ──
    reward_i = jnp.array(reward_i_flat, dtype=jnp.int32).reshape(-1, 1)
    condition = jnp.full((n, num_classes), -1.0, dtype=jnp.float32)
    condition = condition.at[jnp.arange(n), reward_i_flat].set(jnp.array(pred_cond_raw))
    logger.info(
        "Decoder reward prediction done: reward_i=%s, condition=%s",
        reward_i.shape,
        condition.shape,
    )
    return reward_i, condition


# ═══════════════════════════════════════════════════════════════════════════════
#  추론: state_embed → (reward_enum, condition)
# ═══════════════════════════════════════════════════════════════════════════════

def predict_reward_condition(
    apply_fn,
    variables: dict,
    instruction_embedding: jnp.ndarray,
    num_reward_classes: int = 5,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """디코더를 사용하여 instruction embedding으로부터
    reward_enum (1-based)과 condition 벡터를 예측한다.

    Parameters
    ----------
    instruction_embedding : (n_envs, D)
        사전학습된 CLIP encoder를 통과한 latent embedding (e.g. 64-dim).
    """
    raise NotImplementedError(
        "Enum/condition reward prediction has been replaced by direct scalar "
        "transition reward prediction. Use predict_transition_reward instead."
    )
    n_envs = instruction_embedding.shape[0]

    # ContrastiveDecoderModule 내부 decoder 서브모듈을 직접 호출한다.
    reward_logits, _, condition_pred_raw = apply_fn(
        variables,
        instruction_embedding,
        training=False,
        method=lambda m, embed, training=False: m.decoder(embed, training=training),
    )

    reward_logits = reward_logits[:, :num_reward_classes]  # (n_envs, num_classes)
    condition_pred_raw = condition_pred_raw[:, :num_reward_classes]  # (n_envs, num_classes) — 원래 스케일

    # argmax → 0-based → 1-based
    pred_enum_0based = jnp.argmax(reward_logits, axis=-1)    # (n_envs,)
    reward_i = (pred_enum_0based + 1).reshape(-1, 1).astype(jnp.int32)  # (n_envs, 1)

    # condition 벡터 구성: (n_envs, 9)
    # get_reward_batch 는 condition[:, enum-1] 을 사용.
    # 예측된 enum 슬롯에만 값을 채우고 나머지는 -1.
    condition = jnp.full((n_envs, 9), -1.0, dtype=jnp.float32)

    # 각 환경에서 predicted enum에 해당하는 condition 값을 gather
    pred_cond_val = condition_pred_raw[
        jnp.arange(n_envs), pred_enum_0based
    ]  # (n_envs,)

    # condition[:, pred_enum_0based] = pred_cond_val  (vmap-safe scatter)
    condition = condition.at[jnp.arange(n_envs), pred_enum_0based].set(pred_cond_val)

    return jax.lax.stop_gradient(reward_i), jax.lax.stop_gradient(condition)


def _extract_instruction_embedding(instruct_sample, curr_obs):
    """inject 콜백 입력에서 instruction embedding을 안정적으로 꺼낸다."""
    if instruct_sample is not None and hasattr(instruct_sample, "embedding"):
        return instruct_sample.embedding
    if curr_obs is not None and hasattr(curr_obs, "nlp_obs"):
        return curr_obs.nlp_obs
    raise ValueError("Instruction embedding is unavailable for decoder reward prediction")


def predict_from_instruction(
    apply_fn,
    variables: dict,
    instruction_embedding: jnp.ndarray,
    num_reward_classes: int = 5,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """instruction embedding에서 reward_i/condition을 예측한다."""
    return predict_reward_condition(
        apply_fn=apply_fn,
        variables=variables,
        instruction_embedding=instruction_embedding,
        num_reward_classes=num_reward_classes,
    )


def build_decoder_reward_inject_fn(config) -> Callable:
    """train_utils.inject_reward_fn 시그니처에 맞는 콜백을 생성한다."""
    from conf.config import DecoderConfig

    decoder_cfg = DecoderConfig(num_reward_classes=config.decoder.num_reward_classes)
    apply_fn, variables = load_decoder(
        ckpt_dir=config.encoder.ckpt_path,
        encoder_config=config.encoder,
        decoder_config=decoder_cfg,
    )

    def _inject_reward_fn(prev_env_state, curr_env_state, last_obs, curr_obs, instruct_sample, config, env):
        instruction_embedding = _extract_instruction_embedding(instruct_sample, curr_obs)
        return predict_from_instruction(
            apply_fn=apply_fn,
            variables=variables,
            instruction_embedding=instruction_embedding,
            num_reward_classes=config.decoder_reward_classes,
        )

    return _inject_reward_fn


def _env_maps_to_pixel_values(env_maps: jnp.ndarray, num_classes: int) -> jnp.ndarray:
    from instruct_rl.utils.level_processing_utils import add_coord_channel_batch, map2onehot_batch

    zero_based = env_maps.astype(jnp.int32) - 1
    onehot = map2onehot_batch(zero_based, num_classes=num_classes)
    return add_coord_channel_batch(onehot)


def predict_transition_reward(
    apply_fn,
    variables: dict,
    instruction_embedding: jnp.ndarray,
    prev_env_map: jnp.ndarray,
    curr_env_map: jnp.ndarray,
    *,
    num_classes: int = 5,
) -> jnp.ndarray:
    """Predict scalar reward for (instruction, s_t, s_t+1)."""
    prev_pix = _env_maps_to_pixel_values(prev_env_map, num_classes=num_classes)
    curr_pix = _env_maps_to_pixel_values(curr_env_map, num_classes=num_classes)

    reward = apply_fn(
        variables,
        instruction_embedding,
        prev_pix,
        curr_pix,
        False,
        method=lambda m, text_embed, prev_map, curr_map, training=False: m.decoder(
            text_embed, prev_map, curr_map, training=training
        ),
    )
    return jax.lax.stop_gradient(jnp.clip(reward, -2.0, 2.0))


def build_transition_reward_inject_fn(config) -> Callable:
    """Build train_utils reward hook for the universal transition reward model."""
    from conf.config import DecoderConfig
    from evaluator import get_reward_batch

    decoder_cfg = DecoderConfig(
        num_reward_classes=config.decoder.num_reward_classes,
        hidden_dim=getattr(config.decoder, "hidden_dim", DecoderConfig.hidden_dim),
        num_layers=getattr(config.decoder, "num_layers", DecoderConfig.num_layers),
        output_dim=getattr(config.decoder, "output_dim", DecoderConfig.output_dim),
        cnn_reward_enum_onehot=getattr(
            config.decoder,
            "cnn_reward_enum_onehot",
            DecoderConfig.cnn_reward_enum_onehot,
        ),
    )
    apply_fn, variables = load_decoder(
        ckpt_dir=config.encoder.ckpt_path,
        encoder_config=config.encoder,
        decoder_config=decoder_cfg,
    )

    def _inject_reward_fn(prev_env_state, curr_env_state, instruct_sample, network_apply, network_params, last_obs):
        prev_map = prev_env_state.env_state.env_map
        curr_map = curr_env_state.env_state.env_map
        annotated_reward = get_reward_batch(
            instruct_sample.reward_i,
            instruct_sample.condition,
            prev_map,
            curr_map,
            map_size=config.map_width,
            placement_w_amount=config.placement_w_amount,
            placement_w_spread=config.placement_w_spread,
            special_tile_penalty_weight=config.special_tile_penalty_weight,
        )
        model_reward = predict_transition_reward(
            apply_fn,
            variables,
            instruct_sample.embedding,
            prev_map,
            curr_map,
            num_classes=max(1, int(config.clip_input_channel) - 2),
        )
        mask = getattr(instruct_sample, "reward_model_mask", None)
        if mask is None:
            mask = jnp.ones_like(model_reward)
        mask = jnp.ravel(mask).astype(jnp.float32)
        return jnp.where(mask > 0.5, model_reward, annotated_reward)

    logger.info("Transition reward model hook ready: ckpt=%s", config.encoder.ckpt_path)
    return _inject_reward_fn
