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
from typing import Callable, Tuple

import jax
import jax.numpy as jnp
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
    dummy_decoder: bool = False,
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
    dummy_pix = jnp.ones((1, 16, 16, 6), dtype=jnp.float32)
    dummy_reward_enum = jnp.zeros((1,), dtype=jnp.int32)

    mode = "text_state" if encoder_config.state else "text"
    variables_template = module.init(
        rng, dummy_ids, dummy_mask, dummy_pix,
        reward_enum=dummy_reward_enum,
        mode=mode, training=False,
    )

    if dummy_decoder:
        logger.warning("dummy_decoder=True: using randomly initialized decoder (checkpoint restore skipped)")
        return module.apply, variables_template

    if not ckpt_dir:
        raise ValueError("ckpt_dir must be provided unless dummy_decoder=True")

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

    decoder_cfg = DecoderConfig(num_reward_classes=config.decoder_reward_classes)
    apply_fn, variables = load_decoder(
        ckpt_dir=config.decoder_ckpt_path,
        encoder_config=config.encoder,
        decoder_config=decoder_cfg,
        dummy_decoder=getattr(config, "dummy_decoder", False),
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
