"""
encoder/utils/decoder_reward.py
================================
trainingtext CLIP Decoder checkpoint  loadtext,
instruction(text) embedding → (reward_enum, condition) text  textrowtext  utility.

RL training text in  `get_reward_batch` text text  text
reward_enum / condition  as  reward  computetext text text for text.
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
#  Checkpoint load
# ═══════════════════════════════════════════════════════════════════════════════

def load_decoder(
    ckpt_dir: str | None,
    encoder_config,
    decoder_config,
    cond_norm_min: jnp.ndarray | None = None,
    cond_norm_max: jnp.ndarray | None = None,
) -> Tuple:
    """CLIP Decoder checkpoint  loadtext.

    Returns
    -------
    (apply_fn, variables)
        apply_fn : ContrastiveDecoderModule.apply
        variables : {"params": ..., "norm_stats": ...}
    """
    from encoder.clip_model import get_cnnclip_decoder_encoder

    # checkpoint  RL_training=False  to  trainingtext to  sametext mode to  module create
    # → params tree structure  checkpoint and  text
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

    ckpt_dir = os.path.abspath(ckpt_dir)
    if not os.path.exists(ckpt_dir):
        logger.warning(f"Decoder checkpoint path not found: {ckpt_dir}. Falling back to initialized decoder.")
        return module.apply, variables_template

    # latest step directory text (ckpt_dir   ckpts/ leveltext text text)
    from glob import glob
    from os.path import basename, join
    step_dirs = [d for d in glob(join(ckpt_dir, '*')) if basename(d).isdigit()]
    if step_dirs:
        latest_step = max(step_dirs, key=lambda d: int(basename(d)))
        restore_dir = latest_step
    else:
        restore_dir = ckpt_dir

    # ── target based text ──
    # train_clip_decoder   TrainState.create(params=variables)  to  savetext to
    # checkpoint = {"step": ..., "params": <variables>, "opt_state": ...}
    # <variables> = {"params": {...}, "norm_stats": {...}}
    # flax restore_checkpoint  in  target={"params": template}   text
    # checkpoint of  "params" text  template structure to  text text.
    from flax.training.train_state import TrainState
    import optax

    dummy_state = TrainState.create(
        apply_fn=module.apply,
        params=variables_template,
        tx=optax.identity(),  # optimizer  text (text  before  for )
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
#  batch text: text_embeddings → (reward_i, condition)
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
    """CLIP text to  text embedding in  reward_i / condition  batch text.

    Parameters
    ----------
    text_embeddings : (N, D) jnp.ndarray
        CLIP encoder  text and text latent embedding.
    apply_fn : Callable
        ContrastiveDecoderModule.apply (text  same text of  function).
    variables : dict
        text parameter text dictionary (``{"params": ..., "norm_stats": ...}``).
    cond_norm_min, cond_norm_max : Dict[int, float]
        reward_enumtext log-space normalize min/max text.
    num_classes : int
        reward_enum class text.
    batch_size : int
        batch size (default value 256).

    Returns
    -------
    reward_i : (N, 1) jnp.int32
        text reward_enum (0-based).
    condition : (N, num_classes) jnp.float32
        text condition text. selecttext enum slottext text  text remaining  -1.
    """
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

    # ── text text text ──
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
#  text: instruction/text embedding → (reward_enum, condition)
# ═══════════════════════════════════════════════════════════════════════════════

def predict_reward_condition(
    apply_fn,
    variables: dict,
    instruction_embedding: jnp.ndarray,
    num_reward_classes: int = 5,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """text  text for text instruction embedding as text
    reward_enum (1-based) and  condition text  text.

    Parameters
    ----------
    instruction_embedding : (n_envs, D)
        pretrained CLIP encoder  text and text latent embedding (e.g. 64-dim).
    """
    n_envs = instruction_embedding.shape[0]

    # ContrastiveDecoderModule internal decoder text  direct calltext.
    reward_logits, _, condition_pred_raw = apply_fn(
        variables,
        instruction_embedding,
        training=False,
        method=lambda m, embed, training=False: m.decoder(embed, training=training),
    )

    reward_logits = reward_logits[:, :num_reward_classes]  # (n_envs, num_classes)
    condition_pred_raw = condition_pred_raw[:, :num_reward_classes]  # (n_envs, num_classes) — text text

    # argmax → 0-based → 1-based
    pred_enum_0based = jnp.argmax(reward_logits, axis=-1)    # (n_envs,)
    reward_i = (pred_enum_0based + 1).reshape(-1, 1).astype(jnp.int32)  # (n_envs, 1)

    # condition text text: (n_envs, 9)
    # get_reward_batch   condition[:, enum-1]   text for .
    # text enum slot in text text  text remaining  -1.
    condition = jnp.full((n_envs, 9), -1.0, dtype=jnp.float32)

    # each text in  predicted enum in  text  condition text  gather
    pred_cond_val = condition_pred_raw[
        jnp.arange(n_envs), pred_enum_0based
    ]  # (n_envs,)

    # condition[:, pred_enum_0based] = pred_cond_val  (vmap-safe scatter)
    condition = condition.at[jnp.arange(n_envs), pred_enum_0based].set(pred_cond_val)

    return jax.lax.stop_gradient(reward_i), jax.lax.stop_gradient(condition)


def _extract_instruction_embedding(instruct_sample, curr_obs):
    """inject callback text in  instruction embedding  text as  text."""
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
    """instruction embedding in  reward_i/condition  text."""
    return predict_reward_condition(
        apply_fn=apply_fn,
        variables=variables,
        instruction_embedding=instruction_embedding,
        num_reward_classes=num_reward_classes,
    )


def build_decoder_reward_inject_fn(config) -> Callable:
    """train_utils.inject_reward_fn text in  text  callback  createtext."""
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
