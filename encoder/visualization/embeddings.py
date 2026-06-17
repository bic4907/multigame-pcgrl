from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp
from tqdm import tqdm


def batched_encode_state(module, variables, pixel_values, reward_enums, batch_size: int) -> np.ndarray:
    @jax.jit
    def encode(pixels, enums):
        return module.apply(
            variables,
            pixel_values=pixels,
            reward_enum=enums,
            mode="state",
            training=False,
        )["state_embed"]

    parts = []
    for start in tqdm(range(0, len(pixel_values), batch_size), desc="state embeddings"):
        end = min(start + batch_size, len(pixel_values))
        parts.append(np.array(encode(jnp.array(pixel_values[start:end]), jnp.array(reward_enums[start:end]))))
    return np.concatenate(parts, axis=0)


def batched_encode_text(module, variables, input_ids, attention_masks, batch_size: int) -> np.ndarray:
    @jax.jit
    def encode(ids, masks):
        return module.apply(
            variables,
            input_ids=ids,
            attention_mask=masks,
            mode="text",
            training=False,
        )["text_embed"]

    parts = []
    for start in tqdm(range(0, len(input_ids), batch_size), desc="text embeddings"):
        end = min(start + batch_size, len(input_ids))
        parts.append(np.array(encode(jnp.array(input_ids[start:end]), jnp.array(attention_masks[start:end]))))
    return np.concatenate(parts, axis=0)
