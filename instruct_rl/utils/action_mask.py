"""
instruct_rl/utils/action_mask.py
================================
Static action-masking utilities for PCGRL training.

- `resolve_action_mask_config(config)`:
    Evaluate `re01_action_mask` and enable `config.action_mask` when appropriate.
- `build_action_allowed_mask(env)`:
    Create a static (action_dim,) boolean mask allowing only EMPTY/WALL tiles.
- `apply_action_mask(pi, allowed)`:
    Return a new distribution with the mask applied to the final logits axis.
"""
from __future__ import annotations

from typing import Optional

import distrax
import jax.numpy as jnp

from instruct_rl.utils.dataset_loader_helpers.filters import (
    _parse_dataset_reward_enum_filter,
)


def resolve_action_mask_config(config) -> None:
    """Enable config.action_mask in place when re01_action_mask=True and every
    dataset_reward_enum value is 0 or 1."""
    if not getattr(config, "re01_action_mask", False):
        return
    if getattr(config, "action_mask", False):
        return
    parsed = _parse_dataset_reward_enum_filter(
        getattr(config, "dataset_reward_enum", None),
        field_name="dataset_reward_enum",
    )
    if parsed is not None and all(r in (0, 1) for r in parsed):
        config.action_mask = True


def build_action_allowed_mask(env) -> jnp.ndarray:
    """Return a static (action_dim,) boolean mask allowing only EMPTY/WALL tiles.

    Always allow the Turtle representation's move action (build=-1).
    """
    action_dim = int(env.rep.action_space.n)
    builds = getattr(env.rep, "builds", None)
    if builds is None:
        return jnp.ones((action_dim,), dtype=bool)

    builds_np = jnp.asarray(builds)
    n_builds = int(builds_np.shape[0])
    tile_enum = env.prob.tile_enum
    allowed_names = ("EMPTY", "WALL")
    allowed_tile_ids = [
        int(getattr(tile_enum, name))
        for name in allowed_names
        if hasattr(tile_enum, name)
    ]
    if not allowed_tile_ids:
        return jnp.ones((action_dim,), dtype=bool)

    action_idxs = jnp.arange(action_dim)
    if action_dim == n_builds:
        action_tile_ids = builds_np
    elif action_dim % n_builds == 0:
        # Wide-style flat actions: the final axis is the tile-build index
        action_tile_ids = builds_np[action_idxs % n_builds]
    else:
        return jnp.ones((action_dim,), dtype=bool)

    # Always allow Turtle's move action (build=-1)
    is_move_action = action_tile_ids == -1
    is_allowed_tile = jnp.isin(
        action_tile_ids, jnp.array(allowed_tile_ids, dtype=builds_np.dtype)
    )
    return is_move_action | is_allowed_tile


def apply_action_mask(pi, allowed: Optional[jnp.ndarray]):
    """Apply a static mask to the final policy-logits axis; no-op when allowed=None."""
    if allowed is None:
        return pi
    logits = pi.logits
    mask = allowed.reshape((1,) * (logits.ndim - 1) + (-1,))
    masked_logits = jnp.where(mask, logits, -1e9)
    return distrax.Categorical(logits=masked_logits)
