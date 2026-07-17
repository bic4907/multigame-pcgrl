"""
instruct_rl/utils/action_mask.py
================================
PCGRL training for  text action masking utility.

- `resolve_action_mask_config(config)`:
    `re01_action_mask` text  evaluationtext text text `config.action_mask`   True  to  text.
- `build_action_allowed_mask(env)`:
    EMPTY/WALL tiletext text for text  text (action_dim,) bool text create.
- `apply_action_mask(pi, allowed)`:
    distrax.Categorical policy  of  logits text text in  text  applytext text distribution return.
"""
from __future__ import annotations

from typing import Optional

import distrax
import jax.numpy as jnp

from instruct_rl.utils.dataset_loader_helpers.filters import (
    _parse_dataset_reward_enum_filter,
)


def resolve_action_mask_config(config) -> None:
    """re01_action_mask=True  text dataset_reward_enum   text 0/1  text
    config.action_mask   True  to  in-place text."""
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
    """EMPTY/WALL tiletext text for text  text (action_dim,) bool text.

    Turtle representation of  move action (build=-1)  always text for .
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
        # wide-style flat actions: text text  tile build index.
        action_tile_ids = builds_np[action_idxs % n_builds]
    else:
        return jnp.ones((action_dim,), dtype=bool)

    # Turtle of  move action (build=-1)  always text for
    is_move_action = action_tile_ids == -1
    is_allowed_tile = jnp.isin(
        action_tile_ids, jnp.array(allowed_tile_ids, dtype=builds_np.dtype)
    )
    return is_move_action | is_allowed_tile


def apply_action_mask(pi, allowed: Optional[jnp.ndarray]):
    """policy logits text text in  text mask apply. allowed=None  text no-op."""
    if allowed is None:
        return pi
    logits = pi.logits
    mask = allowed.reshape((1,) * (logits.ndim - 1) + (-1,))
    masked_logits = jnp.where(mask, logits, -1e9)
    return distrax.Categorical(logits=masked_logits)
