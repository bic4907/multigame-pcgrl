"""
instruct_rl/utils/action_mask.py
================================
PCGRL 학습용 정적 action masking 유틸.

- `resolve_action_mask_config(config)`:
    `re01_action_mask` 플래그를 평가하여 필요 시 `config.action_mask` 를 True 로 승격.
- `build_action_allowed_mask(env)`:
    EMPTY/WALL 타일만 허용하는 정적 (action_dim,) bool 마스크 생성.
- `apply_action_mask(pi, allowed)`:
    distrax.Categorical policy 의 logits 마지막 축에 마스크를 적용한 새 분포 반환.
"""
from __future__ import annotations

from typing import Optional

import distrax
import jax.numpy as jnp

from instruct_rl.utils.dataset_loader_helpers.filters import (
    _parse_dataset_reward_enum_filter,
)


def resolve_action_mask_config(config) -> None:
    """re01_action_mask=True 이고 dataset_reward_enum 이 모두 0/1 이면
    config.action_mask 를 True 로 in-place 승격시킨다."""
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
    """EMPTY/WALL 타일만 허용하는 정적 (action_dim,) bool 마스크.
    
    Turtle representation의 move action (build=-1)은 항상 허용.
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
        # wide-style flat actions: 마지막 축이 tile build 인덱스.
        action_tile_ids = builds_np[action_idxs % n_builds]
    else:
        return jnp.ones((action_dim,), dtype=bool)

    # Turtle의 move action (build=-1)은 항상 허용
    is_move_action = action_tile_ids == -1
    is_allowed_tile = jnp.isin(
        action_tile_ids, jnp.array(allowed_tile_ids, dtype=builds_np.dtype)
    )
    return is_move_action | is_allowed_tile


def apply_action_mask(pi, allowed: Optional[jnp.ndarray]):
    """policy logits 마지막 축에 정적 mask 적용. allowed=None 이면 no-op."""
    if allowed is None:
        return pi
    logits = pi.logits
    mask = allowed.reshape((1,) * (logits.ndim - 1) + (-1,))
    masked_logits = jnp.where(mask, logits, -1e9)
    return distrax.Categorical(logits=masked_logits)
