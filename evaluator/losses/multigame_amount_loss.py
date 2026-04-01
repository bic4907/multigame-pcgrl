"""evaluator/losses/multigame_amount_loss.py

Multigame 타일(INTERACTIVE, HAZARD, COLLECTABLE) 개수 기반 loss.
"""
import chex
import jax
import jax.numpy as jnp

from ..measures.multigame_amount import (
    get_collectable_count,
    get_hazard_count,
    get_interactive_count,
)


def multigame_amount_loss(
    env_map: chex.Array,
    tile_name: str,
    cond: chex.Array,
    absolute: bool = True,
) -> chex.Array:
    """특정 multigame 타일의 개수와 목표 조건 사이의 loss.

    Parameters
    ----------
    env_map : chex.Array
        (H, W) 정수 맵.
    tile_name : str
        "interactive", "hazard", "collectable" 중 하나.
    cond : chex.Array
        목표 타일 개수 (scalar).
    absolute : bool
        True 면 절댓값 loss, False 면 부호 유지.

    Returns
    -------
    chex.Array : scalar loss.
    """
    count_fn = {
        "interactive": get_interactive_count,
        "hazard": get_hazard_count,
        "collectable": get_collectable_count,
    }[tile_name]

    diff = jnp.subtract(count_fn(env_map), cond).astype(float)

    loss = jax.lax.cond(
        absolute,
        lambda _: jnp.abs(diff),
        lambda _: diff,
        operand=None,
    )

    return loss

