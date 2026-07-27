"""evaluator/losses/multigame_amount_loss.py

Multigame tile(INTERACTIVE, HAZARD, COLLECTABLE) count based loss.
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
    """Loss between the count of a specific multigame tile and its target condition.

    Parameters
    ----------
    env_map : chex.Array
        (H, W) integer map.
    tile_name : str
        One of "interactive", "hazard", or "collectable".
    cond : chex.Array
        Measured tile count as a scalar.
    absolute : bool
        Use absolute loss when True; preserve the sign when False.

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
