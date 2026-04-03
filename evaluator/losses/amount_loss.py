import chex
import jax
import jax.numpy as jnp

from envs.probs.dungeon3 import Dungeon3Tiles

from ..measures import get_amount


def amount_loss(
    env_map: chex.Array,
    tile_type: Dungeon3Tiles,
    cond: chex.Array,
    absolute: bool = True,
) -> chex.Array:
    """
    Function to return the reward value based on the count of a specific tile type in the map.

    Args:
        env_map (chex.Array): Map state
        tile_type (Dungeon3Tiles): Tile type to aggregate
        cond (chex.Array): User-intended number of tiles

    Returns:
        chex.Array: Loss value for the amount metric
    """


    diff = jnp.subtract(get_amount(env_map, tile_type), cond).astype(float)

    loss = jax.lax.cond(
        absolute,
        lambda _: jnp.abs(diff),
        lambda _: diff,
        operand=None,
    )

    return loss
