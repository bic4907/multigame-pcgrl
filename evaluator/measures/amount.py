import chex
import jax.numpy as jnp


def get_amount(
    env_map: chex.Array,
    tile_type: chex.Array,
):
    return jnp.sum(env_map == tile_type, dtype=float)
