from typing import Tuple
from enum import IntEnum

import jax
import jax.numpy as jnp
import chex

from .enum_validator import contains_name


def randomize_map(
    rng: chex.Array,
    init_map: chex.Array,
    map_shape: Tuple[int, int],
    tile_enum: IntEnum,
) -> Tuple[chex.Array, chex.Array]:
    """
    randomize map shape

    Args:
        rng (chex.Array): seed
        init_map (chex.Array): map data
        map_shape (Tuple[int, int]): previous map shape
        tile_enum (IntEnum): tile info

    Returns:
        Tuple[chex.Array, chex.Array]: [newmap, new map shape]
    """

    #check tile
    contains_name(tile_enum, "BORDER")

    new_map_shape = jax.random.randint(rng, (2,), 3, jnp.max(jnp.array(map_shape)) + 1)

    # Use jnp.ogrid to create a grid of indices
    oy, ox = jnp.ogrid[: map_shape[0], : map_shape[1]]
    # Use these indices to create a mask where each dimension is less than the corresponding actual_map_shape
    mask = (oy < new_map_shape[0]) & (ox < new_map_shape[1])

    # Replace the rest with tile_enum.BORDER
    randomized_map = jnp.where(mask, init_map, tile_enum.BORDER)

    return randomized_map, new_map_shape
