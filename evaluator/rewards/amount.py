import chex
import jax

from functools import partial

from envs.probs.dungeon3 import Dungeon3Tiles
from evaluator.losses import amount_loss


@partial(jax.jit)
def get_amount_reward(
    prev_env_map: chex.Array,
    curr_env_map: chex.Array,
    cond: chex.Array,
    tile_type: Dungeon3Tiles
) -> chex.Array:
    """
    Function to return the reward value reflecting improvement in the difference for entities such as monsters or walls.

    Args:
        prev_env_map (chex.Array): Previous map state
        curr_env_map (chex.Array): Current map state
        cond (chex.Array): User-intended number of tiles
        tile_type (Dungeon3Tiles): Tile type to aggregate

    Returns:
        chex.Array: Reward value (1D vector)
    """


    prev_loss = amount_loss(prev_env_map, tile_type, cond, absolute=True)
    curr_loss = amount_loss(curr_env_map, tile_type, cond, absolute=True)

    reward = prev_loss - curr_loss
    reward = reward.astype(float)

    return reward
