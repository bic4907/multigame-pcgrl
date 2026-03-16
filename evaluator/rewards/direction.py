import chex

from envs.probs.dungeon3 import Dungeon3Tiles
from evaluator.losses import direction_loss


def get_direction_reward(
    prev_env_map: chex.Array,
    curr_env_map: chex.Array,
    cond: chex.Array,
    tile_type: chex.Array,
    rows: int = 16,
    cols: int = 16,
):
    """
    Function to evaluate improvement in generating entities in the specified direction.

    Args:
        prev_env_map (chex.Array): Previous map state
        curr_env_map (chex.Array): Current map state
        cond (chex.Array): Desired direction by the user
        tile_type (Dungeon3Tiles): Tile type to aggregate
        rows (int, optional): map size (rows, 16 by default)
        cols (int, optional): map size (cols, 16 by default)

    Returns:
        chex.Array: Value with respect to direction
    """

    prev_loss = direction_loss(prev_env_map, tile_type, cond, rows, cols)
    curr_loss = direction_loss(curr_env_map, tile_type, cond, rows, cols)

    reward: chex.Array = prev_loss - curr_loss
    reward = reward.astype(float)

    reward = reward.clip(-2, 2)

    return reward
