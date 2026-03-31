"""evaluator/rewards/multigame_amount.py

Multigame 타일 개수 기반 reward (이전 맵 대비 개선도).
"""
import chex
import jax
from functools import partial

from evaluator.losses.multigame_amount_loss import multigame_amount_loss


@partial(jax.jit, static_argnames=("tile_name",))
def get_multigame_amount_reward(
    prev_env_map: chex.Array,
    curr_env_map: chex.Array,
    cond: chex.Array,
    tile_name: str = "interactive",
) -> chex.Array:
    """이전 맵 대비 multigame 타일 개수 조건 달성 개선량.

    Parameters
    ----------
    prev_env_map : chex.Array
        이전 (H, W) 정수 맵.
    curr_env_map : chex.Array
        현재 (H, W) 정수 맵.
    cond : chex.Array
        목표 타일 개수.
    tile_name : str
        "interactive", "hazard", "collectable" 중 하나.

    Returns
    -------
    chex.Array : reward (양수 = 개선, 음수 = 악화).
    """
    prev_loss = multigame_amount_loss(prev_env_map, tile_name, cond, absolute=True)
    curr_loss = multigame_amount_loss(curr_env_map, tile_name, cond, absolute=True)

    reward = prev_loss - curr_loss
    return reward.astype(float)

