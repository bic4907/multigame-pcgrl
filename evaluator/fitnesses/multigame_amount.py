"""evaluator/fitnesses/multigame_amount.py

Multigame 타일 개수 기반 fitness (현재 맵의 조건 달성도).
"""
import chex

from evaluator.losses.multigame_amount_loss import multigame_amount_loss


def get_multigame_amount_fitness(
    curr_env_map: chex.Array,
    cond: chex.Array,
    tile_name: str = "interactive",
) -> chex.Array:
    """현재 맵에서 multigame 타일 개수와 목표의 차이 (부호 유지).

    Parameters
    ----------
    curr_env_map : chex.Array
        (H, W) 정수 맵.
    cond : chex.Array
        목표 타일 개수.
    tile_name : str
        "interactive", "hazard", "collectable" 중 하나.

    Returns
    -------
    chex.Array : fitness 값 (cond 보다 많으면 양수, 적으면 음수).
    """
    curr_loss = multigame_amount_loss(curr_env_map, tile_name, cond, absolute=False)
    return curr_loss.astype(float)

