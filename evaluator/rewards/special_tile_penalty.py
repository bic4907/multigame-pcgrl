"""special tile (INTERACTIVE, HAZARD, COLLECTABLE) 이 맵에 존재하는 것 자체에
소량의 패널티를 부여하여, 기본적으로 이 타일 수를 0으로 유지하도록 유도한다.

penalty = (현재 맵의 special tile 총 개수) * weight   (per env)
→ reward 에서 빼 주면 "설치할수록 손해" 시그널이 된다.
"""

import chex
import jax
import jax.numpy as jnp

from envs.probs.multigame import MultigameTiles

_SPECIAL_TILES = jnp.array([
    MultigameTiles.INTERACTIVE,
    MultigameTiles.HAZARD,
    MultigameTiles.COLLECTABLE,
], dtype=jnp.int32)


@jax.jit
def get_special_tile_penalty(
    prev_env_map: chex.Array,
    curr_env_map: chex.Array,
    weight: float = 0.01,
) -> chex.Array:
    """special tile 개수 증가분에 비례하는 패널티(양수 = 증가)를 반환.

    Parameters
    ----------
    prev_env_map : (H, W) int 맵 — 이전 상태.
    curr_env_map : (H, W) int 맵 — 현재 상태.
    weight : 타일 1개 증가당 패널티 크기. 기본 0.01 (소량).

    Returns
    -------
    scalar  (양수 = special tile 증가 → 패널티, 음수 = 감소 → 보상).
    """
    prev_count = jnp.sum(jnp.isin(prev_env_map, _SPECIAL_TILES)).astype(float)
    curr_count = jnp.sum(jnp.isin(curr_env_map, _SPECIAL_TILES)).astype(float)
    return (curr_count - prev_count) * weight

