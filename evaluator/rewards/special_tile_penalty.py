"""special tile (INTERACTIVE, HAZARD, COLLECTABLE) 이 맵에 존재하는 것 자체에
소량의 패널티를 부여하여, 기본적으로 이 타일 수를 0으로 유지하도록 유도한다.

penalty = (현재 맵의 special tile 총 개수) * weight   (per env)
→ reward 에서 빼 주면 "설치할수록 손해" 시그널이 된다.
"""

import chex
import jax
import jax.numpy as jnp

from envs.probs.multigame import MultigameTiles


def _tile_value_or_default(*names: str, default: int = -1) -> int:
    for name in names:
        if hasattr(MultigameTiles, name):
            return int(getattr(MultigameTiles, name))
    return default


_special_tiles_list = [
    _tile_value_or_default("INTERACTIVE", "INTERACTABLE"),
    _tile_value_or_default("HAZARD"),
    _tile_value_or_default("COLLECTABLE", "COLLECTIBLE"),
]
_special_tiles_list = [t for t in _special_tiles_list if t >= 0]
if not _special_tiles_list:
    _special_tiles_list = [-1]
_SPECIAL_TILES = jnp.array(_special_tiles_list, dtype=jnp.int32)


@jax.jit
def get_special_tile_penalty(
    prev_env_map: chex.Array,
    curr_env_map: chex.Array,
    weight: float = 0.01,
    exclude_tiles: chex.Array = jnp.array([-1], dtype=jnp.int32),
) -> chex.Array:
    """special tile 개수 증가분에 비례하는 패널티(양수 = 증가)를 반환.

    Parameters
    ----------
    prev_env_map : (H, W) int 맵 — 이전 상태.
    curr_env_map : (H, W) int 맵 — 현재 상태.
    weight : 타일 1개 증가당 패널티 크기. 기본 0.01 (소량).

    exclude_tiles : 패널티 계산에서 제외할 타일 값 목록.
        예) [INTERACTIVE, -1, -1] 이면 INTERACTIVE 타일 증감은 패널티에서 제외.
        기본값 [-1] 은 제외 없음과 동일.

    Returns
    -------
    scalar  (양수 = special tile 증가 → 패널티, 음수 = 감소 → 보상).
    """
    exclude_tiles = jnp.asarray(exclude_tiles, dtype=jnp.int32)
    excluded_mask = jnp.isin(_SPECIAL_TILES, exclude_tiles)  # (3,)

    prev_counts = jnp.sum(
        prev_env_map[..., None] == _SPECIAL_TILES, axis=(0, 1)
    ).astype(jnp.float32)
    curr_counts = jnp.sum(
        curr_env_map[..., None] == _SPECIAL_TILES, axis=(0, 1)
    ).astype(jnp.float32)

    delta_counts = curr_counts - prev_counts
    delta_counts = jnp.where(excluded_mask, 0.0, delta_counts)
    return jnp.sum(delta_counts) * weight
