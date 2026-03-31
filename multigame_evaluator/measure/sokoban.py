"""
multigame_evaluator/measure/sokoban.py
=======================================
Sokoban (Boxoban) 게임에 맞는 measure 함수 모음.

타일 정의 (dataset/multigame/handlers/boxoban_handler.py 기준)
--------------------------------------------------------------
0  EMPTY   - 빈 바닥 / 목표 칸 (Empty)
1  WALL    - 벽 (Wall)
4  OBJECT  - 상자 (Object)
5  SPAWN   - 플레이어 위치 (Empty)

Note: 타일 ID 2, 3은 Sokoban에서 사용되지 않는다.

분류
--------------------------------------------------------------
Empty   : EMPTY, SPAWN           (통과 가능 지형)
Wall    : WALL                   (막힘)
Object  : OBJECT                 (상자)
Mob     : (없음)
Item    : (없음)

Passable (RG/PL): Empty + Item = Empty
"""
from enum import IntEnum

import chex
import jax
import jax.numpy as jnp

from envs.pathfinding import calc_diameter
from evaluator.types.direction import Direction
from evaluator.utils import init_flood_net


class SokobanTile(IntEnum):
    EMPTY  = 0
    WALL   = 1
    OBJECT = 4   # box
    SPAWN  = 5   # player


SokobanEmpty = jnp.array([
    SokobanTile.EMPTY,
    SokobanTile.SPAWN,
], dtype=jnp.int32)

SokobanWall = jnp.array([
    SokobanTile.WALL,
], dtype=jnp.int32)

SokobanObject = jnp.array([
    SokobanTile.OBJECT,
], dtype=jnp.int32)

SokobanMob = jnp.array([], dtype=jnp.int32)

SokobanItem = jnp.array([], dtype=jnp.int32)

SokobanPassible = jnp.concatenate([SokobanEmpty, SokobanItem])


def get_amount(
    env_map: chex.Array,
    tile_type: int = SokobanTile.OBJECT,
) -> float:
    """지정한 타일 종류의 개수를 반환한다. 기본값: OBJECT(상자)."""
    return jnp.sum(env_map == tile_type, dtype=float)


def get_direction(
    env_map: chex.Array,
    tile_type: int = SokobanTile.OBJECT,  # type: ignore[assignment]
    direction: chex.Array = jnp.array([Direction.east]),
    rows: int = 16,
    cols: int = 16,
) -> float:
    """
    지정한 방향의 절반 영역에 존재하는 tile_type 개수를 반환한다.
    기본값: OBJECT(상자) 타일, east 방향.
    """
    is_direction_sets = [
        lambda _, col: jnp.less(col, jnp.divide(cols, 2)),           # west
        lambda row, _: jnp.less(row, jnp.divide(rows, 2)),           # north
        lambda _, col: jnp.greater_equal(col, jnp.divide(cols, 2)),  # east
        lambda row, _: jnp.greater_equal(row, jnp.divide(rows, 2)),  # south
    ]

    d = direction.flatten()
    indices_col, indices_row = jnp.meshgrid(jnp.arange(rows), jnp.arange(cols))
    direction_map = jax.vmap(
        lambda row, col: jax.lax.switch(d[0], is_direction_sets, row, col),
        in_axes=(0, 0),
    )(indices_row.flatten(), indices_col.flatten()).reshape((rows, cols)).astype(float)

    aggregated = jnp.where(env_map == tile_type, 1, 0)
    return jnp.sum(direction_map * aggregated, dtype=float)


def get_path_length(
    env_map: chex.Array,
    passable_tiles: chex.Array = None,
) -> float:
    """맵에서 가장 긴 경로(diameter)를 반환한다."""
    if passable_tiles is None:
        passable_tiles = SokobanPassible
    region_net, path_net = init_flood_net(env_map.shape)
    path_length, _, _, _ = calc_diameter(region_net, path_net, env_map, passable_tiles)
    return path_length.astype(float)


def get_region(
    env_map: chex.Array,
    passable_tiles: chex.Array = None,
) -> float:
    """맵에서 연결된 통로 영역의 수를 반환한다."""
    if passable_tiles is None:
        passable_tiles = SokobanPassible
    region_net, path_net = init_flood_net(env_map.shape)
    _, _, n_regions, _ = calc_diameter(region_net, path_net, env_map, passable_tiles)
    return n_regions.astype(float)
