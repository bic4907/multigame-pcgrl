"""
multigame_evaluator/measure/doom.py
====================================
Doom 게임에 맞는 measure 함수 모음.

타일 정의 (dataset/multigame/handlers/vglc_games/doom.py 기준)
--------------------------------------------------------------
0  EMPTY   - 빈 공간/경계 (막힘)
1  WALL    - 벽 (막힘)
2  FLOOR   - 바닥 (통과 가능)
3  ENEMY   - 적 (Mob)
4  SPAWN   - 스폰 포인트 / 출구 (통과 가능)
5  ITEM    - 아이템 (통과 가능)
6  DANGER  - 위험 오브젝트 (Object)
7  DOOR    - 문 (Object)

분류
--------------------------------------------------------------
Empty   : FLOOR, SPAWN           (통과 가능 지형)
Wall    : WALL, EMPTY            (막힘)
Object  : DOOR, DANGER           (구조물)
Mob     : ENEMY                  (적)
Item    : ITEM                   (획득 가능)

Passable (RG/PL): Empty + Item
"""
from enum import IntEnum

import chex
import jax
import jax.numpy as jnp

from envs.pathfinding import calc_diameter
from evaluator.types.direction import Direction
from evaluator.utils import init_flood_net


class DoomTile(IntEnum):
    EMPTY  = 0
    WALL   = 1
    FLOOR  = 2
    ENEMY  = 3
    SPAWN  = 4
    ITEM   = 5
    DANGER = 6
    DOOR   = 7


DoomEmpty = jnp.array([
    DoomTile.FLOOR,
    DoomTile.SPAWN,
], dtype=jnp.int32)

DoomWall = jnp.array([
    DoomTile.WALL,
    DoomTile.EMPTY,
], dtype=jnp.int32)

DoomObject = jnp.array([
    DoomTile.DOOR,
    DoomTile.DANGER,
], dtype=jnp.int32)

DoomMob = jnp.array([
    DoomTile.ENEMY,
], dtype=jnp.int32)

DoomItem = jnp.array([
    DoomTile.ITEM,
], dtype=jnp.int32)

DoomPassible = jnp.concatenate([DoomEmpty, DoomItem])


def get_amount(
    env_map: chex.Array,
    tile_type: int = DoomTile.ENEMY,
) -> float:
    """지정한 타일 종류의 개수를 반환한다. 기본값: ENEMY."""
    return jnp.sum(env_map == tile_type, dtype=float)


def get_direction(
    env_map: chex.Array,
    tile_type: int = DoomTile.ENEMY,
    direction: chex.Array = jnp.array([Direction.east]),
    rows: int = 16,
    cols: int = 16,
) -> float:
    """
    지정한 방향의 절반 영역에 존재하는 tile_type 개수를 반환한다.
    기본값: ENEMY 타일, east 방향.
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
        passable_tiles = DoomPassible
    region_net, path_net = init_flood_net(env_map.shape)
    path_length, _, _, _ = calc_diameter(region_net, path_net, env_map, passable_tiles)
    return path_length.astype(float)


def get_region(
    env_map: chex.Array,
    passable_tiles: chex.Array = None,
) -> float:
    """맵에서 연결된 통로 영역의 수를 반환한다."""
    if passable_tiles is None:
        passable_tiles = DoomPassible
    region_net, path_net = init_flood_net(env_map.shape)
    _, _, n_regions, _ = calc_diameter(region_net, path_net, env_map, passable_tiles)
    return n_regions.astype(float)
