"""
multigame_evaluator/measure/dungeon.py
=======================================
Dungeon 게임에 맞는 measure 함수 모음.

타일 정의 (dataset/multigame/handlers/dungeon_handler.py 기준)
--------------------------------------------------------------
0  UNKNOWN  - 패딩/경계 (막힘)
1  FLOOR    - 바닥 (통과 가능)
2  WALL     - 벽 (막힘)
3  ENEMY    - 박쥐 (Mob)
4  TREASURE - 보물 (Collectable, 통과 가능)

분류
--------------------------------------------------------------
Empty   : FLOOR, TREASURE        (통과 가능 지형)
Wall    : WALL, UNKNOWN          (막힘)
Object  : (없음)
Mob     : ENEMY                  (박쥐)
Item    : TREASURE               (보물)

Passable (RG/PL): Empty (FLOOR + TREASURE)
"""
from enum import IntEnum

import chex
import jax
import jax.numpy as jnp

from envs.pathfinding import calc_diameter
from evaluator.types.direction import Direction
from evaluator.utils import init_flood_net


class DungeonTile(IntEnum):
    UNKNOWN  = 0
    FLOOR    = 1
    WALL     = 2
    ENEMY    = 3
    TREASURE = 4


DungeonEmpty = jnp.array([
    DungeonTile.FLOOR,
    DungeonTile.TREASURE,
], dtype=jnp.int32)

DungeonWall = jnp.array([
    DungeonTile.WALL,
    DungeonTile.UNKNOWN,
], dtype=jnp.int32)

DungeonObject = jnp.array([], dtype=jnp.int32)

DungeonMob = jnp.array([
    DungeonTile.ENEMY,
], dtype=jnp.int32)

DungeonItem = jnp.array([
    DungeonTile.TREASURE,
], dtype=jnp.int32)

DungeonPassible = DungeonEmpty


def get_amount(
    env_map: chex.Array,
    tile_type: int = DungeonTile.ENEMY,
) -> float:
    """지정한 타일 종류의 개수를 반환한다. 기본값: ENEMY(박쥐)."""
    return jnp.sum(env_map == tile_type, dtype=float)


def get_direction(
    env_map: chex.Array,
    tile_type: int = DungeonTile.ENEMY,
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
        passable_tiles = DungeonPassible
    region_net, path_net = init_flood_net(env_map.shape)
    path_length, _, _, _ = calc_diameter(region_net, path_net, env_map, passable_tiles)
    return path_length.astype(float)


def get_region(
    env_map: chex.Array,
    passable_tiles: chex.Array = None,
) -> float:
    """맵에서 연결된 통로 영역의 수를 반환한다."""
    if passable_tiles is None:
        passable_tiles = DungeonPassible
    region_net, path_net = init_flood_net(env_map.shape)
    _, _, n_regions, _ = calc_diameter(region_net, path_net, env_map, passable_tiles)
    return n_regions.astype(float)
