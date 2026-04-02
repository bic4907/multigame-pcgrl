"""
multigame_evaluator/measure/zelda.py
=====================================
Zelda 게임에 맞는 measure 함수 모음.

타일 정의 (dataset/multigame/handlers/vglc_games/zelda.py 기준)
--------------------------------------------------------------
0  EMPTY   - 빈 공간/경계 (막힘)
1  WALL    - 벽 (막힘)
2  FLOOR   - 바닥 (통과 가능)
3  DOOR    - 문 (Object)
4  BLOCK   - 장애물 블록 (Object)
5  START   - 플레이어 시작점 (Object)
6  MOB     - 적 (Mob)
7  OBJECT  - 아이템 / 오브젝트 (Item)
8  FLOOD   - 물/용암 위험 지형 (막힘, 구 HAZARD)

분류
--------------------------------------------------------------
Empty   : FLOOR                  (통과 가능 지형)
Wall    : EMPTY, WALL, FLOOD     (막힘)
Object  : BLOCK, DOOR, START     (구조물)
Mob     : MOB                    (적)
Item    : OBJECT                 (획득 가능)

Passable (RG/PL): Empty + Item
"""
from enum import IntEnum

import chex
import jax
import jax.numpy as jnp

from envs.pathfinding import calc_diameter
from evaluator.types.direction import Direction
from evaluator.utils import init_flood_net


class ZeldaTile(IntEnum):
    EMPTY  = 0
    WALL   = 1
    FLOOR  = 2
    DOOR   = 3
    BLOCK  = 4
    START  = 5
    MOB    = 6
    OBJECT = 7
    FLOOD  = 8


ZeldaEmpty = jnp.array([
    ZeldaTile.FLOOR,
], dtype=jnp.int32)

ZeldaWall = jnp.array([
    ZeldaTile.EMPTY,
    ZeldaTile.WALL,
    ZeldaTile.FLOOD,
], dtype=jnp.int32)

ZeldaObject = jnp.array([
    ZeldaTile.BLOCK,
    ZeldaTile.DOOR,
    ZeldaTile.START,
], dtype=jnp.int32)

ZeldaMob = jnp.array([
    ZeldaTile.MOB,
], dtype=jnp.int32)

ZeldaItem = jnp.array([
    ZeldaTile.OBJECT,
], dtype=jnp.int32)

ZeldaPassible = jnp.concatenate([ZeldaEmpty, ZeldaItem])




def get_amount(
    env_map: chex.Array,
    tile_type: int = ZeldaTile.MOB,
) -> float:
    """지정한 타일 종류의 개수를 반환한다. 기본값: MOB."""
    return jnp.sum(env_map == tile_type, dtype=float)


def get_direction(
    env_map: chex.Array,
    tile_type: int = ZeldaTile.MOB,
    direction: chex.Array = jnp.array([Direction.east]),
    rows: int = 16,
    cols: int = 16,
) -> float:
    """
    지정한 방향의 절반 영역에 존재하는 tile_type 개수를 반환한다.
    기본값: MOB 타일, east 방향.
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
        passable_tiles = ZeldaPassible
    region_net, path_net = init_flood_net(env_map.shape)
    path_length, _, _, _ = calc_diameter(region_net, path_net, env_map, passable_tiles)
    return path_length.astype(float)


def get_region(
    env_map: chex.Array,
    passable_tiles: chex.Array = None,
) -> float:
    """맵에서 연결된 통로 영역의 수를 반환한다."""
    if passable_tiles is None:
        passable_tiles = ZeldaPassible
    region_net, path_net = init_flood_net(env_map.shape)
    _, _, n_regions, _ = calc_diameter(region_net, path_net, env_map, passable_tiles)
    return n_regions.astype(float)
