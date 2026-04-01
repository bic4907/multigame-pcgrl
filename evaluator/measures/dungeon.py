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

분류
--------------------------------------------------------------
Empty   : FLOOR                  (통과 가능 지형)
Wall    : WALL, UNKNOWN          (막힘)
Object  : (없음)
Mob     : ENEMY                  (박쥐)
Item    : (없음)

Passable (RG/PL): Empty
"""
from enum import IntEnum

import chex
import jax
import jax.numpy as jnp

from envs.pathfinding import calc_diameter
from evaluator.types.direction import Direction
from evaluator.utils import init_flood_net


class DungeonTile(IntEnum):
    UNKNOWN = 0
    FLOOR   = 1
    WALL    = 2
    ENEMY   = 3


DungeonEmpty = jnp.array([
    DungeonTile.FLOOR,
], dtype=jnp.int32)

DungeonWall = jnp.array([
    DungeonTile.WALL,
    DungeonTile.UNKNOWN,
], dtype=jnp.int32)

DungeonObject = jnp.array([], dtype=jnp.int32)

DungeonMob = jnp.array([
    DungeonTile.ENEMY,
], dtype=jnp.int32)

DungeonItem = jnp.array([], dtype=jnp.int32)

DungeonPassible = DungeonEmpty