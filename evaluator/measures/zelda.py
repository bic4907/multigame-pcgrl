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
8  HAZARD  - 위험 지형 (막힘)

분류
--------------------------------------------------------------
Empty   : FLOOR                  (통과 가능 지형)
Wall    : EMPTY, WALL, HAZARD    (막힘)
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
    HAZARD = 8


ZeldaEmpty = jnp.array([
    ZeldaTile.FLOOR,
], dtype=jnp.int32)

ZeldaWall = jnp.array([
    ZeldaTile.EMPTY,
    ZeldaTile.WALL,
    ZeldaTile.HAZARD,
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

