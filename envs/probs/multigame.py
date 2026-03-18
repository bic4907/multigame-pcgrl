"""envs/probs/multigame.py  (updated)

tile_mapping.json 의 unified categories(_categories 섹션)를 tile_enum으로 삼는
MultigameProblem 및 make_multigame_env() 팩토리.

_categories (7개)
-----------------
  0  EMPTY   – background / void
  1  WALL    – solid, impassable obstacle
  2  FLOOR   – traversable ground
  3  ENEMY   – hostile entity
  4  OBJECT  – item / pickup / collectible
  5  SPAWN   – player start / exit / door
  6  HAZARD  – environmental damage / trap

이 파일 하나만으로 "tile_mapping 스펙과 동일한 action 공간을 가진 env" 를 make할 수 있다.

Usage
-----
    from envs.probs.multigame import make_multigame_env

    env, env_params = make_multigame_env()          # default: narrow, 16x16
    env, env_params = make_multigame_env(
        representation="narrow",
        map_shape=(16, 16),
        rf_shape=(31, 31),
    )

    # n_editable_tiles 가 NUM_CATEGORIES(7) 와 일치함을 확인
    assert env.rep.n_editable_tiles == 7
"""
from __future__ import annotations

import json
from enum import IntEnum
from pathlib import Path
from typing import Optional, Tuple

import chex
from flax import struct
import jax.numpy as jnp
import numpy as np

from envs.probs.problem import Placeholder, Problem, ProblemState

# ── tile_mapping.json 로드 ──────────────────────────────────────────────────────
_MAPPING_FILE = Path(__file__).parent.parent.parent / "dataset" / "multigame" / "tile_mapping.json"
with _MAPPING_FILE.open("r", encoding="utf-8") as _f:
    _MAPPING_CONFIG: dict = json.load(_f)

_CATEGORIES: dict[int, str] = {
    int(k): v for k, v in _MAPPING_CONFIG["_categories"].items()
}
NUM_CATEGORIES: int = len(_CATEGORIES)   # 7


# ── tile_mapping._categories → IntEnum ─────────────────────────────────────────
# BORDER(0) 은 env 규약상 index 0 이어야 하므로, category index를 1-shift 해서
# BORDER=0, categories=1..NUM_CATEGORIES 로 배치한다.
#
#   MultigameTiles
#   --------------
#   BORDER = 0          (env 내부 경계 타일, action 불가)
#   EMPTY  = 1          (category 0)
#   WALL   = 2          (category 1)
#   FLOOR  = 3          (category 2)
#   ENEMY  = 4          (category 3)
#   OBJECT = 5          (category 4)
#   SPAWN  = 6          (category 5)
#   HAZARD = 7          (category 6)

MultigameTiles = IntEnum(
    "MultigameTiles",
    {"BORDER": 0, **{name.upper(): idx + 1 for idx, name in _CATEGORIES.items()}},
)
"""tile_mapping._categories 에서 자동 생성된 tile enum.

BORDER=0, EMPTY=1, WALL=2, ..., HAZARD=7  (총 8개)
에서 BORDER 를 제외하면 editable = 7 = NUM_CATEGORIES.
"""


class MultigameMetrics(IntEnum):
    """멀티게임 env 는 별도 통계 지표를 사용하지 않는다. dummy 1-element."""
    DUMMY = 0


@struct.dataclass
class MultigameState(ProblemState):
    """MultigameProblem 용 dummy state."""
    pass


class MultigameProblem(Problem):
    """tile_mapping.json 의 unified 7-category 를 그대로 action 공간으로 쓰는 Problem.

    - tile_enum  = MultigameTiles  (BORDER + 7 categories = 8개)
    - editable   = 7 (= NUM_CATEGORIES, BORDER 제외)
    - stat/reward 는 null (0) — reward shaping 이 필요하면 서브클래스에서 오버라이드.
    """

    tile_enum = MultigameTiles
    metrics_enum = MultigameMetrics
    region_metrics_enum = Placeholder

    # tile 생성 확률: BORDER 0, 나머지 균등
    tile_probs = tuple(
        [0.0]                                            # BORDER
        + [1.0 / NUM_CATEGORIES] * NUM_CATEGORIES        # 7 categories
    )

    # 고정 개수 없음 (모두 자유 배치)
    tile_nums = tuple([0] * len(MultigameTiles))

    # stat weights / trgs / ctrl_threshes: shape (1,) — dummy, no reward
    stat_weights  = np.zeros(1)
    stat_trgs     = jnp.zeros(1)   # jnp.array 여야 Problem.__init__ 에서 정상 작동
    ctrl_threshes = np.zeros(1)

    unavailable_tiles: list = []

    def __init__(self, map_shape: Tuple[int, int], ctrl_metrics: Tuple, pinpoints: bool):
        super().__init__(map_shape, ctrl_metrics, pinpoints)

    def get_metric_bounds(self, map_shape: Tuple[int, int]):
        """통계 지표 없음 → dummy (1, 2) array."""
        return np.zeros((1, 2), dtype=np.float32)

    def get_curr_stats(self, env_map: chex.Array) -> MultigameState:
        """통계 지표 없음 → zeros stats."""
        stats = jnp.zeros(len(MultigameMetrics))
        return MultigameState(stats=stats)

    def get_stats(self, env_map, prob_state: ProblemState):
        """통계 지표 없음 → zeros (1,)."""
        return np.zeros(1)

    def get_path_coords(self, env_map: chex.Array, prob_state: ProblemState):
        """경로 없음 → empty tuple (render 호환)."""
        return ()

    def draw_path(self, lvl_img, env_map, border_size, path_coords_tpl, tile_size):
        """경로 없음 → 이미지 그대로 반환."""
        return lvl_img

    def init_graphics(self):
        """그래픽 초기화 불필요."""
        pass


# ── 팩토리 함수 ─────────────────────────────────────────────────────────────────

def make_multigame_env(
    representation: str = "narrow",
    map_shape: Tuple[int, int] = (16, 16),
    rf_shape: Tuple[int, int] | None = None,
    act_shape: Tuple[int, int] = (1, 1),
    max_board_scans: float = 3.0,
):
    """tile_mapping._categories 스펙과 동일한 action 공간을 가진 PCGRLEnv 를 반환한다.

    Parameters
    ----------
    representation  : "narrow" | "wide" | "turtle" | "nca"
    map_shape       : (H, W) 맵 크기 (기본 16x16)
    rf_shape        : receptive field 크기. None 이면 2*map_width-1 로 자동 설정.
    act_shape       : action patch 크기 (narrow/turtle 의 경우 (1,1))
    max_board_scans : 보드를 최대 몇 번 스캔할지

    Returns
    -------
    (env, env_params) : (PCGRLEnv, PCGRLEnvParams)

    Guarantees
    ----------
    env.rep.n_editable_tiles == NUM_CATEGORIES  (== 7)
    """
    from envs.pcgrl_env import PCGRLEnv, PCGRLEnvParams, ProbEnum, RepEnum, PROB_CLASSES

    # MultigameProblem 을 PROB_CLASSES 에 동적 등록 (없을 경우에만)
    _MULTIGAME_KEY = max(ProbEnum) + 1
    if _MULTIGAME_KEY not in PROB_CLASSES:
        PROB_CLASSES[_MULTIGAME_KEY] = MultigameProblem

    # rf_shape 자동 계산
    if rf_shape is None:
        rf_size = 2 * map_shape[0] - 1
        rf_shape = (rf_size, rf_size)

    rep_key = RepEnum[representation.upper()]

    env_params = PCGRLEnvParams(
        problem=_MULTIGAME_KEY,
        representation=int(rep_key),
        map_shape=map_shape,
        rf_shape=rf_shape,
        act_shape=act_shape,
        max_board_scans=max_board_scans,
    )
    env = PCGRLEnv(env_params)
    return env, env_params

