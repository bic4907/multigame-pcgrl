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
from functools import partial
from pathlib import Path
from typing import Optional, Tuple

import chex
from flax import struct
import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image

from envs.probs.problem import Placeholder, Problem, ProblemState

# ── tile_mapping.json 로드 ──────────────────────────────────────────────────────
_MAPPING_FILE = Path(__file__).parent.parent.parent / "dataset" / "multigame" / "tile_mapping.json"
with _MAPPING_FILE.open("r", encoding="utf-8") as _f:
    _MAPPING_CONFIG: dict = json.load(_f)

_CATEGORIES: dict[int, str] = {
    int(k): v for k, v in _MAPPING_CONFIG["_categories"].items()
}
_CATEGORY_COLORS: dict[int, tuple] = {
    int(k): tuple(v) for k, v in _MAPPING_CONFIG.get("_category_colors_rgb", {}).items()
}
NUM_CATEGORIES: int = len(_CATEGORIES)   # 7

# ── tile 이미지 파일 매핑: JSON _category_tile_images 에서 로드 ─────────────────
# key "border" → BORDER 타일 (index 0)
# key "0".."N"  → category 인덱스 (MultigameTiles index = cat+1)
_TILE_IMS_DIR = Path(__file__).parent / "tile_ims"

_raw_tile_images: dict = _MAPPING_CONFIG.get("_category_tile_images", {})
_BORDER_IMAGE: str       = _raw_tile_images.get("border", "solid.png")
_CATEGORY_IMAGE_FILES: dict[int, str] = {
    int(k): v
    for k, v in _raw_tile_images.items()
    if k not in ("_comment", "border")
}

_TILE_SIZE = 16


def _make_color_tile(rgb: tuple, size: int = _TILE_SIZE) -> Image.Image:
    """단색 RGBA 타일 이미지 생성."""
    r, g, b = rgb
    arr = np.full((size, size, 4), [r, g, b, 255], dtype=np.uint8)
    return Image.fromarray(arr, mode="RGBA")


def _load_tile_image(filename: str, size: int = _TILE_SIZE) -> Image.Image:
    """envs/probs/tile_ims/<filename> 을 로드. 없으면 보라색 fallback."""
    path = _TILE_IMS_DIR / filename
    if path.exists():
        return Image.open(path).convert("RGBA").resize((size, size))
    import warnings
    warnings.warn(f"[multigame] tile image not found: {path}", stacklevel=2)
    return _make_color_tile((200, 0, 200), size)   # 보라색 = 누락 표시


def _load_or_color_tile(cat_idx: int, size: int = _TILE_SIZE) -> Image.Image:
    """category index → 타일 이미지.
    _CATEGORY_IMAGE_FILES(JSON)에 파일이 지정되어 있으면 그 파일 사용,
    없으면 _CATEGORY_COLORS 로 단색 타일 생성."""
    fname = _CATEGORY_IMAGE_FILES.get(cat_idx)
    if fname:
        return _load_tile_image(fname, size)
    color = _CATEGORY_COLORS.get(cat_idx, (128, 128, 128))
    return _make_color_tile(color, size)


# ── tile_mapping._categories → IntEnum ─────────────────────────────────────────
# BORDER(0) 은 env 규약상 index 0 이어야 하므로, category index를 1-shift 해서
# BORDER=0, categories=1..NUM_CATEGORIES 로 배치한다.
#
#   MultigameTiles
#   --------------
#   BORDER = 0          (env 내부 경계 타일, action 불가)
#   EMPTY  = 1          (category 0)
#   WALL   = 2          (category 1)
#   INTERACTABLE = 3      (category 2)
#   HAZARD = 4          (category 3)
#   COLLECTIBLE = 5      (category 4)

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

    tile_size = _TILE_SIZE
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

    @partial(jax.jit, static_argnums=(0, 3))
    def get_cont_obs(self, env_map, condition, raw_obs: bool = False) -> jnp.array:
        """CPCGRL condition → observation 변환.

        multigame은 게임별 통계를 계산하지 않으므로,
        condition[:4] 값을 그대로 obs로 전달하고,
        condition[4]는 one-hot으로 변환하여 concat한다.
        총 output shape: (8,)  — Dungeon3Problem.get_cont_obs 와 동일.
        """
        _condition = condition[:4]
        mask = jnp.not_equal(_condition, -1).astype(jnp.float32)

        # raw_obs 모드: condition 값 그대로 사용
        # non-raw_obs 모드: 통계가 없으므로 역시 condition 값 그대로
        obs = jnp.where(mask == 1, _condition, 0.0)

        # condition[4]: one-hot 인코딩 (bat_direction 등)
        onehot_cond = condition[4:5]

        def to_onehot(index, num_classes=4):
            return jnp.eye(num_classes)[index]

        expanded_onehot = jax.lax.cond(
            jnp.equal(onehot_cond[0], -1),
            lambda _: jnp.zeros((4,)),
            lambda _: to_onehot(onehot_cond[0].astype(jnp.int32)),
            operand=None,
        )

        obs = jnp.concatenate((obs, expanded_onehot), axis=-1)
        return obs

    def init_graphics(self):
        """tile_mapping.json 의 _category_tile_images 를 읽어 타일 이미지를 초기화한다.

        MultigameTiles 인덱스:
          BORDER = 0  → _category_tile_images["border"]
          EMPTY  = 1  → _category_tile_images["0"]
          WALL   = 2  → _category_tile_images["1"]
          ...
          HAZARD = 7  → _category_tile_images["6"]  (lava.png)
        """
        from envs.utils import idx_dict_to_arr

        graphics: dict = {}

        # BORDER (index 0): JSON "border" 키에서 로드
        graphics[0] = _load_tile_image(_BORDER_IMAGE)

        # category tiles: MultigameTiles index = cat_idx + 1
        for cat_idx in _CATEGORIES:
            graphics[cat_idx + 1] = _load_or_color_tile(cat_idx)

        self.graphics = jnp.array(idx_dict_to_arr(graphics))
        super().init_graphics()


def render_multigame_map(env_map: np.ndarray, tile_size: int = _TILE_SIZE) -> Image.Image:
    """tile_mapping._category_tile_images 에 따라 env_map (H×W int32) 을 PIL Image 로 렌더링한다.

    Parameters
    ----------
    env_map  : (H, W) numpy array, 값은 MultigameTiles 정수
    tile_size: 타일 픽셀 크기 (기본 16)

    Returns
    -------
    PIL.Image.Image  (RGB)
    """
    H, W = env_map.shape
    canvas = Image.new("RGBA", (W * tile_size, H * tile_size), (0, 0, 0, 255))

    tile_imgs = {0: _load_tile_image(_BORDER_IMAGE, tile_size)}
    for cat_idx in _CATEGORIES:
        tile_imgs[cat_idx + 1] = _load_or_color_tile(cat_idx, tile_size)

    for y in range(H):
        for x in range(W):
            t = int(env_map[y, x])
            img = tile_imgs.get(t, _make_color_tile((200, 0, 200), tile_size))
            canvas.paste(img, (x * tile_size, y * tile_size))

    return canvas.convert("RGB")


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

    # MultigameProblem 을 PROB_CLASSES 에 등록 (항상 최신 상태로 갱신)
    _MULTIGAME_KEY = max(ProbEnum) + 1
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

