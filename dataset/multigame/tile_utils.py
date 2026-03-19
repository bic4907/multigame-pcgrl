"""
dataset/multigame/tile_utils.py
================================
Unified tile mapping & one-hot encoding utilities.

All game-specific integer tile IDs are first mapped to a shared 7-category
vocabulary defined in ``tile_mapping.json``, then optionally one-hot encoded.

Unified categories
------------------
  0  empty   – background / void
  1  wall    – solid, impassable obstacle
  2  floor   – traversable ground / rope / ladder
  3  enemy   – hostile entity
  4  object  – item / pickup / collectible
  5  spawn   – player start / exit / door
  6  hazard  – environmental damage / trap

Usage
-----
    from dataset.multigame.tile_utils import (
        UNIFIED_CATEGORIES,
        CATEGORY_COLORS,
        NUM_CATEGORIES,
        to_unified,
        to_onehot,
        to_unified_and_onehot,
        validate_onehot,
    )

    # raw integer array from GameSample
    arr = sample.array                     # (16, 16) int32

    # map to unified category indices
    unified = to_unified(arr, sample.game) # (16, 16) int32, values in [0, 6]

    # one-hot encode
    oh = to_onehot(unified)                # (16, 16, 7) uint8, values in {0, 1}

    # both at once
    unified, oh = to_unified_and_onehot(arr, sample.game)

    # validate
    ok, info = validate_onehot(oh)
    assert ok, info

External dependencies: numpy only (no torch/tf required).
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

# ── 설정 파일 로드 ──────────────────────────────────────────────────────────────
_MAPPING_FILE = Path(__file__).parent / "tile_mapping.json"

with _MAPPING_FILE.open("r", encoding="utf-8") as _f:
    _MAPPING_CONFIG: Dict[str, Any] = json.load(_f)

# ── 공개 상수 ────────────────────────────────────────────────────────────────────
UNIFIED_CATEGORIES: Dict[int, str] = {
    int(k): v for k, v in _MAPPING_CONFIG["_categories"].items()
}
"""int → 카테고리 이름 (e.g. {0: 'empty', 1: 'wall', ...})"""

CATEGORY_COLORS: Dict[int, Tuple[int, int, int]] = {
    int(k): (int(v[0]), int(v[1]), int(v[2]))
    for k, v in _MAPPING_CONFIG["_category_colors_rgb"].items()
}
"""int → RGB 튜플 (렌더링용)"""

NUM_CATEGORIES: int = len(UNIFIED_CATEGORIES)
"""통합 카테고리 수 (7)"""

# ── 내부: 게임별 정수→정수 LUT 빌드 ────────────────────────────────────────────
def _build_lut(game: str) -> Dict[int, int]:
    """tile_mapping.json에서 게임별 LUT(lookup table) 빌드."""
    if game not in _MAPPING_CONFIG:
        return {}
    raw = _MAPPING_CONFIG[game]["mapping"]
    return {int(k): int(v) for k, v in raw.items()}


# LUT 캐시 (런타임 1회 빌드)
_LUT_CACHE: Dict[str, Dict[int, int]] = {}


def _get_lut(game: str) -> Dict[int, int]:
    if game not in _LUT_CACHE:
        _LUT_CACHE[game] = _build_lut(game)
    return _LUT_CACHE[game]


# ── 공개 API ─────────────────────────────────────────────────────────────────────

def to_unified(
    array: np.ndarray,
    game: str,
    *,
    warn_unmapped: bool = True,
) -> np.ndarray:
    """
    게임별 정수 타일 배열을 unified category index 배열로 변환한다.

    Parameters
    ----------
    array        : (H, W) int32 ndarray – GameSample.array
    game         : GameTag 문자열 (e.g. "zelda", "dungeon")
    warn_unmapped: True이면 매핑 규칙에 없는 타일 값에 대해 warning 발생

    Returns
    -------
    (H, W) int32 ndarray, 값 범위 [0, NUM_CATEGORIES-1]
    """
    lut = _get_lut(game)
    if not lut:
        warnings.warn(
            f"[tile_utils] No mapping found for game '{game}'. "
            "Returning zeros (all 'empty').",
            RuntimeWarning,
            stacklevel=2,
        )
        return np.zeros_like(array, dtype=np.int32)

    flat = array.ravel()
    out  = np.empty_like(flat, dtype=np.int32)

    unmapped_values: set = set()
    for i, val in enumerate(flat.tolist()):
        if val in lut:
            out[i] = lut[val]
        else:
            out[i] = 0  # fallback: empty
            unmapped_values.add(val)

    if warn_unmapped and unmapped_values:
        warnings.warn(
            f"[tile_utils] game='{game}': tile values {sorted(unmapped_values)} "
            "have no entry in tile_mapping.json → mapped to 0 ('empty'). "
            "Consider adding them to tile_mapping.json.",
            RuntimeWarning,
            stacklevel=2,
        )

    return out.reshape(array.shape)


def to_onehot(
    unified: np.ndarray,
    *,
    num_categories: int = NUM_CATEGORIES,
) -> np.ndarray:
    """
    Unified category index 배열을 one-hot 배열로 변환한다.

    Parameters
    ----------
    unified        : (H, W) int32 ndarray, 값 범위 [0, num_categories-1]
    num_categories : 카테고리 수 (기본 7)

    Returns
    -------
    (H, W, C) uint8 ndarray, 값 ∈ {0, 1}

    Raises
    ------
    ValueError : unified에 범위 밖 값이 있을 때
    """
    flat = unified.ravel()
    out_of_range = np.where((flat < 0) | (flat >= num_categories))[0]
    if len(out_of_range) > 0:
        bad_vals = sorted(set(flat[out_of_range].tolist()))
        raise ValueError(
            f"[tile_utils] to_onehot: unified array contains out-of-range values "
            f"{bad_vals}. Expected values in [0, {num_categories - 1}]."
        )

    h, w = unified.shape
    oh = np.zeros((h, w, num_categories), dtype=np.uint8)
    oh[np.arange(h)[:, None], np.arange(w)[None, :], unified] = 1
    return oh


def to_unified_and_onehot(
    array: np.ndarray,
    game: str,
    *,
    warn_unmapped: bool = True,
    num_categories: int = NUM_CATEGORIES,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    GameSample.array에서 unified index와 one-hot을 한 번에 반환한다.

    Returns
    -------
    (unified, onehot)
      unified : (H, W)    int32  – category index
      onehot  : (H, W, C) uint8  – one-hot (C = num_categories)
    """
    unified = to_unified(array, game, warn_unmapped=warn_unmapped)
    onehot  = to_onehot(unified, num_categories=num_categories)
    return unified, onehot


def validate_onehot(
    onehot: np.ndarray,
    *,
    num_categories: int = NUM_CATEGORIES,
) -> Tuple[bool, Dict[str, Any]]:
    """
    One-hot 배열의 min/max 및 구조를 검증한다.

    Parameters
    ----------
    onehot         : (H, W, C) ndarray
    num_categories : 예상 카테고리 수

    Returns
    -------
    (ok, info)
      ok   : 모든 검사 통과 여부
      info : 검사 결과 상세 dict
        {
          'shape'           : onehot.shape,
          'min'             : global min value,
          'max'             : global max value,
          'num_categories'  : C axis size,
          'values_valid'    : values ∈ {0, 1},
          'sum_per_cell'    : each cell sums to exactly 1,
          'min_expected'    : 0,
          'max_expected'    : 1,
          'errors'          : list of error strings (empty if ok),
        }
    """
    errors: List[str] = []

    global_min = int(onehot.min())
    global_max = int(onehot.max())
    unique_vals = set(np.unique(onehot).tolist())

    # min/max 범위 체크
    if global_min < 0:
        errors.append(f"min value {global_min} < 0")
    if global_max > 1:
        errors.append(f"max value {global_max} > 1")

    # 값이 {{0, 1}} 만 포함하는지
    invalid_vals = unique_vals - {0, 1}
    if invalid_vals:
        errors.append(f"unexpected values in one-hot: {sorted(invalid_vals)}")

    # 각 셀(H×W)의 합이 1인지
    cell_sums = onehot.sum(axis=-1)  # (H, W)
    bad_cells = int((cell_sums != 1).sum())
    if bad_cells > 0:
        errors.append(
            f"{bad_cells} cell(s) do not sum to 1 "
            f"(got sums: min={cell_sums.min()}, max={cell_sums.max()})"
        )

    # 카테고리 수 체크
    if onehot.ndim != 3:
        errors.append(f"expected 3D array (H, W, C), got ndim={onehot.ndim}")
    elif onehot.shape[2] != num_categories:
        errors.append(
            f"expected {num_categories} categories on axis 2, "
            f"got {onehot.shape[2]}"
        )

    ok = len(errors) == 0
    info: Dict[str, Any] = {
        "shape":          tuple(onehot.shape),
        "min":            global_min,
        "max":            global_max,
        "min_expected":   0,
        "max_expected":   1,
        "num_categories": onehot.shape[2] if onehot.ndim == 3 else None,
        "values_valid":   len(invalid_vals) == 0,
        "sum_per_cell":   bad_cells == 0,
        "errors":         errors,
    }
    return ok, info


def onehot_to_unified(onehot: np.ndarray) -> np.ndarray:
    """
    One-hot (H, W, C) → unified category index (H, W).
    각 셀에서 argmax를 취한다.
    """
    return onehot.argmax(axis=-1).astype(np.int32)


def category_name(idx: int) -> str:
    """카테고리 인덱스 → 이름."""
    return UNIFIED_CATEGORIES.get(idx, f"unknown({idx})")


def category_distribution(
    unified: np.ndarray,
    *,
    normalize: bool = False,
) -> Dict[str, float]:
    """
    Unified 배열의 카테고리별 빈도 분포.

    Parameters
    ----------
    unified   : (H, W) int32
    normalize : True이면 비율(0~1), False이면 카운트

    Returns
    -------
    {'empty': 120, 'wall': 64, ...}
    """
    dist: Dict[str, float] = {name: 0.0 for name in UNIFIED_CATEGORIES.values()}
    flat = unified.ravel()
    total = float(flat.size)
    for val in range(NUM_CATEGORIES):
        count = float(int((flat == val).sum()))
        name  = UNIFIED_CATEGORIES[val]
        dist[name] = (count / total) if normalize else count
    return dist


def available_games() -> List[str]:
    """tile_mapping.json에 정의된 게임 목록."""
    return [k for k in _MAPPING_CONFIG if not k.startswith("_")]


def game_mapping_info(game: str) -> Dict[str, Any]:
    """
    tile_mapping.json의 게임별 매핑 정보를 정규화해 반환한다.

    Returns
    -------
    {
      'game': 'zelda',
      'tile_names': {0: 'EMPTY', ...},
      'mapping': {0: 0, 1: 1, ...},
      'unified_categories': {0: 'empty', ...}
    }
    """
    if game not in _MAPPING_CONFIG or game.startswith("_"):
        raise KeyError(f"[tile_utils] mapping info not found for game: {game!r}")

    entry = _MAPPING_CONFIG[game]
    raw_names = entry.get("_tile_names", {})
    raw_mapping = entry.get("mapping", {})

    tile_names = {int(k): str(v) for k, v in raw_names.items()}
    mapping = {int(k): int(v) for k, v in raw_mapping.items()}

    return {
        "game": game,
        "tile_names": tile_names,
        "mapping": mapping,
        "unified_categories": dict(UNIFIED_CATEGORIES),
    }


def game_mapping_rows(game: str) -> List[Dict[str, Any]]:
    """원본 타일 -> unified 카테고리 매핑을 표 형태 row 리스트로 반환한다."""
    info = game_mapping_info(game)
    tile_names: Dict[int, str] = info["tile_names"]
    mapping: Dict[int, int] = info["mapping"]

    rows: List[Dict[str, Any]] = []
    for raw_id in sorted(mapping.keys()):
        uni_id = mapping[raw_id]
        rows.append({
            "raw_id": raw_id,
            "raw_name": tile_names.get(raw_id, f"TILE_{raw_id}"),
            "unified_id": uni_id,
            "unified_name": UNIFIED_CATEGORIES.get(uni_id, f"unknown({uni_id})"),
        })
    return rows


def render_unified_rgb(
    unified: np.ndarray,
    tile_size: int = 16,
) -> np.ndarray:
    """
    Unified category index 배열을 RGB numpy 이미지로 변환.

    Parameters
    ----------
    unified   : (H, W) int32
    tile_size : 픽셀 단위 타일 크기

    Returns
    -------
    (H*tile_size, W*tile_size, 3) uint8 ndarray
    """
    h, w = unified.shape
    canvas = np.zeros((h * tile_size, w * tile_size, 3), dtype=np.uint8)
    for r in range(h):
        for c in range(w):
            cat = int(unified[r, c])
            color = CATEGORY_COLORS.get(cat, (128, 0, 128))
            y0, y1 = r * tile_size, (r + 1) * tile_size
            x0, x1 = c * tile_size, (c + 1) * tile_size
            canvas[y0:y1, x0:x1] = color
    return canvas

