"""
dataset/multigame/render.py
===========================
타일 그리드 렌더링 유틸리티.

외부 의존: numpy, Pillow (PIL)
Pillow가 없을 경우 numpy 배열만 반환.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .base import GameSample
from .handlers.vglc_games import PALETTES
from .handlers.dungeon_handler import DUNGEON_PALETTE

# ── 전체 팔레트 통합 ─────────────────────────────────────────────────────────────
_ALL_PALETTES: Dict[str, Dict[int, Tuple[int, int, int]]] = {
    **PALETTES,
    "dungeon": DUNGEON_PALETTE,
    "d2": DUNGEON_PALETTE,
}

_DEFAULT_UNKNOWN_COLOR = (255, 0, 255)   # 매핑 없을 때 마젠타
_DEFAULT_TILE_SIZE     = 16              # 픽셀 단위 타일 크기


def get_palette(game: str) -> Dict[int, Tuple[int, int, int]]:
    """게임 태그로 팔레트 dict 반환."""
    return _ALL_PALETTES.get(game, {})


def array_to_rgb(
    array: np.ndarray,
    palette: Dict[int, Tuple[int, int, int]],
    unknown_color: Tuple[int, int, int] = _DEFAULT_UNKNOWN_COLOR,
) -> np.ndarray:
    """
    (H, W) int 배열 → (H, W, 3) uint8 RGB 배열.

    Parameters
    ----------
    array         : (H, W) int32/int64 타일 ID 배열
    palette       : tile_id → (R, G, B) 매핑
    unknown_color : 팔레트에 없는 타일의 색
    """
    h, w = array.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for tile_id, color in palette.items():
        mask = array == tile_id
        rgb[mask] = color
    # 팔레트 미등록 타일
    registered = set(palette.keys())
    for r in range(h):
        for c in range(w):
            if int(array[r, c]) not in registered:
                rgb[r, c] = unknown_color
    return rgb


def render_sample(
    sample: GameSample,
    tile_size: int = _DEFAULT_TILE_SIZE,
    unknown_color: Tuple[int, int, int] = _DEFAULT_UNKNOWN_COLOR,
) -> np.ndarray:
    """
    GameSample → (H*tile_size, W*tile_size, 3) uint8 RGB 배열.

    Returns
    -------
    numpy ndarray (Pillow 없어도 동작)
    """
    palette = get_palette(sample.game)
    small   = array_to_rgb(sample.array, palette, unknown_color)
    if tile_size == 1:
        return small
    # 업스케일
    return np.repeat(np.repeat(small, tile_size, axis=0), tile_size, axis=1)


def render_sample_pil(
    sample: GameSample,
    tile_size: int = _DEFAULT_TILE_SIZE,
    unknown_color: Tuple[int, int, int] = _DEFAULT_UNKNOWN_COLOR,
):
    """
    GameSample → PIL Image.
    Pillow가 설치되지 않은 경우 ImportError 발생.
    """
    from PIL import Image
    rgb = render_sample(sample, tile_size=tile_size, unknown_color=unknown_color)
    return Image.fromarray(rgb, mode="RGB")


def save_rendered(
    sample: GameSample,
    save_path: Path | str,
    tile_size: int = _DEFAULT_TILE_SIZE,
    unknown_color: Tuple[int, int, int] = _DEFAULT_UNKNOWN_COLOR,
) -> Path:
    """
    GameSample을 PNG로 저장.

    Returns
    -------
    저장된 파일 경로
    """
    from PIL import Image
    img = render_sample_pil(sample, tile_size=tile_size, unknown_color=unknown_color)
    out = Path(save_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(out))
    return out


def render_grid(
    samples: List[GameSample],
    cols: int = 4,
    tile_size: int = _DEFAULT_TILE_SIZE,
    gap: int = 2,
    bg_color: Tuple[int, int, int] = (30, 30, 30),
) -> np.ndarray:
    """
    여러 GameSample을 격자 형태로 배치한 이미지 반환.

    Parameters
    ----------
    samples   : GameSample 리스트
    cols      : 열 수
    tile_size : 타일 픽셀 크기
    gap       : 셀 간 간격 (픽셀)
    bg_color  : 배경 색 (R, G, B)

    Returns
    -------
    (total_H, total_W, 3) uint8 RGB 배열
    """
    if not samples:
        return np.zeros((1, 1, 3), dtype=np.uint8)

    rendered = [render_sample(s, tile_size=tile_size) for s in samples]
    max_h = max(r.shape[0] for r in rendered)
    max_w = max(r.shape[1] for r in rendered)

    rows = (len(samples) + cols - 1) // cols
    total_h = rows * max_h + (rows + 1) * gap
    total_w = cols * max_w + (cols + 1) * gap

    canvas = np.full((total_h, total_w, 3), bg_color, dtype=np.uint8)
    for idx, img in enumerate(rendered):
        row_i = idx // cols
        col_i = idx % cols
        y = gap + row_i * (max_h + gap)
        x = gap + col_i * (max_w + gap)
        h, w = img.shape[:2]
        canvas[y:y + h, x:x + w] = img

    return canvas


def save_grid(
    samples: List[GameSample],
    save_path: Path | str,
    cols: int = 4,
    tile_size: int = _DEFAULT_TILE_SIZE,
    gap: int = 2,
    bg_color: Tuple[int, int, int] = (30, 30, 30),
) -> Path:
    """render_grid 결과를 PNG로 저장."""
    from PIL import Image
    canvas = render_grid(samples, cols=cols, tile_size=tile_size,
                         gap=gap, bg_color=bg_color)
    out = Path(save_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(canvas, mode="RGB").save(str(out))
    return out

