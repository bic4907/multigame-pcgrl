"""
tests/test_render.py
====================
render.py 유틸 함수 테스트 (Pillow 없어도 array 기반 테스트는 통과).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_DATASET_ROOT = Path(__file__).parent.parent.parent
if str(_DATASET_ROOT) not in sys.path:
    sys.path.insert(0, str(_DATASET_ROOT))

from multigame.base import GameSample, GameTag
from multigame.render import (
    array_to_rgb,
    render_sample,
    render_grid,
    get_palette,
)

# Pillow 설치 여부 확인
try:
    from PIL import Image
    HAS_PILLOW = True
except ImportError:
    HAS_PILLOW = False


def _make_dungeon_sample(h=16, w=16):
    arr = np.zeros((h, w), dtype=np.int32)
    arr[1:-1, 1:-1] = 1   # floor
    arr[0, :] = 2          # wall
    arr[-1, :] = 2
    arr[5, 5] = 3          # enemy
    return GameSample(
        game=GameTag.DUNGEON,
        source_id="test_dungeon",
        array=arr,
    )


def _make_zelda_sample():
    arr = np.zeros((11, 11), dtype=np.int32)
    arr[0, :] = 1   # wall
    arr[1:-1, 1:-1] = 2   # floor
    return GameSample(
        game=GameTag.ZELDA,
        source_id="test_zelda",
        array=arr,
    )


class TestGetPalette:
    def test_known_game(self):
        p = get_palette(GameTag.DUNGEON)
        assert len(p) > 0

    def test_unknown_game_empty(self):
        p = get_palette("unknown_game_xyz")
        assert p == {}


class TestArrayToRgb:
    def test_output_shape(self):
        arr = np.array([[1, 2], [2, 3]], dtype=np.int32)
        palette = {1: (200, 180, 120), 2: (80, 80, 80), 3: (220, 50, 50)}
        rgb = array_to_rgb(arr, palette)
        assert rgb.shape == (2, 2, 3)
        assert rgb.dtype == np.uint8

    def test_correct_color_mapping(self):
        arr = np.array([[1, 2]], dtype=np.int32)
        palette = {1: (255, 0, 0), 2: (0, 255, 0)}
        rgb = array_to_rgb(arr, palette)
        np.testing.assert_array_equal(rgb[0, 0], [255, 0, 0])
        np.testing.assert_array_equal(rgb[0, 1], [0, 255, 0])

    def test_unknown_tile_gets_fallback_color(self):
        arr = np.array([[99]], dtype=np.int32)
        palette = {1: (255, 0, 0)}
        fallback = (128, 0, 128)
        rgb = array_to_rgb(arr, palette, unknown_color=fallback)
        np.testing.assert_array_equal(rgb[0, 0], fallback)


class TestRenderSample:
    def test_tile_size_1(self):
        sample = _make_dungeon_sample()
        rgb = render_sample(sample, tile_size=1)
        assert rgb.shape == (16, 16, 3)

    def test_tile_size_16(self):
        sample = _make_dungeon_sample()
        rgb = render_sample(sample, tile_size=16)
        assert rgb.shape == (16 * 16, 16 * 16, 3)

    def test_zelda_render(self):
        sample = _make_zelda_sample()
        rgb = render_sample(sample, tile_size=8)
        assert rgb.shape == (11 * 8, 11 * 8, 3)

    def test_dtype_uint8(self):
        sample = _make_dungeon_sample()
        rgb = render_sample(sample)
        assert rgb.dtype == np.uint8


class TestRenderGrid:
    def test_single_sample(self):
        samples = [_make_dungeon_sample()]
        canvas = render_grid(samples, cols=1, tile_size=4)
        assert canvas.ndim == 3
        assert canvas.shape[2] == 3

    def test_multiple_samples_shape(self):
        samples = [_make_dungeon_sample() for _ in range(6)]
        canvas = render_grid(samples, cols=3, tile_size=4, gap=2)
        # 2행 x 3열 구성 예상
        assert canvas.shape[0] > 0
        assert canvas.shape[1] > 0

    def test_empty_list(self):
        canvas = render_grid([], cols=4, tile_size=4)
        assert canvas.shape == (1, 1, 3)


@pytest.mark.skipif(not HAS_PILLOW, reason="Pillow not installed")
class TestPilRendering:
    def test_render_sample_pil_returns_image(self):
        from multigame.render import render_sample_pil
        sample = _make_dungeon_sample()
        img = render_sample_pil(sample, tile_size=8)
        assert img.mode == "RGB"
        assert img.size == (16 * 8, 16 * 8)

    def test_save_rendered(self, tmp_path):
        from multigame.render import save_rendered
        sample = _make_dungeon_sample()
        out = tmp_path / "test_render.png"
        result = save_rendered(sample, out, tile_size=4)
        assert result.exists()

    def test_save_grid(self, tmp_path):
        from multigame.render import save_grid
        samples = [_make_dungeon_sample() for _ in range(4)]
        out = tmp_path / "test_grid.png"
        result = save_grid(samples, out, cols=2, tile_size=4)
        assert result.exists()

