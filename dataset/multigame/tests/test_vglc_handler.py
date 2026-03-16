"""
tests/test_vglc_handler.py
==========================
VGLCHandler / VGLCGameHandler 단위 테스트.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# ── 패키지 경로 설정 (dataset/ 폴더가 sys.path에 없는 경우 대비) ─────────────────
_DATASET_ROOT = Path(__file__).parent.parent.parent   # dataset/
if str(_DATASET_ROOT) not in sys.path:
    sys.path.insert(0, str(_DATASET_ROOT))

from multigame.handlers.vglc_handler import VGLCHandler, VGLCGameHandler
from multigame.handlers.vglc_games import SUPPORTED_GAMES
from multigame.base import GameTag, GameSample

VGLC_ROOT = _DATASET_ROOT / "TheVGLC"


# ── fixture ─────────────────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def zelda_handler():
    return VGLCGameHandler(GameTag.ZELDA, vglc_root=VGLC_ROOT)

@pytest.fixture(scope="module")
def mario_handler():
    return VGLCGameHandler(GameTag.MARIO, vglc_root=VGLC_ROOT)

@pytest.fixture(scope="module")
def vglc_handler():
    return VGLCHandler(
        vglc_root=VGLC_ROOT,
        selected_games=[GameTag.ZELDA, GameTag.MARIO, GameTag.LODE_RUNNER],
    )


# ── VGLCGameHandler 테스트 ──────────────────────────────────────────────────────
class TestVGLCGameHandler:
    def test_game_tag(self, zelda_handler):
        assert zelda_handler.game_tag == GameTag.ZELDA

    def test_list_entries_nonempty(self, zelda_handler):
        entries = zelda_handler.list_entries()
        assert len(entries) > 0, "Zelda 레벨 파일이 없습니다"

    def test_list_entries_are_txt(self, zelda_handler):
        for path in zelda_handler.list_entries():
            assert path.endswith(".txt"), f"txt 파일이 아님: {path}"

    def test_load_sample_returns_gamesample(self, zelda_handler):
        entry = zelda_handler.list_entries()[0]
        sample = zelda_handler.load_sample(entry, order=0)
        assert isinstance(sample, GameSample)

    def test_load_sample_game_tag(self, zelda_handler):
        entry = zelda_handler.list_entries()[0]
        sample = zelda_handler.load_sample(entry, order=0)
        assert sample.game == GameTag.ZELDA

    def test_load_sample_array_is_2d_int(self, zelda_handler):
        entry = zelda_handler.list_entries()[0]
        sample = zelda_handler.load_sample(entry, order=0)
        assert sample.array.ndim == 2
        assert np.issubdtype(sample.array.dtype, np.integer)

    def test_load_sample_shape_is_16x16(self, zelda_handler):
        entry = zelda_handler.list_entries()[0]
        sample = zelda_handler.load_sample(entry, order=0)
        assert sample.array.shape == (16, 16)

    def test_load_sample_char_grid(self, zelda_handler):
        entry = zelda_handler.list_entries()[0]
        sample = zelda_handler.load_sample(entry, order=0)
        assert sample.char_grid is not None
        assert len(sample.char_grid) <= 16
        assert len(sample.char_grid[0]) <= 16

    def test_load_sample_order(self, zelda_handler):
        entry = zelda_handler.list_entries()[0]
        sample = zelda_handler.load_sample(entry, order=42)
        assert sample.order == 42

    def test_iterate_all(self, zelda_handler):
        samples = list(zelda_handler)
        assert len(samples) == len(zelda_handler)

    def test_mario_levels_nonempty(self, mario_handler):
        assert len(mario_handler) > 0

    def test_mario_array_no_negative(self, mario_handler):
        entry = mario_handler.list_entries()[0]
        sample = mario_handler.load_sample(entry)
        assert (sample.array >= 0).all()

    def test_unsupported_game_raises(self):
        with pytest.raises(ValueError):
            VGLCGameHandler("not_a_game", vglc_root=VGLC_ROOT)


# ── VGLCHandler 테스트 ──────────────────────────────────────────────────────────
class TestVGLCHandler:
    def test_selected_games(self, vglc_handler):
        assert set(vglc_handler.selected_games) == {
            GameTag.ZELDA, GameTag.MARIO, GameTag.LODE_RUNNER
        }

    def test_total_len(self, vglc_handler):
        total = sum(
            len(vglc_handler.game_handler(g))
            for g in vglc_handler.selected_games
        )
        assert len(vglc_handler) == total

    def test_iter_game_tags(self, vglc_handler):
        games_seen = set()
        for sample in vglc_handler:
            games_seen.add(sample.game)
        assert games_seen == set(vglc_handler.selected_games)

    def test_order_monotonic(self, vglc_handler):
        orders = [s.order for s in vglc_handler]
        assert orders == list(range(len(orders)))

    def test_invalid_game_raises(self):
        with pytest.raises(ValueError):
            VGLCHandler(vglc_root=VGLC_ROOT, selected_games=["invalid_game"])

    def test_list_entries_by_game(self, vglc_handler):
        zelda_entries = vglc_handler.list_entries(GameTag.ZELDA)
        assert all(GameTag.ZELDA in e or "Zelda" in e for e in zelda_entries)

    def test_discover_excludes_readme_files(self, vglc_handler):
        zelda_entries = vglc_handler.list_entries(GameTag.ZELDA)
        assert all("readme" not in e.lower() for e in zelda_entries)

    def test_repr(self, vglc_handler):
        r = repr(vglc_handler)
        assert "VGLCHandler" in r

