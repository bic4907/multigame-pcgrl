"""
tests/test_dungeon_handler.py
==============================
DungeonHandler 단위 테스트.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_DATASET_ROOT = Path(__file__).parent.parent.parent
if str(_DATASET_ROOT) not in sys.path:
    sys.path.insert(0, str(_DATASET_ROOT))

from multigame.handlers.dungeon_handler import DungeonHandler, DungeonTile
from multigame.base import GameTag, GameSample

DUNGEON_ROOT = _DATASET_ROOT / "dungeon_level_dataset"


@pytest.fixture(scope="module")
def handler():
    return DungeonHandler(root=DUNGEON_ROOT)


class TestDungeonHandler:
    def test_game_tag(self, handler):
        assert handler.game_tag == GameTag.DUNGEON

    def test_total_samples(self, handler):
        assert len(handler) == 5052

    def test_list_entries_are_zero_padded(self, handler):
        entries = handler.list_entries()
        for key in entries[:10]:
            assert key.isdigit() and len(key) == 6, f"키 형식 오류: {key}"

    def test_load_by_key(self, handler):
        entry = handler.list_entries()[0]
        sample = handler.load_sample(entry)
        assert isinstance(sample, GameSample)

    def test_array_shape(self, handler):
        entry = handler.list_entries()[0]
        sample = handler.load_sample(entry)
        assert sample.shape == (16, 16)

    def test_array_dtype(self, handler):
        entry = handler.list_entries()[0]
        sample = handler.load_sample(entry)
        assert np.issubdtype(sample.array.dtype, np.integer)

    def test_tile_values_in_range(self, handler):
        entry = handler.list_entries()[0]
        sample = handler.load_sample(entry)
        valid = {DungeonTile.FLOOR, DungeonTile.WALL, DungeonTile.ENEMY}
        unique = set(sample.array.flatten().tolist())
        assert unique.issubset(valid), f"예상 외 타일 값: {unique - valid}"

    def test_instruction_not_none(self, handler):
        entry = handler.list_entries()[0]
        sample = handler.load_sample(entry)
        assert sample.instruction is not None
        assert len(sample.instruction) > 0

    def test_meta_fields(self, handler):
        entry = handler.list_entries()[0]
        sample = handler.load_sample(entry)
        for field in ("instruction_slug", "level_id", "sample_id"):
            assert field in sample.meta, f"meta에 {field!r} 없음"

    def test_order_assigned(self, handler):
        entry = handler.list_entries()[0]
        sample = handler.load_sample(entry, order=99)
        assert sample.order == 99

    def test_filter_by_instruction(self, handler):
        results = handler.filter_by_instruction("bat")
        assert len(results) > 0
        for s in results:
            assert "bat" in s.instruction.lower()

    def test_filter_returns_gamesample(self, handler):
        results = handler.filter_by_instruction("wall")
        for s in results:
            assert isinstance(s, GameSample)

    def test_group_by_instruction(self, handler):
        groups = handler.group_by_instruction()
        assert len(groups) == 160  # 160개 카테고리

    def test_group_values_are_lists(self, handler):
        groups = handler.group_by_instruction()
        for slug, samples in groups.items():
            assert isinstance(samples, list)
            assert len(samples) > 0

    def test_category_names_count(self, handler):
        names = handler.category_names()
        assert len(names) == 160

    def test_iterate_all(self, handler):
        count = sum(1 for _ in handler)
        assert count == len(handler)

    def test_invalid_key_raises(self, handler):
        with pytest.raises(KeyError):
            handler.load_sample("999999")

    def test_repr(self, handler):
        r = repr(handler)
        assert "DungeonHandler" in r

