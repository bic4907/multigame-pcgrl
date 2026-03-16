"""
tests/test_multigame_dataset.py
================================
MultiGameDataset 통합 테스트.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_DATASET_ROOT = Path(__file__).parent.parent.parent
if str(_DATASET_ROOT) not in sys.path:
    sys.path.insert(0, str(_DATASET_ROOT))

from multigame.dataset import MultiGameDataset
from multigame.base import GameTag, GameSample

VGLC_ROOT    = _DATASET_ROOT / "TheVGLC"
DUNGEON_ROOT = _DATASET_ROOT / "dungeon_level_dataset"


@pytest.fixture(scope="module")
def ds_zelda_dungeon():
    """Zelda + Dungeon만 포함한 경량 데이터셋."""
    return MultiGameDataset(
        vglc_root=VGLC_ROOT,
        dungeon_root=DUNGEON_ROOT,
        vglc_games=[GameTag.ZELDA],
        include_dungeon=True,
    )


@pytest.fixture(scope="module")
def ds_vglc_only():
    """VGLC 3개 게임만."""
    return MultiGameDataset(
        vglc_root=VGLC_ROOT,
        dungeon_root=DUNGEON_ROOT,
        vglc_games=[GameTag.ZELDA, GameTag.MARIO, GameTag.LODE_RUNNER],
        include_dungeon=False,
    )


class TestMultiGameDatasetBasic:
    def test_len_positive(self, ds_zelda_dungeon):
        assert len(ds_zelda_dungeon) > 0

    def test_getitem_returns_gamesample(self, ds_zelda_dungeon):
        sample = ds_zelda_dungeon[0]
        assert isinstance(sample, GameSample)

    def test_iter_count(self, ds_zelda_dungeon):
        count = sum(1 for _ in ds_zelda_dungeon)
        assert count == len(ds_zelda_dungeon)

    def test_order_unique_and_monotonic(self, ds_zelda_dungeon):
        orders = [s.order for s in ds_zelda_dungeon]
        assert orders == list(range(len(orders)))

    def test_available_games_contains_expected(self, ds_zelda_dungeon):
        games = ds_zelda_dungeon.available_games()
        assert GameTag.ZELDA in games
        assert GameTag.DUNGEON in games

    def test_repr(self, ds_zelda_dungeon):
        r = repr(ds_zelda_dungeon)
        assert "MultiGameDataset" in r


class TestMultiGameDatasetFilters:
    def test_by_game_zelda(self, ds_zelda_dungeon):
        zelda = ds_zelda_dungeon.by_game(GameTag.ZELDA)
        assert len(zelda) > 0
        assert all(s.game == GameTag.ZELDA for s in zelda)

    def test_by_game_dungeon(self, ds_zelda_dungeon):
        dungeon = ds_zelda_dungeon.by_game(GameTag.DUNGEON)
        assert len(dungeon) == 5052

    def test_by_games_multi(self, ds_zelda_dungeon):
        result = ds_zelda_dungeon.by_games([GameTag.ZELDA, GameTag.DUNGEON])
        assert len(result) == len(ds_zelda_dungeon)

    def test_with_instruction(self, ds_zelda_dungeon):
        result = ds_zelda_dungeon.with_instruction()
        assert all(s.instruction is not None for s in result)

    def test_without_instruction(self, ds_zelda_dungeon):
        result = ds_zelda_dungeon.without_instruction()
        assert all(s.instruction is None for s in result)

    def test_by_instruction_keyword(self, ds_zelda_dungeon):
        result = ds_zelda_dungeon.by_instruction("bat")
        assert all("bat" in s.instruction.lower() for s in result)

    def test_by_order_range(self, ds_zelda_dungeon):
        result = ds_zelda_dungeon.by_order(0, 10)
        assert len(result) == 10
        assert all(0 <= s.order < 10 for s in result)

    def test_by_meta(self, ds_zelda_dungeon):
        # 실제 dungeon dataset의 slug는 snake_case 전체 문장 형태
        result = ds_zelda_dungeon.by_meta(
            "instruction_slug",
            "a_balanced_path_length_with_narrow_characteristics_is_created",
        )
        assert len(result) > 0

    def test_filter_lambda(self, ds_zelda_dungeon):
        result = ds_zelda_dungeon.filter(
            lambda s: s.array.shape[0] > 10
        )
        assert all(s.array.shape[0] > 10 for s in result)


class TestMultiGameDatasetAggregation:
    def test_count_by_game(self, ds_zelda_dungeon):
        counts = ds_zelda_dungeon.count_by_game()
        assert counts[GameTag.DUNGEON] == 5052
        assert counts.get(GameTag.ZELDA, 0) > 0

    def test_group_by_game_keys(self, ds_zelda_dungeon):
        groups = ds_zelda_dungeon.group_by_game()
        assert GameTag.ZELDA in groups
        assert GameTag.DUNGEON in groups

    def test_group_by_instruction(self, ds_zelda_dungeon):
        groups = ds_zelda_dungeon.group_by_instruction()
        # Zelda는 instruction=None → "__no_instruction__" 키에 합산
        assert "__no_instruction__" in groups

    def test_summary_keys(self, ds_zelda_dungeon):
        s = ds_zelda_dungeon.summary()
        for key in ("total", "by_game", "with_instruction",
                    "without_instruction", "unique_instructions"):
            assert key in s

    def test_summary_total_consistent(self, ds_zelda_dungeon):
        s = ds_zelda_dungeon.summary()
        assert s["total"] == len(ds_zelda_dungeon)
        assert s["with_instruction"] + s["without_instruction"] == s["total"]


class TestMultiGameDatasetTags:
    def test_get_tags(self, ds_zelda_dungeon):
        t = ds_zelda_dungeon.get_tags(0)
        assert "game" in t and "order" in t

    def test_all_tags_length(self, ds_zelda_dungeon):
        all_t = ds_zelda_dungeon.all_tags()
        assert len(all_t) == len(ds_zelda_dungeon)


class TestMultiGameDatasetSampling:
    def test_sample_count(self, ds_zelda_dungeon):
        sampled = ds_zelda_dungeon.sample(10, seed=42)
        assert len(sampled) == 10

    def test_sample_no_duplicate(self, ds_zelda_dungeon):
        sampled = ds_zelda_dungeon.sample(50, seed=0)
        ids = [s.source_id + str(s.order) for s in sampled]
        assert len(set(ids)) == len(ids)

    def test_sample_by_game(self, ds_zelda_dungeon):
        sampled = ds_zelda_dungeon.sample(5, game=GameTag.ZELDA, seed=1)
        assert all(s.game == GameTag.ZELDA for s in sampled)

    def test_sample_clamp_to_pool(self, ds_zelda_dungeon):
        zelda_total = len(ds_zelda_dungeon.by_game(GameTag.ZELDA))
        sampled = ds_zelda_dungeon.sample(99999, game=GameTag.ZELDA)
        assert len(sampled) == zelda_total


class TestVglcOnlyDataset:
    def test_no_dungeon(self, ds_vglc_only):
        dungeon = ds_vglc_only.by_game(GameTag.DUNGEON)
        assert len(dungeon) == 0

    def test_three_games_present(self, ds_vglc_only):
        games = set(ds_vglc_only.available_games())
        assert {GameTag.ZELDA, GameTag.MARIO, GameTag.LODE_RUNNER}.issubset(games)

