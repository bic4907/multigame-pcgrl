"""
tests/test_tags.py
==================
tags.py 유틸 함수 단위 테스트.
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
from multigame import tags


def _make_sample(game, instruction=None, order=None, meta=None):
    return GameSample(
        game=game,
        source_id=f"{game}_test",
        array=np.zeros((8, 8), dtype=np.int32),
        instruction=instruction,
        order=order,
        meta=meta or {},
    )


@pytest.fixture
def mixed_samples():
    return [
        _make_sample(GameTag.ZELDA,   instruction=None,           order=0),
        _make_sample(GameTag.ZELDA,   instruction="clear path",   order=1),
        _make_sample(GameTag.MARIO,   instruction=None,           order=2),
        _make_sample(GameTag.DUNGEON, instruction="bat swarm",    order=3, meta={"instruction_slug": "bat_swarm"}),
        _make_sample(GameTag.DUNGEON, instruction="dense wall",   order=4, meta={"instruction_slug": "dense_wall"}),
        _make_sample(GameTag.DOOM,    instruction=None,           order=5),
    ]


class TestBuildTags:
    def test_keys_present(self, mixed_samples):
        t = tags.build_tags(mixed_samples[0])
        for key in ("game", "instruction", "order", "source_id",
                    "has_instruction", "shape"):
            assert key in t

    def test_has_instruction_false(self, mixed_samples):
        t = tags.build_tags(mixed_samples[0])
        assert t["has_instruction"] is False

    def test_has_instruction_true(self, mixed_samples):
        t = tags.build_tags(mixed_samples[1])
        assert t["has_instruction"] is True

    def test_meta_merged(self, mixed_samples):
        t = tags.build_tags(mixed_samples[3])
        assert "instruction_slug" in t
        assert t["instruction_slug"] == "bat_swarm"


class TestExtractByGame:
    def test_single_game(self, mixed_samples):
        result = tags.extract_by_game(mixed_samples, GameTag.ZELDA)
        assert len(result) == 2
        assert all(s.game == GameTag.ZELDA for s in result)

    def test_multiple_games(self, mixed_samples):
        result = tags.extract_by_games(mixed_samples, [GameTag.ZELDA, GameTag.MARIO])
        assert len(result) == 3

    def test_no_match_returns_empty(self, mixed_samples):
        result = tags.extract_by_game(mixed_samples, "nonexistent")
        assert result == []


class TestExtractByInstruction:
    def test_keyword_match(self, mixed_samples):
        result = tags.extract_by_instruction(mixed_samples, "bat")
        assert len(result) == 1
        assert result[0].instruction == "bat swarm"

    def test_case_insensitive(self, mixed_samples):
        result = tags.extract_by_instruction(mixed_samples, "BAT")
        assert len(result) == 1

    def test_case_sensitive_no_match(self, mixed_samples):
        result = tags.extract_by_instruction(
            mixed_samples, "BAT", case_sensitive=True
        )
        assert len(result) == 0

    def test_with_instruction(self, mixed_samples):
        result = tags.extract_with_instruction(mixed_samples)
        assert len(result) == 3

    def test_without_instruction(self, mixed_samples):
        result = tags.extract_without_instruction(mixed_samples)
        assert len(result) == 3


class TestExtractByOrder:
    def test_range(self, mixed_samples):
        result = tags.extract_by_order(mixed_samples, 2, 5)
        orders = [s.order for s in result]
        assert orders == [2, 3, 4]

    def test_empty_range(self, mixed_samples):
        result = tags.extract_by_order(mixed_samples, 10, 20)
        assert result == []


class TestExtractByMeta:
    def test_meta_match(self, mixed_samples):
        result = tags.extract_by_meta(mixed_samples, "instruction_slug", "bat_swarm")
        assert len(result) == 1

    def test_meta_no_match(self, mixed_samples):
        result = tags.extract_by_meta(mixed_samples, "instruction_slug", "nonexistent")
        assert result == []


class TestGroupBy:
    def test_group_by_game_keys(self, mixed_samples):
        groups = tags.group_by_game(mixed_samples)
        assert set(groups.keys()) == {
            GameTag.ZELDA, GameTag.MARIO, GameTag.DUNGEON, GameTag.DOOM
        }

    def test_group_by_game_counts(self, mixed_samples):
        groups = tags.group_by_game(mixed_samples)
        assert len(groups[GameTag.ZELDA]) == 2
        assert len(groups[GameTag.DUNGEON]) == 2

    def test_group_by_instruction(self, mixed_samples):
        groups = tags.group_by_instruction(mixed_samples)
        assert "__no_instruction__" in groups
        assert len(groups["__no_instruction__"]) == 3

    def test_count_by_game(self, mixed_samples):
        counts = tags.count_by_game(mixed_samples)
        assert counts[GameTag.ZELDA] == 2
        assert counts[GameTag.MARIO] == 1


class TestSummary:
    def test_summary_keys(self, mixed_samples):
        s = tags.summary(mixed_samples)
        for key in ("total", "by_game", "with_instruction",
                    "without_instruction", "unique_instructions"):
            assert key in s

    def test_summary_total(self, mixed_samples):
        s = tags.summary(mixed_samples)
        assert s["total"] == 6

    def test_summary_unique_instructions(self, mixed_samples):
        s = tags.summary(mixed_samples)
        assert s["unique_instructions"] == 3   # "clear path", "bat swarm", "dense wall"

