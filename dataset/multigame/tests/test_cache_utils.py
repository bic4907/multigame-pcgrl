"""tests for local cache helpers."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_DATASET_ROOT = Path(__file__).parent.parent.parent
if str(_DATASET_ROOT) not in sys.path:
    sys.path.insert(0, str(_DATASET_ROOT))

from multigame.base import GameSample
from multigame.cache_utils import (
    build_cache_key,
    load_samples_from_cache,
    save_samples_to_cache,
)


def _sample(game: str, source_id: str, value: int) -> GameSample:
    arr = np.full((16, 16), value, dtype=np.int32)
    return GameSample(game=game, source_id=source_id, array=arr, instruction="txt", order=value)


def test_cache_save_and_load_roundtrip(tmp_path):
    cache_dir = tmp_path / "cache"
    key = "abc123"
    samples = [_sample("dungeon", "000001", 1), _sample("dungeon", "000002", 2)]

    save_samples_to_cache(cache_dir, key, samples)
    loaded = load_samples_from_cache(cache_dir, key)

    assert loaded is not None
    assert len(loaded) == 2
    assert loaded[0].game == "dungeon"
    assert loaded[0].array.shape == (16, 16)
    assert int(loaded[1].array[0, 0]) == 2


def test_build_cache_key_changes_with_args():
    code_root = Path(__file__).parent.parent
    k1 = build_cache_key({"include_dungeon": True, "vglc_games": []}, code_root=code_root)
    k2 = build_cache_key({"include_dungeon": False, "vglc_games": []}, code_root=code_root)
    assert k1 != k2


def test_build_cache_key_stable_for_same_args():
    code_root = Path(__file__).parent.parent
    args = {"include_dungeon": True, "vglc_games": ["zelda"]}
    k1 = build_cache_key(args, code_root=code_root)
    k2 = build_cache_key(args, code_root=code_root)
    assert k1 == k2

