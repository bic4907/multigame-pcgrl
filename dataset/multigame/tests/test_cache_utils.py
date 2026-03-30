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
    # legacy
    build_cache_key,
    load_samples_from_cache,
    save_samples_to_cache,
    # per-game (v2)
    build_per_game_cache_key,
    load_game_samples_from_cache,
    save_game_samples_to_cache,
    load_any_game_cache,
    list_cached_games,
)


def _sample(game: str, source_id: str, value: int) -> GameSample:
    arr = np.full((16, 16), value, dtype=np.int32)
    return GameSample(game=game, source_id=source_id, array=arr, instruction="txt", order=value)


# ═══════════════════════════════════════════════════════════════════════════════
#  Legacy (v1) 테스트
# ═══════════════════════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════════════════════
#  Per-game (v2) 테스트
# ═══════════════════════════════════════════════════════════════════════════════

def test_per_game_cache_roundtrip(tmp_path):
    """게임별 캐시 저장/로드 roundtrip."""
    cache_dir = tmp_path / "artifacts"
    game = "dungeon"
    key = "testkey123"
    samples = [_sample("dungeon", "d001", 1), _sample("dungeon", "d002", 2)]

    save_game_samples_to_cache(cache_dir, game, key, samples)
    loaded = load_game_samples_from_cache(cache_dir, game, key)

    assert loaded is not None
    assert len(loaded) == 2
    assert loaded[0].game == "dungeon"
    assert int(loaded[1].array[0, 0]) == 2
    # 파일이 game 서브디렉토리에 생성되었는지 확인
    assert (cache_dir / game).exists()
    assert any((cache_dir / game).glob("*.npz"))


def test_per_game_key_isolation(tmp_path):
    """한 게임의 파라미터를 바꿔도 다른 게임의 캐시 키가 바뀌지 않는다."""
    hc1 = {"rotate_90": False, "max_samples": 1000}
    hc2 = {"rotate_90": True, "max_samples": 1000}

    k_dungeon_1 = build_per_game_cache_key("dungeon", "/data/dungeon", hc1)
    k_dungeon_2 = build_per_game_cache_key("dungeon", "/data/dungeon", hc1)
    assert k_dungeon_1 == k_dungeon_2  # 같은 파라미터 → 같은 키

    k_pokemon_1 = build_per_game_cache_key("pokemon", "/data/pokemon", hc1)
    k_pokemon_2 = build_per_game_cache_key("pokemon", "/data/pokemon", hc2)
    assert k_pokemon_1 != k_pokemon_2  # 파라미터 변경 → 키 변경

    # dungeon 키는 pokemon 파라미터 변경에 영향받지 않음
    assert k_dungeon_1 == build_per_game_cache_key("dungeon", "/data/dungeon", hc1)


def test_artifact_only_fallback(tmp_path):
    """원본 데이터 없이 캐시 artifact만으로 로드 가능."""
    cache_dir = tmp_path / "artifacts"
    game = "dungeon"
    key = "old_key_abc"
    samples = [_sample("dungeon", "d999", 9)]

    # 캐시 저장 (실제 학습 후 생성된 것처럼)
    save_game_samples_to_cache(cache_dir, game, key, samples)

    # 다른 키로 시도 → 매칭 안 됨
    wrong_key = "new_key_xyz"
    loaded = load_game_samples_from_cache(cache_dir, game, wrong_key)
    assert loaded is None  # 정확한 키로는 못 찾음

    # artifact-only fallback: 키를 모를 때 아무 캐시나 로드
    fallback = load_any_game_cache(cache_dir, game)
    assert fallback is not None
    assert len(fallback) == 1
    assert fallback[0].source_id == "d999"


def test_list_cached_games(tmp_path):
    """캐시 디렉토리에서 게임 목록 조회."""
    cache_dir = tmp_path / "artifacts"
    save_game_samples_to_cache(
        cache_dir, "dungeon", "k1", [_sample("dungeon", "d1", 1)]
    )
    save_game_samples_to_cache(
        cache_dir, "pokemon", "k2", [_sample("pokemon", "p1", 2)]
    )

    games = list_cached_games(cache_dir)
    assert "dungeon" in games
    assert "pokemon" in games
    assert len(games) == 2


def test_multiple_games_independent_caches(tmp_path):
    """여러 게임이 독립적으로 캐시된다."""
    cache_dir = tmp_path / "artifacts"
    dungeon_key = "dk1"
    pokemon_key = "pk1"

    dungeon_samples = [_sample("dungeon", "d1", 1), _sample("dungeon", "d2", 2)]
    pokemon_samples = [_sample("pokemon", "p1", 3)]

    save_game_samples_to_cache(cache_dir, "dungeon", dungeon_key, dungeon_samples)
    save_game_samples_to_cache(cache_dir, "pokemon", pokemon_key, pokemon_samples)

    # 각각 독립적으로 로드
    d = load_game_samples_from_cache(cache_dir, "dungeon", dungeon_key)
    p = load_game_samples_from_cache(cache_dir, "pokemon", pokemon_key)

    assert d is not None and len(d) == 2
    assert p is not None and len(p) == 1
    assert d[0].game == "dungeon"
    assert p[0].game == "pokemon"


