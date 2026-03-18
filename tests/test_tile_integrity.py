"""tests/test_tile_integrity.py

타일 무결성 통합 테스트.

검증 항목
---------
1. tile_mapping.json 구조 검증
   - 모든 게임 섹션에 'mapping' 키 존재
   - mapping의 모든 value가 _categories 범위 안에 있을 것

2. dataset handler ↔ tile_mapping 일치 검증
   - 각 VGLC 게임 handler의 Tile 클래스 정의 ID 집합이
     tile_mapping.json의 해당 게임 mapping key 집합과 일치하는지
   - dungeon handler의 DungeonTile ID 집합도 동일하게 검증

3. env ↔ tile_mapping (dungeon) 일치 검증
   - PCGRLEnv(dungeon)의 n_editable_tiles가
     tile_mapping["dungeon"]["mapping"]에서 UNKNOWN(0)을 제외한
     고유 타일 ID 수와 일치하는지

4. 실제 데이터셋 tile 값 범위 검증 (캐시 있을 때만)
   - MultiGameDataset에서 샘플링한 array 값이
     각 게임의 tile_mapping mapping key 범위 안에 있을 것
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

# ── 프로젝트 루트를 sys.path에 추가 ──────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

_MAPPING_PATH = _PROJECT_ROOT / "dataset" / "multigame" / "tile_mapping.json"

# ── fixture: tile_mapping.json 로드 ──────────────────────────────────────────────
@pytest.fixture(scope="session")
def tile_mapping() -> dict:
    assert _MAPPING_PATH.exists(), f"tile_mapping.json not found: {_MAPPING_PATH}"
    return json.loads(_MAPPING_PATH.read_text(encoding="utf-8"))


# ── fixture: multigame env tile info ─────────────────────────────────────────────
@pytest.fixture()
def multigame_env_info() -> dict:
    """make_multigame_env() 로 생성한 env의 tile 정보."""
    from envs.probs.multigame import make_multigame_env
    env, _ = make_multigame_env()
    return {
        "all_tiles":         [t.name for t in env.tile_enum],
        "n_all_tiles":       len(env.tile_enum),
        "editable_tiles":    [t.name for t in env.rep.editable_tile_enum],
        "n_editable_tiles":  env.rep.n_editable_tiles,
        "unavailable_tiles": env.unavailable_tiles,
    }





# ═══════════════════════════════════════════════════════════════════════════════
# 1. tile_mapping.json 구조 검증
# ═══════════════════════════════════════════════════════════════════════════════

def test_mapping_has_categories(tile_mapping):
    """_categories 섹션이 존재하고 비어있지 않아야 한다."""
    assert "_categories" in tile_mapping
    assert len(tile_mapping["_categories"]) > 0


def test_all_games_have_mapping_key(tile_mapping):
    """내부 키(언더스코어로 시작)가 아닌 모든 게임 섹션에 'mapping' 키가 있어야 한다."""
    for game, section in tile_mapping.items():
        if game.startswith("_"):
            continue
        assert "mapping" in section, f"[{game}] 'mapping' 키 없음"


def test_mapping_values_within_category_range(tile_mapping):
    """모든 게임의 mapping value가 _categories 인덱스 범위 안이어야 한다."""
    valid_cat_ids = {int(k) for k in tile_mapping["_categories"].keys()}
    for game, section in tile_mapping.items():
        if game.startswith("_"):
            continue
        for tile_id, cat_id in section["mapping"].items():
            assert cat_id in valid_cat_ids, (
                f"[{game}] tile_id={tile_id} → category={cat_id} 가 "
                f"_categories 범위({valid_cat_ids}) 밖"
            )


# ═══════════════════════════════════════════════════════════════════════════════
# 2. dataset handler ↔ tile_mapping 일치 검증
# ═══════════════════════════════════════════════════════════════════════════════

# VGLC 게임별 Tile 클래스 임포트
def _vglc_tile_ids(game: str) -> set[int]:
    """game handler의 XxxTile 클래스에서 정수 tile ID 집합을 반환한다."""
    from dataset.multigame.handlers.vglc_games import (
        zelda, mario, lode_runner, kid_icarus, doom, mega_man
    )
    module_map = {
        "zelda":       (zelda,       "ZeldaTile"),
        "mario":       (mario,       "MarioTile"),
        "lode_runner": (lode_runner, "LodeRunnerTile"),
        "kid_icarus":  (kid_icarus,  "KidIcarusTile"),
        "doom":        (doom,        "DoomTile"),
        "mega_man":    (mega_man,    "MegaManTile"),
    }
    mod, cls_name = module_map[game]
    tile_cls = getattr(mod, cls_name)
    return {v for k, v in vars(tile_cls).items() if not k.startswith("_")}


@pytest.mark.parametrize("game", [
    "zelda", "mario", "lode_runner", "kid_icarus", "doom", "mega_man"
])
def test_vglc_handler_tile_ids_match_mapping(tile_mapping, game):
    """VGLC handler의 Tile 정의 ID 집합 ⊆ tile_mapping의 mapping key 집합이어야 한다."""
    mapping_keys = {int(k) for k in tile_mapping[game]["mapping"].keys()}
    handler_ids  = _vglc_tile_ids(game)
    missing = handler_ids - mapping_keys
    assert not missing, (
        f"[{game}] handler에는 있지만 tile_mapping에 없는 tile ID: {missing}"
    )


def test_dungeon_handler_tile_ids_match_mapping(tile_mapping):
    """dungeon handler의 DungeonTile ID 집합 ⊆ tile_mapping["dungeon"] mapping key 집합."""
    from dataset.multigame.handlers.dungeon_handler import DungeonTile
    mapping_keys = {int(k) for k in tile_mapping["dungeon"]["mapping"].keys()}
    handler_ids  = {v for k, v in vars(DungeonTile).items() if not k.startswith("_")}
    missing = handler_ids - mapping_keys
    assert not missing, (
        f"[dungeon] handler에는 있지만 tile_mapping에 없는 tile ID: {missing}"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 3. env ↔ tile_mapping (dungeon) 일치 검증
# ═══════════════════════════════════════════════════════════════════════════════

def test_num_categories_matches_tile_mapping(tile_mapping):
    """tile_utils.NUM_CATEGORIES 가 tile_mapping._categories 수와 일치해야 한다."""
    from dataset.multigame.tile_utils import NUM_CATEGORIES
    n_defined = len(tile_mapping["_categories"])
    assert NUM_CATEGORIES == n_defined, (
        f"tile_utils.NUM_CATEGORIES({NUM_CATEGORIES}) != "
        f"tile_mapping._categories 수({n_defined})"
    )


def test_all_categories_used_in_mapping(tile_mapping):
    """tile_mapping 전체 게임에 걸쳐 _categories의 모든 index가 최소 한 번 사용되어야 한다."""
    valid_cat_ids = {int(k) for k in tile_mapping["_categories"].keys()}
    used_cat_ids: set[int] = set()
    for game, section in tile_mapping.items():
        if game.startswith("_"):
            continue
        used_cat_ids.update(section["mapping"].values())
    unused = valid_cat_ids - used_cat_ids
    assert not unused, (
        f"_categories에 정의됐지만 어떤 게임에도 사용 안 된 category: "
        f"{[tile_mapping['_categories'][str(i)] for i in unused]}"
    )


def test_dungeon_env_editable_covers_all_categories(tile_mapping, multigame_env_info):
    """multigame env의 n_editable_tiles 가 tile_mapping의 NUM_CATEGORIES와 정확히 일치해야 한다."""
    from dataset.multigame.tile_utils import NUM_CATEGORIES
    n_editable = multigame_env_info["n_editable_tiles"]
    assert n_editable == NUM_CATEGORIES, (
        f"multigame env editable tiles({n_editable}) != "
        f"unified NUM_CATEGORIES({NUM_CATEGORIES})"
    )


def test_dungeon_dataset_categories_subset_of_all_categories(tile_mapping):
    """dungeon dataset에서 사용하는 unified categories가 _categories 정의 범위 안이어야 한다."""
    valid_cat_ids = {int(k) for k in tile_mapping["_categories"].keys()}
    dungeon_cats = set(tile_mapping["dungeon"]["mapping"].values())
    out_of_range = dungeon_cats - valid_cat_ids
    assert not out_of_range, (
        f"dungeon mapping에 _categories 밖의 category가 있음: {out_of_range}"
    )


def test_multigame_env_all_tile_names(multigame_env_info):
    """MultigameProblem tile_enum 에 BORDER 포함 전체 타일이 있어야 한다."""
    all_tiles = multigame_env_info["all_tiles"]
    assert "BORDER" in all_tiles, "BORDER tile 없음"
    assert len(all_tiles) == multigame_env_info["n_all_tiles"]
    assert multigame_env_info["n_editable_tiles"] == multigame_env_info["n_all_tiles"] - 1, (
        "editable = all - 1(BORDER) 이어야 함 (unavailable_tiles가 없을 때)"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 4. 실제 데이터셋 tile 값 범위 검증
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="function")
def dataset_samples():
    """MultiGameDataset 로드 (캐시 없으면 건너뜀)."""
    try:
        from dataset.multigame import MultiGameDataset
        ds = MultiGameDataset(include_dungeon=True)
        return ds._samples   # GameSample 리스트 (내부 속성)
    except Exception as e:
        pytest.skip(f"MultiGameDataset 로드 실패: {e}")


def test_dataset_samples_loaded(dataset_samples):
    """데이터셋이 비어있지 않아야 한다."""
    assert len(dataset_samples) > 0, "샘플이 0개"


def test_dataset_tile_values_within_mapping(tile_mapping, dataset_samples):
    """각 게임 샘플의 tile 값이 tile_mapping에 정의된 key 범위 안에 있어야 한다."""
    errors: list[str] = []
    for sample in dataset_samples:
        game = sample.game
        if game not in tile_mapping:
            continue  # _comment 등 메타 키는 건너뜀
        valid_ids = {int(k) for k in tile_mapping[game]["mapping"].keys()}
        unique_vals = set(np.unique(sample.array).tolist())
        out_of_range = unique_vals - valid_ids
        if out_of_range:
            errors.append(
                f"[{game}] source_id={sample.source_id} → "
                f"tile 값 {out_of_range} 가 mapping 범위({valid_ids}) 밖"
            )
    assert not errors, "\n".join(errors[:20])  # 최대 20개만 출력


def test_dataset_array_shape(dataset_samples):
    """모든 샘플 array가 (16, 16) shape이어야 한다."""
    bad = [
        (s.game, s.source_id, s.array.shape)
        for s in dataset_samples
        if s.array.shape != (16, 16)
    ]
    assert not bad, f"shape 불일치 샘플: {bad[:10]}"


def test_dataset_array_dtype(dataset_samples):
    """모든 샘플 array dtype이 int32이어야 한다."""
    bad = [
        (s.game, s.source_id, s.array.dtype)
        for s in dataset_samples
        if s.array.dtype != np.int32
    ]
    assert not bad, f"dtype 불일치 샘플: {bad[:10]}"


def test_dataset_games_all_present(tile_mapping, dataset_samples):
    """tile_mapping에 정의된 모든 게임이 데이터셋에 존재해야 한다."""
    mapping_games = {g for g in tile_mapping if not g.startswith("_")}
    dataset_games = {s.game for s in dataset_samples}
    missing = mapping_games - dataset_games
    assert not missing, f"tile_mapping에 있지만 dataset에 없는 게임: {missing}"

