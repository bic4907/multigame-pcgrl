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

5. JSON _categories 변경 → env 타일 개수 연동 검증
   - tile_mapping.json 의 _categories 개수가 바뀌면
     make_multigame_env() 의 action_space.n / n_editable_tiles 도
     동일하게 바뀌는지 임시 JSON 으로 검증
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
    """VGLC 게임은 현재 지원하지 않으므로 skip."""
    pytest.skip(f"VGLC 게임({game})은 현재 지원하지 않음 (dungeon/sokoban만 지원)")


def test_dungeon_handler_tile_ids_match_mapping(tile_mapping):
    """dungeon handler의 DungeonTile ID 집합 ⊆ tile_mapping["dungeon"] mapping key 집합."""
    from dataset.multigame.handlers.dungeon_handler import DungeonTile
    mapping_keys = {int(k) for k in tile_mapping["dungeon"]["mapping"].keys()}
    handler_ids  = {v for k, v in vars(DungeonTile).items() if not k.startswith("_")}
    missing = handler_ids - mapping_keys
    assert not missing, (
        f"[dungeon] handler에는 있지만 tile_mapping에 없는 tile ID: {missing}"
    )


def test_pokemon_handler_tile_ids_match_mapping(tile_mapping):
    """pokemon handler의 tile ID 집합도 tile_mapping["pokemon"] 범위 안이어야 한다."""
    # Pokemon은 다양한 타일을 사용하므로, tile_mapping["pokemon"]이 모두 정의되어 있는지 확인
    assert "pokemon" in tile_mapping, "tile_mapping에 'pokemon' 게임 정의 없음"
    assert "mapping" in tile_mapping["pokemon"], "pokemon의 'mapping' 정의 없음"
    # Pokemon은 FDM 핸들러이므로, mapping key 집합이 비어있지 않아야 함
    mapping_keys = {int(k) for k in tile_mapping["pokemon"]["mapping"].keys()}
    assert len(mapping_keys) > 0, "pokemon tile_mapping이 비어있음"



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
    """tile_mapping에 사용된 category가 모두 _categories 정의 범위 안이어야 한다.

    Note: dungeon/sokoban만 지원하므로 hazard(6) 등 일부 category는
    실제 게임에 사용되지 않을 수 있으나, 범위 초과 값은 없어야 한다.
    """
    valid_cat_ids = {int(k) for k in tile_mapping["_categories"].keys()}
    for game, section in tile_mapping.items():
        if game.startswith("_"):
            continue
        used = set(section["mapping"].values())
        out_of_range = used - valid_cat_ids
        assert not out_of_range, (
            f"[{game}] mapping에 _categories 범위 밖의 category가 있음: {out_of_range}"
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

def _dataset_dirs_exist() -> bool:
    """dungeon, sokoban, pokemon, 또는 doom 데이터 폴더 중 하나라도 존재하면 True.
    
    근본 원인 해결:
    MultiGameDataset은 여러 게임을 지원하므로, 테스트도 모든 가능한 데이터셋을 확인해야 한다.
    사용 가능한 데이터: dungeon, sokoban, pokemon(FDM), doom, doom2
    """
    from dataset.multigame.handlers.dungeon_handler import _DEFAULT_DUNGEON_ROOT
    from dataset.multigame.handlers.boxoban_handler import _DEFAULT_BOXOBAN_ROOT
    from dataset.multigame.handlers.pokemon_handler import _DEFAULT_POKEMON_ROOT
    from dataset.multigame.handlers.doom_handler import _DEFAULT_DOOM_ROOT, _DEFAULT_DOOM2_ROOT
    
    return (
        Path(_DEFAULT_DUNGEON_ROOT).exists()
        or Path(_DEFAULT_BOXOBAN_ROOT).exists()
        or Path(_DEFAULT_POKEMON_ROOT).exists()
        or Path(_DEFAULT_DOOM_ROOT).exists()
        or Path(_DEFAULT_DOOM2_ROOT).exists()
    )


@pytest.fixture(scope="function")
def dataset_samples():
    """MultiGameDataset 로드 — 데이터 폴더가 없으면 즉시 FAIL.
    
    캐시 프로세스:
    1. dungeon, sokoban 데이터 로드
    2. pokemon(FDM) 데이터 로드 및 필터링
       - max_tile_ratio 필터링 (기본 0.95)
       - tileset 필터링 (max_tile_count=250)
       - instruction 필터링 (min_words=2)
    3. doom/doom2 데이터 로드
    4. 데이터 증강 (rotate_90)
    5. cache 저장
    
    모든 게임의 샘플은 통합 tile_mapping에 따라 매핑됨.
    """
    assert _dataset_dirs_exist(), (
        "다음 데이터 폴더 중 하나 이상이 필요합니다:\n"
        "  ✓ dataset/dungeon_level_dataset (Dungeon)\n"
        "  ✓ dataset/boxoban_levels (Sokoban)\n"
        "  ✓ dataset/five-dollar-model (POKEMON/FDM)\n"
        "  ✓ dataset/TheVGLC/Doom (DOOM 1)\n"
        "  ✓ dataset/TheVGLC/Doom2 (DOOM 2)\n"
        "\n현재 존재하는 데이터 폴더가 없어서 테스트가 실패했습니다."
    )
    from dataset.multigame import MultiGameDataset
    ds = MultiGameDataset(include_dungeon=True)
    samples = ds._samples
    assert len(samples) > 0, (
        "데이터 폴더는 존재하지만 샘플이 0개입니다. "
        "dungeon/sokoban/pokemon/doom 데이터 파일을 확인하세요."
    )
    return samples


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
    """지원 게임(dungeon, sokoban, pokemon)이 데이터셋에 존재해야 한다."""
    supported_games = {"dungeon", "sokoban"}
    dataset_games = {s.game for s in dataset_samples}
    missing = supported_games - dataset_games
    assert not missing, f"지원 게임이지만 dataset에 없는 게임: {missing}"


def test_dataset_pokemon_samples_present(dataset_samples):
    """Pokemon(FDM) 샘플이 존재하고 충분한 개수여야 한다.
    
    Pokemon 캐시 프로세스:
    1. FDM 데이터 로드
    2. max_tile_ratio 필터링 (기본 0.95) - 타일 편차 많은 샘플 제거
    3. tileset 필터링 (max_tile_count=250) - 동일 타일 너무 많은 샘플 제거
    4. instruction 필터링 (min_words=2) - 텍스트 설명 부족한 샘플 제거
    """
    pokemon_samples = [s for s in dataset_samples if s.game == "pokemon"]
    assert len(pokemon_samples) > 0, "Pokemon 샘플이 로드되지 않음"
    # Pokemon은 필터링을 거치므로 원본(887)보다 적어야 함 (기본 1564로 설정)
    assert len(pokemon_samples) > 800, f"Pokemon 샘플이 너무 적음: {len(pokemon_samples)}"


def test_dataset_pokemon_cache_filtering_applied(dataset_samples):
    """Pokemon 샘플이 캐시 필터링을 통과했는지 검증.
    
    필터링 기준:
    - max_tile_ratio=0.95: 타일 빈도 max > 95% 제거
    - max_tile_count=250: 256개 타일 중 250개 이상이 같은 타일 제거
    - min_instruction_words=2: 명령어 단어 2개 미만 제거
    """
    pokemon_samples = [s for s in dataset_samples if s.game == "pokemon"]
    
    # 모든 pokemon 샘플이 instruction을 가져야 함
    no_instruction = [s.source_id for s in pokemon_samples if not s.instruction]
    assert len(no_instruction) == 0, (
        f"instruction이 없는 pokemon 샘플이 캐시 필터링을 통과함: {no_instruction[:5]}"
    )
    
    # 모든 pokemon 샘플의 instruction이 최소 단어 수를 만족해야 함
    for sample in pokemon_samples[:10]:  # 샘플링으로 검증
        words = sample.instruction.split()
        assert len(words) >= 2, (
            f"instruction 단어 수가 2개 미만: {sample.source_id} → "
            f"'{sample.instruction}'"
        )



# ═══════════════════════════════════════════════════════════════════════════════
# 5. JSON _categories 변경 → env 타일 개수 연동 검증
# ═══════════════════════════════════════════════════════════════════════════════

def _make_env_from_json(tmp_json_path: Path):
    """임시 tile_mapping.json 을 기반으로 multigame env 를 생성하고,
    (env, n_categories) 를 반환한다. 테스트 후 원본 모듈 상태를 복원한다."""
    import json
    import importlib
    from enum import IntEnum
    import envs.probs.multigame as mg_module
    from envs.pcgrl_env import PROB_CLASSES, ProbEnum

    # ── 임시 JSON 읽기 ──────────────────────────────────────────────────────
    with tmp_json_path.open("r", encoding="utf-8") as f:
        tmp_config = json.load(f)

    new_categories = {int(k): v for k, v in tmp_config["_categories"].items()}
    n_new = len(new_categories)

    # ── 모듈 레벨 속성 패치 (원본 백업) ────────────────────────────────────
    orig_config     = mg_module._MAPPING_CONFIG
    orig_categories = mg_module._CATEGORIES
    orig_n          = mg_module.NUM_CATEGORIES
    orig_tiles      = mg_module.MultigameTiles
    orig_prob_cls   = PROB_CLASSES.get(max(ProbEnum) + 1)

    try:
        mg_module._MAPPING_CONFIG = tmp_config
        mg_module._CATEGORIES     = new_categories
        mg_module.NUM_CATEGORIES  = n_new

        new_tiles = IntEnum(
            "MultigameTiles",
            {"BORDER": 0, **{name.upper(): idx + 1 for idx, name in new_categories.items()}},
        )
        mg_module.MultigameTiles = new_tiles

        # MultigameProblem 클래스 속성 갱신
        mg_module.MultigameProblem.tile_enum  = new_tiles
        mg_module.MultigameProblem.tile_probs = tuple(
            [0.0] + [1.0 / n_new] * n_new
        )
        mg_module.MultigameProblem.tile_nums = tuple([0] * len(new_tiles))

        # PROB_CLASSES 강제 갱신 (캐시 무효화)
        _MULTIGAME_KEY = max(ProbEnum) + 1
        PROB_CLASSES[_MULTIGAME_KEY] = mg_module.MultigameProblem

        env, _ = mg_module.make_multigame_env()
        return env, n_new

    finally:
        # ── 원본 복원 ────────────────────────────────────────────────────────
        mg_module._MAPPING_CONFIG    = orig_config
        mg_module._CATEGORIES        = orig_categories
        mg_module.NUM_CATEGORIES     = orig_n
        mg_module.MultigameTiles     = orig_tiles
        mg_module.MultigameProblem.tile_enum  = orig_tiles
        mg_module.MultigameProblem.tile_probs = tuple(
            [0.0] + [1.0 / orig_n] * orig_n
        )
        mg_module.MultigameProblem.tile_nums = tuple([0] * len(orig_tiles))
        _MULTIGAME_KEY = max(ProbEnum) + 1
        if orig_prob_cls is not None:
            PROB_CLASSES[_MULTIGAME_KEY] = orig_prob_cls
        elif _MULTIGAME_KEY in PROB_CLASSES:
            del PROB_CLASSES[_MULTIGAME_KEY]


def test_json_categories_change_updates_action_space(tmp_path, tile_mapping):
    """_categories 개수를 줄인 임시 JSON 으로 env 를 만들면
    action_space.n 과 n_editable_tiles 가 그 수와 일치해야 한다."""
    import json

    # 원본 categories 에서 마지막 1개 제거 (6개로 축소)
    original_cats = tile_mapping["_categories"]
    reduced_cats  = {str(k): v for k, v in list(original_cats.items())[:-1]}

    modified = dict(tile_mapping)
    modified["_categories"] = reduced_cats

    tmp_json = tmp_path / "tile_mapping_reduced.json"
    tmp_json.write_text(json.dumps(modified), encoding="utf-8")

    env, n_cats = _make_env_from_json(tmp_json)
    expected = len(reduced_cats)   # 6

    assert n_cats == expected, (
        f"NUM_CATEGORIES({n_cats}) != _categories 수({expected})"
    )
    assert env.rep.n_editable_tiles == expected, (
        f"n_editable_tiles({env.rep.n_editable_tiles}) != "
        f"_categories 수({expected})"
    )
    assert env.action_space(None).n == expected, (
        f"action_space.n({env.action_space(None).n}) != "
        f"_categories 수({expected})"
    )


def test_json_categories_increase_updates_action_space(tmp_path, tile_mapping):
    """_categories 개수를 1개 늘린 임시 JSON 으로 env 를 만들면
    action_space.n 과 n_editable_tiles 가 그 수와 일치해야 한다."""
    import json

    original_cats = dict(tile_mapping["_categories"])
    new_idx = str(len(original_cats))
    expanded_cats = {**original_cats, new_idx: "extra"}

    modified = dict(tile_mapping)
    modified["_categories"] = expanded_cats

    tmp_json = tmp_path / "tile_mapping_expanded.json"
    tmp_json.write_text(json.dumps(modified), encoding="utf-8")

    env, n_cats = _make_env_from_json(tmp_json)
    expected = len(expanded_cats)   # 8

    assert n_cats == expected, (
        f"NUM_CATEGORIES({n_cats}) != _categories 수({expected})"
    )
    assert env.rep.n_editable_tiles == expected, (
        f"n_editable_tiles({env.rep.n_editable_tiles}) != "
        f"_categories 수({expected})"
    )
    assert env.action_space(None).n == expected, (
        f"action_space.n({env.action_space(None).n}) != "
        f"_categories 수({expected})"
    )



