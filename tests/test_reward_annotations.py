"""tests/test_reward_annotations.py

reward annotation 무결성 테스트.

검증 항목
---------
dungeon
  - 모든 샘플에 reward_enum / feature_name / sub_condition / conditions가 있어야 함
  - reward_enum 값이 1~5 범위
  - conditions가 일반 dict (not _WarningConditionsDict)
  - conditions[reward_enum] 값이 존재해야 함

sokoban / zelda / doom / pokemon
  - reward annotation 자체가 없는 것이 정상 (text annotation 미지원)
    → meta에 reward_enum / conditions 키가 없어야 함
  - 단, placeholder CSV가 로드된 경우 conditions는 _WarningConditionsDict이어야 함
    → 접근 시 WARNING 로그가 발생해야 함 (실제로는 값은 있지만 "미완성"임을 알 수 있음)
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ── 데이터 존재 여부 체크 ───────────────────────────────────────────────────────
def _dungeon_exists() -> bool:
    from dataset.multigame.handlers.dungeon_handler import _DEFAULT_DUNGEON_ROOT
    return Path(_DEFAULT_DUNGEON_ROOT).exists()

def _sokoban_exists() -> bool:
    from dataset.multigame.handlers.boxoban_handler import _DEFAULT_BOXOBAN_ROOT
    return Path(_DEFAULT_BOXOBAN_ROOT).exists()

def _zelda_exists() -> bool:
    from dataset.multigame.handlers.zelda_handler import _DEFAULT_ZELDA_ROOT
    return Path(_DEFAULT_ZELDA_ROOT).exists()

def _doom_exists() -> bool:
    from dataset.multigame.handlers.doom_handler import _DEFAULT_DOOM_ROOT, _DEFAULT_DOOM2_ROOT
    return Path(_DEFAULT_DOOM_ROOT).exists() or Path(_DEFAULT_DOOM2_ROOT).exists()

def _pokemon_exists() -> bool:
    from dataset.multigame.handlers.pokemon_handler import _DEFAULT_POKEMON_ROOT
    return Path(_DEFAULT_POKEMON_ROOT).exists()


# ── fixture: 각 게임별 raw samples ─────────────────────────────────────────────
@pytest.fixture(scope="module")
def dungeon_samples():
    """dungeon 게임 raw samples (reward annotation 포함)."""
    if not _dungeon_exists():
        pytest.skip("dungeon 데이터셋 없음")
    from dataset.multigame import MultiGameDataset
    ds = MultiGameDataset(
        include_dungeon=True,
        include_sokoban=False,
        include_zelda=False,
        include_doom=False,
        include_doom2=False,
        include_pokemon=False,
        use_cache=False,
    )
    return [s for s in ds._samples if s.game == "dungeon"]


@pytest.fixture(scope="module")
def sokoban_samples():
    """sokoban 게임 raw samples."""
    if not _sokoban_exists():
        pytest.skip("sokoban(boxoban) 데이터셋 없음")
    from dataset.multigame import MultiGameDataset
    ds = MultiGameDataset(
        include_dungeon=False,
        include_sokoban=True,
        include_zelda=False,
        include_doom=False,
        include_doom2=False,
        include_pokemon=False,
        use_cache=False,
    )
    return [s for s in ds._samples if s.game == "sokoban"]


@pytest.fixture(scope="module")
def zelda_samples():
    """zelda 게임 raw samples."""
    if not _zelda_exists():
        pytest.skip("zelda 데이터셋 없음")
    from dataset.multigame import MultiGameDataset
    ds = MultiGameDataset(
        include_dungeon=False,
        include_sokoban=False,
        include_zelda=True,
        include_doom=False,
        include_doom2=False,
        include_pokemon=False,
        use_cache=False,
    )
    return [s for s in ds._samples if s.game == "zelda"]


@pytest.fixture(scope="module")
def doom_samples():
    """doom 게임 raw samples."""
    if not _doom_exists():
        pytest.skip("doom 데이터셋 없음")
    from dataset.multigame import MultiGameDataset
    ds = MultiGameDataset(
        include_dungeon=False,
        include_sokoban=False,
        include_zelda=False,
        include_doom=True,
        include_doom2=True,
        include_pokemon=False,
        use_cache=False,
    )
    return [s for s in ds._samples if s.game == "doom"]


@pytest.fixture(scope="module")
def pokemon_samples():
    """pokemon 게임 raw samples."""
    if not _pokemon_exists():
        pytest.skip("pokemon(FDM) 데이터셋 없음")
    from dataset.multigame import MultiGameDataset
    ds = MultiGameDataset(
        include_dungeon=False,
        include_sokoban=False,
        include_zelda=False,
        include_doom=False,
        include_doom2=False,
        include_pokemon=True,
        use_cache=False,
    )
    return [s for s in ds._samples if s.game == "pokemon"]


# ══════════════════════════════════════════════════════════════════════════════
# dungeon: per-sample reward annotation이 있어야 함
# ══════════════════════════════════════════════════════════════════════════════

def test_dungeon_all_samples_have_reward_enum(dungeon_samples):
    """dungeon 전체 샘플에 reward_enum이 있어야 한다."""
    missing = [s.source_id for s in dungeon_samples if "reward_enum" not in s.meta]
    assert not missing, f"reward_enum 없는 dungeon 샘플: {missing[:10]}"


def test_dungeon_reward_enum_range(dungeon_samples):
    """dungeon reward_enum 값이 1~5 범위여야 한다."""
    out_of_range = [
        (s.source_id, s.meta["reward_enum"])
        for s in dungeon_samples
        if s.meta.get("reward_enum") not in range(1, 6)
    ]
    assert not out_of_range, f"reward_enum 범위 초과: {out_of_range[:10]}"


def test_dungeon_all_samples_have_feature_name(dungeon_samples):
    """dungeon 전체 샘플에 feature_name이 있어야 한다."""
    valid = {"region", "path_length", "block", "bat_amount", "bat_direction"}
    bad = [
        (s.source_id, s.meta.get("feature_name"))
        for s in dungeon_samples
        if s.meta.get("feature_name") not in valid
    ]
    assert not bad, f"feature_name 이상: {bad[:10]}"


def test_dungeon_conditions_is_plain_dict(dungeon_samples):
    """dungeon conditions는 일반 dict이어야 한다 (WarningConditionsDict가 아님)."""
    from dataset.multigame.dataset import _WarningConditionsDict
    bad = [
        s.source_id
        for s in dungeon_samples
        if isinstance(s.meta.get("conditions"), _WarningConditionsDict)
    ]
    assert not bad, f"dungeon에 _WarningConditionsDict가 설정된 샘플: {bad[:5]}"


def test_dungeon_conditions_contains_reward_enum_key(dungeon_samples):
    """dungeon conditions[reward_enum] 값이 존재해야 한다."""
    bad = []
    for s in dungeon_samples:
        reward_enum = s.meta.get("reward_enum")
        conditions = s.meta.get("conditions", {})
        if reward_enum not in conditions:
            bad.append((s.source_id, reward_enum, list(conditions.keys())))
    assert not bad, f"conditions에 reward_enum 키 없는 샘플: {bad[:5]}"


def test_dungeon_conditions_value_is_float(dungeon_samples):
    """dungeon conditions 값이 float이어야 한다."""
    bad = []
    for s in dungeon_samples:
        for k, v in s.meta.get("conditions", {}).items():
            if not isinstance(v, float):
                bad.append((s.source_id, k, v, type(v).__name__))
    assert not bad, f"conditions 값이 float이 아닌 샘플: {bad[:5]}"


def test_dungeon_feature_distribution(dungeon_samples):
    """dungeon 각 feature(1~5)에 샘플이 존재해야 한다."""
    from collections import Counter
    counts = Counter(s.meta["reward_enum"] for s in dungeon_samples)
    missing_enums = [e for e in range(1, 6) if counts[e] == 0]
    assert not missing_enums, (
        f"reward_enum {missing_enums}에 해당하는 샘플이 없음. "
        f"분포: {dict(counts)}"
    )


# ══════════════════════════════════════════════════════════════════════════════
# 나머지 게임: text annotation 없음 → conditions 없거나 placeholder(Warning) 이어야 함
# ══════════════════════════════════════════════════════════════════════════════

def _assert_no_real_annotation(samples: list, game: str) -> None:
    """
    text annotation이 없는 게임 샘플 검증.

    placeholder CSV가 있으면 conditions가 _WarningConditionsDict이어야 하고,
    없으면 conditions 키 자체가 없어야 함.
    두 경우 모두 '실제 per-sample annotation이 없음'을 의미.
    """
    from dataset.multigame.dataset import _WarningConditionsDict

    _annotations_dir = _PROJECT_ROOT / "dataset" / "reward_annotations"
    placeholder_exists = (_annotations_dir / f"{game}_reward_annotations_placeholder.csv").exists()

    for s in samples:
        conditions = s.meta.get("conditions")

        if placeholder_exists:
            # placeholder CSV가 있으면 _WarningConditionsDict이어야 함
            assert isinstance(conditions, _WarningConditionsDict), (
                f"[{game}] source_id={s.source_id}: placeholder CSV가 있는데 "
                f"conditions 타입이 {type(conditions).__name__}임 "
                f"(기대: _WarningConditionsDict)"
            )
        else:
            # placeholder CSV도 없으면 conditions 키 자체가 없어야 함
            assert conditions is None, (
                f"[{game}] source_id={s.source_id}: annotation이 없어야 하는데 "
                f"conditions={conditions} 가 설정되어 있음"
            )


def _assert_warning_on_conditions_access(samples: list, game: str) -> None:
    """conditions 접근 시 logging.WARNING이 발생하는지 검증."""
    from dataset.multigame.dataset import _WarningConditionsDict

    _annotations_dir = _PROJECT_ROOT / "dataset" / "reward_annotations"
    if not (_annotations_dir / f"{game}_reward_annotations_placeholder.csv").exists():
        pytest.skip(f"{game} placeholder CSV 없음 - warning 테스트 스킵")

    if not samples:
        pytest.skip(f"{game} 샘플 없음")

    sample = samples[0]
    conditions = sample.meta.get("conditions")
    assert isinstance(conditions, _WarningConditionsDict), (
        f"[{game}] conditions가 _WarningConditionsDict가 아님: {type(conditions)}"
    )

    # 새 인스턴스로 테스트 (_warned 초기화 보장)
    records: list[logging.LogRecord] = []

    class _Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    logger_name = "dataset.multigame.dataset"
    logger = logging.getLogger(logger_name)
    handler = _Capture()
    handler.setLevel(logging.WARNING)
    original_level = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.WARNING)

    try:
        fresh = _WarningConditionsDict(dict(conditions), game=game, logger=logger)
        _ = fresh[next(iter(fresh))]  # 첫 번째 키 접근 → WARNING 발생
    finally:
        logger.removeHandler(handler)
        logger.setLevel(original_level)

    assert len(records) > 0, f"[{game}] conditions 접근 시 WARNING이 발생하지 않음"
    assert records[0].levelno == logging.WARNING


# sokoban 테스트
def test_sokoban_no_real_annotation(sokoban_samples):
    """sokoban은 text annotation이 없으므로 per-sample conditions가 없어야 한다."""
    _assert_no_real_annotation(sokoban_samples, "sokoban")


def test_sokoban_conditions_warning_on_access(sokoban_samples):
    """sokoban placeholder conditions 접근 시 WARNING이 발생해야 한다."""
    _assert_warning_on_conditions_access(sokoban_samples, "sokoban")


# zelda 테스트
def test_zelda_no_real_annotation(zelda_samples):
    """zelda는 text annotation이 없으므로 per-sample conditions가 없어야 한다."""
    _assert_no_real_annotation(zelda_samples, "zelda")


def test_zelda_conditions_warning_on_access(zelda_samples):
    """zelda placeholder conditions 접근 시 WARNING이 발생해야 한다."""
    _assert_warning_on_conditions_access(zelda_samples, "zelda")


# doom 테스트
def test_doom_no_real_annotation(doom_samples):
    """doom은 text annotation이 없으므로 per-sample conditions가 없어야 한다."""
    _assert_no_real_annotation(doom_samples, "doom")


def test_doom_conditions_warning_on_access(doom_samples):
    """doom placeholder conditions 접근 시 WARNING이 발생해야 한다."""
    _assert_warning_on_conditions_access(doom_samples, "doom")


# pokemon 테스트
def test_pokemon_no_real_annotation(pokemon_samples):
    """pokemon은 text annotation이 없으므로 per-sample conditions가 없어야 한다."""
    _assert_no_real_annotation(pokemon_samples, "pokemon")


def test_pokemon_conditions_warning_on_access(pokemon_samples):
    """pokemon placeholder conditions 접근 시 WARNING이 발생해야 한다."""
    _assert_warning_on_conditions_access(pokemon_samples, "pokemon")


# ══════════════════════════════════════════════════════════════════════════════
# 공통: reward_enum 범위가 모든 게임에서 1~5 이어야 함
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("game,fixture_name", [
    ("dungeon", "dungeon_samples"),
    ("sokoban", "sokoban_samples"),
    ("zelda",   "zelda_samples"),
    ("doom",    "doom_samples"),
    ("pokemon", "pokemon_samples"),
])
def test_reward_enum_always_1_to_5(game, fixture_name, request):
    """reward annotation이 있는 모든 게임에서 reward_enum이 1~5 범위여야 한다."""
    samples = request.getfixturevalue(fixture_name)
    annotated = [s for s in samples if "reward_enum" in s.meta]
    if not annotated:
        pytest.skip(f"{game}: reward annotation 없음 (정상)")
    out_of_range = [
        (s.source_id, s.meta["reward_enum"])
        for s in annotated
        if s.meta["reward_enum"] not in range(1, 6)
    ]
    assert not out_of_range, (
        f"[{game}] reward_enum 1~5 범위 초과: {out_of_range[:5]}"
    )

