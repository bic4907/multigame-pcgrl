"""
dataset/multigame/tests/test_tile_utils.py
==========================================
tile_utils (unified mapping + one-hot) 테스트 모음.

실행:
    pytest dataset/multigame/tests/test_tile_utils.py -v
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from dataset.multigame.tile_utils import (
    NUM_CATEGORIES,
    UNIFIED_CATEGORIES,
    available_games,
    category_distribution,
    category_name,
    onehot_to_unified,
    render_unified_rgb,
    to_onehot,
    to_unified,
    to_unified_and_onehot,
    validate_onehot,
)

# ── 상수 검사 ────────────────────────────────────────────────────────────────────

class TestConstants:
    def test_num_categories(self):
        assert NUM_CATEGORIES == 7

    def test_unified_categories_keys(self):
        assert set(UNIFIED_CATEGORIES.keys()) == set(range(7))

    def test_unified_category_names(self):
        expected = {"empty", "wall", "floor", "enemy", "object", "spawn", "hazard"}
        assert set(UNIFIED_CATEGORIES.values()) == expected

    def test_available_games_includes_all(self):
        games = available_games()
        for g in ["zelda", "mario", "lode_runner", "kid_icarus", "doom", "mega_man", "dungeon"]:
            assert g in games, f"'{g}' missing from available_games()"

    def test_category_name(self):
        assert category_name(0) == "empty"
        assert category_name(1) == "wall"
        assert category_name(6) == "hazard"

    def test_category_name_unknown(self):
        name = category_name(999)
        assert "unknown" in name.lower()


# ── to_unified ──────────────────────────────────────────────────────────────────

class TestToUnified:
    def _make_array(self, values: list[list[int]]) -> np.ndarray:
        return np.array(values, dtype=np.int32)

    def test_zelda_empty(self):
        arr = self._make_array([[0]])
        result = to_unified(arr, "zelda")
        assert result[0, 0] == 0  # empty

    def test_zelda_wall(self):
        arr = self._make_array([[1]])
        result = to_unified(arr, "zelda")
        assert result[0, 0] == 1  # wall

    def test_zelda_mob_is_enemy(self):
        arr = self._make_array([[6]])
        result = to_unified(arr, "zelda")
        assert result[0, 0] == 3  # enemy

    def test_zelda_door_is_spawn(self):
        arr = self._make_array([[3]])
        result = to_unified(arr, "zelda")
        assert result[0, 0] == 5  # spawn

    def test_mario_enemy_is_enemy(self):
        arr = self._make_array([[4]])
        result = to_unified(arr, "mario")
        assert result[0, 0] == 3  # enemy

    def test_mario_cannon_is_hazard(self):
        arr = self._make_array([[7]])
        result = to_unified(arr, "mario")
        assert result[0, 0] == 6  # hazard

    def test_dungeon_floor(self):
        arr = self._make_array([[1]])
        result = to_unified(arr, "dungeon")
        assert result[0, 0] == 2  # floor

    def test_dungeon_wall(self):
        arr = self._make_array([[2]])
        result = to_unified(arr, "dungeon")
        assert result[0, 0] == 1  # wall

    def test_dungeon_enemy(self):
        arr = self._make_array([[3]])
        result = to_unified(arr, "dungeon")
        assert result[0, 0] == 3  # enemy

    def test_shape_preserved(self):
        arr = np.zeros((16, 16), dtype=np.int32)
        result = to_unified(arr, "zelda")
        assert result.shape == (16, 16)

    def test_output_dtype_int32(self):
        arr = np.zeros((4, 4), dtype=np.int32)
        result = to_unified(arr, "mario")
        assert result.dtype == np.int32

    def test_output_range(self):
        """결과 값은 반드시 [0, NUM_CATEGORIES-1] 범위여야 한다."""
        for game in available_games():
            arr = np.zeros((4, 4), dtype=np.int32)
            result = to_unified(arr, game)
            assert result.min() >= 0
            assert result.max() < NUM_CATEGORIES

    def test_unknown_tile_warns_and_falls_back_to_empty(self):
        arr = np.array([[9999]], dtype=np.int32)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = to_unified(arr, "zelda", warn_unmapped=True)
            assert any("9999" in str(warning.message) for warning in w), \
                "Warning about unmapped tile value not raised"
        assert result[0, 0] == 0  # empty fallback

    def test_unknown_game_warns_and_returns_zeros(self):
        arr = np.ones((3, 3), dtype=np.int32)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = to_unified(arr, "nonexistent_game")
            assert len(w) > 0
        assert (result == 0).all()

    def test_known_unknown_tile_id_99(self):
        """99 (UNKNOWN) 타일은 각 게임에서 empty(0)로 매핑되어야 한다."""
        for game in ["zelda", "mario", "lode_runner"]:
            arr = np.array([[99]], dtype=np.int32)
            result = to_unified(arr, game)
            assert result[0, 0] == 0, f"game={game}: tile 99 should map to 0(empty)"


# ── to_onehot ────────────────────────────────────────────────────────────────────

class TestToOnehot:
    def test_shape(self):
        unified = np.zeros((16, 16), dtype=np.int32)
        oh = to_onehot(unified)
        assert oh.shape == (16, 16, NUM_CATEGORIES)

    def test_dtype_uint8(self):
        unified = np.zeros((4, 4), dtype=np.int32)
        oh = to_onehot(unified)
        assert oh.dtype == np.uint8

    def test_values_binary(self):
        unified = np.arange(NUM_CATEGORIES, dtype=np.int32).reshape(1, NUM_CATEGORIES)
        oh = to_onehot(unified)
        unique = set(np.unique(oh).tolist())
        assert unique <= {0, 1}

    def test_each_cell_sums_to_one(self):
        unified = np.array([[0, 1, 2, 3, 4, 5, 6, 0]], dtype=np.int32)
        oh = to_onehot(unified)
        assert (oh.sum(axis=-1) == 1).all()

    def test_correct_channel_activated(self):
        for cat in range(NUM_CATEGORIES):
            unified = np.full((1, 1), cat, dtype=np.int32)
            oh = to_onehot(unified)
            assert oh[0, 0, cat] == 1
            other_channels = list(range(NUM_CATEGORIES))
            other_channels.remove(cat)
            assert (oh[0, 0, other_channels] == 0).all()

    def test_out_of_range_raises(self):
        unified = np.array([[NUM_CATEGORIES]], dtype=np.int32)
        with pytest.raises(ValueError, match="out-of-range"):
            to_onehot(unified)

    def test_negative_raises(self):
        unified = np.array([[-1]], dtype=np.int32)
        with pytest.raises(ValueError):
            to_onehot(unified)


# ── to_unified_and_onehot ────────────────────────────────────────────────────────

class TestToUnifiedAndOnehot:
    def test_returns_tuple(self):
        arr = np.zeros((16, 16), dtype=np.int32)
        result = to_unified_and_onehot(arr, "zelda")
        assert isinstance(result, tuple) and len(result) == 2

    def test_unified_shape(self):
        arr = np.zeros((16, 16), dtype=np.int32)
        unified, _ = to_unified_and_onehot(arr, "mario")
        assert unified.shape == (16, 16)

    def test_onehot_shape(self):
        arr = np.zeros((16, 16), dtype=np.int32)
        _, oh = to_unified_and_onehot(arr, "dungeon")
        assert oh.shape == (16, 16, NUM_CATEGORIES)

    def test_consistency_unified_and_onehot(self):
        arr = np.array([[0, 1, 2], [3, 1, 0]], dtype=np.int32)
        unified, oh = to_unified_and_onehot(arr, "dungeon")
        # argmax of onehot must equal unified
        assert (oh.argmax(axis=-1) == unified).all()

    @pytest.mark.parametrize("game", ["zelda", "mario", "lode_runner",
                                       "kid_icarus", "doom", "mega_man", "dungeon"])
    def test_all_games_produce_valid_onehot(self, game):
        arr = np.zeros((16, 16), dtype=np.int32)
        _, oh = to_unified_and_onehot(arr, game)
        ok, info = validate_onehot(oh)
        assert ok, f"game={game} produced invalid one-hot: {info['errors']}"


# ── validate_onehot ──────────────────────────────────────────────────────────────

class TestValidateOnehot:
    def _valid_oh(self, shape=(4, 4)):
        unified = np.zeros(shape, dtype=np.int32)
        return to_onehot(unified)

    def test_valid_returns_true(self):
        oh = self._valid_oh()
        ok, info = validate_onehot(oh)
        assert ok
        assert info["errors"] == []

    def test_min_max_reported(self):
        oh = self._valid_oh()
        ok, info = validate_onehot(oh)
        assert info["min"] == 0
        assert info["max"] == 1

    def test_min_expected_is_zero(self):
        oh = self._valid_oh()
        _, info = validate_onehot(oh)
        assert info["min_expected"] == 0

    def test_max_expected_is_one(self):
        oh = self._valid_oh()
        _, info = validate_onehot(oh)
        assert info["max_expected"] == 1

    def test_invalid_value_detected(self):
        oh = self._valid_oh()
        oh[0, 0, 0] = 2  # corrupt
        ok, info = validate_onehot(oh)
        assert not ok
        assert any("2" in e for e in info["errors"])

    def test_sum_not_one_detected(self):
        oh = self._valid_oh()
        oh[1, 1, :] = 0  # all-zero cell → sum = 0
        ok, info = validate_onehot(oh)
        assert not ok

    def test_wrong_num_categories_detected(self):
        oh = np.zeros((4, 4, 3), dtype=np.uint8)
        oh[:, :, 0] = 1
        ok, info = validate_onehot(oh, num_categories=NUM_CATEGORIES)
        assert not ok

    def test_shape_in_info(self):
        oh = self._valid_oh((3, 5))
        _, info = validate_onehot(oh)
        assert info["shape"] == (3, 5, NUM_CATEGORIES)

    def test_num_categories_in_info(self):
        oh = self._valid_oh()
        _, info = validate_onehot(oh)
        assert info["num_categories"] == NUM_CATEGORIES


# ── onehot_to_unified ────────────────────────────────────────────────────────────

class TestOnehotToUnified:
    def test_roundtrip(self):
        original = np.random.randint(0, NUM_CATEGORIES, (16, 16), dtype=np.int32)
        oh = to_onehot(original)
        recovered = onehot_to_unified(oh)
        np.testing.assert_array_equal(original, recovered)

    def test_shape(self):
        oh = to_onehot(np.zeros((16, 16), dtype=np.int32))
        result = onehot_to_unified(oh)
        assert result.shape == (16, 16)

    def test_dtype(self):
        oh = to_onehot(np.zeros((4, 4), dtype=np.int32))
        result = onehot_to_unified(oh)
        assert result.dtype == np.int32


# ── category_distribution ────────────────────────────────────────────────────────

class TestCategoryDistribution:
    def test_keys_are_category_names(self):
        unified = np.zeros((4, 4), dtype=np.int32)
        dist = category_distribution(unified)
        assert set(dist.keys()) == set(UNIFIED_CATEGORIES.values())

    def test_count_mode(self):
        unified = np.zeros((4, 4), dtype=np.int32)
        dist = category_distribution(unified)
        assert dist["empty"] == 16.0  # 4×4 = 16 cells

    def test_normalize_mode_sums_to_one(self):
        unified = np.random.randint(0, NUM_CATEGORIES, (16, 16), dtype=np.int32)
        dist = category_distribution(unified, normalize=True)
        total = sum(dist.values())
        assert abs(total - 1.0) < 1e-6


# ── render_unified_rgb ───────────────────────────────────────────────────────────

class TestRenderUnifiedRgb:
    def test_output_shape(self):
        unified = np.zeros((16, 16), dtype=np.int32)
        img = render_unified_rgb(unified, tile_size=8)
        assert img.shape == (128, 128, 3)

    def test_output_dtype(self):
        unified = np.zeros((4, 4), dtype=np.int32)
        img = render_unified_rgb(unified, tile_size=4)
        assert img.dtype == np.uint8

    def test_known_color(self):
        """empty(0) 타일은 항상 정해진 색상이어야 한다."""
        from dataset.multigame.tile_utils import CATEGORY_COLORS
        unified = np.zeros((1, 1), dtype=np.int32)
        img = render_unified_rgb(unified, tile_size=4)
        expected = CATEGORY_COLORS[0]
        # 타일 중앙 픽셀 색상 확인
        pixel = tuple(img[2, 2].tolist())
        assert pixel == expected, f"expected {expected}, got {pixel}"


# ── 통합: GameSample 흐름 시뮬레이션 ────────────────────────────────────────────

class TestEndToEnd:
    """GameSample.array 기준 end-to-end 검증."""

    def _make_sample_array(self, game: str, tile_ids: list[int]) -> np.ndarray:
        """주어진 tile_ids로 구성된 (4, 4) 배열 생성."""
        data = np.array(tile_ids * ((16 // len(tile_ids)) + 1), dtype=np.int32)[:16]
        return data.reshape(4, 4)

    @pytest.mark.parametrize("game,tile_ids", [
        ("zelda",       [0, 1, 2, 6]),
        ("mario",       [0, 1, 4, 7]),
        ("lode_runner", [0, 1, 5, 6]),
        ("kid_icarus",  [0, 1, 2, 3]),
        ("doom",        [0, 1, 2, 4]),
        ("mega_man",    [0, 1, 3, 7]),
        ("dungeon",     [0, 1, 2, 3]),
    ])
    def test_full_pipeline(self, game, tile_ids):
        arr = self._make_sample_array(game, tile_ids)
        unified, oh = to_unified_and_onehot(arr, game)

        # shape
        assert unified.shape == arr.shape
        assert oh.shape == (*arr.shape, NUM_CATEGORIES)

        # validate
        ok, info = validate_onehot(oh)
        assert ok, f"game={game}: {info['errors']}"

        # min/max
        assert info["min"] == 0
        assert info["max"] == 1

        # roundtrip
        recovered = onehot_to_unified(oh)
        np.testing.assert_array_equal(unified, recovered)


# ── shape / min-max / render 전용 검증 ──────────────────────────────────────────
# CI action에서 이 섹션을 명시적으로 실행한다.

ALL_GAMES = ["zelda", "mario", "lode_runner", "kid_icarus", "doom", "mega_man", "dungeon"]

# 각 게임에서 실제로 나타날 수 있는 타일 ID 목록
_GAME_TILE_IDS: dict[str, list[int]] = {
    "zelda":       [0, 1, 2, 3, 4, 5, 6, 7, 99],
    "mario":       [0, 1, 2, 3, 4, 5, 6, 7, 8, 99],
    "lode_runner": [0, 1, 2, 3, 4, 5, 6, 7, 99],
    "kid_icarus":  [0, 1, 2, 3, 4, 99],
    "doom":        [0, 1, 2, 3, 4, 5, 99],
    "mega_man":    [0, 1, 2, 3, 4, 5, 6, 7, 99],
    "dungeon":     [0, 1, 2, 3],
}


def _make_16x16(tile_ids: list[int]) -> np.ndarray:
    """tile_ids를 반복 배치하여 (16, 16) int32 배열 생성."""
    flat = np.array((tile_ids * 256)[:256], dtype=np.int32)
    return flat.reshape(16, 16)


class TestMappedShape:
    """매핑된 배열이 항상 (16, 16) shape를 유지하는지 검증."""

    @pytest.mark.parametrize("game", ALL_GAMES)
    def test_unified_shape_is_16x16(self, game):
        arr = _make_16x16(_GAME_TILE_IDS[game])
        unified = to_unified(arr, game)
        assert unified.shape == (16, 16), (
            f"[{game}] unified shape {unified.shape} != (16, 16)"
        )

    @pytest.mark.parametrize("game", ALL_GAMES)
    def test_onehot_shape_is_16x16xC(self, game):
        arr = _make_16x16(_GAME_TILE_IDS[game])
        _, oh = to_unified_and_onehot(arr, game)
        assert oh.shape == (16, 16, NUM_CATEGORIES), (
            f"[{game}] one-hot shape {oh.shape} != (16, 16, {NUM_CATEGORIES})"
        )

    @pytest.mark.parametrize("game", ALL_GAMES)
    def test_unified_ndim_is_2(self, game):
        arr = _make_16x16(_GAME_TILE_IDS[game])
        unified = to_unified(arr, game)
        assert unified.ndim == 2, f"[{game}] unified ndim={unified.ndim}, expected 2"

    @pytest.mark.parametrize("game", ALL_GAMES)
    def test_onehot_ndim_is_3(self, game):
        arr = _make_16x16(_GAME_TILE_IDS[game])
        _, oh = to_unified_and_onehot(arr, game)
        assert oh.ndim == 3, f"[{game}] one-hot ndim={oh.ndim}, expected 3"


class TestMappedMinMax:
    """매핑된 unified 배열 및 one-hot 값의 min/max가 정의된 범위 내에 있는지 검증."""

    @pytest.mark.parametrize("game", ALL_GAMES)
    def test_unified_min_is_0(self, game):
        arr = _make_16x16(_GAME_TILE_IDS[game])
        unified = to_unified(arr, game)
        assert int(unified.min()) >= 0, (
            f"[{game}] unified min={unified.min()} < 0"
        )

    @pytest.mark.parametrize("game", ALL_GAMES)
    def test_unified_max_within_num_categories(self, game):
        arr = _make_16x16(_GAME_TILE_IDS[game])
        unified = to_unified(arr, game)
        assert int(unified.max()) < NUM_CATEGORIES, (
            f"[{game}] unified max={unified.max()} >= NUM_CATEGORIES={NUM_CATEGORIES}"
        )

    @pytest.mark.parametrize("game", ALL_GAMES)
    def test_onehot_min_is_0(self, game):
        arr = _make_16x16(_GAME_TILE_IDS[game])
        _, oh = to_unified_and_onehot(arr, game)
        ok, info = validate_onehot(oh)
        assert info["min"] == 0, (
            f"[{game}] one-hot min={info['min']} != 0"
        )

    @pytest.mark.parametrize("game", ALL_GAMES)
    def test_onehot_max_is_1(self, game):
        arr = _make_16x16(_GAME_TILE_IDS[game])
        _, oh = to_unified_and_onehot(arr, game)
        ok, info = validate_onehot(oh)
        assert info["max"] == 1, (
            f"[{game}] one-hot max={info['max']} != 1"
        )

    @pytest.mark.parametrize("game", ALL_GAMES)
    def test_onehot_validate_passes(self, game):
        arr = _make_16x16(_GAME_TILE_IDS[game])
        _, oh = to_unified_and_onehot(arr, game)
        ok, info = validate_onehot(oh)
        assert ok, f"[{game}] validate_onehot failed: {info['errors']}"

    @pytest.mark.parametrize("game", ALL_GAMES)
    def test_onehot_each_cell_sums_to_1(self, game):
        arr = _make_16x16(_GAME_TILE_IDS[game])
        _, oh = to_unified_and_onehot(arr, game)
        cell_sums = oh.sum(axis=-1)  # (16, 16)
        bad = int((cell_sums != 1).sum())
        assert bad == 0, (
            f"[{game}] {bad} cell(s) have one-hot sum != 1"
        )


class TestMappedRender:
    """render_unified_rgb 출력의 shape/dtype/픽셀값 검증."""

    @pytest.mark.parametrize("game", ALL_GAMES)
    def test_render_shape(self, game):
        arr = _make_16x16(_GAME_TILE_IDS[game])
        unified = to_unified(arr, game)
        img = render_unified_rgb(unified, tile_size=4)
        assert img.shape == (64, 64, 3), (
            f"[{game}] render shape {img.shape} != (64, 64, 3)"
        )

    @pytest.mark.parametrize("game", ALL_GAMES)
    def test_render_dtype_uint8(self, game):
        arr = _make_16x16(_GAME_TILE_IDS[game])
        unified = to_unified(arr, game)
        img = render_unified_rgb(unified, tile_size=4)
        assert img.dtype == np.uint8, (
            f"[{game}] render dtype={img.dtype}, expected uint8"
        )

    @pytest.mark.parametrize("game", ALL_GAMES)
    def test_render_pixel_range(self, game):
        """RGB 값은 [0, 255] 범위여야 한다."""
        arr = _make_16x16(_GAME_TILE_IDS[game])
        unified = to_unified(arr, game)
        img = render_unified_rgb(unified, tile_size=4)
        assert int(img.min()) >= 0, f"[{game}] pixel min={img.min()} < 0"
        assert int(img.max()) <= 255, f"[{game}] pixel max={img.max()} > 255"

    @pytest.mark.parametrize("game", ALL_GAMES)
    def test_render_colors_match_category_palette(self, game):
        """각 카테고리 타일 블록의 색상이 CATEGORY_COLORS와 일치해야 한다."""
        from dataset.multigame.tile_utils import CATEGORY_COLORS
        tile_size = 8
        for cat_idx in range(NUM_CATEGORIES):
            unified = np.full((1, 1), cat_idx, dtype=np.int32)
            img = render_unified_rgb(unified, tile_size=tile_size)
            expected_color = CATEGORY_COLORS[cat_idx]
            center = tile_size // 2
            actual_color = tuple(img[center, center].tolist())
            assert actual_color == expected_color, (
                f"[{game}] category {cat_idx}: "
                f"expected color {expected_color}, got {actual_color}"
            )


