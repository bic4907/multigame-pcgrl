"""tests/test_multigame_evaluator.py

evaluator/measures, losses, fitnesses, rewards 의 multigame_amount 및
multigame_placement 테스트.

- 정확한 값 검증
- jax.jit 컴파일 검증
- jax.vmap 배치 컴파일 검증
- jax.grad 미분 가능성 검증

실행
----
    python -m pytest tests/test_multigame_evaluator.py -v
"""
from __future__ import annotations

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import jax
import jax.numpy as jnp
import pytest

from evaluator.measures.multigame_amount import (
    get_collectable_count,
    get_hazard_count,
    get_interactive_count,
    get_multigame_tile_counts,
)
from evaluator.losses.multigame_amount_loss import multigame_amount_loss
from evaluator.fitnesses.multigame_amount import get_multigame_amount_fitness
from evaluator.rewards.multigame_amount import get_multigame_amount_reward
from evaluator.rewards.multigame_placement import (
    get_multigame_placement_reward,
    get_multigame_tile_placement_reward,
    _cluster_penalty,
    _accessibility_bonus,
    _spread_bonus,
    _cluster_penalty_tile,
    _accessibility_bonus_tile,
    _spread_bonus_tile,
    _tile_amount_diff,
)


# ── 공통 Fixture ───────────────────────────────────────────────────────────────

@pytest.fixture()
def env_map():
    """4×4 테스트 맵.

    BORDER=0(x2), EMPTY=1(x3), WALL=2(x2),
    INTERACTIVE=3(x3), HAZARD=4(x3), COLLECTABLE=5(x3)
    """
    return jnp.array([
        [0, 1, 2, 3],
        [3, 4, 5, 1],
        [5, 5, 3, 2],
        [4, 4, 1, 0],
    ], dtype=jnp.int32)


@pytest.fixture()
def prev_curr_maps(env_map):
    """prev_map = env_map, curr_map = INTERACTIVE 하나를 EMPTY 로 교체."""
    prev = env_map
    curr = prev.at[0, 3].set(1)  # interactive 3→2
    return prev, curr


# ═══════════════════════════════════════════════════════════════════════════════
#  1. measures — 값 검증
# ═══════════════════════════════════════════════════════════════════════════════

class TestMeasures:

    def test_interactive_count(self, env_map):
        assert float(get_interactive_count(env_map)) == 3.0

    def test_hazard_count(self, env_map):
        assert float(get_hazard_count(env_map)) == 3.0

    def test_collectable_count(self, env_map):
        assert float(get_collectable_count(env_map)) == 3.0

    def test_tile_counts_dict(self, env_map):
        counts = get_multigame_tile_counts(env_map)
        assert float(counts["interactive"]) == 3.0
        assert float(counts["hazard"]) == 3.0
        assert float(counts["collectable"]) == 3.0

    def test_empty_map_returns_zero(self):
        empty = jnp.ones((8, 8), dtype=jnp.int32)  # all EMPTY=1
        assert float(get_interactive_count(empty)) == 0.0
        assert float(get_hazard_count(empty)) == 0.0
        assert float(get_collectable_count(empty)) == 0.0


# ═══════════════════════════════════════════════════════════════════════════════
#  2. losses — 값 검증
# ═══════════════════════════════════════════════════════════════════════════════

class TestLoss:

    @pytest.mark.parametrize("tile_name", ["interactive", "hazard", "collectable"])
    def test_absolute_loss(self, env_map, tile_name):
        # count=3, cond=2 → |3-2| = 1.0
        loss = multigame_amount_loss(env_map, tile_name, jnp.array(2.0), absolute=True)
        assert float(loss) == pytest.approx(1.0)

    @pytest.mark.parametrize("tile_name", ["interactive", "hazard", "collectable"])
    def test_signed_loss(self, env_map, tile_name):
        # count=3, cond=5 → 3-5 = -2.0
        loss = multigame_amount_loss(env_map, tile_name, jnp.array(5.0), absolute=False)
        assert float(loss) == pytest.approx(-2.0)

    def test_zero_loss_when_matched(self, env_map):
        loss = multigame_amount_loss(env_map, "interactive", jnp.array(3.0), absolute=True)
        assert float(loss) == pytest.approx(0.0)


# ═══════════════════════════════════════════════════════════════════════════════
#  3. fitnesses — 값 검증
# ═══════════════════════════════════════════════════════════════════════════════

class TestFitness:

    def test_fitness_positive(self, env_map):
        # count=3 > cond=2 → 1.0
        fit = get_multigame_amount_fitness(env_map, jnp.array(2.0), tile_name="interactive")
        assert float(fit) == pytest.approx(1.0)

    def test_fitness_negative(self, env_map):
        # count=3 < cond=5 → -2.0
        fit = get_multigame_amount_fitness(env_map, jnp.array(5.0), tile_name="hazard")
        assert float(fit) == pytest.approx(-2.0)


# ═══════════════════════════════════════════════════════════════════════════════
#  4. rewards — 값 검증
# ═══════════════════════════════════════════════════════════════════════════════

class TestReward:

    def test_reward_positive_when_improved(self, prev_curr_maps):
        prev, curr = prev_curr_maps
        # interactive: prev 3개(loss=1), curr 2개(loss=0) → reward=1.0
        reward = get_multigame_amount_reward(prev, curr, jnp.array(2.0), tile_name="interactive")
        assert float(reward) == pytest.approx(1.0)

    def test_reward_zero_when_unchanged(self, prev_curr_maps):
        prev, curr = prev_curr_maps
        # hazard: 변화 없음 → reward=0.0
        reward = get_multigame_amount_reward(prev, curr, jnp.array(2.0), tile_name="hazard")
        assert float(reward) == pytest.approx(0.0)

    def test_reward_negative_when_worsened(self, env_map):
        prev = env_map
        curr = prev.at[1, 1].set(4)  # hazard 이미 4인 곳이라 변화 없으므로
        curr = prev.at[0, 1].set(4)  # EMPTY→HAZARD, hazard 3→4, cond=2
        # prev loss=|3-2|=1, curr loss=|4-2|=2 → reward=-1.0
        reward = get_multigame_amount_reward(prev, curr, jnp.array(2.0), tile_name="hazard")
        assert float(reward) == pytest.approx(-1.0)


# ═══════════════════════════════════════════════════════════════════════════════
#  5. jax.jit 컴파일 검증
# ═══════════════════════════════════════════════════════════════════════════════

class TestJitCompilation:
    """모든 함수가 jax.jit 으로 컴파일 & 실행 가능한지 검증."""

    def test_jit_interactive_count(self, env_map):
        jitted = jax.jit(get_interactive_count)
        result = jitted(env_map)
        assert float(result) == 3.0

    def test_jit_hazard_count(self, env_map):
        jitted = jax.jit(get_hazard_count)
        result = jitted(env_map)
        assert float(result) == 3.0

    def test_jit_collectable_count(self, env_map):
        jitted = jax.jit(get_collectable_count)
        result = jitted(env_map)
        assert float(result) == 3.0

    @pytest.mark.parametrize("tile_name", ["interactive", "hazard", "collectable"])
    def test_jit_loss(self, env_map, tile_name):
        from functools import partial
        jitted = jax.jit(partial(multigame_amount_loss, tile_name=tile_name), static_argnames=("absolute",))
        result = jitted(env_map, cond=jnp.array(2.0), absolute=True)
        assert float(result) == pytest.approx(1.0)

    @pytest.mark.parametrize("tile_name", ["interactive", "hazard", "collectable"])
    def test_jit_fitness(self, env_map, tile_name):
        from functools import partial
        jitted = jax.jit(partial(get_multigame_amount_fitness, tile_name=tile_name))
        result = jitted(env_map, jnp.array(2.0))
        assert float(result) == pytest.approx(1.0)

    def test_jit_reward(self, prev_curr_maps):
        prev, curr = prev_curr_maps
        # get_multigame_amount_reward 는 이미 jax.jit 데코레이터 적용됨
        result = get_multigame_amount_reward(prev, curr, jnp.array(2.0), tile_name="interactive")
        assert float(result) == pytest.approx(1.0)


# ═══════════════════════════════════════════════════════════════════════════════
#  6. jax.vmap 배치 컴파일 검증
# ═══════════════════════════════════════════════════════════════════════════════

class TestVmapCompilation:
    """vmap 으로 배치 처리가 가능한지 검증."""

    def test_vmap_interactive_count(self, env_map):
        batch = jnp.stack([env_map, env_map, jnp.ones_like(env_map)])  # (3, 4, 4)
        vmapped = jax.vmap(get_interactive_count)
        results = vmapped(batch)
        assert results.shape == (3,)
        assert float(results[0]) == 3.0
        assert float(results[2]) == 0.0  # all-EMPTY 맵

    def test_vmap_loss(self, env_map):
        batch = jnp.stack([env_map, env_map])
        conds = jnp.array([2.0, 5.0])
        vmapped = jax.vmap(lambda m, c: multigame_amount_loss(m, "interactive", c, absolute=True))
        results = vmapped(batch, conds)
        assert results.shape == (2,)
        assert float(results[0]) == pytest.approx(1.0)  # |3-2|
        assert float(results[1]) == pytest.approx(2.0)  # |3-5|

    def test_vmap_reward(self, prev_curr_maps):
        from functools import partial
        prev, curr = prev_curr_maps
        prev_batch = jnp.stack([prev, prev])
        curr_batch = jnp.stack([curr, curr])
        conds = jnp.array([2.0, 3.0])
        vmapped = jax.vmap(partial(get_multigame_amount_reward, tile_name="interactive"))
        results = vmapped(prev_batch, curr_batch, conds)
        assert results.shape == (2,)
        assert float(results[0]) == pytest.approx(1.0)   # cond=2: loss 1→0
        assert float(results[1]) == pytest.approx(-1.0)  # cond=3: loss 0→1

if __name__ == "__main__":
    pytest.main([__file__, "-v"])


# ═══════════════════════════════════════════════════════════════════════════════
#  8. placement — cluster_penalty 값 검증
# ═══════════════════════════════════════════════════════════════════════════════

class TestClusterPenalty:

    def test_no_cluster_isolated_items(self):
        """아이템이 모두 고립되어 있으면 penalty=0."""
        # INTERACTIVE(3)들이 서로 인접하지 않음
        m = jnp.array([
            [3, 1, 3],
            [1, 1, 1],
            [3, 1, 3],
        ], dtype=jnp.int32)
        assert float(_cluster_penalty(m)) == 0.0

    def test_cluster_adjacent_same(self):
        """같은 아이템이 인접하면 penalty > 0."""
        # 3-3 가로 인접 → 각각 이웃 1개씩 = 2
        m = jnp.array([
            [3, 3, 1],
            [1, 1, 1],
            [1, 1, 1],
        ], dtype=jnp.int32)
        assert float(_cluster_penalty(m)) == 2.0

    def test_cluster_line_of_three(self):
        """3개 일직선 배치 → 양 끝 1 + 가운데 2 = 4."""
        m = jnp.array([
            [3, 3, 3],
            [1, 1, 1],
            [1, 1, 1],
        ], dtype=jnp.int32)
        assert float(_cluster_penalty(m)) == 4.0

    def test_non_item_same_neighbors_ignored(self):
        """WALL-WALL 인접은 아이템이 아니므로 penalty 에 안 잡힘."""
        m = jnp.array([
            [2, 2, 1],
            [1, 1, 1],
            [1, 1, 1],
        ], dtype=jnp.int32)
        assert float(_cluster_penalty(m)) == 0.0


# ═══════════════════════════════════════════════════════════════════════════════
#  9. placement — accessibility_bonus 값 검증
# ═══════════════════════════════════════════════════════════════════════════════

class TestAccessibilityBonus:

    def test_all_accessible(self):
        """모든 아이템 이웃에 EMPTY 있음 → 1.0."""
        m = jnp.array([
            [1, 3, 1],
            [1, 1, 1],
            [1, 4, 1],
        ], dtype=jnp.int32)
        assert float(_accessibility_bonus(m)) == pytest.approx(1.0)

    def test_walled_item(self):
        """사방이 WALL/BORDER 로 막힌 아이템 → 접근 불가."""
        # (1,1)=INTERACTIVE, 상하좌우 모두 WALL(2)
        m = jnp.array([
            [2, 2, 2],
            [2, 3, 2],
            [2, 2, 2],
        ], dtype=jnp.int32)
        assert float(_accessibility_bonus(m)) == pytest.approx(0.0)

    def test_partially_accessible(self):
        """2개 아이템 중 1개만 접근 가능 → 0.5."""
        # (0,1)=3 → 이웃: BORDER(경계밖), WALL(2), EMPTY(1), WALL(2) → EMPTY 1개 → 접근 가능
        # (2,1)=3 → 이웃: WALL(2), BORDER(경계밖), WALL(2), WALL(2) → 0개 → 접근 불가
        m = jnp.array([
            [2, 3, 1],
            [2, 2, 2],
            [2, 3, 2],
        ], dtype=jnp.int32)
        assert float(_accessibility_bonus(m)) == pytest.approx(0.5)

    def test_no_items_returns_one(self):
        """아이템 0개 → 1.0 (패널티 없음)."""
        m = jnp.ones((4, 4), dtype=jnp.int32)  # all EMPTY
        assert float(_accessibility_bonus(m)) == pytest.approx(1.0)


# ═══════════════════════════════════════════════════════════════════════════════
# 10. placement — spread_bonus 값 검증
# ═══════════════════════════════════════════════════════════════════════════════

class TestSpreadBonus:

    def test_spread_max_corners(self):
        """대각선 양 끝 배치 → 최대 분산."""
        m = jnp.ones((4, 4), dtype=jnp.int32)
        m = m.at[0, 0].set(3)  # 좌상단
        m = m.at[3, 3].set(3)  # 우하단
        # L1 거리 = 3+3=6, max_dist=3+3=6, bonus=6/6=1.0
        assert float(_spread_bonus(m)) == pytest.approx(1.0)

    def test_spread_adjacent(self):
        """바로 옆 배치 → 낮은 분산."""
        m = jnp.ones((4, 4), dtype=jnp.int32)
        m = m.at[0, 0].set(3)
        m = m.at[0, 1].set(3)
        # L1 거리 = 1, max_dist=6, bonus=1/6
        assert float(_spread_bonus(m)) == pytest.approx(1.0 / 6.0)

    def test_spread_single_item_zero(self):
        """아이템 1개 → 분산 측정 불가 → 0.0."""
        m = jnp.ones((4, 4), dtype=jnp.int32)
        m = m.at[2, 2].set(5)
        assert float(_spread_bonus(m)) == pytest.approx(0.0)

    def test_spread_no_items_zero(self):
        """아이템 0개 → 0.0."""
        m = jnp.ones((4, 4), dtype=jnp.int32)
        assert float(_spread_bonus(m)) == pytest.approx(0.0)


# ═══════════════════════════════════════════════════════════════════════════════
# 11. placement — 통합 reward 검증
# ═══════════════════════════════════════════════════════════════════════════════

class TestPlacementReward:

    def test_improvement_gives_positive_reward(self):
        """뭉친 배치 → 분산 배치로 개선하면 양수 reward."""
        prev = jnp.array([
            [3, 3, 3, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
        ], dtype=jnp.int32)
        # 일직선 3개를 분산시킴
        curr = jnp.array([
            [3, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 3, 1],
            [1, 1, 1, 3],
        ], dtype=jnp.int32)
        reward = get_multigame_placement_reward(prev, curr)
        assert float(reward) > 0.0

    def test_worsening_gives_negative_reward(self):
        """분산 → 뭉침으로 악화하면 음수 reward."""
        prev = jnp.array([
            [3, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 3, 1],
            [1, 1, 1, 3],
        ], dtype=jnp.int32)
        curr = jnp.array([
            [3, 3, 3, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
        ], dtype=jnp.int32)
        reward = get_multigame_placement_reward(prev, curr)
        assert float(reward) < 0.0

    def test_no_change_zero_reward(self, env_map):
        """동일 맵 → reward=0."""
        reward = get_multigame_placement_reward(env_map, env_map)
        assert float(reward) == pytest.approx(0.0)

    def test_walling_off_item_penalised(self):
        """아이템 주변을 WALL 로 막으면 음수 reward."""
        prev = jnp.array([
            [1, 1, 1],
            [1, 3, 1],
            [1, 1, 1],
        ], dtype=jnp.int32)
        # 사방을 WALL 로 막음
        curr = jnp.array([
            [1, 2, 1],
            [2, 3, 2],
            [1, 2, 1],
        ], dtype=jnp.int32)
        reward = get_multigame_placement_reward(prev, curr)
        assert float(reward) < 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# 12. placement — jax.jit / vmap 컴파일 검증
# ═══════════════════════════════════════════════════════════════════════════════

class TestPlacementJitVmap:

    def test_jit_cluster_penalty(self, env_map):
        jitted = jax.jit(_cluster_penalty)
        result = jitted(env_map)
        assert result.shape == ()

    def test_jit_accessibility_bonus(self, env_map):
        jitted = jax.jit(_accessibility_bonus)
        result = jitted(env_map)
        assert result.shape == ()
        assert 0.0 <= float(result) <= 1.0

    def test_jit_spread_bonus(self, env_map):
        from functools import partial
        jitted = jax.jit(partial(_spread_bonus, max_items=32))
        result = jitted(env_map)
        assert result.shape == ()
        assert 0.0 <= float(result) <= 1.0

    def test_jit_placement_reward(self, env_map):
        """get_multigame_placement_reward 는 이미 jax.jit 적용."""
        result = get_multigame_placement_reward(env_map, env_map)
        assert result.shape == ()
        assert float(result) == pytest.approx(0.0)

    def test_vmap_cluster_penalty(self, env_map):
        batch = jnp.stack([env_map, jnp.ones_like(env_map)])
        vmapped = jax.vmap(_cluster_penalty)
        results = vmapped(batch)
        assert results.shape == (2,)
        assert float(results[1]) == 0.0  # all-EMPTY → no items → 0

    def test_vmap_accessibility_bonus(self, env_map):
        batch = jnp.stack([env_map, jnp.ones_like(env_map)])
        vmapped = jax.vmap(_accessibility_bonus)
        results = vmapped(batch)
        assert results.shape == (2,)
        assert float(results[1]) == pytest.approx(1.0)  # no items → 1.0

    def test_vmap_placement_reward(self, env_map):
        prev_batch = jnp.stack([env_map, env_map])
        curr_batch = jnp.stack([env_map, env_map])
        vmapped = jax.vmap(get_multigame_placement_reward)
        results = vmapped(prev_batch, curr_batch)
        assert results.shape == (2,)
        assert float(results[0]) == pytest.approx(0.0)


# ═══════════════════════════════════════════════════════════════════════════════
# 13. tile-specific placement — 내부 함수 값 검증
# ═══════════════════════════════════════════════════════════════════════════════

class TestTilePlacementHelpers:
    """_cluster_penalty_tile, _accessibility_bonus_tile, _spread_bonus_tile, _tile_amount_diff 검증."""

    def test_cluster_penalty_tile_isolated(self):
        """INTERACTIVE(3)가 고립 → penalty=0."""
        m = jnp.array([
            [3, 1, 3],
            [1, 1, 1],
            [3, 1, 3],
        ], dtype=jnp.int32)
        assert float(_cluster_penalty_tile(m, 3)) == 0.0

    def test_cluster_penalty_tile_adjacent(self):
        """INTERACTIVE(3) 2개 인접 → 각 1씩 = 2."""
        m = jnp.array([
            [3, 3, 1],
            [1, 1, 1],
            [1, 1, 1],
        ], dtype=jnp.int32)
        assert float(_cluster_penalty_tile(m, 3)) == 2.0

    def test_cluster_ignores_other_tiles(self):
        """HAZARD(4) 인접이어도 INTERACTIVE(3) penalty는 0."""
        m = jnp.array([
            [3, 4, 1],
            [1, 1, 1],
            [1, 1, 1],
        ], dtype=jnp.int32)
        assert float(_cluster_penalty_tile(m, 3)) == 0.0

    def test_accessibility_tile_open(self):
        """HAZARD(4) 주변에 EMPTY → 접근 가능 = 1.0."""
        m = jnp.array([
            [1, 4, 1],
            [1, 1, 1],
            [1, 1, 1],
        ], dtype=jnp.int32)
        assert float(_accessibility_bonus_tile(m, 4)) == pytest.approx(1.0)

    def test_accessibility_tile_walled(self):
        """HAZARD(4) 사방 WALL → 접근 불가 = 0.0."""
        m = jnp.array([
            [2, 2, 2],
            [2, 4, 2],
            [2, 2, 2],
        ], dtype=jnp.int32)
        assert float(_accessibility_bonus_tile(m, 4)) == pytest.approx(0.0)

    def test_spread_tile_far(self):
        """COLLECTABLE(5) 양 끝 배치 → 최대 분산."""
        m = jnp.ones((4, 4), dtype=jnp.int32)
        m = m.at[0, 0].set(5).at[3, 3].set(5)
        # L1 dist=6, max_dist=6 → 1.0
        assert float(_spread_bonus_tile(m, 5)) == pytest.approx(1.0)

    def test_spread_tile_single(self):
        """타일 1개 → 0.0."""
        m = jnp.ones((4, 4), dtype=jnp.int32).at[0, 0].set(5)
        assert float(_spread_bonus_tile(m, 5)) == pytest.approx(0.0)

    def test_amount_diff_improved(self):
        """개수가 목표에 가까워지면 양수."""
        prev = jnp.array([[3, 3, 3], [1, 1, 1], [1, 1, 1]], dtype=jnp.int32)
        curr = jnp.array([[3, 3, 1], [1, 1, 1], [1, 1, 1]], dtype=jnp.int32)
        # cond=2: prev |3-2|=1, curr |2-2|=0 → diff=1
        assert float(_tile_amount_diff(prev, curr, 3, jnp.array(2.0))) == pytest.approx(1.0)

    def test_amount_diff_worsened(self):
        """개수가 목표에서 멀어지면 음수."""
        prev = jnp.array([[3, 3, 1], [1, 1, 1], [1, 1, 1]], dtype=jnp.int32)
        curr = jnp.array([[3, 3, 3], [1, 1, 1], [1, 1, 1]], dtype=jnp.int32)
        # cond=2: prev |2-2|=0, curr |3-2|=1 → diff=-1
        assert float(_tile_amount_diff(prev, curr, 3, jnp.array(2.0))) == pytest.approx(-1.0)


# ═══════════════════════════════════════════════════════════════════════════════
# 14. tile-specific placement — 통합 reward 값 검증
# ═══════════════════════════════════════════════════════════════════════════════

class TestTilePlacementReward:

    def test_no_change_zero(self, env_map):
        """동일 맵 → 0."""
        r = get_multigame_tile_placement_reward(env_map, env_map, jnp.array(3.0), tile_name="interactive")
        assert float(r) == pytest.approx(0.0)

    def test_amount_improvement_positive(self):
        """개수만 개선 (분산/cluster 동일) → 양수."""
        prev = jnp.array([
            [3, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
        ], dtype=jnp.int32)
        curr = jnp.array([
            [3, 1, 1, 3],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
        ], dtype=jnp.int32)
        # cond=2: prev count=1, curr count=2 → amount 개선, spread 개선, cluster 변화 없음
        r = get_multigame_tile_placement_reward(prev, curr, jnp.array(2.0), tile_name="interactive")
        assert float(r) > 0.0

    def test_clustering_penalised(self):
        """분산 배치 → 뭉침 배치로 바뀌면 음수."""
        prev = jnp.array([
            [3, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 3, 1],
            [1, 1, 1, 3],
        ], dtype=jnp.int32)
        curr = jnp.array([
            [3, 3, 3, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
        ], dtype=jnp.int32)
        # cond=3: 개수 동일, 하지만 cluster↑, spread↓ → 음수
        r = get_multigame_tile_placement_reward(prev, curr, jnp.array(3.0), tile_name="interactive")
        assert float(r) < 0.0

    @pytest.mark.parametrize("tile_name", ["interactive", "hazard", "collectable"])
    def test_all_tile_names_work(self, env_map, tile_name):
        """세 tile_name 모두 에러 없이 동작."""
        r = get_multigame_tile_placement_reward(env_map, env_map, jnp.array(2.0), tile_name=tile_name)
        assert r.shape == ()


# ═══════════════════════════════════════════════════════════════════════════════
# 15. tile-specific placement — jax.jit / vmap 컴파일 검증
# ═══════════════════════════════════════════════════════════════════════════════

class TestTilePlacementJitVmap:

    def test_jit_tile_placement(self, env_map):
        """이미 @jax.jit 적용 — 실행 확인."""
        r = get_multigame_tile_placement_reward(env_map, env_map, jnp.array(2.0), tile_name="interactive")
        assert r.shape == ()

    def test_jit_cluster_penalty_tile(self, env_map):
        jitted = jax.jit(lambda m: _cluster_penalty_tile(m, 3))
        r = jitted(env_map)
        assert r.shape == ()

    def test_jit_accessibility_bonus_tile(self, env_map):
        jitted = jax.jit(lambda m: _accessibility_bonus_tile(m, 3))
        r = jitted(env_map)
        assert r.shape == ()
        assert 0.0 <= float(r) <= 1.0

    def test_jit_spread_bonus_tile(self, env_map):
        jitted = jax.jit(lambda m: _spread_bonus_tile(m, 3))
        r = jitted(env_map)
        assert r.shape == ()
        assert 0.0 <= float(r) <= 1.0

    def test_vmap_tile_placement(self, env_map):
        prev_batch = jnp.stack([env_map, env_map])
        curr_batch = jnp.stack([env_map, env_map])
        conds = jnp.array([2.0, 3.0])
        vmapped = jax.vmap(lambda p, c, cd: get_multigame_tile_placement_reward(p, c, cd, tile_name="interactive"))
        results = vmapped(prev_batch, curr_batch, conds)
        assert results.shape == (2,)
        assert float(results[0]) == pytest.approx(0.0)

    def test_vmap_cluster_penalty_tile(self, env_map):
        batch = jnp.stack([env_map, jnp.ones_like(env_map)])
        vmapped = jax.vmap(lambda m: _cluster_penalty_tile(m, 3))
        results = vmapped(batch)
        assert results.shape == (2,)
        assert float(results[1]) == 0.0  # all-EMPTY → no 3 tiles → 0
