"""tests/test_multigame_evaluator.py

evaluator/measures, losses, fitnesses, rewards 의 multigame_amount 테스트.

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

