"""tests/test_multigame_rl.py

MultigameProblem + PCGRLEnv 에 대한 JAX PPO 스타일 통합 테스트.

목적
----
1. make_multigame_env() 로 환경 생성 → action_space / observation_space 검증
2. JAX vmap reset / step 이 정상 동작하는지 확인
3. 짧은 PPO 롤아웃(miniPPO) 을 실제 실행하여 loss 가 수렴 방향으로 변하는지 확인
4. action_space.n == NUM_CATEGORIES(7) 임을 명시적으로 검증

실행
----
    python -m pytest tests/test_multigame_rl.py -v
    # 또는
    python tests/test_multigame_rl.py
"""

from __future__ import annotations

import sys
import os

# ── 프로젝트 루트를 PYTHONPATH 에 추가 ────────────────────────────────────────
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import math
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
import flax.linen as nn
from flax.training.train_state import TrainState

import pytest

# ── 프로젝트 내부 임포트 ───────────────────────────────────────────────────────
from envs.pcgrl_env import (
    PCGRLEnv,
    PCGRLEnvParams,
    PCGRLObs,
    gen_dummy_queued_state,
)
from envs.probs.multigame import (
    NUM_CATEGORIES,
    MultigameTiles,
    make_multigame_env,
)
from purejaxrl.wrappers import LogWrapper


# ═══════════════════════════════════════════════════════════════════════════════
# 헬퍼: 경량 actor-critic 네트워크 (train.py 의 ConvForward2 를 단순화)
# ═══════════════════════════════════════════════════════════════════════════════

class _MiniConv(nn.Module):
    """가벼운 conv→dense actor-critic (테스트 전용)."""
    action_dim: int
    hidden: int = 64

    @nn.compact
    def __call__(self, map_x: jnp.ndarray, flat_x: jnp.ndarray):
        x = map_x.reshape((map_x.shape[0], -1))
        x = jnp.concatenate([x, flat_x], axis=-1)
        x = nn.Dense(self.hidden, kernel_init=orthogonal(np.sqrt(2)),
                     bias_init=constant(0.0))(x)
        x = nn.relu(x)
        act = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01),
                       bias_init=constant(0.0))(x)
        val = nn.Dense(1, kernel_init=orthogonal(1.0),
                       bias_init=constant(0.0))(x)
        return act, jnp.squeeze(val, axis=-1)


class _MiniActorCritic(nn.Module):
    """MultigameEnv 전용 thin wrapper — act_shape=(1,1), n_agents=1."""
    subnet: nn.Module
    n_editable_tiles: int          # == NUM_CATEGORIES

    @nn.compact
    def __call__(self, x: PCGRLObs):
        act, val = self.subnet(x.map_obs, x.flat_obs)
        # reshape → (batch, n_agents=1, act_h=1, act_w=1, n_tiles)
        act = act.reshape((act.shape[0], 1, 1, 1, self.n_editable_tiles))
        try:
            import distrax
            pi = distrax.Categorical(logits=act)
        except ImportError:
            raise ImportError("distrax 가 설치되어 있어야 합니다: pip install distrax")
        return pi, val


class _Transition(NamedTuple):
    obs: PCGRLObs
    action: jnp.ndarray
    reward: jnp.ndarray
    done: jnp.ndarray
    log_prob: jnp.ndarray
    value: jnp.ndarray


# ═══════════════════════════════════════════════════════════════════════════════
# Fixture
# ═══════════════════════════════════════════════════════════════════════════════

N_ENVS   = 4    # 병렬 환경 수
N_STEPS  = 16   # 롤아웃 길이
N_EPOCHS = 2    # PPO 미니배치 업데이트 횟수
LR       = 3e-4
GAMMA    = 0.99
GAE_LAMBDA = 0.95
CLIP_EPS   = 0.2
ENT_COEF   = 0.01
VF_COEF    = 0.5
MAX_GRAD_NORM = 0.5


@pytest.fixture(scope="module")
def multigame_setup():
    """환경·파라미터·더미 QueuedState·네트워크를 한 번만 생성."""
    env, env_params = make_multigame_env(
        representation="narrow",
        map_shape=(16, 16),
        rf_shape=(31, 31),
        act_shape=(1, 1),
    )
    env = LogWrapper(env)
    env.init_graphics()

    n_tiles = NUM_CATEGORIES           # 7
    action_dim = n_tiles               # narrow rep: n_editable_tiles

    subnet = _MiniConv(action_dim=action_dim)
    network = _MiniActorCritic(subnet=subnet, n_editable_tiles=n_tiles)

    rng = jax.random.PRNGKey(42)
    dummy_qs = gen_dummy_queued_state(env._env)

    # dummy obs 로 파라미터 초기화
    rng, _rng = jax.random.split(rng)
    reset_rng = jax.random.split(_rng, N_ENVS)
    vmap_reset = jax.vmap(env.reset, in_axes=(0, None, None))
    init_obs, _ = vmap_reset(reset_rng, env_params, dummy_qs)

    # map_obs: (N_ENVS, H, W, C) → (1, H, W, C) 슬라이스
    dummy_map  = init_obs.map_obs[:1]
    dummy_flat = init_obs.flat_obs[:1]
    dummy_obs  = PCGRLObs(
        map_obs=dummy_map,
        past_map_obs=dummy_map,
        flat_obs=dummy_flat,
        nlp_obs=jnp.zeros((1, 1)),
        input_ids=None,
        attention_mask=None,
        pixel_values=None,
    )
    network_params = network.init(_rng, dummy_obs)

    tx = optax.chain(
        optax.clip_by_global_norm(MAX_GRAD_NORM),
        optax.adam(LR, eps=1e-5),
    )
    train_state = TrainState.create(
        apply_fn=network.apply,
        params=network_params,
        tx=tx,
    )

    return dict(
        env=env,
        env_params=env_params,
        network=network,
        train_state=train_state,
        dummy_qs=dummy_qs,
        rng=rng,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# ① action_space 검증
# ═══════════════════════════════════════════════════════════════════════════════

def test_action_space_equals_num_categories(multigame_setup):
    """action_space.n 이 NUM_CATEGORIES(7) 와 같아야 한다."""
    env_raw: PCGRLEnv = multigame_setup["env"]._env
    env_params: PCGRLEnvParams = multigame_setup["env_params"]

    n = env_raw.action_space(env_params).n
    print(f"\n[action_space] action_space.n = {n}, NUM_CATEGORIES = {NUM_CATEGORIES}")

    assert n == NUM_CATEGORIES, (
        f"action_space.n({n}) != NUM_CATEGORIES({NUM_CATEGORIES})"
    )


def test_n_editable_tiles_equals_num_categories(multigame_setup):
    """rep.n_editable_tiles 가 NUM_CATEGORIES(7) 와 같아야 한다."""
    env_raw: PCGRLEnv = multigame_setup["env"]._env
    n = env_raw.rep.n_editable_tiles
    print(f"\n[n_editable_tiles] rep.n_editable_tiles = {n}, NUM_CATEGORIES = {NUM_CATEGORIES}")
    assert n == NUM_CATEGORIES, (
        f"rep.n_editable_tiles({n}) != NUM_CATEGORIES({NUM_CATEGORIES})"
    )


def test_multigame_tile_enum_length(multigame_setup):
    """MultigameTiles 의 크기 = NUM_CATEGORIES + 1(BORDER)."""
    env_raw: PCGRLEnv = multigame_setup["env"]._env
    n_tiles = len(env_raw.tile_enum)
    expected = NUM_CATEGORIES + 1   # BORDER 포함
    print(f"\n[tile_enum] len(tile_enum) = {n_tiles}, expected = {expected}")
    assert n_tiles == expected, f"tile_enum size {n_tiles} != {expected}"


def test_action_shape_output(multigame_setup):
    """env.action_shape() 이 (n_agents=1, act_h=1, act_w=1, n_tiles=7) 이어야 한다."""
    env_raw: PCGRLEnv = multigame_setup["env"]._env
    shape = env_raw.action_shape()
    print(f"\n[action_shape] env.action_shape() = {shape}")
    assert shape == (1, 1, 1, NUM_CATEGORIES), (
        f"action_shape {shape} != (1, 1, 1, {NUM_CATEGORIES})"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# ② vmap reset / step 검증
# ═══════════════════════════════════════════════════════════════════════════════

def test_vmap_reset(multigame_setup):
    """vmap reset 이 올바른 obs shape 을 반환해야 한다."""
    env      = multigame_setup["env"]
    env_params = multigame_setup["env_params"]
    dummy_qs = multigame_setup["dummy_qs"]
    rng      = multigame_setup["rng"]

    rng, _rng = jax.random.split(rng)
    reset_rng = jax.random.split(_rng, N_ENVS)
    vmap_reset = jax.vmap(env.reset, in_axes=(0, None, None))
    obs, state = vmap_reset(reset_rng, env_params, dummy_qs)

    # map_obs shape = (N_ENVS, rf_H, rf_W, n_tile_types+1)
    expected_channels = len(MultigameTiles) + 1   # +1 for static-tile channel
    print(f"\n[vmap_reset] obs.map_obs.shape = {obs.map_obs.shape}")
    print(f"             expected channels  = {expected_channels}")

    assert obs.map_obs.shape[0] == N_ENVS
    assert obs.map_obs.shape[-1] == expected_channels, (
        f"channel mismatch: {obs.map_obs.shape[-1]} != {expected_channels}"
    )


def test_vmap_step(multigame_setup):
    """vmap step 이 reward, done 를 올바른 shape 으로 반환해야 한다."""
    env      = multigame_setup["env"]
    env_params = multigame_setup["env_params"]
    dummy_qs = multigame_setup["dummy_qs"]
    rng      = multigame_setup["rng"]

    rng, _rng = jax.random.split(rng)
    reset_rng = jax.random.split(_rng, N_ENVS)
    vmap_reset = jax.vmap(env.reset, in_axes=(0, None, None))
    _, env_state = vmap_reset(reset_rng, env_params, dummy_qs)

    # 랜덤 action: shape (N_ENVS, n_agents=1, act_h=1, act_w=1)
    rng, _rng = jax.random.split(rng)
    action = jax.random.randint(_rng, (N_ENVS, 1, 1, 1), 0, NUM_CATEGORIES)

    rng, _rng = jax.random.split(rng)
    step_rng = jax.random.split(_rng, N_ENVS)
    vmap_step = jax.vmap(env.step, in_axes=(0, 0, 0, None))
    obs, next_state, reward, done, info = vmap_step(step_rng, env_state, action, env_params)

    print(f"\n[vmap_step] reward.shape={reward.shape}, done.shape={done.shape}")
    print(f"            action.shape={action.shape}")

    assert reward.shape == (N_ENVS,), f"reward shape {reward.shape} != ({N_ENVS},)"
    assert done.shape   == (N_ENVS,), f"done shape {done.shape} != ({N_ENVS},)"


# ═══════════════════════════════════════════════════════════════════════════════
# ③ 네트워크 forward pass 검증
# ═══════════════════════════════════════════════════════════════════════════════

def test_network_forward(multigame_setup):
    """network.apply 가 올바른 pi, value 를 반환해야 한다."""
    env        = multigame_setup["env"]
    env_params = multigame_setup["env_params"]
    dummy_qs   = multigame_setup["dummy_qs"]
    network    = multigame_setup["network"]
    train_state = multigame_setup["train_state"]
    rng        = multigame_setup["rng"]

    rng, _rng = jax.random.split(rng)
    reset_rng = jax.random.split(_rng, N_ENVS)
    vmap_reset = jax.vmap(env.reset, in_axes=(0, None, None))
    obs, _ = vmap_reset(reset_rng, env_params, dummy_qs)

    pi, val = network.apply(train_state.params, obs)

    action = pi.sample(seed=_rng)
    log_prob = pi.log_prob(action)

    print(f"\n[network] pi.logits.shape = {pi.logits.shape}")
    print(f"          val.shape        = {val.shape}")
    print(f"          action.shape     = {action.shape}")
    print(f"          log_prob.shape   = {log_prob.shape}")

    # logits shape: (N_ENVS, n_agents=1, act_h=1, act_w=1, n_tiles=7)
    assert pi.logits.shape == (N_ENVS, 1, 1, 1, NUM_CATEGORIES), (
        f"pi.logits.shape {pi.logits.shape} != ({N_ENVS}, 1, 1, 1, {NUM_CATEGORIES})"
    )
    assert val.shape == (N_ENVS,), f"val.shape {val.shape} != ({N_ENVS},)"
    # action shape: (N_ENVS, n_agents=1, act_h=1, act_w=1)
    assert action.shape == (N_ENVS, 1, 1, 1), (
        f"action.shape {action.shape} != ({N_ENVS}, 1, 1, 1)"
    )
    # action 값은 [0, NUM_CATEGORIES) 범위
    assert int(action.min()) >= 0 and int(action.max()) < NUM_CATEGORIES, (
        f"action 범위 초과: min={int(action.min())}, max={int(action.max())}"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# ④ 미니 PPO 롤아웃 + 업데이트 (실제 JAX scan 루프)
# ═══════════════════════════════════════════════════════════════════════════════

def _collect_rollout(env, env_params, network, train_state, dummy_qs, rng, n_envs, n_steps):
    """N_STEPS 동안 rollout 수집 (python loop, 테스트 전용)."""
    rng, _rng = jax.random.split(rng)
    reset_rng = jax.random.split(_rng, n_envs)
    vmap_reset = jax.vmap(env.reset, in_axes=(0, None, None))
    obs, env_state = vmap_reset(reset_rng, env_params, dummy_qs)

    transitions = []
    for _ in range(n_steps):
        rng, _rng = jax.random.split(rng)
        pi, val = network.apply(train_state.params, obs)
        action   = pi.sample(seed=_rng)
        log_prob = pi.log_prob(action)

        rng, _rng = jax.random.split(rng)
        step_rng = jax.random.split(_rng, n_envs)
        vmap_step = jax.vmap(env.step, in_axes=(0, 0, 0, None))
        next_obs, env_state, reward, done, _ = vmap_step(
            step_rng, env_state, action, env_params
        )
        transitions.append(_Transition(
            obs=obs, action=action, reward=reward,
            done=done, log_prob=log_prob, value=val,
        ))
        obs = next_obs

    return transitions, obs, env_state


def _compute_gae(transitions, last_val, gamma=GAMMA, gae_lambda=GAE_LAMBDA):
    """Generalized Advantage Estimation."""
    advantages = []
    gae = jnp.zeros_like(transitions[-1].reward)
    for t in reversed(transitions):
        delta = t.reward + gamma * last_val * (1.0 - t.done) - t.value
        gae   = delta + gamma * gae_lambda * (1.0 - t.done) * gae
        advantages.insert(0, gae)
        last_val = t.value
    advantages = jnp.stack(advantages, axis=0)   # (n_steps, n_envs)
    returns    = advantages + jnp.stack([t.value for t in transitions], axis=0)
    return advantages, returns


def _ppo_loss(params, network, obs_batch, act_batch, old_log_prob_batch,
              adv_batch, ret_batch):
    """단일 미니배치에 대한 PPO loss."""
    pi, val = network.apply(params, obs_batch)
    log_prob = pi.log_prob(act_batch)
    entropy  = pi.entropy().mean()

    ratio    = jnp.exp(log_prob - old_log_prob_batch)
    adv_norm = (adv_batch - adv_batch.mean()) / (adv_batch.std() + 1e-8)

    pg_loss1 = -adv_norm * ratio
    pg_loss2 = -adv_norm * jnp.clip(ratio, 1.0 - CLIP_EPS, 1.0 + CLIP_EPS)
    actor_loss = jnp.maximum(pg_loss1, pg_loss2).mean()

    val_loss = jnp.mean((val - ret_batch) ** 2)

    total_loss = actor_loss + VF_COEF * val_loss - ENT_COEF * entropy
    return total_loss, (actor_loss, val_loss, entropy)


def _stack_obs(transitions):
    """transition 리스트의 obs 를 배치 차원으로 concat."""
    def _cat(attr):
        arrs = [getattr(t.obs, attr) for t in transitions
                if getattr(t.obs, attr) is not None]
        return jnp.concatenate(arrs, axis=0) if arrs else None

    return PCGRLObs(
        map_obs      =_cat("map_obs"),
        past_map_obs =_cat("past_map_obs"),
        flat_obs     =_cat("flat_obs"),
        nlp_obs      =_cat("nlp_obs"),
        input_ids    =None,
        attention_mask=None,
        pixel_values  =None,
    )


def test_mini_ppo_rollout_and_update(multigame_setup):
    """
    짧은 PPO 롤아웃 + 1-step 업데이트가 실제로 돌아가는지 확인.
    loss 가 NaN/Inf 가 아니면 통과.
    """
    env         = multigame_setup["env"]
    env_params  = multigame_setup["env_params"]
    network     = multigame_setup["network"]
    train_state = multigame_setup["train_state"]
    dummy_qs    = multigame_setup["dummy_qs"]
    rng         = multigame_setup["rng"]

    # ── 롤아웃 수집 ────────────────────────────────────────────────────────────
    transitions, last_obs, _ = _collect_rollout(
        env, env_params, network, train_state, dummy_qs, rng, N_ENVS, N_STEPS
    )

    _, last_val = network.apply(train_state.params, last_obs)
    advantages, returns = _compute_gae(transitions, last_val)

    # (n_steps * n_envs,) 로 flatten
    flat_adv = advantages.reshape(-1)
    flat_ret = returns.reshape(-1)
    flat_act = jnp.concatenate(
        [t.action.reshape(N_ENVS, -1) for t in transitions], axis=0
    ).reshape(N_STEPS * N_ENVS, 1, 1, 1)  # action shape 유지

    flat_log_prob = jnp.concatenate(
        [t.log_prob.reshape(N_ENVS, -1) for t in transitions], axis=0
    ).reshape(N_STEPS * N_ENVS, 1, 1, 1)

    flat_obs = _stack_obs(transitions)  # (n_steps * n_envs, ...)

    # ── PPO 업데이트 ────────────────────────────────────────────────────────────
    grad_fn = jax.value_and_grad(_ppo_loss, argnums=0, has_aux=True)
    (total_loss, (actor_loss, val_loss, entropy)), grads = grad_fn(
        train_state.params, network,
        flat_obs, flat_act, flat_log_prob, flat_adv, flat_ret,
    )

    new_train_state = train_state.apply_gradients(grads=grads)

    total_loss_val = float(total_loss)
    actor_loss_val = float(actor_loss)
    val_loss_val   = float(val_loss)
    entropy_val    = float(entropy)

    print(f"\n[mini-PPO] total_loss={total_loss_val:.4f}  "
          f"actor_loss={actor_loss_val:.4f}  "
          f"val_loss={val_loss_val:.4f}  "
          f"entropy={entropy_val:.4f}")

    assert not math.isnan(total_loss_val), "total_loss 가 NaN"
    assert not math.isinf(total_loss_val), "total_loss 가 Inf"
    assert not math.isnan(val_loss_val),   "val_loss 가 NaN"
    assert entropy_val > 0.0, "entropy 가 0 이하 — 정책이 collapse 된 것 같습니다"

    # 파라미터가 실제로 업데이트 됐는지 확인
    old_p = jax.tree_util.tree_leaves(train_state.params)[0]
    new_p = jax.tree_util.tree_leaves(new_train_state.params)[0]
    assert not jnp.array_equal(old_p, new_p), "파라미터가 업데이트 되지 않았습니다"


# ═══════════════════════════════════════════════════════════════════════════════
# ⑤ 여러 스텝에 걸친 PPO 반복 — loss 추이 출력
# ═══════════════════════════════════════════════════════════════════════════════

def test_ppo_multi_update_no_nan(multigame_setup):
    """
    N_EPOCHS 번의 PPO 업데이트 동안 loss 가 NaN/Inf 없이 진행되는지 확인.
    (수렴 여부는 검사하지 않음 — steps 가 너무 짧으므로)
    """
    env         = multigame_setup["env"]
    env_params  = multigame_setup["env_params"]
    network     = multigame_setup["network"]
    train_state = multigame_setup["train_state"]
    dummy_qs    = multigame_setup["dummy_qs"]
    rng         = multigame_setup["rng"]

    losses = []
    for epoch_i in range(N_EPOCHS):
        rng, _rng = jax.random.split(rng)
        transitions, last_obs, _ = _collect_rollout(
            env, env_params, network, train_state, dummy_qs, _rng, N_ENVS, N_STEPS
        )
        _, last_val = network.apply(train_state.params, last_obs)
        advantages, returns = _compute_gae(transitions, last_val)

        flat_adv      = advantages.reshape(-1)
        flat_ret      = returns.reshape(-1)
        flat_act      = jnp.concatenate(
            [t.action.reshape(N_ENVS, -1) for t in transitions], axis=0
        ).reshape(N_STEPS * N_ENVS, 1, 1, 1)
        flat_log_prob = jnp.concatenate(
            [t.log_prob.reshape(N_ENVS, -1) for t in transitions], axis=0
        ).reshape(N_STEPS * N_ENVS, 1, 1, 1)
        flat_obs = _stack_obs(transitions)

        grad_fn = jax.value_and_grad(_ppo_loss, argnums=0, has_aux=True)
        (total_loss, _), grads = grad_fn(
            train_state.params, network,
            flat_obs, flat_act, flat_log_prob, flat_adv, flat_ret,
        )
        train_state = train_state.apply_gradients(grads=grads)
        losses.append(float(total_loss))
        print(f"  epoch {epoch_i+1}/{N_EPOCHS}  loss={float(total_loss):.4f}")

    print(f"\n[multi-update] losses = {[f'{l:.4f}' for l in losses]}")
    for i, loss in enumerate(losses):
        assert not math.isnan(loss), f"epoch {i} loss 가 NaN"
        assert not math.isinf(loss), f"epoch {i} loss 가 Inf"


# ═══════════════════════════════════════════════════════════════════════════════
# 단독 실행 시 간단 summary 출력
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print(" MultigameEnv JAX RL Integration Test")
    print("=" * 70)

    env_raw, env_params = make_multigame_env()
    env = LogWrapper(env_raw)
    env.init_graphics()

    print(f"\nEnvironment Info")
    print(f"  NUM_CATEGORIES        : {NUM_CATEGORIES}")
    print(f"  action_space.n        : {env_raw.action_space(env_params).n}")
    print(f"  rep.n_editable_tiles  : {env_raw.rep.n_editable_tiles}")
    print(f"  action_shape()        : {env_raw.action_shape()}")
    print(f"  tile_enum             : {list(env_raw.tile_enum)}")
    print(f"  map_shape             : {env_params.map_shape}")
    print(f"  rf_shape              : {env_params.rf_shape}")

    obs_space = env_raw.observation_space(env_params)
    print(f"  observation_space     : {obs_space.shape}")

    # 환경 단일 reset/step (raw env 직접 사용)
    rng = jax.random.PRNGKey(0)
    dummy_qs = gen_dummy_queued_state(env_raw)
    rng, _rng = jax.random.split(rng)
    obs, state = env_raw.reset(_rng, env_params, dummy_qs)
    print(f"\n  reset obs.map_obs.shape : {obs.map_obs.shape}")
    print(f"  reset obs.flat_obs.shape: {obs.flat_obs.shape}")

    # 랜덤 step
    rng, _rng = jax.random.split(rng)
    action = jax.random.randint(_rng, (1, 1, 1), 0, NUM_CATEGORIES)
    rng, _rng = jax.random.split(rng)
    obs2, state2, reward, done, info = env_raw.step(_rng, state, action, env_params)
    print(f"  step reward={float(reward):.4f}  done={bool(done)}")
    print(f"  step action={int(action.flatten()[0])}  "
          f"(valid range: 0–{NUM_CATEGORIES-1})")

    print("\n✅ All basic checks passed")
    print("  Run full tests with: pytest tests/test_multigame_rl.py -v")

