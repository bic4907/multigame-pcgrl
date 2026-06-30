"""
encoder/few_shot_adapter.py
===========================
Few-shot adapters for the CLIP → RewardDecoder pipeline.

세 가지 adapter type:
  film : 전역 affine 변환 (γ, β) 을 support set으로 gradient 최적화
  lora : 저랭크 잔차 행렬 (A @ B) 을 support set으로 gradient 최적화
  moe  : Mixture-of-Experts 잔차 변환을 support set으로 gradient 최적화

모두 frozen RewardDecoder의 입력 embedding space에서 동작한다.
frozen decoder는 수정하지 않는다.
"""

from __future__ import annotations

import logging
import os
import pickle
from typing import Callable, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn
from flax.training.train_state import TrainState

logger = logging.getLogger(__name__)

# W&B 로깅 step offset (adapter 학습용 독립 카운터)
_adapter_log_step = 0


def _wandb_log(metrics: dict, step: int) -> None:
    """wandb가 초기화된 경우에만 로깅한다."""
    try:
        import wandb
        if wandb.run is not None:
            wandb.log(metrics, step=step)
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────────
#  Shared: frozen decoder 호출 + support loss + support accuracy
# ─────────────────────────────────────────────────────────────────────────────

def _decoder_forward(
    embeddings: jnp.ndarray,
    apply_fn: Callable,
    variables: dict,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """frozen decoder → (reward_logits (B, C), condition_pred_raw (B, C))."""
    reward_logits, _cond_norm, condition_pred_raw = apply_fn(
        variables,
        embeddings,
        training=False,
        method=lambda m, e, training=False: m.decoder(e, training=training),
    )
    return reward_logits, condition_pred_raw


def _support_loss(
    adapted: jnp.ndarray,
    support_reward_i: jnp.ndarray,   # (K,) int 0-based
    support_condition: jnp.ndarray,  # (K, num_classes)
    apply_fn: Callable,
    variables: dict,
) -> jnp.ndarray:
    """CE (reward_enum) + 0.01 * MSE (condition) on support set."""
    reward_logits, cond_raw = _decoder_forward(adapted, apply_fn, variables)

    ce = optax.softmax_cross_entropy_with_integer_labels(
        reward_logits, support_reward_i
    ).mean()

    K = support_reward_i.shape[0]
    target_cond = support_condition[jnp.arange(K), support_reward_i]
    pred_cond   = cond_raw[jnp.arange(K), support_reward_i]
    mse = jnp.mean((pred_cond - target_cond) ** 2)

    return ce + 0.01 * mse


def _support_accuracy(
    adapted: jnp.ndarray,
    support_reward_i: jnp.ndarray,
    apply_fn: Callable,
    variables: dict,
) -> float:
    """support set에서 reward_enum 분류 정확도."""
    reward_logits, _ = _decoder_forward(adapted, apply_fn, variables)
    pred = jnp.argmax(reward_logits, axis=-1)
    return float(jnp.mean(pred == support_reward_i))


# ─────────────────────────────────────────────────────────────────────────────
#  공통 학습 루프
# ─────────────────────────────────────────────────────────────────────────────

def save_adapter_state(params, save_path: str) -> None:
    """adapter params를 pickle로 저장한다."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cpu_params = jax.tree_util.tree_map(np.asarray, params)
    with open(save_path, "wb") as f:
        pickle.dump(cpu_params, f)
    logger.info("Adapter state saved → %s", save_path)


def load_adapter_state(load_path: str) -> Optional[dict]:
    """저장된 adapter params를 로드한다. 파일 없으면 None 반환."""
    if not os.path.exists(load_path):
        return None
    with open(load_path, "rb") as f:
        params = pickle.load(f)
    logger.info("Adapter state loaded ← %s", load_path)
    return jax.tree_util.tree_map(jnp.array, params)


def _run_adapter(
    module: nn.Module,
    params,
    support: jnp.ndarray,
    support_reward_i: jnp.ndarray,
    support_condition: jnp.ndarray,
    apply_fn: Callable,
    variables: dict,
    lr: float,
    steps: int,
    tag: str,
    save_path: Optional[str] = None,
) -> TrainState:
    """공통 gradient 최적화 루프. loss 곡선과 최종 accuracy를 W&B에 기록한다."""
    global _adapter_log_step

    state = TrainState.create(apply_fn=module.apply, params=params, tx=optax.adam(lr))
    log_freq = max(1, steps // 100)  # 최대 100개 데이터포인트

    @jax.jit
    def step_fn(state):
        def loss_fn(p):
            return _support_loss(
                state.apply_fn(p, support),
                support_reward_i, support_condition, apply_fn, variables,
            )
        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        return state.apply_gradients(grads=grads), loss

    for i in range(steps):
        state, loss = step_fn(state)

        if i % log_freq == 0 or i == steps - 1:
            loss_val = float(loss)
            adapted_i = module.apply(state.params, support)
            acc_i = _support_accuracy(adapted_i, support_reward_i, apply_fn, variables)
            logger.debug("[%s] step %d/%d  loss=%.5f  acc=%.3f", tag, i, steps, loss_val, acc_i)
            _wandb_log({
                f"adapter/{tag}/loss":     loss_val,
                f"adapter/{tag}/accuracy": acc_i,
                f"adapter/{tag}/step":     i,
            }, step=_adapter_log_step)
            _adapter_log_step += 1

    # 최종 support accuracy
    adapted_support = module.apply(state.params, support)
    acc = _support_accuracy(adapted_support, support_reward_i, apply_fn, variables)
    final_loss = float(loss)

    logger.info(
        "[%s] done — final_loss=%.5f  support_accuracy=%.3f  (steps=%d, K=%d)",
        tag, final_loss, acc, steps, support.shape[0],
    )
    _wandb_log({
        f"adapter/{tag}/final_loss":         final_loss,
        f"adapter/{tag}/support_accuracy":   acc,
        f"adapter/{tag}/support_size":       support.shape[0],
        f"adapter/{tag}/steps":              steps,
    }, step=_adapter_log_step)

    if save_path is not None:
        save_adapter_state(state.params, save_path)

    return state


# ─────────────────────────────────────────────────────────────────────────────
#  FiLM: γ * x + β  (파라미터 2D = 128)
# ─────────────────────────────────────────────────────────────────────────────

class _FiLMModule(nn.Module):
    embed_dim: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        gamma = self.param("gamma", nn.initializers.ones,  (self.embed_dim,))
        beta  = self.param("beta",  nn.initializers.zeros, (self.embed_dim,))
        return gamma * x + beta


def film_adapt(
    query: jnp.ndarray,
    support: jnp.ndarray,
    support_reward_i: jnp.ndarray,
    support_condition: jnp.ndarray,
    apply_fn: Callable,
    variables: dict,
    lr: float = 1e-2,
    steps: int = 500,
    save_path: Optional[str] = None,
    load_path: Optional[str] = None,
) -> jnp.ndarray:
    D = query.shape[-1]
    module = _FiLMModule(embed_dim=D)
    saved = load_adapter_state(load_path) if load_path else None
    if saved is not None:
        return module.apply(saved, query)
    params = module.init(jax.random.PRNGKey(0), support)
    state = _run_adapter(
        module, params, support, support_reward_i, support_condition,
        apply_fn, variables, lr=lr, steps=steps, tag="film", save_path=save_path,
    )
    return module.apply(state.params, query)


# ─────────────────────────────────────────────────────────────────────────────
#  LoRA: x + x @ A @ B  (파라미터 2·D·rank = 512)
# ─────────────────────────────────────────────────────────────────────────────

class _LoRAModule(nn.Module):
    embed_dim: int
    rank: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        A = self.param("A", nn.initializers.normal(0.02), (self.embed_dim, self.rank))
        B = self.param("B", nn.initializers.zeros,        (self.rank, self.embed_dim))
        return x + x @ A @ B


def lora_adapt(
    query: jnp.ndarray,
    support: jnp.ndarray,
    support_reward_i: jnp.ndarray,
    support_condition: jnp.ndarray,
    apply_fn: Callable,
    variables: dict,
    rank: int = 4,
    lr: float = 1e-3,
    steps: int = 500,
    save_path: Optional[str] = None,
    load_path: Optional[str] = None,
) -> jnp.ndarray:
    D = query.shape[-1]
    module = _LoRAModule(embed_dim=D, rank=rank)
    saved = load_adapter_state(load_path) if load_path else None
    if saved is not None:
        return module.apply(saved, query)
    params = module.init(jax.random.PRNGKey(0), support)
    state = _run_adapter(
        module, params, support, support_reward_i, support_condition,
        apply_fn, variables, lr=lr, steps=steps, tag="lora", save_path=save_path,
    )
    return module.apply(state.params, query)


# ─────────────────────────────────────────────────────────────────────────────
#  MoE: x + Σ_k gate_k · expert_k(x)  (파라미터 ≈ K·D²)
# ─────────────────────────────────────────────────────────────────────────────

class _MoEModule(nn.Module):
    embed_dim: int
    num_experts: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        gates = jax.nn.softmax(nn.Dense(self.num_experts)(x), axis=-1)  # (N, K)
        expert_outs = jnp.stack(
            [nn.Dense(self.embed_dim, name=f"expert_{k}")(x)
             for k in range(self.num_experts)],
            axis=1,
        )  # (N, K, D)
        return x + jnp.einsum("nk,nkd->nd", gates, expert_outs)


def moe_adapt(
    query: jnp.ndarray,
    support: jnp.ndarray,
    support_reward_i: jnp.ndarray,
    support_condition: jnp.ndarray,
    apply_fn: Callable,
    variables: dict,
    num_experts: int = 4,
    lr: float = 1e-3,
    steps: int = 500,
    save_path: Optional[str] = None,
    load_path: Optional[str] = None,
) -> jnp.ndarray:
    D = query.shape[-1]
    module = _MoEModule(embed_dim=D, num_experts=num_experts)
    saved = load_adapter_state(load_path) if load_path else None
    if saved is not None:
        return module.apply(saved, query)
    params = module.init(jax.random.PRNGKey(0), support)
    state = _run_adapter(
        module, params, support, support_reward_i, support_condition,
        apply_fn, variables, lr=lr, steps=steps, tag="moe", save_path=save_path,
    )
    return module.apply(state.params, query)


# ─────────────────────────────────────────────────────────────────────────────
#  통합 진입점
# ─────────────────────────────────────────────────────────────────────────────

def adapt_embeddings(
    query: jnp.ndarray,
    support: jnp.ndarray,
    support_reward_i: jnp.ndarray,
    support_condition: jnp.ndarray,
    apply_fn: Callable,
    variables: dict,
    *,
    adapter_type: str,
    rank: int = 4,
    num_experts: int = 4,
    hidden_dim: int = 64,
    lr: float = 1e-3,
    steps: int = 500,
    save_path: Optional[str] = None,
    load_path: Optional[str] = None,
) -> jnp.ndarray:
    if adapter_type == "none":
        return query

    # 사전 학습된 adapter state가 있으면 support 없이도 적용 가능
    _has_pretrained = bool(load_path and os.path.exists(load_path))
    if support.shape[0] == 0 and not _has_pretrained:
        logger.warning("adapt_embeddings: support set is empty — skipping adapter")
        return query

    logger.info(
        "Few-shot adapter: type=%s  support=%d  query=%d  steps=%d  pretrained=%s",
        adapter_type, support.shape[0], query.shape[0], steps, _has_pretrained,
    )

    if _has_pretrained:
        logger.info("Pre-computed adapter state found — skipping training: %s", load_path)

    if adapter_type == "film":
        return film_adapt(
            query, support, support_reward_i, support_condition,
            apply_fn, variables, lr=lr, steps=steps,
            save_path=save_path, load_path=load_path,
        )
    elif adapter_type == "lora":
        return lora_adapt(
            query, support, support_reward_i, support_condition,
            apply_fn, variables, rank=rank, lr=lr, steps=steps,
            save_path=save_path, load_path=load_path,
        )
    elif adapter_type == "moe":
        return moe_adapt(
            query, support, support_reward_i, support_condition,
            apply_fn, variables, num_experts=num_experts, lr=lr, steps=steps,
            save_path=save_path, load_path=load_path,
        )
    else:
        raise ValueError(
            f"Unknown adapter_type={adapter_type!r}. Choose from 'film', 'lora', 'moe'."
        )
