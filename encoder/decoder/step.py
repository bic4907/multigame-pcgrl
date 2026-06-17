from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState
from jax import jit

from encoder.data.clip_batch import CLIPDecoderBatch


@partial(jit, static_argnums=(3, 4, 5, 6, 7, 8, 9))
def train_step(
    train_state: TrainState,
    batch: CLIPDecoderBatch,
    rng_key: jax.random.PRNGKey,
    is_train: bool = True,
    mode: str = "text_state",
    contrastive_weight: float = 1.0,
    cls_weight: float = 1.0,
    reg_weight: float = 0.1,
    num_reward_classes: int = 5,
    regression_loss: str = "mae",
    norm_min_arr: jnp.ndarray = None,
    norm_max_arr: jnp.ndarray = None,
):
    rng_key, dropout_rng = jax.random.split(rng_key)

    def pairwise_contrastive_loss_accuracy(a, b, temperature):
        logits = jnp.matmul(a, b.T) / jnp.exp(temperature)
        a2b_logps = jax.nn.log_softmax(logits, axis=1)
        b2a_logps = jax.nn.log_softmax(logits, axis=0)

        a2b_pos_logps = a2b_logps - 1e9 * (1 - batch.duplicate_matrix)
        b2a_pos_logps = b2a_logps - 1e9 * (1 - batch.duplicate_matrix)

        a2b_loss = -jnp.mean(jax.scipy.special.logsumexp(a2b_pos_logps, axis=1))
        b2a_loss = -jnp.mean(jax.scipy.special.logsumexp(b2a_pos_logps, axis=0))

        a2b_correct_pr = jnp.mean(
            jnp.sum(jnp.exp(a2b_logps) * batch.duplicate_matrix, axis=1)
        )
        b2a_correct_pr = jnp.mean(
            jnp.sum(jnp.exp(b2a_logps) * batch.duplicate_matrix, axis=0)
        )

        a2b_top1_accuracy = jnp.mean(
            jnp.max(a2b_logps, axis=1) == jnp.max(a2b_pos_logps, axis=1)
        )
        b2a_top1_accuracy = jnp.mean(
            jnp.max(b2a_logps, axis=0) == jnp.max(b2a_pos_logps, axis=0)
        )

        return a2b_loss, b2a_loss, a2b_correct_pr, b2a_correct_pr, a2b_top1_accuracy, b2a_top1_accuracy

    def loss_fn(params):
        outputs = train_state.apply_fn(
            params,
            batch.input_ids,
            batch.attention_mask,
            batch.pixel_values,
            reward_enum=batch.reward_enum_target,
            mode=mode,
            training=is_train,
            rngs={"dropout": dropout_rng},
        )

        text_embed = outputs["text_embed"]
        state_embed = outputs.get("state_embed", jnp.zeros_like(text_embed))
        state_mask = jnp.any(state_embed != 0).astype(jnp.float32)
        text_state_temperature = outputs["text_state_temperature"]

        # ── Contrastive Loss ──
        temperature = jnp.clip(text_state_temperature, jnp.log(0.01), jnp.log(100))
        s2t_loss, t2s_loss, s2t_correct_pr, t2s_correct_pr, s2t_top1, t2s_top1 = pairwise_contrastive_loss_accuracy(
            state_embed, text_embed, temperature
        )
        contrastive_loss = state_mask * (s2t_loss + t2s_loss) / 2.0

        # ── Decoder: reward_enum classification ──
        reward_logits = outputs["reward_logits"]
        reward_target = batch.reward_enum_target
        cls_loss = jnp.mean(
            optax.softmax_cross_entropy_with_integer_labels(reward_logits, reward_target)
        )
        reward_pred = jnp.argmax(reward_logits, axis=-1)
        reward_accuracy = jnp.mean(reward_pred == reward_target)

        # ── (3) Decoder: condition regression loss (huber or mae) ──
        condition_pred = outputs["condition_pred"]    # (B, num_classes) — [0,1] 정규화
        condition_target = batch.condition_target      # (B,) — [0,1] 정규화
        # 각 샘플의 predicted condition을 gt reward_enum 인덱스로 gather
        per_sample_cond = condition_pred[jnp.arange(condition_pred.shape[0]), reward_target]
        abs_diff = jnp.abs(per_sample_cond - condition_target)

        # 원본 스케일로 변환한 값/타깃 및 오차 (로깅용)
        condition_pred_raw = outputs["condition_pred_raw"]   # (B, num_classes) — 원래 linear 스케일
        per_sample_cond_raw = condition_pred_raw[jnp.arange(condition_pred_raw.shape[0]), reward_target]
        target_log = condition_target * (norm_max_arr[reward_target] - norm_min_arr[reward_target]) + norm_min_arr[reward_target]
        target_raw = jnp.expm1(jnp.maximum(target_log, 0.0))
        abs_diff_raw = jnp.abs(per_sample_cond_raw - target_raw)

        if regression_loss == "huber":
            reg_per_sample = jnp.where(abs_diff <= 1.0, 0.5 * abs_diff ** 2, abs_diff - 0.5)
            reg_per_sample_raw = jnp.where(abs_diff_raw <= 1.0, 0.5 * abs_diff_raw ** 2, abs_diff_raw - 0.5)
        else:  # mae
            reg_per_sample = abs_diff
            reg_per_sample_raw = abs_diff_raw
        reg_loss = jnp.mean(reg_per_sample)
        reg_loss_raw = jnp.mean(reg_per_sample_raw)
        # linear 공간 normalized [0,1] MAE (모니터링용 — gradient 계산에 불포함)
        # norm_min/max는 log1p 공간이므로 expm1로 linear 스케일로 복원 후 정규화
        linear_min = jnp.expm1(norm_min_arr[reward_target])
        linear_max = jnp.expm1(norm_max_arr[reward_target])
        linear_range = linear_max - linear_min + 1e-8
        condition_mae_normalized = jnp.mean(jnp.abs(per_sample_cond_raw - target_raw) / linear_range)

        # ── Per-reward_enum regression 메트릭 ──
        per_enum_huber = jnp.zeros(num_reward_classes)
        per_enum_mae = jnp.zeros(num_reward_classes)
        per_enum_huber_raw = jnp.zeros(num_reward_classes)
        per_enum_mae_raw = jnp.zeros(num_reward_classes)
        per_enum_count = jnp.zeros(num_reward_classes)

        for eidx in range(num_reward_classes):
            mask = (reward_target == eidx).astype(jnp.float32)        # (B,)
            count = jnp.sum(mask) + 1e-8                               # 0-div 방지
            per_enum_huber = per_enum_huber.at[eidx].set(jnp.sum(reg_per_sample * mask) / count)
            per_enum_mae = per_enum_mae.at[eidx].set(jnp.sum(abs_diff * mask) / count)
            per_enum_huber_raw = per_enum_huber_raw.at[eidx].set(jnp.sum(reg_per_sample_raw * mask) / count)
            per_enum_mae_raw = per_enum_mae_raw.at[eidx].set(jnp.sum(abs_diff_raw * mask) / count)
            per_enum_count = per_enum_count.at[eidx].set(jnp.sum(mask))

        # ── Total Loss ──
        total_loss = (
            contrastive_weight * contrastive_loss
            + cls_weight * cls_loss
            + reg_weight * reg_loss
        )

        metrics = {
            "contrastive_loss": contrastive_loss,
            "state2text_loss": s2t_loss * state_mask,
            "text2state_loss": t2s_loss * state_mask,
            "state2text_correct_pr": s2t_correct_pr * state_mask,
            "text2state_correct_pr": t2s_correct_pr * state_mask,
            "state2text_top1_accuracy": s2t_top1 * state_mask,
            "text2state_top1_accuracy": t2s_top1 * state_mask,
            "text_state_temperature": text_state_temperature,
            "cls_loss": cls_loss,
            "reg_loss": reg_loss,
            "reg_loss_raw": reg_loss_raw,
            "reward_accuracy": reward_accuracy,
            "reward_pred": reward_pred,  # (B,) per-sample predictions
            "abs_diff": abs_diff,        # (B,) per-sample |pred_cond - target_cond| (normalized)
            "abs_diff_raw": abs_diff_raw, # (B,) per-sample |pred_cond_raw - target_raw|
            "per_enum_reg_loss": per_enum_huber,
            "per_enum_reg_loss_mae": per_enum_mae,
            "per_enum_reg_loss_raw": per_enum_huber_raw,
            "per_enum_reg_loss_raw_mae": per_enum_mae_raw,
            "per_enum_count": per_enum_count,
            # ── per-sample scatter / per-enum 통계 용 ──
            "per_sample_cond_norm": per_sample_cond,           # (B,) normalized [0,1] pred
            "per_sample_cond_target_norm": condition_target,   # (B,) normalized [0,1] target
            "per_sample_cond_raw": per_sample_cond_raw,        # (B,) linear-scale pred
            "per_sample_cond_target_raw": target_raw,      # (B,) linear-scale target
        }
        return total_loss, metrics

    (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(
        train_state.params
    )
    train_state = jax.lax.cond(
        is_train,
        lambda _: train_state.apply_gradients(grads=grads),
        lambda _: train_state,
        operand=None,
    )
    return train_state, loss, metrics, rng_key
