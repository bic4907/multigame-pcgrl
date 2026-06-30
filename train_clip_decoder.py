"""
train_clip_decoder_unseen.py
============================
Seen/Unseen 게임 분리 + Few-shot Ratio Sweep 실험 스크립트.

Seen 게임의 전체 학습 데이터 + Unseen 게임의 가변 비율 학습 데이터로
CLIP Decoder 모델을 학습하고, **고정된** 테스트셋에서 게임별 reward_accuracy를 측정한다.

최종 출력: few-shot ratio (x) vs. per-game reward accuracy (y) 그래프

Usage
-----
    python train_clip_decoder_unseen.py game=all unseen_games=zd
    python train_clip_decoder_unseen.py game=all unseen_games=pkzd \\
        'unseen_ratios=[0.0,0.1,0.5,1.0]' n_epochs=30
"""

import datetime
import json
import math
import os
import shutil
import logging
from collections import deque
from functools import partial
from os.path import basename
from typing import Dict, List, Optional, Set, Tuple

import hydra
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import wandb
from flax.training.train_state import TrainState
from jax import jit
from tqdm import tqdm
from transformers import CLIPProcessor

from conf.config import CLIPDecoderTrainConfig
from conf.game_utils import GAME_ABBR
from dataset.multigame import MultiGameDataset
from encoder.clip_model import get_cnnclip_decoder_encoder
from encoder.data.clip_batch import (
    CLIPDataset,
    CLIPDecoderBatch,
    CLIPDatasetBuilder,
    create_clip_decoder_batch,
)
from encoder.schedular import create_learning_rate_fn
from encoder.utils.path import init_config
from encoder.utils.training import (
    build_multigame_dataset,
    save_encoder_checkpoint,
    save_norm_stats,
)
from instruct_rl.utils.format_utils import simple_table
from instruct_rl.utils.logger import get_wandb_name


log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logger = logging.getLogger(basename(__file__))
logger.setLevel(getattr(logging, log_level, logging.INFO))
logging.getLogger("absl").setLevel(logging.ERROR)


# ═══════════════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def parse_unseen_game_names(unseen_str: Optional[str]) -> Set[str]:
    """2글자 약어 문자열 → full game name set.

    None 또는 빈 문자열이면 빈 set을 반환한다.

    Examples
    --------
    parse_unseen_game_names(None)   -> set()
    parse_unseen_game_names("")     -> set()
    parse_unseen_game_names("zd")   -> {'zelda'}
    parse_unseen_game_names("pkzd") -> {'pokemon', 'zelda'}
    """
    if not unseen_str:
        return set()
    names: Set[str] = set()
    for i in range(0, len(unseen_str), 2):
        abbr = unseen_str[i : i + 2]
        names.update(GAME_ABBR.get(abbr, []))
    return names


def subset_clip_dataset(dataset: CLIPDataset, indices: np.ndarray) -> CLIPDataset:
    """CLIPDataset에서 주어진 인덱스의 서브셋을 추출한다."""
    idx = np.asarray(indices, dtype=int)
    return CLIPDataset(
        class_ids=dataset.class_ids[idx],
        reward_cond=dataset.reward_cond[idx],
        input_ids=dataset.input_ids[idx],
        attention_masks=dataset.attention_masks[idx],
        pixel_values=dataset.pixel_values[idx],
        is_train=dataset.is_train[idx],
        reward_enum_targets=dataset.reward_enum_targets[idx],
        condition_targets=dataset.condition_targets[idx],
        quantized_condition_targets=dataset.quantized_condition_targets[idx],
        game_ids=dataset.game_ids[idx] if dataset.game_ids is not None else None,
    )


def split_dataset_by_game(
    full_dataset: CLIPDataset,
    unseen_game_names: Set[str],
    test_ratio: float,
    test_seed: int,
) -> Tuple[
    Dict[str, np.ndarray],  # game → train pool indices
    Dict[str, np.ndarray],  # game → test indices
    np.ndarray,             # all game names (per sample)
]:
    """전체 데이터셋을 게임별로 train pool / test 로 분할한다.

    - 모든 게임(seen + unseen)에서 ``test_ratio`` 만큼 테스트 세트로 분리
    - 분할은 ``test_seed`` 로 결정 → 동일한 시드에서 항상 같은 테스트셋
    - train pool 내 unseen 게임 데이터의 실제 사용량은 sweep ratio 에 의해 결정

    Returns
    -------
    game_train_pool : {game_name: ndarray of indices}
    game_test       : {game_name: ndarray of indices}
    all_game_names  : ndarray of str  (길이 = len(full_dataset.class_ids))
    """
    all_game_names = np.array(
        [rc["game_name"] for rc in full_dataset.reward_cond]
    )
    unique_games = sorted(set(all_game_names))

    rng = np.random.RandomState(test_seed)

    game_train_pool: Dict[str, np.ndarray] = {}
    game_test: Dict[str, np.ndarray] = {}

    for game in unique_games:
        game_indices = np.where(all_game_names == game)[0]
        perm = rng.permutation(game_indices)
        n_test = max(1, int(len(perm) * test_ratio))
        game_test[game] = perm[:n_test]
        game_train_pool[game] = perm[n_test:]  # 고정 순서 (ratio 서브셋은 prefix)
        tag = "(unseen)" if game in unseen_game_names else "(seen)"
        logger.debug(
            "split_dataset_by_game [%s] %s: total=%d, train_pool=%d, test=%d",
            game, tag, len(game_indices), len(game_train_pool[game]), len(game_test[game]),
        )

    return game_train_pool, game_test, all_game_names


def build_train_indices_for_ratio(
    game_train_pool: Dict[str, np.ndarray],
    unseen_game_names: Set[str],
    ratio: float,
    seen_ratio: float = 1.0,
) -> np.ndarray:
    """주어진 few-shot ``ratio`` 에 대해 학습 인덱스를 구성한다.

    - Seen 게임: train pool 중 seen_ratio 비율만큼 (prefix) 사용
    - Unseen 게임: train pool 중 ratio 비율만큼 (prefix) 사용
    - ratio=0.0 이면 unseen 게임의 학습 데이터 = 0
    - seen_ratio=0.0 이면 seen 게임의 학습 데이터 = 0
    """
    train_indices: List[np.ndarray] = []
    for game, pool in sorted(game_train_pool.items()):
        if game in unseen_game_names:
            n_use = int(len(pool) * ratio)
            if n_use > 0:
                train_indices.append(pool[:n_use])
        else:
            n_use = int(len(pool) * seen_ratio)
            if n_use > 0:
                train_indices.append(pool[:n_use])
    if train_indices:
        return np.concatenate(train_indices)
    return np.array([], dtype=int)


# ═══════════════════════════════════════════════════════════════════════════════
#  Train Step (JIT) — reward_pred 추가
# ═══════════════════════════════════════════════════════════════════════════════

@partial(jit, static_argnums=(3, 4, 5, 6, 7, 8, 9, 12, 13, 14, 15, 16))
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
    delta_weight: float = 0.0,
    num_games: int = 1,
    delta_min_count: int = 2,
    delta_var_eps: float = 1e-4,
    compute_delta_when_zero: bool = True,
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

    def safe_l2_normalize(x, axis=-1, eps=1e-6):
        """0-vector 근처에서도 gradient가 NaN으로 터지지 않는 L2 normalize.

        jnp.linalg.norm()은 zero-vector에서 gradient가 NaN이 되므로,
        sum-of-squares + jax.lax.rsqrt(maximum(...)) 형태로 우회한다.
        """
        sq_norm = jnp.sum(x * x, axis=axis, keepdims=True)
        inv_norm = jax.lax.rsqrt(jnp.maximum(sq_norm, eps * eps))
        return x * inv_norm

    def continuous_direction_alignment(text_embed, game_id, reward_target, condition_target):
        """Continuous Task-wise Cross-game Direction Alignment Loss.

        각 (game, task) 그룹에서 condition 값과 (L2-normalized) text embedding 사이의
        slope vector를 OLS-style centered regression 으로 추정한 후, 같은 task 안의
        서로 다른 game 간 방향 벡터를 cosine distance 로 정렬한다.

        주의: invalid (game, task) 그룹은 normalize **이전에** slope를 0으로 차단해야
        한다. slope=0 인 그룹에 normalize gradient가 흐르면 NaN이 발생하고, 이후
        mask를 곱해도 (NaN * 0 = NaN) 으로 parameter 전체가 오염된다.

        Returns: (loss, valid_pair_count)
        """
        z = safe_l2_normalize(text_embed, axis=-1, eps=1e-6)
        c = condition_target.astype(jnp.float32)
        G = num_games
        T = num_reward_classes

        game_mask = (game_id[None, :] == jnp.arange(G)[:, None])           # (G, B)
        task_mask = (reward_target[None, :] == jnp.arange(T)[:, None])     # (T, B)
        m = (game_mask[:, None, :] & task_mask[None, :, :]).astype(jnp.float32)  # (G, T, B)

        n_gt = m.sum(axis=-1)                                              # (G, T)
        safe_n = jnp.maximum(n_gt, 1.0)

        c_b = c[None, None, :]                                             # (1, 1, B)
        z_b = z[None, None, :, :]                                          # (1, 1, B, D)

        c_mean = (m * c_b).sum(-1) / safe_n                                # (G, T)
        z_mean = (m[..., None] * z_b).sum(-2) / safe_n[..., None]          # (G, T, D)

        dc = c_b - c_mean[..., None]                                       # (G, T, B)
        dz = z_b - z_mean[:, :, None, :]                                   # (G, T, B, D)

        m_dc = m * dc                                                      # (G, T, B)
        slope_num = (m_dc[..., None] * dz).sum(-2)                         # (G, T, D)
        slope_den = (m_dc * dc).sum(-1)                                    # (G, T) = sum m*dc^2

        # ── valid group 판정 (slope 계산 전) ──
        c_var = slope_den / safe_n                                         # (G, T)
        valid = (n_gt >= float(delta_min_count)) & (c_var > delta_var_eps) # (G, T)

        # ── slope_den이 매우 작은 invalid group에서 slope 폭주 방지 ──
        safe_slope_den = jnp.where(valid, slope_den, 1.0)                  # (G, T)
        slope = slope_num / safe_slope_den[..., None]                      # (G, T, D)
        # invalid group은 normalize 이전에 완전히 0으로 차단
        slope = jnp.where(valid[..., None], slope, 0.0)                    # (G, T, D)

        # ── NaN-safe normalize + invalid 방향 재차단 ──
        d_dir = safe_l2_normalize(slope, axis=-1, eps=1e-6)               # (G, T, D)
        d_dir = jnp.where(valid[..., None], d_dir, 0.0)                    # (G, T, D)

        # task별 cross-game cosine: (G, G, T)
        cos_mat = jnp.einsum('gtd,htd->ght', d_dir, d_dir)
        pair_valid = valid[:, None, :] & valid[None, :, :]                 # (G, G, T)
        tri = jnp.triu(jnp.ones((G, G), dtype=bool), k=1)                  # (G, G)
        pair_valid = pair_valid & tri[:, :, None]

        # NaN * 0 방지: multiply mask 대신 where 사용
        pair_loss = jnp.where(pair_valid, 1.0 - cos_mat, 0.0)
        total_pairs = pair_valid.astype(jnp.float32).sum()
        delta_loss = pair_loss.sum() / jnp.maximum(total_pairs, 1.0)
        return delta_loss, total_pairs

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

        # ── Continuous Task-wise Cross-game Direction Alignment ──
        # 기본값은 delta_weight=0.0이어도 alignment metric을 모니터링한다.
        # compute_delta_when_zero=False일 때만 baseline/eval의 불필요한 계산을 건너뛴다.
        if delta_weight != 0.0 or compute_delta_when_zero:
            delta_loss, delta_valid_pairs = continuous_direction_alignment(
                text_embed, batch.game_id, reward_target, condition_target
            )
        else:
            delta_loss = jnp.array(0.0, dtype=text_embed.dtype)
            delta_valid_pairs = jnp.array(0.0, dtype=text_embed.dtype)

        # ── Total Loss ──
        total_loss = (
            contrastive_weight * contrastive_loss
            + cls_weight * cls_loss
            + reg_weight * reg_loss
            + delta_weight * delta_loss
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
            # ── Continuous direction alignment ──
            "continuous_delta_loss": delta_loss,                          # raw alignment loss (weight 미적용, 항상 모니터링)
            "continuous_delta_loss_weighted": delta_weight * delta_loss,  # objective에 실제 반영된 값
            "valid_direction_pair_count": delta_valid_pairs
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


# ═══════════════════════════════════════════════════════════════════════════════
#  Sequential evaluation (테스트셋 순서 보존)
# ═══════════════════════════════════════════════════════════════════════════════

# reward_enum 이름 매핑 (0-based: CSV reward_enum은 0-indexed)
_REWARD_ENUM_NAMES = {
    0: "region",
    1: "path_length",
    2: "interactable_count",
    3: "hazard_count",
    4: "collectable_count",
}


def _log_reward_condition_summary(dataset: MultiGameDataset):
    """학습 시작 전에 reward_enum별 condition 범위를 출력한다 (게임 구분 없이 enum 기준)."""
    from collections import defaultdict

    # reward_enum → [(game, condition_value)]
    enum_stats: dict = defaultdict(list)
    # game → reward_enum → [condition_values]  (게임별 분해용)
    game_enum_stats: dict = defaultdict(lambda: defaultdict(list))

    for s in dataset._samples:
        reward_enum = s.meta.get("reward_enum")
        if reward_enum is None:
            continue
        game = s.game
        conditions = s.meta.get("conditions", {})
        cond_val = list(conditions.values())[0] if conditions else None
        re_id = int(reward_enum)
        enum_stats[re_id].append(cond_val)
        game_enum_stats[game][re_id].append(cond_val)

    logger.info("=" * 80)
    logger.info("  Reward Enum & Condition Range Summary  (raw, before normalization)")
    logger.info("=" * 80)

    # ── reward_enum별 전체 통계 ──
    logger.info(f"  {'enum':>5}  {'name':<22} {'count':>6}  {'min':>10}  {'max':>10}  {'mean':>10}  {'std':>10}")
    logger.info(f"  {'-'*5}  {'-'*22} {'-'*6}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}")

    for re_id in sorted(enum_stats.keys()):
        vals = enum_stats[re_id]
        valid_vals = [v for v in vals if v is not None]
        name = _REWARD_ENUM_NAMES.get(re_id, f"unknown_{re_id}")
        count = len(vals)

        if valid_vals:
            v_min = min(valid_vals)
            v_max = max(valid_vals)
            v_mean = sum(valid_vals) / len(valid_vals)
            v_std = (sum((v - v_mean) ** 2 for v in valid_vals) / len(valid_vals)) ** 0.5
            logger.info(f"  {re_id:>5}  {name:<22} {count:>6}  {v_min:>10.2f}  {v_max:>10.2f}  {v_mean:>10.2f}  {v_std:>10.2f}")
        else:
            logger.info(f"  {re_id:>5}  {name:<22} {count:>6}  {'N/A':>10}  {'N/A':>10}  {'N/A':>10}  {'N/A':>10}")

    logger.info("")

    # ── 게임별 분해 ──
    for game in sorted(game_enum_stats.keys()):
        enum_dict = game_enum_stats[game]
        n_total = sum(len(v) for v in enum_dict.values())
        logger.info(f"  [{game}]  ({n_total} samples)")
        for re_id in sorted(enum_dict.keys()):
            vals = enum_dict[re_id]
            valid_vals = [v for v in vals if v is not None]
            name = _REWARD_ENUM_NAMES.get(re_id, f"unknown_{re_id}")
            if valid_vals:
                logger.info(f"    enum {re_id} ({name}): "
                            f"n={len(vals)}, "
                            f"range=[{min(valid_vals):.2f}, {max(valid_vals):.2f}], "
                            f"mean={sum(valid_vals)/len(valid_vals):.2f}")
            else:
                logger.info(f"    enum {re_id} ({name}): n={len(vals)}, range=N/A")
    logger.info("")

    # 전체 요약
    all_enums = set(enum_stats.keys())
    logger.info(f"  Total games: {len(game_enum_stats)},  "
                f"Unique reward_enums (0-based): {sorted(all_enums)},  "
                f"num_reward_classes should be >= {max(all_enums) + 1 if all_enums else 0}")
    logger.info("=" * 80)
    logger.info("  ※ Condition values will be min-max normalized per reward_enum to [0, 1]")
    logger.info("=" * 80)

def make_train(config: CLIPDecoderTrainConfig):
    def train(rng_key):
        rng_key, subkey = jax.random.split(rng_key)
        dataset = build_multigame_dataset(config)

        # ── 학습 전 reward_enum / condition 범위 요약 출력 ──
        _log_reward_condition_summary(dataset)

        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        dataset_builder = CLIPDatasetBuilder(
            processor=processor,
            paired_data=dataset,
            rng_key=subkey,
            max_len=config.encoder.token_max_len,
            train_ratio=config.train_ratio,
            max_samples=config.max_samples,
            instruction_prefix=config.instruction_prefix,
            longtail_cut=config.longtail_cut,
        )

        train_clip_dataset, test_clip_dataset = dataset_builder.get_split_dataset()
        class_id2reward_cond = dataset_builder.get_class_id2reward_cond()
        cond_norm_min, cond_norm_max = dataset_builder.get_condition_norm_stats()

        # ── Save norm stats to ckpt directory (used for denorm during inference) ──
        save_norm_stats(config, cond_norm_min, cond_norm_max)

        # scatter plot용 class_id → game_name 매핑
        class_id2game_name = {}
        full_ds = dataset_builder.get_dataset()
        for cid, rc in zip(full_ds.class_ids, full_ds.reward_cond):
            class_id2game_name[int(cid)] = rc.get("game_name", "unknown")

        # ── 정규화 파라미터 출력 ──
        logger.info("  Per-reward_enum condition normalization applied:")
        logger.info(f"  {'enum(0idx)':>10}  {'name':<22} {'raw_min':>10}  {'raw_max':>10}  {'→ normalized':>12}")
        for eidx in sorted(cond_norm_min.keys()):
            name = _REWARD_ENUM_NAMES.get(eidx, f"unknown_{eidx}")
            r_min, r_max = cond_norm_min[eidx], cond_norm_max[eidx]
            logger.info(f"  {eidx:>10}  {name:<22} {r_min:>10.2f}  {r_max:>10.2f}  {'[0.0, 1.0]':>12}")
        logger.info("")


        n_train = len(train_clip_dataset.class_ids)
        n_test = len(test_clip_dataset.class_ids)

        n_train_batch = math.ceil(n_train / config.batch_size)
        n_test_batch = math.ceil(n_test / config.batch_size)

        config.steps_per_epoch = n_train_batch

        mode = "text"
        if config.encoder.state:
            mode += "_state"
        config.encoder.mode = mode

        # ── norm stats를 jnp 배열로 변환 (모델 내 역변환용) ──
        num_cls = config.decoder.num_reward_classes
        norm_min_arr = jnp.array([cond_norm_min.get(i, 0.0) for i in range(num_cls)], dtype=jnp.float32)
        norm_max_arr = jnp.array([cond_norm_max.get(i, 1.0) for i in range(num_cls)], dtype=jnp.float32)

        train_state, lr_schedular = get_train_state(
            config, subkey,
            cond_norm_min=norm_min_arr,
            cond_norm_max=norm_max_arr,
        )

        logger.info("Start training CLIP + Decoder model")
        logger.info(f"  contrastive_weight={config.contrastive_weight}, "
                    f"cls_weight={config.cls_weight}, reg_weight={config.reg_weight}")

        train_embed_queue = deque(maxlen=config.n_max_points)
        val_embed_queue = deque(maxlen=config.n_max_points)
def evaluate_per_game(
    train_state: TrainState,
    test_ds: CLIPDataset,
    test_game_names: np.ndarray,
    unseen_game_names: Set[str],
    config,
    rng_key: jax.random.PRNGKey,
    num_cls: int,
    mode: str,
    norm_min_arr: jnp.ndarray = None,
    norm_max_arr: jnp.ndarray = None,
    num_games: int = 1,
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, Dict[int, float]], Dict[int, Dict[str, np.ndarray]], Dict[int, float]]:
    """고정된 테스트셋에서 **게임별** reward accuracy 와 reg_loss를 계산한다.

    Returns
    -------
    per_game_acc       : {game: acc, "overall", "seen_overall", "unseen_overall"}
    per_game_reg_loss  : {game: reg, "overall", "seen_overall", "unseen_overall"}
    per_game_enum_diff : {game: {reward_enum: mean_abs_diff}}
    scatter_data       : {reward_enum: {"pred_norm","target_norm","pred_raw","target_raw"}}
    per_enum_reg_loss  : {reward_enum: mean_abs_diff (raw space)}
    """
    n_test = len(test_ds.input_ids)
    batch_size = config.batch_size

    all_preds: List[int] = []
    all_targets: List[int] = []
    all_reg_losses: List[float] = []  # per-sample reg loss
    all_abs_diffs: List[float] = []   # per-sample |pred_cond - target_cond| (normalized)
    all_abs_diffs_raw: List[float] = []  # per-sample |pred_cond_raw - target_raw|
    all_reward_enums: List[int] = []  # per-sample reward_enum target
    all_pred_norm: List[float] = []
    all_target_norm: List[float] = []
    all_pred_raw: List[float] = []
    all_target_raw: List[float] = []

    for start_idx in range(0, n_test, batch_size):
        end_idx = min(start_idx + batch_size, n_test)
        indices = np.arange(start_idx, end_idx)
        actual_size = len(indices)

        # 마지막 배치 패딩
        if actual_size < batch_size:
            pad = np.arange(batch_size - actual_size) % n_test
            indices = np.concatenate([indices, pad])

        class_ids = test_ds.class_ids[indices].squeeze()
        input_ids = test_ds.input_ids[indices]
        attention_mask = test_ds.attention_masks[indices]
        pixel_values = test_ds.pixel_values[indices]
        duplicate_matrix = np.equal.outer(class_ids, class_ids).astype(np.float32)
        reward_enum_target = test_ds.reward_enum_targets[indices]
        condition_target = test_ds.condition_targets[indices]
        game_id = (
            test_ds.game_ids[indices]
            if test_ds.game_ids is not None
            else np.zeros(len(indices), dtype=np.int32)
        )

        batch = CLIPDecoderBatch(
            class_ids=class_ids,
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            duplicate_matrix=duplicate_matrix,
            reward_enum_target=reward_enum_target,
            condition_target=condition_target,
            game_id=game_id,
        )
        batch = jax.device_put(batch)

        _, _, metrics, _ = train_step(
            train_state,
            batch,
            rng_key=rng_key,
            is_train=False,
            mode=mode,
            contrastive_weight=config.contrastive_weight,
            cls_weight=config.cls_weight,
            reg_weight=config.reg_weight,
            num_reward_classes=num_cls,
            regression_loss=config.regression_loss,
            norm_min_arr=norm_min_arr,
            norm_max_arr=norm_max_arr,
            delta_weight=float(getattr(config, "delta_weight", 0.0)),
            num_games=int(num_games),
            delta_min_count=int(getattr(config, "delta_min_group_samples", 2)),
            delta_var_eps=float(getattr(config, "delta_var_eps", 1e-4)),
            compute_delta_when_zero=bool(getattr(config, "compute_delta_when_zero", True)),
        )

        preds = np.array(jax.device_get(metrics["reward_pred"]))
        targets = np.array(jax.device_get(reward_enum_target))
        batch_reg = float(jax.device_get(metrics["reg_loss"]))
        batch_abs_diff = np.array(jax.device_get(metrics["abs_diff"]))
        batch_abs_diff_raw = np.array(jax.device_get(metrics["abs_diff_raw"]))
        batch_pred_norm = np.array(jax.device_get(metrics["per_sample_cond_norm"]))
        batch_target_norm = np.array(jax.device_get(metrics["per_sample_cond_target_norm"]))
        batch_pred_raw = np.array(jax.device_get(metrics["per_sample_cond_raw"]))
        batch_target_raw = np.array(jax.device_get(metrics["per_sample_cond_target_raw"]))
        all_preds.extend(preds[:actual_size].tolist())
        all_targets.extend(targets[:actual_size].tolist())
        # batch-level reg_loss를 actual_size만큼 복제 (batch 평균이므로)
        all_reg_losses.extend([batch_reg] * actual_size)
        all_abs_diffs.extend(batch_abs_diff[:actual_size].tolist())
        all_abs_diffs_raw.extend(batch_abs_diff_raw[:actual_size].tolist())
        all_reward_enums.extend(targets[:actual_size].tolist())
        all_pred_norm.extend(batch_pred_norm[:actual_size].tolist())
        all_target_norm.extend(batch_target_norm[:actual_size].tolist())
        all_pred_raw.extend(batch_pred_raw[:actual_size].tolist())
        all_target_raw.extend(batch_target_raw[:actual_size].tolist())

    # ── Per-game accuracy 집계 ──
    all_preds_arr = np.array(all_preds[:n_test])
    all_targets_arr = np.array(all_targets[:n_test])
    all_reg_arr = np.array(all_reg_losses[:n_test])
    all_abs_diff_arr = np.array(all_abs_diffs[:n_test])
    all_abs_diff_raw_arr = np.array(all_abs_diffs_raw[:n_test])
    all_reward_enum_arr = np.array(all_reward_enums[:n_test])
    correct = all_preds_arr == all_targets_arr

    per_game_acc: Dict[str, float] = {}
    per_game_reg: Dict[str, float] = {}
    per_game_enum_diff: Dict[str, Dict[int, float]] = {}
    unique_test_games = sorted(set(test_game_names))
    for game in unique_test_games:
        mask = test_game_names == game
        if mask.sum() > 0:
            per_game_acc[game] = float(correct[mask].mean())
            per_game_reg[game] = float(all_reg_arr[mask].mean())
            # per reward_enum mean abs diff
            enum_diff: Dict[int, float] = {}
            for e in sorted(set(all_reward_enum_arr[mask])):
                emask = mask & (all_reward_enum_arr == e)
                if emask.sum() > 0:
                    enum_diff[int(e)] = float(all_abs_diff_arr[emask].mean())
            per_game_enum_diff[game] = enum_diff

    per_game_acc["overall"] = float(correct.mean())
    per_game_reg["overall"] = float(all_reg_arr.mean())

    # seen / unseen overall
    seen_mask = np.array([g not in unseen_game_names for g in test_game_names])
    unseen_mask = ~seen_mask
    if seen_mask.sum() > 0:
        per_game_acc["seen_overall"] = float(correct[seen_mask].mean())
        per_game_reg["seen_overall"] = float(all_reg_arr[seen_mask].mean())
    if unseen_mask.sum() > 0:
        per_game_acc["unseen_overall"] = float(correct[unseen_mask].mean())
        per_game_reg["unseen_overall"] = float(all_reg_arr[unseen_mask].mean())

    # ── Per reward_enum 통계 (전체 테스트셋 기준) ──
    all_pred_norm_arr = np.array(all_pred_norm[:n_test])
    all_target_norm_arr = np.array(all_target_norm[:n_test])
    all_pred_raw_arr = np.array(all_pred_raw[:n_test])
    all_target_raw_arr = np.array(all_target_raw[:n_test])

    per_enum_reg_loss: Dict[int, float] = {}
    scatter_data: Dict[int, Dict[str, np.ndarray]] = {}
    for e in sorted(set(all_reward_enum_arr.tolist())):
        emask = all_reward_enum_arr == int(e)
        if emask.sum() == 0:
            continue
        per_enum_reg_loss[int(e)] = float(all_abs_diff_raw_arr[emask].mean())
        scatter_data[int(e)] = {
            "pred_norm": all_pred_norm_arr[emask],
            "target_norm": all_target_norm_arr[emask],
            "pred_raw": all_pred_raw_arr[emask],
            "target_raw": all_target_raw_arr[emask],
        }

    return per_game_acc, per_game_reg, per_game_enum_diff, scatter_data, per_enum_reg_loss


def _build_scatter_data_from_arrays(
    reward_enums: np.ndarray,
    pred_norm: np.ndarray,
    target_norm: np.ndarray,
    pred_raw: np.ndarray,
    target_raw: np.ndarray,
    game_names: Optional[np.ndarray] = None,
) -> Dict[int, Dict[str, np.ndarray]]:
    """훈련 중 수집한 per-sample 예측/타깃으로 enum별 scatter dict 생성."""
    if len(reward_enums) == 0:
        return {}

    reward_enums = np.asarray(reward_enums)
    pred_norm = np.asarray(pred_norm)
    target_norm = np.asarray(target_norm)
    pred_raw = np.asarray(pred_raw)
    target_raw = np.asarray(target_raw)
    if game_names is None:
        game_names_arr = np.array([], dtype=object)
    else:
        game_names_arr = np.asarray(game_names, dtype=object)
        if game_names_arr.shape != reward_enums.shape:
            raise ValueError(
                "game_names shape must match reward_enums shape when provided."
            )

    scatter_data: Dict[int, Dict[str, np.ndarray]] = {}
    for e in sorted(set(reward_enums.tolist())):
        emask = reward_enums == int(e)
        if emask.sum() == 0:
            continue
        scatter_data[int(e)] = {
            "pred_norm": pred_norm[emask],
            "target_norm": target_norm[emask],
            "pred_raw": pred_raw[emask],
            "target_raw": target_raw[emask],
        }
        if len(game_names_arr) > 0:
            scatter_data[int(e)]["game_names"] = game_names_arr[emask]
    return scatter_data


def _compute_train_set_metrics_from_arrays(
    reward_pred: np.ndarray,
    reward_target: np.ndarray,
    reg_loss: np.ndarray,
    abs_diff: np.ndarray,
    abs_diff_raw: np.ndarray,
    reward_enums: np.ndarray,
    train_game_names: np.ndarray,
    unseen_game_names: Set[str],
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, Dict[int, float]], Dict[int, float]]:
    """학습 중 수집한 per-sample 값으로 train-set 지표를 재구성한다."""
    reward_pred_arr = np.asarray(reward_pred, dtype=np.int64)
    reward_target_arr = np.asarray(reward_target, dtype=np.int64)
    reg_loss_arr = np.asarray(reg_loss, dtype=np.float32)
    abs_diff_arr = np.asarray(abs_diff, dtype=np.float32)
    abs_diff_raw_arr = np.asarray(abs_diff_raw, dtype=np.float32)
    reward_enums_arr = np.asarray(reward_enums, dtype=np.int64)

    n = len(reward_target_arr)
    if n == 0:
        return {}, {}, {}, {}

    correct = reward_pred_arr == reward_target_arr
    correct_f = correct.astype(np.float32)

    per_game_acc: Dict[str, float] = {}
    per_game_reg: Dict[str, float] = {}
    per_game_enum_diff: Dict[str, Dict[int, float]] = {}

    for game in sorted(set(train_game_names)):
        mask = (np.asarray(train_game_names) == game)
        if mask.sum() == 0:
            continue
        per_game_acc[game] = float(correct_f[mask].mean())
        per_game_reg[game] = float(reg_loss_arr[mask].mean())

        enum_diff: Dict[int, float] = {}
        for e in sorted(set(reward_enums_arr[mask])):
            emask = mask & (reward_enums_arr == int(e))
            if emask.sum() > 0:
                enum_diff[int(e)] = float(abs_diff_arr[emask].mean())
        per_game_enum_diff[game] = enum_diff

    per_game_acc["overall"] = float(correct_f.mean())
    per_game_reg["overall"] = float(reg_loss_arr.mean())

    seen_mask = np.array([g not in unseen_game_names for g in train_game_names], dtype=bool)
    unseen_mask = ~seen_mask
    if seen_mask.sum() > 0:
        per_game_acc["seen_overall"] = float(correct_f[seen_mask].mean())
        per_game_reg["seen_overall"] = float(reg_loss_arr[seen_mask].mean())
    if unseen_mask.sum() > 0:
        per_game_acc["unseen_overall"] = float(correct_f[unseen_mask].mean())
        per_game_reg["unseen_overall"] = float(reg_loss_arr[unseen_mask].mean())

    per_enum_reg_loss: Dict[int, float] = {}
    for e in sorted(set(reward_enums_arr.tolist())):
        emask = reward_enums_arr == int(e)
        if emask.sum() == 0:
            continue
        per_enum_reg_loss[int(e)] = float(abs_diff_raw_arr[emask].mean())

    return per_game_acc, per_game_reg, per_game_enum_diff, per_enum_reg_loss


# ═══════════════════════════════════════════════════════════════════════════════
#  Model Init (get_train_state)
# ═══════════════════════════════════════════════════════════════════════════════

def get_train_state(config, rng_key, cond_norm_min=None, cond_norm_max=None):
    lr_schedular = create_learning_rate_fn(config, config.lr, config.steps_per_epoch)

    def create_train_state(module, rng_key, pretrained_params):
        def replace_params(params, key, replacement):
            for k in params.keys():
                if k == key:
                    params[k] = replacement
                    logging.info(f"replaced {key} in params")
                    return
                if isinstance(params[k], type(params)):
                    replace_params(params[k], key, replacement)

        rng_key, init_rng = jax.random.split(rng_key)
        input_ids = jnp.ones((1, config.encoder.token_max_len), dtype=jnp.int32)
        attention_mask = jnp.ones((1, config.encoder.token_max_len), dtype=jnp.int32)

        if config.encoder.model == "cnnclip":
            pixel_values = jnp.ones(
                (1, 16, 16, config.clip_input_channel), dtype=jnp.float32
            )
        elif config.encoder.model == "clip":
            pixel_values = jnp.ones(
                (1, 224, 224, config.clip_input_channel), dtype=jnp.float32
            )
        else:
            raise NotImplementedError(f"Model not implemented: {config.encoder.model}")

        variables = module.init(
            init_rng,
            input_ids,
            attention_mask,
            pixel_values,
            reward_enum=jnp.zeros((1,), dtype=jnp.int32),
            mode=config.encoder.mode,
            training=False,
        )

        if pretrained_params is not None:
            for key in pretrained_params:
                replace_params(variables, key, pretrained_params[key])

        def _create_mask(variables):
            import jax.tree_util as jtu

            flat = jtu.tree_map(lambda _: True, variables.get("params", {}))
            frozen = jtu.tree_map(lambda _: False, variables.get("norm_stats", {}))
            return {"params": flat, "norm_stats": frozen}

        mask = _create_mask(variables)
        tx = optax.masked(
            optax.adamw(learning_rate=lr_schedular, weight_decay=config.weight_decay),
            mask,
        )
        return TrainState.create(apply_fn=module.apply, params=variables, tx=tx)

    module, pretrained_params = get_cnnclip_decoder_encoder(
        config.encoder,
        decoder_config=config.decoder,
        cond_norm_min=cond_norm_min,
        cond_norm_max=cond_norm_max,
        RL_training=False,
    )

    state = create_train_state(module, rng_key=rng_key, pretrained_params=pretrained_params)
    return state, lr_schedular


# ═══════════════════════════════════════════════════════════════════════════════
#  Single-ratio 학습 + 평가
# ═══════════════════════════════════════════════════════════════════════════════

def train_and_evaluate_ratio(
    config: CLIPDecoderTrainConfig,
    rng_key: jax.random.PRNGKey,
    train_ds: CLIPDataset,
    test_ds: CLIPDataset,
    test_game_names: np.ndarray,
    unseen_game_names: Set[str],
    cond_norm_min: dict,
    cond_norm_max: dict,
    ratio: float,
    unseen_eval_ds: Optional[CLIPDataset] = None,
    unseen_eval_game_names: Optional[np.ndarray] = None,
    num_games: int = 1,
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, Dict[int, float]], Dict[int, Dict[str, np.ndarray]], Dict[int, float]]:
    """하나의 few-shot ratio에 대해 모델을 처음부터 학습하고 평가한다.

    Returns
    -------
    per_game_acc, per_game_reg_loss, per_game_enum_diff, scatter_data, per_enum_reg_loss

    NOTE: 성능은 **train set** 기준으로 리포트한다.
    """

    n_train = len(train_ds.class_ids)
    n_train_batch = max(1, math.ceil(n_train / config.batch_size))
    num_cls = config.decoder.num_reward_classes

    # steps_per_epoch 업데이트
    config.steps_per_epoch = n_train_batch

    mode = "text"
    if config.encoder.state:
        mode += "_state"
    config.encoder.mode = mode

    # norm stats → jnp array
    norm_min_arr = jnp.array(
        [cond_norm_min.get(i, 0.0) for i in range(num_cls)], dtype=jnp.float32
    )
    norm_max_arr = jnp.array(
        [cond_norm_max.get(i, 1.0) for i in range(num_cls)], dtype=jnp.float32
    )

    rng_key, init_key = jax.random.split(rng_key)
    train_state, lr_sched = get_train_state(
        config, init_key, cond_norm_min=norm_min_arr, cond_norm_max=norm_max_arr
    )


    # ── train_ds 기준 평가용 game_names ──
    train_game_names = np.array(
        [rc["game_name"] for rc in train_ds.reward_cond]
    )

    if n_train == 0:
        logger.warning("  ⚠ No training data for ratio=%.2f — skipping training", ratio)
        return {}, {}, {}, {}, {}

    # ── Training Loop ──
    n_train_batch = math.ceil(len(train_ds.class_ids) / config.batch_size)
    scatter_freq: int = int(getattr(config, "scatter_freq", 1000))  # scatter plot 업로드 주기
    max_pts: int = int(getattr(config, "n_max_points", 1000))

    # ── Unseen game 평가 서브셋 ──
    # unseen_eval_ds가 주입된 경우(전체 unseen pool 기반): 그대로 사용
    # 아닌 경우: test set에서 unseen 샘플만 필터링 (fallback)
    if unseen_eval_ds is not None and unseen_eval_game_names is not None and len(unseen_eval_ds.class_ids) > 0:
        unseen_test_ds = unseen_eval_ds
        unseen_test_game_names_arr = unseen_eval_game_names
        logger.info("  Unseen eval pool: %d samples (injected, eval_unseen_ratio applied)", len(unseen_test_ds.class_ids))
    elif unseen_game_names:
        _unseen_test_mask = np.array([g in unseen_game_names for g in test_game_names], dtype=bool)
        _unseen_test_indices = np.where(_unseen_test_mask)[0]
        if len(_unseen_test_indices) > 0:
            unseen_test_ds = subset_clip_dataset(test_ds, _unseen_test_indices)
            unseen_test_game_names_arr = test_game_names[_unseen_test_indices]
            logger.info("  Unseen eval pool: %d samples (test set fallback)", len(_unseen_test_indices))
        else:
            unseen_test_ds = None
            unseen_test_game_names_arr = np.array([])
    else:
        unseen_test_ds = None
        unseen_test_game_names_arr = np.array([])
    epoch_scatter_data: Dict[int, Dict[str, np.ndarray]] = {}
    epoch_per_game_acc: Dict[str, float] = {}
    epoch_per_game_reg: Dict[str, float] = {}
    epoch_per_game_enum_diff: Dict[str, Dict[int, float]] = {}
    epoch_per_enum_reg_loss: Dict[int, float] = {}
    train_class_ids = np.asarray(train_ds.class_ids).reshape(-1)
    class_id2game_name = {
        int(cid): rc.get("game_name", "unknown")
        for cid, rc in zip(train_class_ids, train_ds.reward_cond)
    }

    for epoch in range(config.n_epochs):
        rng_key, subkey = jax.random.split(rng_key)
        epoch_loss = 0.0
        epoch_acc = 0.0
        epoch_reg_loss = 0.0
        epoch_reg_loss_raw = 0.0
        epoch_cls_loss = 0.0
        epoch_contrastive_loss = 0.0
        epoch_s2t_top1 = 0.0
        epoch_t2s_top1 = 0.0
        epoch_s2t_correct_pr = 0.0
        epoch_t2s_correct_pr = 0.0
        epoch_temperature = 0.0
        epoch_per_enum_reg_raw: np.ndarray = np.zeros(num_cls)
        epoch_per_enum_huber_raw: np.ndarray = np.zeros(num_cls)
        epoch_per_enum_cnt: np.ndarray = np.zeros(num_cls)
        epoch_delta_loss = 0.0
        epoch_delta_pair_count = 0.0
        epoch_delta_loss_weighted = 0.0
        n_batches = 0
        epoch_reward_targets: List[int] = []
        epoch_reward_preds: List[int] = []
        epoch_reg_losses_norm: List[float] = []
        epoch_abs_diff: List[float] = []
        epoch_abs_diff_raw: List[float] = []
        epoch_reward_enums: List[int] = []
        epoch_game_names: List[str] = []
        epoch_pred_norm: List[float] = []
        epoch_target_norm: List[float] = []
        epoch_pred_raw: List[float] = []
        epoch_target_raw: List[float] = []
        batch_idx = 0

        with tqdm(total=n_train_batch, desc=f"Epoch {epoch + 1}/{config.n_epochs}", leave=False) as pbar:
            for batch in create_clip_decoder_batch(
                train_ds, config.batch_size, rng_key=subkey
            ):
                batch = jax.device_put(batch)
                train_state, loss, metrics, rng_key = train_step(
                    train_state,
                    batch,
                    rng_key=subkey,
                    is_train=True,
                    mode=mode,
                    contrastive_weight=config.contrastive_weight,
                    cls_weight=config.cls_weight,
                    reg_weight=config.reg_weight,
                    num_reward_classes=num_cls,
                    regression_loss=config.regression_loss,
                    norm_min_arr=norm_min_arr,
                    norm_max_arr=norm_max_arr,
                    delta_weight=float(getattr(config, "delta_weight", 0.0)),
                    num_games=int(num_games),
                    delta_min_count=int(getattr(config, "delta_min_group_samples", 2)),
                    delta_var_eps=float(getattr(config, "delta_var_eps", 1e-4)),
                    compute_delta_when_zero=bool(getattr(config, "compute_delta_when_zero", False)),
                )
                epoch_loss += float(loss)
                epoch_acc += float(metrics["reward_accuracy"])
                epoch_reg_loss += float(metrics["reg_loss"])
                epoch_reg_loss_raw += float(metrics["reg_loss_raw"])
                epoch_cls_loss += float(metrics["cls_loss"])
                epoch_contrastive_loss += float(metrics["contrastive_loss"])
                epoch_s2t_top1 += float(metrics["state2text_top1_accuracy"])
                epoch_t2s_top1 += float(metrics["text2state_top1_accuracy"])
                epoch_s2t_correct_pr += float(metrics["state2text_correct_pr"])
                epoch_t2s_correct_pr += float(metrics["text2state_correct_pr"])
                epoch_temperature += float(metrics["text_state_temperature"])
                batch_per_enum_raw = np.array(jax.device_get(metrics["per_enum_reg_loss_raw"]))
                batch_per_enum_raw_mae = np.array(jax.device_get(metrics["per_enum_reg_loss_raw_mae"]))
                batch_per_enum_cnt = np.array(jax.device_get(metrics["per_enum_count"]))
                epoch_per_enum_huber_raw += batch_per_enum_raw * batch_per_enum_cnt
                epoch_per_enum_reg_raw += batch_per_enum_raw_mae * batch_per_enum_cnt
                epoch_per_enum_cnt += batch_per_enum_cnt
                epoch_delta_loss += float(metrics["continuous_delta_loss"])
                epoch_delta_pair_count += float(metrics["valid_direction_pair_count"])
                epoch_delta_loss_weighted += float(metrics["continuous_delta_loss_weighted"])

                # ── Scatter / per-sample 집계 (실제 train 샘플만) ──
                actual_size = min(config.batch_size, max(0, n_train - batch_idx * config.batch_size))
                if actual_size > 0:
                    batch_reward_target = np.array(jax.device_get(batch.reward_enum_target))[:actual_size].astype(int).tolist()
                    batch_reward_pred = np.array(jax.device_get(metrics["reward_pred"]))[:actual_size].astype(int).tolist()
                    batch_class_ids = np.array(jax.device_get(batch.class_ids)).reshape(-1)[:actual_size].astype(int)
                    batch_reg_loss = float(jax.device_get(metrics["reg_loss"]))
                    batch_abs_diff = np.array(jax.device_get(metrics["abs_diff"]))[:actual_size].tolist()
                    batch_abs_diff_raw = np.array(jax.device_get(metrics["abs_diff_raw"]))[:actual_size].tolist()
                    batch_pred_norm = np.array(jax.device_get(metrics["per_sample_cond_norm"]))[:actual_size].tolist()
                    batch_target_norm = np.array(jax.device_get(metrics["per_sample_cond_target_norm"]))[:actual_size].tolist()
                    batch_pred_raw = np.array(jax.device_get(metrics["per_sample_cond_raw"]))[:actual_size].tolist()
                    batch_target_raw = np.array(jax.device_get(metrics["per_sample_cond_target_raw"]))[:actual_size].tolist()
                    batch_game_names = [
                        class_id2game_name.get(int(cid), "unknown")
                        for cid in batch_class_ids.tolist()
                    ]

                    epoch_reward_targets.extend(batch_reward_target)
                    epoch_reward_preds.extend(batch_reward_pred)
                    epoch_reg_losses_norm.extend([batch_reg_loss] * actual_size)
                    epoch_abs_diff.extend(batch_abs_diff)
                    epoch_abs_diff_raw.extend(batch_abs_diff_raw)
                    epoch_reward_enums.extend(batch_reward_target)
                    epoch_game_names.extend(batch_game_names)
                    epoch_pred_norm.extend(batch_pred_norm)
                    epoch_target_norm.extend(batch_target_norm)
                    epoch_pred_raw.extend(batch_pred_raw)
                    epoch_target_raw.extend(batch_target_raw)

                n_batches += 1
                batch_idx += 1
                pbar.update(1)
                pbar.set_postfix({"loss": f"{epoch_loss / n_batches:.4f}", "acc": f"{epoch_acc / n_batches:.3f}", "reg": f"{epoch_reg_loss_raw / n_batches:.4f}"})

        if n_batches > 0:
            epoch_loss /= n_batches
            epoch_acc /= n_batches
            epoch_reg_loss /= n_batches
            epoch_reg_loss_raw /= n_batches
            epoch_cls_loss /= n_batches
            epoch_contrastive_loss /= n_batches
            epoch_s2t_top1 /= n_batches
            epoch_t2s_top1 /= n_batches
            epoch_s2t_correct_pr /= n_batches
            epoch_t2s_correct_pr /= n_batches
            epoch_temperature /= n_batches
            epoch_delta_loss /= n_batches
            epoch_delta_pair_count /= n_batches
            epoch_delta_loss_weighted /= n_batches

        # ── 에폭 단위 scatter + epoch 기반 train-set 지표 ──
        epoch_scatter_data = _build_scatter_data_from_arrays(
            np.array(epoch_reward_enums, dtype=np.int64),
            np.array(epoch_pred_norm, dtype=np.float32),
            np.array(epoch_target_norm, dtype=np.float32),
            np.array(epoch_pred_raw, dtype=np.float32),
            np.array(epoch_target_raw, dtype=np.float32),
            np.array(epoch_game_names, dtype=object),
        )

        (
            epoch_per_game_acc,
            epoch_per_game_reg,
            epoch_per_game_enum_diff,
            epoch_per_enum_reg_loss,
        ) = _compute_train_set_metrics_from_arrays(
            np.array(epoch_reward_preds, dtype=np.int64),
            np.array(epoch_reward_targets, dtype=np.int64),
            np.array(epoch_reg_losses_norm, dtype=np.float32),
            np.array(epoch_abs_diff, dtype=np.float32),
            np.array(epoch_abs_diff_raw, dtype=np.float32),
            np.array(epoch_reward_enums, dtype=np.int64),
            train_game_names[: len(epoch_reward_targets)],
            unseen_game_names,
        )

        if (epoch + 1) % max(1, config.n_epochs // 5) == 0 or epoch == 0:
            logger.info(
                "  [ratio=%.2f] epoch %d/%d — loss: %.4f, train_acc: %.3f, reg: %.4f, cls: %.4f, contrastive: %.4f",
                ratio, epoch + 1, config.n_epochs, epoch_loss, epoch_acc,
                epoch_reg_loss_raw, epoch_cls_loss, epoch_contrastive_loss,
            )

        # ── W&B 스칼라 로깅 (매 에폭) ──
        if wandb.run is not None:
            selected_reg_per_enum = {}
            total_reg_cnt = 0.0
            total_reg_sum = 0.0
            for e in range(num_cls):
                if epoch_per_enum_cnt[e] > 0:
                    if config.regression_loss == "huber":
                        selected_reg_per_enum[e] = float(epoch_per_enum_huber_raw[e] / epoch_per_enum_cnt[e])
                    else:
                        selected_reg_per_enum[e] = float(epoch_per_enum_reg_raw[e] / epoch_per_enum_cnt[e])
                    total_reg_sum += selected_reg_per_enum[e] * float(epoch_per_enum_cnt[e])
                    total_reg_cnt += float(epoch_per_enum_cnt[e])
                else:
                    selected_reg_per_enum[e] = float(np.nan)

            overall_reg = float(total_reg_sum / total_reg_cnt) if total_reg_cnt > 0 else float(np.nan)
            wandb.log(
                {
                    "total/train_loss": epoch_loss,
                    "total/epoch": epoch,
                    "total/lr": lr_sched(train_state.step),

                    "train(text-state)/contrastive_loss": epoch_contrastive_loss,
                    "train(text-state)/state-text_temperature": epoch_temperature,
                    "train(text-state)/state2text_top1_accuracy": epoch_s2t_top1,
                    "train(text-state)/text2state_top1_accuracy": epoch_t2s_top1,
                    "train(text-state)/state2text_correct_pr": epoch_s2t_correct_pr,
                    "train(text-state)/text2state_correct_pr": epoch_t2s_correct_pr,

                    "train(decoder)/reward_accuracy": epoch_acc,
                    "train(decoder)/cls_loss": epoch_cls_loss,
                    "train(decoder)/reg_loss": epoch_reg_loss_raw,
                    "train(decoder)/reg_loss_normalized": epoch_reg_loss,
                    "train(direction)/continuous_delta_loss": epoch_delta_loss,
                    "train(direction)/continuous_delta_loss_weighted": epoch_delta_loss_weighted,
                    "train(direction)/valid_direction_pair_count": epoch_delta_pair_count,
                    **{
                        f"seen/regression/enum_{e}": selected_reg_per_enum[e]
                        for e in selected_reg_per_enum
                    },
                    "seen/regression/overall": overall_reg,
                }
            )

        # ── W&B Scatter plot 업로드 (scatter_freq 에폭마다, raw 공간만) ──
        if wandb.run is not None and scatter_freq > 0 and (epoch + 1) % scatter_freq == 0:
            scatter_data_mid = epoch_scatter_data
            regression_scatter_paths = create_regression_scatter_plots_per_enum(
                scatter_data_mid,
                out_dir=getattr(config, "exp_dir", "."),
                max_points=max_pts,
                seed=getattr(config, "seed", 0),
                space="raw",
            )
            epoch_imgs: Dict[str, object] = {}
            for e, path in regression_scatter_paths.items():
                epoch_imgs[f"seen/regression_scatter/enum_{int(e)}"] = wandb.Image(path)
            if epoch_imgs:
                epoch_imgs["total/epoch"] = epoch
                wandb.log(epoch_imgs)
                logger.info("  Scatter plot uploaded to wandb (epoch %d)", epoch + 1)

        # ── W&B Unseen 평가 (unseen_eval_freq / unseen_scatter_freq 에폭마다, test set 기반) ──
        # test set 기반이므로 unseen_ratio 설정과 무관하게 동작
        _unseen_eval_freq: int = int(getattr(config, "unseen_eval_freq", 100))
        _unseen_scatter_freq: int = int(getattr(config, "unseen_scatter_freq", 500))
        _do_unseen_eval = _unseen_eval_freq > 0 and (epoch + 1) % _unseen_eval_freq == 0
        _do_unseen_scatter = _unseen_scatter_freq > 0 and (epoch + 1) % _unseen_scatter_freq == 0

        if wandb.run is not None and unseen_test_ds is not None and (_do_unseen_eval or _do_unseen_scatter):
            rng_key, _eval_key = jax.random.split(rng_key)
            _, _, _, _unseen_scatter_data, _unseen_per_enum_reg = evaluate_per_game(
                train_state,
                unseen_test_ds,
                unseen_test_game_names_arr,
                unseen_game_names,
                config,
                _eval_key,
                num_cls,
                mode,
                norm_min_arr,
                norm_max_arr,
                num_games=num_games,
            )

            if _do_unseen_eval and _unseen_per_enum_reg:
                _unseen_overall_reg = float(np.mean(list(_unseen_per_enum_reg.values())))
                wandb.log({
                    **{f"unseen/regression/enum_{_e}": _v for _e, _v in _unseen_per_enum_reg.items()},
                    "unseen/regression/overall": _unseen_overall_reg,
                })

            if _do_unseen_scatter and _unseen_scatter_data:
                _unseen_scatter_paths = create_regression_scatter_plots_per_enum(
                    _unseen_scatter_data,
                    out_dir=getattr(config, "exp_dir", "."),
                    max_points=max_pts,
                    seed=getattr(config, "seed", 0),
                    space="raw",
                )
                _unseen_imgs: Dict[str, object] = {}
                for _e, _path in _unseen_scatter_paths.items():
                    _unseen_imgs[f"unseen/regression_scatter/enum_{int(_e)}"] = wandb.Image(_path)
                if _unseen_imgs:
                    _unseen_imgs["total/epoch"] = epoch
                    wandb.log(_unseen_imgs)
                    logger.info("  Unseen scatter plot uploaded to wandb (epoch %d)", epoch + 1)

        # ── Checkpoint 저장 ──
        if hasattr(config, 'ckpt_freq') and config.ckpt_freq > 0:
            if (epoch + 1) % config.ckpt_freq == 0:
                save_encoder_checkpoint(config, train_state, step=epoch + 1)

    per_game_acc = epoch_per_game_acc
    per_game_reg = epoch_per_game_reg
    per_game_enum_diff = epoch_per_game_enum_diff
    scatter_data = epoch_scatter_data
    per_enum_reg_loss = epoch_per_enum_reg_loss

    return per_game_acc, per_game_reg, per_game_enum_diff, scatter_data, per_enum_reg_loss, train_state



# ═══════════════════════════════════════════════════════════════════════════════
#  Post-encoder Adapter Training
# ═══════════════════════════════════════════════════════════════════════════════

def _post_train_adapter(config, train_state, full_dataset, unseen_game_set, train_indices):
    """encoder 학습 완료 후 decoder.adapter_type이 설정된 경우 adapter를 연속 학습한다.

    support = encoder가 실제 학습한 unseen 샘플 (train_indices 내 unseen 게임 샘플)
    query   = full_dataset의 나머지 모든 unseen 샘플

    adapter 저장 위치: {exp_dir}_adp-{type}/
      - adapter_state.pkl      : adapter 파라미터
      - dataset_setting.json   : unseen_support_source_ids 포함
      - norm_stats.json        : base encoder에서 복사
      - ckpts/                 : base encoder ckpts 심볼릭 링크
    """
    import shutil
    from collections import defaultdict
    from encoder.few_shot_adapter import adapt_embeddings

    _at = getattr(config.decoder, "adapter_type", None)
    if not _at or str(_at).lower() in ("none", "null"):
        return

    adp_ur = float(config.unseen_ratio)
    if adp_ur <= 0.0:
        logger.warning("Post-encoder adapter skipped: unseen_ratio=0.0 (adp_ur must be > 0)")
        return

    adapter_dir = f"{config.exp_dir}_adp-{_at}"
    adapter_state_path = os.path.join(adapter_dir, "adapter_state.pkl")

    if os.path.exists(adapter_state_path) and not getattr(config, "overwrite", False):
        logger.info("Adapter already exists, skipping: %s", adapter_state_path)
        return

    os.makedirs(adapter_dir, exist_ok=True)
    logger.info(
        "=== Post-encoder adapter training: type=%s  adp_ur=%.4f ===", _at, adp_ur
    )

    # ── 1. 전체 샘플 text embedding 추출 (batch inference) ──────────────────
    n = len(full_dataset.input_ids)
    batch_size = config.batch_size
    all_embeddings = []
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        out = train_state.apply_fn(
            train_state.params,
            full_dataset.input_ids[start:end],
            full_dataset.attention_masks[start:end],
            full_dataset.pixel_values[start:end],
            reward_enum=None,
            mode="text",
            training=False,
        )
        all_embeddings.append(np.asarray(out["text_embed"]))
    all_embeddings = np.concatenate(all_embeddings, axis=0)  # (N, D)
    logger.info("Extracted embeddings: shape=%s", all_embeddings.shape)

    # ── 2. encoder가 학습한 unseen 샘플 = support, 나머지 unseen = query ────
    all_game_names = np.array([rc["game_name"] for rc in full_dataset.reward_cond])
    train_set = set(train_indices)

    unseen_support_source_ids: dict = defaultdict(list)
    support_indices, query_indices = [], []

    for i, gname in enumerate(all_game_names):
        if gname not in unseen_game_set:
            continue
        if i in train_set:
            support_indices.append(i)
            unseen_support_source_ids[gname].append(full_dataset.reward_cond[i]["source_id"])
        else:
            query_indices.append(i)

    unseen_support_source_ids = dict(unseen_support_source_ids)

    for game in sorted(unseen_support_source_ids.keys()):
        n_sup = len(unseen_support_source_ids[game])
        n_qry = sum(1 for i in query_indices if all_game_names[i] == game)
        logger.info(
            "  game=%-12s  support=%d  query=%d",
            game, n_sup, n_qry,
        )

    if not support_indices:
        logger.error("No support samples — check unseen_game_set and unseen_ratio")
        return

    # ── 3. support GT 레이블 ────────────────────────────────────────────────
    num_classes = config.decoder.num_reward_classes
    sup_reward_i = np.array(
        [full_dataset.reward_cond[i]["reward_enum"] for i in support_indices], dtype=np.int32
    )
    sup_condition = np.full((len(support_indices), num_classes), -1.0, dtype=np.float32)
    for local_i, orig_i in enumerate(support_indices):
        rc = full_dataset.reward_cond[orig_i]
        enum_i = int(rc["reward_enum"])
        cval = rc.get("condition_value")
        if cval is not None:
            sup_condition[local_i, enum_i] = float(cval)

    query_arr = (
        jnp.array(all_embeddings[query_indices])
        if query_indices
        else jnp.array(all_embeddings[support_indices[:1]])
    )

    # ── 4. adapter 학습 ─────────────────────────────────────────────────────
    adapt_embeddings(
        query=query_arr,
        support=jnp.array(all_embeddings[support_indices]),
        support_reward_i=jnp.array(sup_reward_i),
        support_condition=jnp.array(sup_condition),
        apply_fn=train_state.apply_fn,
        variables=train_state.params,
        adapter_type=_at,
        rank=config.decoder.adapter_rank,
        num_experts=config.decoder.adapter_num_experts,
        lr=config.decoder.adapter_lr,
        steps=config.decoder.adapter_steps,
        save_path=adapter_state_path,
        load_path=None,
    )

    # ── 5. adapter encoder 형식 저장 ────────────────────────────────────────
    ckpts_link = os.path.join(adapter_dir, "ckpts")
    if not os.path.lexists(ckpts_link):
        base_ckpts = os.path.abspath(os.path.join(config.exp_dir, "ckpts"))
        if os.path.exists(base_ckpts):
            os.symlink(base_ckpts, ckpts_link)

    norm_src = os.path.join(config.exp_dir, "norm_stats.json")
    norm_dst = os.path.join(adapter_dir, "norm_stats.json")
    if os.path.exists(norm_src) and not os.path.exists(norm_dst):
        shutil.copy(norm_src, norm_dst)

    base_setting_path = os.path.join(config.exp_dir, "dataset_setting.json")
    adapter_setting: dict = {}
    if os.path.exists(base_setting_path):
        with open(base_setting_path) as _f:
            adapter_setting = json.load(_f)
    adapter_setting["unseen_ratio"] = adp_ur
    adapter_setting["unseen_support_source_ids"] = unseen_support_source_ids
    adapter_setting["adapter_type"] = _at
    adapter_setting["base_encoder_ckpt_name"] = os.path.basename(config.exp_dir)
    with open(os.path.join(adapter_dir, "dataset_setting.json"), "w") as _f:
        json.dump(adapter_setting, _f, indent=2, ensure_ascii=False)

    logger.info("=== Adapter encoder saved to %s ===", adapter_dir)


# ═══════════════════════════════════════════════════════════════════════════════
#  Visualization: Few-shot Ratio vs. Per-game Reward Accuracy
# ═══════════════════════════════════════════════════════════════════════════════

# ── 고정 게임별 색상 팔레트 (visualizer/plots.py와 동일) ─────────────────────
_GAME_SCATTER_COLORS = {
    "dungeon": "#4C72B0",
    "sokoban": "#DD8452",
    "zelda": "#55A868",
    "pokemon": "#C44E52",
    "doom": "#8172B3",
}


def _get_game_color(game: str, color_map: Optional[Dict[str, str]] = None, fallback_seed: int = 0) -> str:
    """게임별 고정 색상을 반환한다."""
    if color_map is None:
        color_map = _GAME_SCATTER_COLORS
    if game in color_map:
        return color_map[game]

    # 새 게임 이름은 순환 팔레트로 fallback
    fallback_colors = [
        "#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3",
        "#393b79", "#637939", "#8c6d31", "#843c39", "#8c564b",
    ]
    return fallback_colors[fallback_seed % len(fallback_colors)]


def _compute_scatter_trendline(
    x: np.ndarray,
    y: np.ndarray,
) -> Tuple[float, float, float]:
    """Return (r, slope, intercept) for finite (x, y) pairs.

    If not enough finite pairs exist, returns NaN values.
    """
    x_arr = np.asarray(x, dtype=float).reshape(-1)
    y_arr = np.asarray(y, dtype=float).reshape(-1)
    finite_mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_arr = x_arr[finite_mask]
    y_arr = y_arr[finite_mask]

    if x_arr.size < 2 or y_arr.size < 2:
        return float("nan"), float("nan"), float("nan")
    if np.std(x_arr) < 1e-12 or np.std(y_arr) < 1e-12:
        return float("nan"), float("nan"), float("nan")

    try:
        r = float(np.corrcoef(x_arr, y_arr)[0, 1])
        slope, intercept = np.polyfit(x_arr, y_arr, 1)
    except Exception:
        return float("nan"), float("nan"), float("nan")
    return r, float(slope), float(intercept)


def create_fewshot_plot(
    results: Dict[float, Dict[str, float]],
    reg_results: Dict[float, Dict[str, float]],
    unseen_game_names: Set[str],
    out_dir: str,
) -> str:
    """Few-shot ratio sweep 결과를 reg_loss 단일 패널로 시각화한다.

    Reward Accuracy는 wandb 스칼라로만 기록하고 이미지에는 포함하지 않는다.
    """
    os.makedirs(out_dir, exist_ok=True)

    ratios = sorted([r for r in results.keys() if r < 1.0])
    all_games = sorted(
        {g for r in reg_results.values() for g in r
         if g not in ("overall", "seen_overall", "unseen_overall")}
    )

    unseen_tag = ", ".join(sorted(unseen_game_names))

    fig, ax = plt.subplots(figsize=(3.8, 2.6))

    # ── Seen / Unseen (굵은 선, legend) ──
    seen_ov = [reg_results[r].get("seen_overall", float("nan")) for r in ratios]
    unseen_ov = [reg_results[r].get("unseen_overall", float("nan")) for r in ratios]
    ax.plot(ratios, seen_ov, marker="s", markersize=4, linewidth=2.4,
            linestyle="--", color="#b2182b", label="Seen")
    ax.plot(ratios, unseen_ov, marker="o", markersize=4, linewidth=2.4,
            linestyle="-", color="#2166ac", label="Unseen")

    ax.set_xlabel("Few-shot Ratio", fontsize=8)
    ax.set_ylabel("Regression Loss (Huber)", fontsize=8)
    ax.set_title(f"Unseen: {unseen_tag}", fontsize=8.5)
    ax.set_xlim(-0.02, 1.02)
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right", fontsize=6, framealpha=0.85)

    path = os.path.join(out_dir, "fewshot_ratio_vs_reward_accuracy.png")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    logger.info("Few-shot plot saved: %s", path)
    return path


def create_scatter_plots(
    scatter_data: Dict[int, Dict[str, np.ndarray]],
    out_dir: str,
    max_points: int = 1000,
    seed: int = 0,
    space: str = "norm",
    game_colors: Optional[Dict[str, str]] = None,
) -> Optional[str]:
    """Per reward_enum scatter plot (pred vs target).

    Parameters
    ----------
    scatter_data : evaluate_per_game()의 반환값
    max_points   : 서브플롯별 최대 점 개수 (초과 시 random sampling)
    space        : "norm" (정규화된 [0,1] 공간) or "raw" (linear 스케일)
    """
    if not scatter_data:
        logger.warning("create_scatter_plots: empty scatter_data — skipping")
        return None

    os.makedirs(out_dir, exist_ok=True)
    pred_key = f"pred_{space}"
    target_key = f"target_{space}"

    enums = sorted(scatter_data.keys())
    n = len(enums)
    ncols = min(3, n)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, squeeze=False)
    fig.set_size_inches(2.9 * ncols, 2.3 * nrows)

    color_map = {**_GAME_SCATTER_COLORS, **(game_colors or {})}
    rng = np.random.RandomState(seed)
    for i, e in enumerate(enums):
        ax = axes[i // ncols][i % ncols]
        pred = np.asarray(scatter_data[e][pred_key])
        target = np.asarray(scatter_data[e][target_key])
        game_names = np.asarray(scatter_data[e].get("game_names", np.array([])), dtype=object)

        n_pts = len(pred)
        if n_pts > max_points:
            sel = rng.choice(n_pts, size=max_points, replace=False)
            pred = pred[sel]
            target = target[sel]
            if len(game_names) > 0:
                game_names = game_names[sel]

        if len(game_names) == len(pred) and len(set(game_names)) > 1:
            for gi, gname in enumerate(sorted(set(game_names))):
                gmask = game_names == gname
                if not gmask.any():
                    continue
                ax.scatter(
                    target[gmask],
                    pred[gmask],
                    s=6, alpha=0.45, edgecolors="none",
                    color=_get_game_color(gname, color_map, gi),
                    label=gname, rasterized=True,
                )

            handles, labels = ax.get_legend_handles_labels()
            if labels:
                by_label = dict(zip(labels, handles))
                ax.legend(
                    by_label.values(),
                    by_label.keys(),
                    fontsize=6,
                    loc="upper right",
                    framealpha=0.8,
                )
        else:
            ax.scatter(target, pred, s=6, alpha=0.45, edgecolors="none", color="#2166ac")

        # y=x 기준선
        lo = float(min(target.min(), pred.min())) if len(pred) else 0.0
        hi = float(max(target.max(), pred.max())) if len(pred) else 1.0
        ax.plot([lo, hi], [lo, hi], linestyle="--", color="#888", linewidth=1)
        r, slope, intercept = _compute_scatter_trendline(target, pred)
        if np.isfinite(r):
            ax.plot(
                [lo, hi],
                [slope * lo + intercept, slope * hi + intercept],
                linestyle="-",
                color="#1b9e77",
                linewidth=1,
            )

        name = _REWARD_ENUM_NAMES.get(int(e), f"enum_{e}")
        mae = float(np.mean(np.abs(pred - target))) if len(pred) else float("nan")
        r_txt = f"{r:.4f}" if np.isfinite(r) else "nan"
        ax.set_title(f"{name}\nMAE={mae:.4f} | r={r_txt}", fontsize=8)
        ax.set_xlabel("target", fontsize=7)
        ax.set_ylabel("pred", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.grid(alpha=0.25)

    # 빈 subplot 숨김
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].set_visible(False)

    fig.suptitle(f"Train-set Prediction Scatter ({space} space)", fontsize=10)
    fig.tight_layout()
    path = os.path.join(out_dir, f"train_scatter_{space}.png")
    fig.savefig(path, dpi=220)
    plt.close(fig)
    logger.info("Scatter plot saved: %s", path)
    return path


def create_regression_scatter_plots_per_enum(
    scatter_data: Dict[int, Dict[str, np.ndarray]],
    out_dir: str,
    max_points: int = 1000,
    seed: int = 0,
    space: str = "raw",
    game_colors: Optional[Dict[str, str]] = None,
) -> Dict[int, str]:
    """Enum별 regression scatter (pred vs target) 이미지를 개별 저장해 경로를 반환한다.

    Returns
    -------
    {reward_enum: image_path}
    """
    if not scatter_data:
        logger.warning("create_regression_scatter_plots_per_enum: empty scatter_data — skipping")
        return {}

    os.makedirs(out_dir, exist_ok=True)
    pred_key = f"pred_{space}"
    target_key = f"target_{space}"

    color_map = {**_GAME_SCATTER_COLORS, **(game_colors or {})}
    pred_paths: Dict[int, str] = {}
    rng = np.random.RandomState(seed)
    for e in sorted(scatter_data.keys()):
        pred = np.asarray(scatter_data[e].get(pred_key, np.array([])))
        target = np.asarray(scatter_data[e].get(target_key, np.array([])))
        game_names = np.asarray(scatter_data[e].get("game_names", np.array([])), dtype=object)

        n_pts = len(pred)
        if n_pts == 0:
            continue

        if n_pts > max_points:
            sel = rng.choice(n_pts, size=max_points, replace=False)
            pred = pred[sel]
            target = target[sel]
            if len(game_names) > 0:
                game_names = game_names[sel]

        fig, ax = plt.subplots()
        fig.set_size_inches(2.9, 2.5)
        if len(game_names) == len(pred) and len(set(game_names)) > 1:
            for gi, gname in enumerate(sorted(set(game_names))):
                gmask = game_names == gname
                if not gmask.any():
                    continue
                ax.scatter(
                    target[gmask], pred[gmask],
                    s=6, alpha=0.45, edgecolors="none",
                    color=_get_game_color(gname, color_map, gi),
                    label=gname, rasterized=True,
                )
            handles, labels = ax.get_legend_handles_labels()
            if labels:
                by_label = dict(zip(labels, handles))
                ax.legend(
                    by_label.values(),
                    by_label.keys(),
                    fontsize=6,
                    loc="upper right",
                    framealpha=0.8,
                )
        else:
            ax.scatter(target, pred, s=6, alpha=0.45, edgecolors="none", color="#2166ac")

        lo = float(min(target.min(), pred.min()))
        hi = float(max(target.max(), pred.max()))
        ax.plot([lo, hi], [lo, hi], linestyle="--", color="#888", linewidth=1)
        r, slope, intercept = _compute_scatter_trendline(target, pred)
        if np.isfinite(r):
            ax.plot(
                [lo, hi],
                [slope * lo + intercept, slope * hi + intercept],
                linestyle="-",
                color="#1b9e77",
                linewidth=1,
            )

        name = _REWARD_ENUM_NAMES.get(int(e), f"enum_{e}")
        mae = float(np.mean(np.abs(pred - target)))
        r_txt = f"{r:.4f}" if np.isfinite(r) else "nan"
        ax.set_title(f"{name}\\nMAE={mae:.4f} | r={r_txt}", fontsize=8)
        ax.set_xlabel("target", fontsize=7)
        ax.set_ylabel("pred", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.grid(alpha=0.25)

        fig.tight_layout()
        path = os.path.join(out_dir, f"train_regression_scatter_{space}_enum_{int(e)}.png")
        fig.savefig(path, dpi=220)
        plt.close(fig)
        pred_paths[int(e)] = path

    if pred_paths:
        logger.info("Regression scatter plots (per enum, %s space) saved: %d", space, len(pred_paths))
    return pred_paths


# ═══════════════════════════════════════════════════════════════════════════════
#  Main Sweep
# ═══════════════════════════════════════════════════════════════════════════════

def make_train_unseen(config: CLIPDecoderTrainConfig):
    def train(rng_key):
        rng_key, subkey = jax.random.split(rng_key)

        # ── 1. 전체 데이터셋 빌드 (한 번만) ──
        dataset = build_multigame_dataset(config)
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

        dataset_builder = CLIPDatasetBuilder(
            processor=processor,
            paired_data=dataset,
            rng_key=subkey,
            max_len=config.encoder.token_max_len,
            train_ratio=1.0,  # 자체 split 수행 → 빌더의 split 사용 안 함
            max_samples=config.max_samples,
            instruction_prefix=config.instruction_prefix,
            longtail_cut=config.longtail_cut,
        )

        full_dataset = dataset_builder.get_dataset()
        cond_norm_min, cond_norm_max = dataset_builder.get_condition_norm_stats()

        # ── Save norm stats to ckpt directory (used for denorm during inference) ──
        save_norm_stats(config, cond_norm_min, cond_norm_max)

        # ── 2. Seen/Unseen 게임 파싱 ──
        unseen_game_set = parse_unseen_game_names(config.unseen_games)
        all_game_names = np.array(
            [rc["game_name"] for rc in full_dataset.reward_cond]
        )
        unique_games = sorted(set(all_game_names))
        seen_games = [g for g in unique_games if g not in unseen_game_set]
        unseen_games = [g for g in unique_games if g in unseen_game_set]

        logger.info("=" * 70)
        logger.info("  Seen/Unseen Split")
        logger.info("  Seen games  : %s", seen_games)
        logger.info("  Unseen games: %s", unseen_games)
        logger.info("  Seen ratio  : %.2f", config.seen_ratio)
        logger.info("  Total samples: %d", len(full_dataset.class_ids))
        logger.info("=" * 70)

        # ── N_SEEN_GAMES / N_UNSEEN_GAMES를 wandb.config에 기록 ──
        if wandb.run is not None:
            wandb.config.update(
                {
                    "N_SEEN_GAMES": len(seen_games),
                    "N_UNSEEN_GAMES": len(unseen_games),
                },
                allow_val_change=True,
            )

        # ── 데이터셋 설정 JSON 저장 ──
        os.makedirs(config.exp_dir, exist_ok=True)
        dataset_setting = {
            "all_games": unique_games,
            "seen_games": seen_games,
            "unseen_games": unseen_games,
            "unseen_ratio": config.unseen_ratio,
            "seen_ratio": config.seen_ratio,
        }
        dataset_setting_path = os.path.join(config.exp_dir, "dataset_setting.json")
        with open(dataset_setting_path, "w") as f:
            json.dump(dataset_setting, f, indent=2, ensure_ascii=False)
        logger.info("Dataset setting saved: %s", dataset_setting_path)
        
        # ── Encoder 학습 설정 JSON 저장 (pcgrl train/eval에서 참조용) ──
        encoder_config = {
            "delta_weight": config.delta_weight,
            "delta_min_group_samples": config.delta_min_group_samples,
            "delta_var_eps": config.delta_var_eps,
            "compute_delta_when_zero": config.compute_delta_when_zero,
            "batch_size": config.batch_size,
            "lr": config.lr,
            "weight_decay": config.weight_decay,
            "n_epochs": config.n_epochs,
            "train_ratio": config.train_ratio,
            "unseen_ratio": config.unseen_ratio,
            "seen_ratio": config.seen_ratio,
            "exp_name": config.exp_name,
            "seed": config.seed,
        }
        encoder_config_path = os.path.join(config.exp_dir, "encoder_config.json")
        with open(encoder_config_path, "w") as f:
            json.dump(encoder_config, f, indent=2, ensure_ascii=False)
        logger.info("Encoder config saved: %s", encoder_config_path)

        if not unseen_games:
            logger.warning("No unseen games found in dataset — treating all games as seen.")
            unseen_game_set = set()

        # ── 3. 게임별 train pool / test 분할 (seed 고정) ──
        game_train_pool, game_test, _ = split_dataset_by_game(
            full_dataset,
            unseen_game_set,
            test_ratio=1.0 - config.train_ratio,
            test_seed=config.split_seed,
        )

        # 고정 테스트 인덱스 (모든 게임)
        test_indices = np.concatenate(
            [game_test[g] for g in sorted(game_test.keys())]
        )
        test_ds = subset_clip_dataset(full_dataset, test_indices)
        test_game_names = np.array(
            [rc["game_name"] for rc in test_ds.reward_cond]
        )

        # 로깅: 분할 요약
        logger.info("  Test set (fixed, seed=%d):", config.split_seed)
        for g in sorted(game_test.keys()):
            tag = "(unseen)" if g in unseen_game_set else "(seen)"
            logger.info(
                "    %-12s %s  train_pool=%d, test=%d",
                g, tag, len(game_train_pool[g]), len(game_test[g]),
            )
        logger.info("  Total test: %d", len(test_indices))

        # ── Unseen 평가 풀: 전체 unseen game 데이터에서 eval_unseen_ratio만큼 샘플링 ──
        # unseen_ratio(학습)와 완전히 독립 — 전체 full_dataset에서 직접 샘플링
        _eval_unseen_ratio = float(getattr(config, "eval_unseen_ratio", 1.0))
        if unseen_game_set and _eval_unseen_ratio > 0.0:
            _all_unseen_mask = np.array([g in unseen_game_set for g in all_game_names], dtype=bool)
            _all_unseen_indices = np.where(_all_unseen_mask)[0]
            if _eval_unseen_ratio < 1.0:
                _eval_n = max(1, int(round(len(_all_unseen_indices) * _eval_unseen_ratio)))
                _eval_rng = np.random.RandomState(config.split_seed)
                _chosen = _eval_rng.choice(len(_all_unseen_indices), size=_eval_n, replace=False)
                _chosen.sort()
                _unseen_eval_indices = _all_unseen_indices[_chosen]
            else:
                _unseen_eval_indices = _all_unseen_indices
            unseen_eval_ds = subset_clip_dataset(full_dataset, _unseen_eval_indices)
            unseen_eval_game_names_arr = all_game_names[_unseen_eval_indices]
            logger.info(
                "  Unseen eval pool: %d / %d total unseen samples (eval_unseen_ratio=%.2f)",
                len(_unseen_eval_indices), len(_all_unseen_indices), _eval_unseen_ratio,
            )
        else:
            unseen_eval_ds = None
            unseen_eval_game_names_arr = np.array([])

        # ── 4. 단일 unseen_ratio 학습 ──
        ratio = config.unseen_ratio

        train_indices = build_train_indices_for_ratio(
            game_train_pool, unseen_game_set, ratio,
            seen_ratio=config.seen_ratio,
        )

        if len(train_indices) == 0:
            logger.warning(
                "ratio=%.2f: 0 training samples — evaluating untrained model",
                ratio,
            )
            train_ds = subset_clip_dataset(full_dataset, np.array([0]))
        else:
            train_ds = subset_clip_dataset(full_dataset, train_indices)

        _train_games = np.array(
            [rc["game_name"] for rc in train_ds.reward_cond]
        )
        _game_counts = {g: int(np.sum(_train_games == g)) for g in sorted(set(_train_games))}
        logger.info("  Train set = %d samples %s", len(train_indices), _game_counts)

        rng_key, ratio_key = jax.random.split(rng_key)
        per_game_acc, per_game_reg, per_game_enum_diff, scatter_data, per_enum_reg_loss, train_state = train_and_evaluate_ratio(
            config=config,
            rng_key=ratio_key,
            train_ds=train_ds,
            test_ds=test_ds,
            test_game_names=test_game_names,
            unseen_game_names=unseen_game_set,
            cond_norm_min=cond_norm_min,
            cond_norm_max=cond_norm_max,
            ratio=ratio,
            unseen_eval_ds=unseen_eval_ds,
            unseen_eval_game_names=unseen_eval_game_names_arr,
            num_games=len(unique_games),
        )

        _post_train_adapter(config, train_state, full_dataset, unseen_game_set, train_indices)

        # W&B 로깅 (unseen 로그 제거)

        # ── Scatter plots (최대 포인트 개수 제한) ──
        max_pts = int(getattr(config, "n_max_points", 1000))
        regression_scatter_paths = create_regression_scatter_plots_per_enum(
            scatter_data, out_dir=config.exp_dir,
            max_points=max_pts, seed=config.seed, space="raw",
        )
        if wandb.run is not None:
            wb_imgs = {}
            for e, path in regression_scatter_paths.items():
                wb_imgs[f"seen/regression_scatter/enum_{int(e)}"] = wandb.Image(path)
            if wb_imgs:
                wandb.log(wb_imgs)

        # ── 5. 결과 저장 ──
        save_data = {
            str(ratio): {
                "accuracy": per_game_acc,
                "reg_loss": per_game_reg,
                "per_enum_reg_loss": {str(k): v for k, v in per_enum_reg_loss.items()},
                "per_game_enum_diff": {
                    g: {str(k): v for k, v in d.items()}
                    for g, d in per_game_enum_diff.items()
                },
            }
        }
        results_path = os.path.join(config.exp_dir, "fewshot_results.json")
        with open(results_path, "w") as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)
        # ── 최종 요약 테이블 ──
        summary_games = [g for g in sorted(unique_games)] + ["overall", "seen_overall", "unseen_overall"]
        rows = [
            (g, f"{per_game_acc.get(g, float('nan')):.4f}", f"{per_game_reg.get(g, float('nan')):.4f}")
            for g in summary_games
        ]
        table_str = simple_table(rows, headers=["game", "train_acc", "train_reg_loss"])
        logger.info("── Per-game performance (train set) ──")
        for line in table_str.splitlines():
            logger.info(line)
        logger.info("Results saved: %s", results_path)


    return lambda rng_key: train(rng_key)


# ═══════════════════════════════════════════════════════════════════════════════
#  Entry Point
# ═══════════════════════════════════════════════════════════════════════════════

@hydra.main(version_base=None, config_path="./conf", config_name="train_clip_decoder")
def main(config: CLIPDecoderTrainConfig):
    if config.encoder.model is None:
        config.encoder.model = "cnnclip"
        logger.warning("encoder.model is None, using default value: cnnclip")

    config = init_config(config)

    rng_key = jax.random.PRNGKey(config.seed)
    np.random.seed(config.seed)

    from instruct_rl.utils.env_loader import get_wandb_key

    wandb_key = get_wandb_key()
    if wandb_key:
        dt = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        wandb_id = f"{get_wandb_name(config)}-{dt}"
        wandb.login(key=wandb_key)
        wandb.init(
            project=config.wandb_project,
            group=config.instruct,
            entity=config.wandb_entity,
            name=get_wandb_name(config),
            id=wandb_id,
            save_code=True,
        )
        wandb.config.update(dict(config), allow_val_change=True)

    exp_dir = config.exp_dir
    logger.info(f"jax devices: {jax.devices()}")
    logger.info(f"running experiment at {exp_dir}")

    if config.overwrite and os.path.exists(exp_dir):
        shutil.rmtree(exp_dir)

    os.makedirs(exp_dir, exist_ok=True)

    make_train_unseen(config)(rng_key)

    if wandb.run:
        wandb.finish()


if __name__ == "__main__":
    main()
