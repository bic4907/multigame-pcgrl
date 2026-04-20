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
from functools import partial
from os.path import basename
from collections import deque
from copy import deepcopy
from tqdm import tqdm
import wandb
from typing import Dict, List, Set, Tuple
import hydra
import logging
import shutil
import numpy as np
from functools import partial
import os
from flax.training import checkpoints
from flax.training.train_state import TrainState
from jax import jit
import jax
import jax.numpy as jnp
import optax
import matplotlib.pyplot as plt
import numpy as np
import optax
import wandb
from flax.training.train_state import TrainState
from jax import jit
from transformers import CLIPProcessor

from dataset.multigame import MultiGameDataset
from conf.config import CLIPDecoderUnseenConfig
from conf.game_utils import GAME_ABBR
from encoder.clip_model import get_cnnclip_decoder_encoder
from encoder.utils.training import save_encoder_checkpoint
from encoder.data.clip_batch import (
    CLIPDataset,
    CLIPDecoderBatch,
    CLIPDatasetBuilder,
    create_clip_decoder_batch,
)
from encoder.schedular import create_learning_rate_fn
from encoder.utils.path import init_config
from encoder.utils.training import build_multigame_dataset
from instruct_rl.utils.logger import get_wandb_name
from encoder.utils.path import get_ckpt_dir, init_config
from encoder.utils.training import build_multigame_dataset, save_encoder_checkpoint, setup_wandb
from encoder.data import CLIPDatasetBuilder, CLIPEmbedData, CLIPDataset
from encoder.data.clip_batch import create_clip_decoder_batch, CLIPDecoderBatch

from conf.config import CLIPDecoderTrainConfig

from encoder.utils.visualize import create_clip_embedding_figures
from encoder.utils.game_palette import palette_for_games

from transformers import CLIPProcessor

log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logger = logging.getLogger(basename(__file__))
logger.setLevel(getattr(logging, log_level, logging.INFO))
logging.getLogger("absl").setLevel(logging.ERROR)


# ═══════════════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def parse_unseen_game_names(unseen_str: str) -> Set[str]:
    """2글자 약어 문자열 → full game name set.

    Examples
    --------
    parse_unseen_game_names("zd")   -> {'zelda'}
    parse_unseen_game_names("pkzd") -> {'pokemon', 'zelda'}
    """
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

        a2b_top1_accuracy = jnp.mean(
            jnp.max(a2b_logps, axis=1) == jnp.max(a2b_pos_logps, axis=1)
        )
        b2a_top1_accuracy = jnp.mean(
            jnp.max(b2a_logps, axis=0) == jnp.max(b2a_pos_logps, axis=0)
        )

        return a2b_loss, b2a_loss, a2b_top1_accuracy, b2a_top1_accuracy

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
        s2t_loss, t2s_loss, s2t_top1, t2s_top1 = pairwise_contrastive_loss_accuracy(
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
        if regression_loss == "huber":
            reg_loss = jnp.mean(jnp.where(abs_diff <= 1.0, 0.5 * abs_diff ** 2, abs_diff - 0.5))
        else:  # mae
            reg_loss = jnp.mean(abs_diff)

        # normalized [0,1] 공간 MAE (모니터링용 — gradient 계산에 불포함)
        condition_pred_raw = outputs["condition_pred_raw"]   # (B, num_classes)
        per_sample_cond_raw = condition_pred_raw[jnp.arange(condition_pred_raw.shape[0]), reward_target]
        condition_mae_normalized = jnp.mean(abs_diff)

        # ── Per-reward_enum regression 메트릭 ──
        per_enum_huber = jnp.zeros(num_reward_classes)
        per_enum_mae = jnp.zeros(num_reward_classes)
        per_enum_count = jnp.zeros(num_reward_classes)

        mae_per_sample = abs_diff

        for eidx in range(num_reward_classes):
            mask = (reward_target == eidx).astype(jnp.float32)        # (B,)
            count = jnp.sum(mask) + 1e-8                               # 0-div 방지
            per_enum_huber = per_enum_huber.at[eidx].set(jnp.sum(mae_per_sample * mask) / count)
            per_enum_mae = per_enum_mae.at[eidx].set(jnp.sum(mae_per_sample * mask) / count)
            per_enum_count = per_enum_count.at[eidx].set(jnp.sum(mask))

        # ── Total Loss ──
        total_loss = (
            contrastive_weight * contrastive_loss
            + cls_weight * cls_loss
            + reg_weight * reg_loss
        )

        metrics = {
            "contrastive_loss": contrastive_loss,
            "cls_loss": cls_loss,
            "reg_loss": reg_loss,
            "reward_accuracy": reward_accuracy,
            "reward_pred": reward_pred,  # (B,) per-sample predictions
            "abs_diff": abs_diff,        # (B,) per-sample |pred_cond - target_cond|
            "state2text_top1_accuracy": s2t_top1 * state_mask,
            "text2state_top1_accuracy": t2s_top1 * state_mask,
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
            prepend_game_prefix=config.prepend_game_prefix,
            prepend_game_desc=config.prepend_game_desc,
            longtail_cut=config.longtail_cut,
        )

        train_clip_dataset, test_clip_dataset = dataset_builder.get_split_dataset()
        class_id2reward_cond = dataset_builder.get_class_id2reward_cond()
        cond_norm_min, cond_norm_max = dataset_builder.get_condition_norm_stats()

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
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, Dict[int, float]]]:
    """고정된 테스트셋에서 **게임별** reward accuracy 와 reg_loss를 계산한다.

    Returns
    -------
    per_game_acc      : {game: acc, "overall", "seen_overall", "unseen_overall"}
    per_game_reg_loss : {game: reg, "overall", "seen_overall", "unseen_overall"}
    per_game_enum_diff: {game: {reward_enum: mean_abs_diff}}
    """
    n_test = len(test_ds.input_ids)
    batch_size = config.batch_size

    all_preds: List[int] = []
    all_targets: List[int] = []
    all_reg_losses: List[float] = []  # per-sample reg loss
    all_abs_diffs: List[float] = []   # per-sample |pred_cond - target_cond|
    all_reward_enums: List[int] = []  # per-sample reward_enum target

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

        batch = CLIPDecoderBatch(
            class_ids=class_ids,
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            duplicate_matrix=duplicate_matrix,
            reward_enum_target=reward_enum_target,
            condition_target=condition_target,
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
        )

        preds = np.array(jax.device_get(metrics["reward_pred"]))
        targets = np.array(jax.device_get(reward_enum_target))
        batch_reg = float(jax.device_get(metrics["reg_loss"]))
        batch_abs_diff = np.array(jax.device_get(metrics["abs_diff"]))
        all_preds.extend(preds[:actual_size].tolist())
        all_targets.extend(targets[:actual_size].tolist())
        # batch-level reg_loss를 actual_size만큼 복제 (batch 평균이므로)
        all_reg_losses.extend([batch_reg] * actual_size)
        all_abs_diffs.extend(batch_abs_diff[:actual_size].tolist())
        all_reward_enums.extend(targets[:actual_size].tolist())

    # ── Per-game accuracy 집계 ──
    all_preds_arr = np.array(all_preds[:n_test])
    all_targets_arr = np.array(all_targets[:n_test])
    all_reg_arr = np.array(all_reg_losses[:n_test])
    all_abs_diff_arr = np.array(all_abs_diffs[:n_test])
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

    return per_game_acc, per_game_reg, per_game_enum_diff


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
    config: CLIPDecoderUnseenConfig,
    rng_key: jax.random.PRNGKey,
    train_ds: CLIPDataset,
    test_ds: CLIPDataset,
    test_game_names: np.ndarray,
    unseen_game_names: Set[str],
    cond_norm_min: dict,
    cond_norm_max: dict,
    ratio: float,
    ratio_idx: int,
    total_ratios: int,
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, Dict[int, float]]]:
    """하나의 few-shot ratio에 대해 모델을 처음부터 학습하고 평가한다.

    Returns (per_game_acc, per_game_reg_loss, per_game_enum_diff)
    """

    n_train = len(train_ds.class_ids)
    n_test = len(test_ds.class_ids)
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

    logger.info(
        "=" * 70 + "\n"
        f"  Ratio {ratio_idx + 1}/{total_ratios}: unseen_ratio={ratio:.2f}\n"
        f"  Train samples: {n_train}, Test samples: {n_test}\n"
        + "=" * 70
    )

    if n_train == 0:
        logger.warning("  ⚠ No training data for ratio=%.2f — skipping training", ratio)
        per_game_acc, per_game_reg, per_game_enum_diff = evaluate_per_game(
            train_state, test_ds, test_game_names, unseen_game_names,
            config, rng_key, num_cls, mode,
        )
        return per_game_acc, per_game_reg, per_game_enum_diff

    # ── Training Loop ──
    for epoch in range(config.n_epochs):
        rng_key, subkey = jax.random.split(rng_key)
        epoch_loss = 0.0
        epoch_acc = 0.0
        n_batches = 0

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
            )
            epoch_loss += float(loss)
            epoch_acc += float(metrics["reward_accuracy"])
            n_batches += 1

        if n_batches > 0:
            epoch_loss /= n_batches
            epoch_acc /= n_batches

        if (epoch + 1) % max(1, config.n_epochs // 5) == 0 or epoch == 0:
            logger.info(
                "  [ratio=%.2f] epoch %d/%d — loss: %.4f, train_acc: %.3f",
                ratio, epoch + 1, config.n_epochs, epoch_loss, epoch_acc,
            )

        # W&B 로깅 (per-epoch)
        if wandb.run is not None:
            wandb.log(
                {
                    f"ratio_{ratio:.2f}/epoch": epoch,
                    f"ratio_{ratio:.2f}/train_loss": epoch_loss,
                    f"ratio_{ratio:.2f}/train_reward_acc": epoch_acc,
                    f"ratio_{ratio:.2f}/lr": lr_sched(train_state.step),
                }
            )

        # ── Checkpoint 저장 ──
        if hasattr(config, 'ckpt_freq') and config.ckpt_freq > 0:
            if (epoch + 1) % config.ckpt_freq == 0:
                save_encoder_checkpoint(config, train_state, step=epoch + 1)

    # ── Evaluation ──
    per_game_acc, per_game_reg, per_game_enum_diff = evaluate_per_game(
        train_state, test_ds, test_game_names, unseen_game_names,
        config, rng_key, num_cls, mode,
    )

    logger.info("  [ratio=%.2f] Evaluation results:", ratio)
    for game in sorted(set(per_game_acc.keys()) | set(per_game_reg.keys())):
        acc = per_game_acc.get(game, float("nan"))
        reg = per_game_reg.get(game, float("nan"))
        logger.info("    %-12s  reward_acc = %.4f  reg_loss = %.4f", game, acc, reg)

    return per_game_acc, per_game_reg, per_game_enum_diff


# ═══════════════════════════════════════════════════════════════════════════════
#  Visualization: Few-shot Ratio vs. Per-game Reward Accuracy
# ═══════════════════════════════════════════════════════════════════════════════

# ── Seen/Unseen 색상 팔레트 ──────────────────────────────────────────────────
# seen 게임: 붉은색 계열,  unseen 게임: 푸른색 계열
_SEEN_COLORS = ["#e41a1c", "#ff7f00", "#d95f02", "#b2182b", "#cb181d"]
_UNSEEN_COLORS = ["#377eb8", "#4daf4a", "#6baed6", "#2171b5", "#08519c"]


def _game_color(game: str, unseen_game_names: Set[str], game_list: List[str]) -> str:
    """게임 이름 → 색상.  seen=red계열, unseen=blue계열."""
    seen_games = [g for g in game_list if g not in unseen_game_names]
    unseen_games = [g for g in game_list if g in unseen_game_names]
    if game in unseen_game_names:
        idx = unseen_games.index(game) % len(_UNSEEN_COLORS)
        return _UNSEEN_COLORS[idx]
    else:
        idx = seen_games.index(game) % len(_SEEN_COLORS)
        return _SEEN_COLORS[idx]


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


# ═══════════════════════════════════════════════════════════════════════════════
#  Main Sweep
# ═══════════════════════════════════════════════════════════════════════════════

def make_train_unseen(config: CLIPDecoderUnseenConfig):
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
            prepend_game_prefix=config.prepend_game_prefix,
            prepend_game_desc=config.prepend_game_desc,
        )

        full_dataset = dataset_builder.get_dataset()
        cond_norm_min, cond_norm_max = dataset_builder.get_condition_norm_stats()

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

        if not unseen_games:
            logger.warning("No unseen games found in dataset — treating all games as seen.")
            unseen_game_set = set()

        # ── 3. 게임별 train pool / test 분할 (seed 고정) ──
        game_train_pool, game_test, _ = split_dataset_by_game(
            full_dataset,
            unseen_game_set,
            test_ratio=config.unseen_test_ratio,
            test_seed=config.unseen_test_seed,
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
        logger.info("  Test set (fixed, seed=%d):", config.unseen_test_seed)
        for g in sorted(game_test.keys()):
            tag = "(unseen)" if g in unseen_game_set else "(seen)"
            logger.info(
                "    %-12s %s  train_pool=%d, test=%d",
                g, tag, len(game_train_pool[g]), len(game_test[g]),
            )
        logger.info("  Total test: %d", len(test_indices))

        # ── 4. Few-shot ratio sweep ──
        ratios = list(config.unseen_ratios)
        results: Dict[float, Dict[str, float]] = {}
        reg_results: Dict[float, Dict[str, float]] = {}
        enum_diff_results: Dict[float, Dict[str, Dict[int, float]]] = {}

        for ratio_idx, ratio in enumerate(ratios):
            # 학습 인덱스 구성
            train_indices = build_train_indices_for_ratio(
                game_train_pool, unseen_game_set, ratio,
                seen_ratio=config.seen_ratio,
            )

            if len(train_indices) == 0:
                logger.warning(
                    "ratio=%.2f: 0 training samples — evaluating untrained model",
                    ratio,
                )
                # seen game도 없는 특이 케이스: 최소 1개 샘플로 모델 초기화만 수행
                train_ds = subset_clip_dataset(full_dataset, np.array([0]))
            else:
                train_ds = subset_clip_dataset(full_dataset, train_indices)

            # train pool 구성 로그
            _train_games = np.array(
                [rc["game_name"] for rc in train_ds.reward_cond]
            )
            _game_counts = {g: int(np.sum(_train_games == g)) for g in sorted(set(_train_games))}
            logger.info(
                "  Ratio %.2f: train set = %d samples %s",
                ratio, len(train_indices), _game_counts,
            )

            rng_key, ratio_key = jax.random.split(rng_key)
            per_game_acc, per_game_reg, per_game_enum_diff = train_and_evaluate_ratio(
                config=config,
                rng_key=ratio_key,
                train_ds=train_ds,
                test_ds=test_ds,
                test_game_names=test_game_names,
                unseen_game_names=unseen_game_set,
                cond_norm_min=cond_norm_min,
                cond_norm_max=cond_norm_max,
                ratio=ratio,
                ratio_idx=ratio_idx,
                total_ratios=len(ratios),
            )
            results[ratio] = per_game_acc
            reg_results[ratio] = per_game_reg
            enum_diff_results[ratio] = per_game_enum_diff

            # W&B 로깅 (per-ratio 최종 결과 + incremental plot)
            if wandb.run is not None:
                log_dict = {"unseen/ratio": ratio, "unseen/seen_ratio": config.seen_ratio}
                for g, acc in per_game_acc.items():
                    log_dict[f"unseen/acc_{g}"] = acc
                for g, reg in per_game_reg.items():
                    log_dict[f"unseen/reg_{g}"] = reg
                wandb.log(log_dict)

        # ── 5. 결과 저장 ──
        save_data = {
            str(r): {"accuracy": results[r], "reg_loss": reg_results[r]}
            for r in results
        }
        results_path = os.path.join(config.exp_dir, "fewshot_results.json")
        with open(results_path, "w") as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)
        logger.info("Results saved: %s", results_path)

        # ── 6. 최종 그래프 생성 (결과 디렉토리에 저장) ──
        plot_path = create_fewshot_plot(
            results, reg_results, unseen_game_set, config.exp_dir
        )

        if wandb.run is not None:

            # reward_enum 전체 목록 수집
            all_enums = sorted({
                e for ratio_ed in enum_diff_results.values()
                for game_ed in ratio_ed.values()
                for e in game_ed.keys()
            })

            # 게임 이름 (overall 등 제외)
            game_names_sorted = sorted(
                {g for r in results.values() for g in r
                 if g not in ("overall", "seen_overall", "unseen_overall")}
            )

            # ratio=1.0 baseline diff 계산 (정규화 기준)
            baseline_ratio = 1.0
            baseline_seen_diff = None
            baseline_unseen_diff = None
            if baseline_ratio in enum_diff_results:
                _s, _u = [], []
                for g in game_names_sorted:
                    game_ed = enum_diff_results[baseline_ratio].get(g, {})
                    vals = [v for v in game_ed.values() if v is not None]
                    if vals:
                        avg = sum(vals) / len(vals)
                        if g in unseen_game_set:
                            _u.append(avg)
                        else:
                            _s.append(avg)
                baseline_seen_diff = sum(_s) / len(_s) if _s else None
                baseline_unseen_diff = sum(_u) / len(_u) if _u else None

            # 테이블: 각 행 = 하나의 ratio
            # norm_diff = raw_diff / baseline_diff (ratio=1.0 대비 상대 에러)
            columns = [
                "ratio", "seen_ratio", "game", "unseen_games",
                "seen_acc", "unseen_acc",
                "seen_reg", "unseen_reg",
                "seen_avg_diff", "unseen_avg_diff",
                "seen_norm_diff", "unseen_norm_diff",
            ]
            for g in game_names_sorted:
                for e in all_enums:
                    columns.append(f"{g}_enum_{e}")

            table = wandb.Table(columns=columns)
            for ratio_val in sorted(results.keys()):
                # seen/unseen 집계
                seen_acc = results[ratio_val].get("seen_overall", None)
                unseen_acc = results[ratio_val].get("unseen_overall", None)
                seen_reg = reg_results[ratio_val].get("seen_overall", None)
                unseen_reg = reg_results[ratio_val].get("unseen_overall", None)

                # 게임별 enum diff 평균 → seen/unseen raw avg diff
                seen_diffs, unseen_diffs = [], []
                for g in game_names_sorted:
                    game_ed = enum_diff_results[ratio_val].get(g, {})
                    vals = [v for v in game_ed.values() if v is not None]
                    if vals:
                        avg = sum(vals) / len(vals)
                        if g in unseen_game_set:
                            unseen_diffs.append(avg)
                        else:
                            seen_diffs.append(avg)
                seen_avg = sum(seen_diffs) / len(seen_diffs) if seen_diffs else None
                unseen_avg = sum(unseen_diffs) / len(unseen_diffs) if unseen_diffs else None

                # ratio=1.0 대비 정규화 (1.0 = baseline과 동일, >1.0 = 더 나쁨)
                seen_norm = (seen_avg / baseline_seen_diff
                             if seen_avg is not None and baseline_seen_diff else None)
                unseen_norm = (unseen_avg / baseline_unseen_diff
                               if unseen_avg is not None and baseline_unseen_diff else None)

                row = [
                    float(ratio_val), float(config.seen_ratio), config.game, config.unseen_games,
                    seen_acc, unseen_acc,
                    seen_reg, unseen_reg,
                    seen_avg, unseen_avg,
                    seen_norm, unseen_norm,
                ]
                for g in game_names_sorted:
                    game_ed = enum_diff_results[ratio_val].get(g, {})
                    for e in all_enums:
                        row.append(game_ed.get(e, None))
                table.add_data(*row)

            wandb.log({
                "table/results": table,
                "table/fewshot_plot": wandb.Image(plot_path),
            })

        # ── 최종 요약 출력 ──
        logger.info("\n" + "=" * 70)
        logger.info("  SWEEP COMPLETE — Summary")
        logger.info("=" * 70)
        header = f"  {'ratio':>6}"
        for g in sorted(unique_games):
            header += f"  {g:>10}"
        header += f"  {'overall':>10}"
        logger.info(header)
        logger.info("  " + "-" * (len(header) - 2))
        for ratio_val in sorted(results.keys()):
            row = f"  {ratio_val:>6.2f}"
            for g in sorted(unique_games):
                row += f"  {results[ratio_val].get(g, float('nan')):>10.4f}"
            row += f"  {results[ratio_val].get('overall', float('nan')):>10.4f}"
            logger.info(row)
        logger.info("=" * 70)

    return lambda rng_key: train(rng_key)


# ═══════════════════════════════════════════════════════════════════════════════
#  Entry Point
# ═══════════════════════════════════════════════════════════════════════════════

@hydra.main(version_base=None, config_path="./conf", config_name="train_clip_decoder_unseen")
def main(config: CLIPDecoderUnseenConfig):
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

