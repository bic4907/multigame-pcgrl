"""
train_clip_decoder.py
=====================
CLIP Contrastive Encoder + Reward Decoder 학습 스크립트.

기존 contrastive loss (state↔text) 에 더해,
state embedding으로부터 reward_enum(분류) 과 condition(회귀) 을 예측하는
디코더 브랜치를 추가로 학습한다.

Usage
-----
    python train_clip_decoder.py game=dg
    python train_clip_decoder.py game=dgdm decoder.hidden_dim=256 cls_weight=2.0
"""

import datetime
import math
from os.path import basename
from collections import deque
from copy import deepcopy
from tqdm import tqdm
import wandb
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

from dataset.multigame import MultiGameDataset
from encoder.schedular import create_learning_rate_fn
from instruct_rl.utils.logger import get_wandb_name
from encoder.utils.path import get_ckpt_dir, init_config
from encoder.data import CLIPDatasetBuilder, CLIPEmbedData, CLIPDataset
from encoder.data.clip_batch import create_clip_decoder_batch, CLIPDecoderBatch

from conf.config import CLIPDecoderTrainConfig

from encoder.clip_model import get_cnnclip_decoder_encoder
from encoder.utils.visualize import create_clip_embedding_figures
from encoder.utils.game_palette import palette_for_games

from transformers import CLIPProcessor

log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
logger = logging.getLogger(basename(__file__))
logger.setLevel(getattr(logging, log_level, logging.INFO))
logging.getLogger('absl').setLevel(logging.ERROR)


# ═══════════════════════════════════════════════════════════════════════════════
#  Train Step (JIT)
# ═══════════════════════════════════════════════════════════════════════════════

@partial(jit, static_argnums=(3, 4, 5, 6, 7, 8))
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

        a2b_correct_pr = jnp.mean(jnp.sum(jnp.exp(a2b_logps) * batch.duplicate_matrix, axis=1))
        b2a_correct_pr = jnp.mean(jnp.sum(jnp.exp(b2a_logps) * batch.duplicate_matrix, axis=0))

        a2b_top1_accuracy = jnp.mean(jnp.max(a2b_logps, axis=1) == jnp.max(a2b_pos_logps, axis=1))
        b2a_top1_accuracy = jnp.mean(jnp.max(b2a_logps, axis=0) == jnp.max(b2a_pos_logps, axis=0))

        return a2b_loss, b2a_loss, a2b_correct_pr, b2a_correct_pr, a2b_top1_accuracy, b2a_top1_accuracy

    def loss_fn(params):
        outputs = train_state.apply_fn(
            params,
            batch.input_ids,
            batch.attention_mask,
            batch.pixel_values,
            mode=mode,
            training=is_train,
            rngs={"dropout": dropout_rng},
        )

        text_embed = outputs["text_embed"]
        state_embed = outputs.get("state_embed", jnp.zeros_like(text_embed))
        state_mask = jnp.any(state_embed != 0).astype(jnp.float32)
        text_state_temperature = outputs["text_state_temperature"]

        # ── (1) Contrastive Loss ──
        temperature = jnp.clip(text_state_temperature, jnp.log(0.01), jnp.log(100))
        s2t_loss, t2s_loss, s2t_pr, t2s_pr, s2t_top1, t2s_top1 = \
            pairwise_contrastive_loss_accuracy(state_embed, text_embed, temperature)

        contrastive_loss = state_mask * (s2t_loss + t2s_loss) / 2.0

        # ── (2) Decoder: reward_enum classification loss ──
        reward_logits = outputs["reward_logits"]     # (B, num_classes)
        reward_target = batch.reward_enum_target      # (B,) 0-indexed
        cls_loss = jnp.mean(
            optax.softmax_cross_entropy_with_integer_labels(reward_logits, reward_target)
        )
        reward_pred = jnp.argmax(reward_logits, axis=-1)
        reward_accuracy = jnp.mean(reward_pred == reward_target)

        # ── (3) Decoder: condition regression loss (Huber / Smooth L1) ──
        condition_pred = outputs["condition_pred"]    # (B, num_classes) — [0,1] 정규화
        condition_target = batch.condition_target      # (B,) — [0,1] 정규화
        # 각 샘플의 predicted condition을 gt reward_enum 인덱스로 gather
        per_sample_cond = condition_pred[jnp.arange(condition_pred.shape[0]), reward_target]
        # Huber loss (δ=1.0): outlier에 robust한 regression loss (정규화 공간)
        abs_diff = jnp.abs(per_sample_cond - condition_target)
        reg_loss = jnp.mean(
            jnp.where(abs_diff <= 1.0, 0.5 * abs_diff ** 2, abs_diff - 0.5)
        )

        # 원래 스케일 MAE (모니터링용 — gradient 계산에 불포함)
        condition_pred_raw = outputs["condition_pred_raw"]   # (B, num_classes) — 원래 스케일
        per_sample_cond_raw = condition_pred_raw[jnp.arange(condition_pred_raw.shape[0]), reward_target]
        condition_mae_normalized = jnp.mean(jnp.abs(per_sample_cond - condition_target))

        # ── Per-reward_enum regression 메트릭 ──
        per_enum_huber = jnp.zeros(num_reward_classes)
        per_enum_mae = jnp.zeros(num_reward_classes)
        per_enum_count = jnp.zeros(num_reward_classes)

        huber_per_sample = jnp.where(abs_diff <= 1.0, 0.5 * abs_diff ** 2, abs_diff - 0.5)
        mae_per_sample = jnp.abs(per_sample_cond - condition_target)

        for eidx in range(num_reward_classes):
            mask = (reward_target == eidx).astype(jnp.float32)        # (B,)
            count = jnp.sum(mask) + 1e-8                               # 0-div 방지
            per_enum_huber = per_enum_huber.at[eidx].set(jnp.sum(huber_per_sample * mask) / count)
            per_enum_mae = per_enum_mae.at[eidx].set(jnp.sum(mae_per_sample * mask) / count)
            per_enum_count = per_enum_count.at[eidx].set(jnp.sum(mask))

        # ── Total Loss ──
        total_loss = (
            contrastive_weight * contrastive_loss +
            cls_weight * cls_loss +
            reg_weight * reg_loss
        )

        metrics = {
            "state_embed": state_embed,
            "text_embed": text_embed,
            "text_state_temperature": text_state_temperature,
            "contrastive_loss": contrastive_loss,
            "state2text_loss": s2t_loss * state_mask,
            "text2state_loss": t2s_loss * state_mask,
            "state2text_correct_pr": s2t_pr * state_mask,
            "text2state_correct_pr": t2s_pr * state_mask,
            "state2text_top1_accuracy": s2t_top1 * state_mask,
            "text2state_top1_accuracy": t2s_top1 * state_mask,
            "cls_loss": cls_loss,
            "reward_accuracy": reward_accuracy,
            "reg_loss": reg_loss,
            "condition_mae_norm": condition_mae_normalized,
            "per_enum_huber": per_enum_huber,       # (num_classes,)
            "per_enum_mae": per_enum_mae,           # (num_classes,)
            "per_enum_count": per_enum_count,       # (num_classes,)
            "per_sample_cond_raw": per_sample_cond_raw,
        }

        return total_loss, metrics

    (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(train_state.params)
    train_state = jax.lax.cond(
        is_train,
        lambda _: train_state.apply_gradients(grads=grads),
        lambda _: train_state,
        operand=None,
    )
    return train_state, loss, metrics, rng_key


# ═══════════════════════════════════════════════════════════════════════════════
#  Training Loop
# ═══════════════════════════════════════════════════════════════════════════════

# reward_enum 이름 매핑 (1-based)
_REWARD_ENUM_NAMES = {
    1: "region",
    2: "path_length",
    3: "interactable_count",
    4: "hazard_count",
    5: "collectable_count",
    6: "item_count",
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
                f"Unique reward_enums (1-based): {sorted(all_enums)},  "
                f"num_reward_classes should be >= {max(all_enums) if all_enums else 0}")
    logger.info("=" * 80)
    logger.info("  ※ Condition values will be min-max normalized per reward_enum to [0, 1]")
    logger.info("=" * 80)

def make_train(config: CLIPDecoderTrainConfig):
    def train(rng_key):
        rng_key, subkey = jax.random.split(rng_key)
        dataset = MultiGameDataset(
            include_dungeon=config.include_dungeon,
            include_pokemon=config.include_pokemon,
            include_sokoban=config.include_sokoban,
            include_doom=config.include_doom,
            include_doom2=config.include_doom2,
            include_zelda=config.include_zelda,
        )

        # ── 학습 전 reward_enum / condition 범위 요약 출력 ──
        _log_reward_condition_summary(dataset)

        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        dataset_builder = CLIPDatasetBuilder(
            processor=processor,
            paired_data=dataset,
            rng_key=subkey,
            max_len=config.encoder.token_max_len,
            train_ratio=config.train_ratio,
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
            name = _REWARD_ENUM_NAMES.get(eidx + 1, f"unknown_{eidx+1}")
            r_min, r_max = cond_norm_min[eidx], cond_norm_max[eidx]
            logger.info(f"  {eidx:>10}  {name:<22} {r_min:>10.2f}  {r_max:>10.2f}  {'[0.0, 1.0]':>12}")
        logger.info("")

        # dry-run: 데이터 개수 제한
        if config.max_samples is not None:
            n = config.max_samples
            n_train_orig = len(train_clip_dataset.class_ids)
            n_test_orig = len(test_clip_dataset.class_ids)

            def _slice_dataset(ds, n):
                n = min(n, len(ds.class_ids))
                return CLIPDataset(
                    class_ids=ds.class_ids[:n],
                    reward_cond=ds.reward_cond[:n],
                    input_ids=ds.input_ids[:n],
                    attention_masks=ds.attention_masks[:n],
                    pixel_values=ds.pixel_values[:n],
                    is_train=ds.is_train[:n],
                    reward_enum_targets=ds.reward_enum_targets[:n],
                    condition_targets=ds.condition_targets[:n],
                )
            train_clip_dataset = _slice_dataset(train_clip_dataset, n)
            test_clip_dataset = _slice_dataset(test_clip_dataset, n)
            logger.info(f"[dry-run] max_samples={n} → "
                        f"train: {n_train_orig} → {len(train_clip_dataset.class_ids)}, "
                        f"test: {n_test_orig} → {len(test_clip_dataset.class_ids)}")

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

        for epoch in range(config.n_epochs):
            # ── 메트릭 초기화 ──
            num_cls = config.decoder.num_reward_classes
            train_losses = {k: jnp.zeros(()) for k in [
                "total", "contrastive", "state2text", "text2state", "cls", "reg"
            ]}
            train_metrics = {k: jnp.zeros(()) for k in [
                "state2text_correct_pr", "text2state_correct_pr",
                "state2text_top1_accuracy", "text2state_top1_accuracy",
                "text_state_temperature", "reward_accuracy",
                "condition_mae_norm",
            ]}
            train_per_enum_huber = jnp.zeros(num_cls)
            train_per_enum_mae = jnp.zeros(num_cls)
            train_per_enum_count = jnp.zeros(num_cls)

            val_losses = deepcopy(train_losses)
            val_metrics = deepcopy(train_metrics)
            val_per_enum_huber = jnp.zeros(num_cls)
            val_per_enum_mae = jnp.zeros(num_cls)
            val_per_enum_count = jnp.zeros(num_cls)

            # scatter plot 버퍼 (raw scale)
            val_true_raw_buf = []
            val_pred_raw_buf = []
            val_enum_buf = []
            val_game_buf = []

            i = 1

            with tqdm(total=n_train_batch + n_test_batch, desc=f"Epoch {epoch + 1}") as pbar:
                rng_key, subkey = jax.random.split(rng_key)

                # ── Training Loop ──
                for batch in create_clip_decoder_batch(train_clip_dataset, config.batch_size, rng_key=subkey):
                    class_ids = batch.class_ids
                    batch = jax.device_put(batch)

                    train_state, loss, metrics, rng_key = train_step(
                        train_state, batch, rng_key=subkey,
                        is_train=True, mode=config.encoder.mode,
                        contrastive_weight=config.contrastive_weight,
                        cls_weight=config.cls_weight,
                        reg_weight=config.reg_weight,
                        num_reward_classes=num_cls,
                    )

                    embeddings = [
                        CLIPEmbedData(class_ids=c, state_embeddings=st, text_embeddings=t)
                        for c, st, t in zip(class_ids, metrics["state_embed"], metrics["text_embed"])
                    ]
                    train_embed_queue.extend(embeddings)

                    train_losses["total"] += loss
                    train_losses["contrastive"] += metrics["contrastive_loss"]
                    train_losses["state2text"] += metrics["state2text_loss"]
                    train_losses["text2state"] += metrics["text2state_loss"]
                    train_losses["cls"] += metrics["cls_loss"]
                    train_losses["reg"] += metrics["reg_loss"]

                    train_metrics["state2text_correct_pr"] += metrics["state2text_correct_pr"]
                    train_metrics["text2state_correct_pr"] += metrics["text2state_correct_pr"]
                    train_metrics["state2text_top1_accuracy"] += metrics["state2text_top1_accuracy"]
                    train_metrics["text2state_top1_accuracy"] += metrics["text2state_top1_accuracy"]
                    train_metrics["text_state_temperature"] += metrics["text_state_temperature"]
                    train_metrics["reward_accuracy"] += metrics["reward_accuracy"]
                    train_metrics["condition_mae_norm"] += metrics["condition_mae_norm"]

                    # per-enum: count-weighted 누적 (나중에 count로 나눔)
                    train_per_enum_huber += metrics["per_enum_huber"] * metrics["per_enum_count"]
                    train_per_enum_mae += metrics["per_enum_mae"] * metrics["per_enum_count"]
                    train_per_enum_count += metrics["per_enum_count"]

                    pbar.update(1)
                    pbar.set_postfix({
                        "Train": f"{train_losses['total'] / i:.4f}",
                        "cls_acc": f"{train_metrics['reward_accuracy'] / i:.3f}",
                    })
                    i += 1

                train_losses = {k: float(v / n_train_batch) for k, v in train_losses.items()}
                train_metrics = {k: float(v / n_train_batch) for k, v in train_metrics.items()}
                train_per_enum_huber = train_per_enum_huber / (train_per_enum_count + 1e-8)
                train_per_enum_mae = train_per_enum_mae / (train_per_enum_count + 1e-8)

                # ── Validation Loop ──
                i = 1
                for batch in create_clip_decoder_batch(test_clip_dataset, config.batch_size, rng_key=subkey):
                    class_ids = batch.class_ids
                    batch = jax.device_put(batch)

                    _, loss, metrics, rng_key = train_step(
                        train_state, batch,
                        is_train=False, rng_key=subkey, mode=config.encoder.mode,
                        contrastive_weight=config.contrastive_weight,
                        cls_weight=config.cls_weight,
                        reg_weight=config.reg_weight,
                        num_reward_classes=num_cls,
                    )

                    embeddings = [
                        CLIPEmbedData(class_ids=c, state_embeddings=st, text_embeddings=t)
                        for c, st, t in zip(class_ids, metrics["state_embed"], metrics["text_embed"])
                    ]
                    val_embed_queue.extend(embeddings)

                    val_losses["total"] += loss
                    val_losses["contrastive"] += metrics["contrastive_loss"]
                    val_losses["state2text"] += metrics["state2text_loss"]
                    val_losses["text2state"] += metrics["text2state_loss"]
                    val_losses["cls"] += metrics["cls_loss"]
                    val_losses["reg"] += metrics["reg_loss"]

                    val_metrics["state2text_correct_pr"] += metrics["state2text_correct_pr"]
                    val_metrics["text2state_correct_pr"] += metrics["text2state_correct_pr"]
                    val_metrics["state2text_top1_accuracy"] += metrics["state2text_top1_accuracy"]
                    val_metrics["text2state_top1_accuracy"] += metrics["text2state_top1_accuracy"]
                    val_metrics["text_state_temperature"] += metrics["text_state_temperature"]
                    val_metrics["reward_accuracy"] += metrics["reward_accuracy"]
                    val_metrics["condition_mae_norm"] += metrics["condition_mae_norm"]

                    val_per_enum_huber += metrics["per_enum_huber"] * metrics["per_enum_count"]
                    val_per_enum_mae += metrics["per_enum_mae"] * metrics["per_enum_count"]
                    val_per_enum_count += metrics["per_enum_count"]

                    # scatter plot 버퍼 누적
                    reward_enum_1based = np.array(jax.device_get(batch.reward_enum_target)) + 1
                    pred_raw = np.array(jax.device_get(metrics["per_sample_cond_raw"]))
                    target_norm = np.array(jax.device_get(batch.condition_target))
                    target_raw = target_norm * (norm_max_arr[reward_enum_1based - 1] - norm_min_arr[reward_enum_1based - 1]) + norm_min_arr[reward_enum_1based - 1]
                    games = [class_id2game_name.get(int(cid), "unknown") for cid in np.array(jax.device_get(batch.class_ids))]

                    val_true_raw_buf.append(np.array(target_raw))
                    val_pred_raw_buf.append(pred_raw)
                    val_enum_buf.append(reward_enum_1based)
                    val_game_buf.extend(games)

                    pbar.update(1)
                    pbar.set_postfix({
                        "Train": f"{train_losses['total']:.4f}",
                        "Val": f"{val_losses['total'] / i:.4f}",
                    })
                    i += 1

                val_losses = {k: float(v / n_test_batch) for k, v in val_losses.items()}
                val_metrics = {k: float(v / n_test_batch) for k, v in val_metrics.items()}
                val_per_enum_huber = val_per_enum_huber / (val_per_enum_count + 1e-8)
                val_per_enum_mae = val_per_enum_mae / (val_per_enum_count + 1e-8)

            # ── Checkpoint ──
            if (epoch + 1) % config.ckpt_freq == 0:
                save_checkpoint(config, train_state, step=epoch + 1)

            # ── Embedding 시각화 ──
            if (epoch + 1) % config.embed_visualize_freq == 0:
                task_train_embed_paths = create_clip_embedding_figures(
                    train_embed_queue, class_id2reward_cond, epoch, config, postfix='_train')
                task_val_embed_paths = create_clip_embedding_figures(
                    val_embed_queue, class_id2reward_cond, epoch, config, postfix='_val')
                aux_dict = {
                    "train_tsne/embed_all": wandb.Image(task_train_embed_paths),
                    "val_tsne/embed_all": wandb.Image(task_val_embed_paths),
                }
            else:
                aux_dict = dict()

            # ── W&B Logging ──
            if wandb.run is not None:
                wandb.log({
                    # Total
                    "total/train_loss": train_losses["total"],
                    "total/val_loss": val_losses["total"],
                    "total/epoch": epoch,
                    "total/lr": lr_schedular(train_state.step),

                    # Contrastive
                    "train(contrastive)/loss": train_losses["contrastive"],
                    "train(contrastive)/temperature": train_metrics["text_state_temperature"],
                    "train(contrastive)/state2text_loss": train_losses["state2text"],
                    "train(contrastive)/text2state_loss": train_losses["text2state"],
                    "train(contrastive)/state2text_correct_pr": train_metrics["state2text_correct_pr"],
                    "train(contrastive)/text2state_correct_pr": train_metrics["text2state_correct_pr"],
                    "train(contrastive)/state2text_top1_accuracy": train_metrics["state2text_top1_accuracy"],
                    "train(contrastive)/text2state_top1_accuracy": train_metrics["text2state_top1_accuracy"],

                    "val(contrastive)/loss": val_losses["contrastive"],
                    "val(contrastive)/state2text_loss": val_losses["state2text"],
                    "val(contrastive)/text2state_loss": val_losses["text2state"],
                    "val(contrastive)/state2text_correct_pr": val_metrics["state2text_correct_pr"],
                    "val(contrastive)/text2state_correct_pr": val_metrics["text2state_correct_pr"],
                    "val(contrastive)/state2text_top1_accuracy": val_metrics["state2text_top1_accuracy"],
                    "val(contrastive)/text2state_top1_accuracy": val_metrics["text2state_top1_accuracy"],

                    # Decoder - classification
                    "train(decoder)/cls_loss": train_losses["cls"],
                    "train(decoder)/reward_accuracy": train_metrics["reward_accuracy"],
                    "val(decoder)/cls_loss": val_losses["cls"],
                    "val(decoder)/reward_accuracy": val_metrics["reward_accuracy"],

                    # Decoder - regression
                    "train(decoder)/reg_loss": train_losses["reg"],
                    "train(decoder)/condition_mae_norm": train_metrics["condition_mae_norm"],
                    "val(decoder)/reg_loss": val_losses["reg"],
                    "val(decoder)/condition_mae_norm": val_metrics["condition_mae_norm"],

                    **aux_dict,
                })

            # ── Condition Scatter Plots (reward_enum별) ──
            if (epoch + 1) % config.embed_visualize_freq == 0 and len(val_true_raw_buf) > 0:
                all_true_raw = np.concatenate(val_true_raw_buf, axis=0)
                all_pred_raw = np.concatenate(val_pred_raw_buf, axis=0)
                all_enum = np.concatenate(val_enum_buf, axis=0)
                scatter_dir = os.path.join(config.exp_dir, "scatter")
                scatter_paths = _create_condition_scatter_plots(
                    true_raw=all_true_raw,
                    pred_raw=all_pred_raw,
                    reward_enum_1based=all_enum,
                    game_names=val_game_buf,
                    num_reward_classes=num_cls,
                    epoch=epoch,
                    out_dir=scatter_dir,
                )
                if wandb.run is not None:
                    wandb.log({
                        **{f"val_scatter/enum_{i+1}": wandb.Image(p) for i, p in enumerate(scatter_paths)}
                    })

    return lambda rng_key: train(rng_key)


# ═══════════════════════════════════════════════════════════════════════════════
#  Model Init
# ═══════════════════════════════════════════════════════════════════════════════

def get_train_state(config: CLIPDecoderTrainConfig, rng_key: jax.random.PRNGKey,
                    cond_norm_min=None, cond_norm_max=None):
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

        if config.encoder.model == 'cnnclip':
            pixel_values = jnp.ones((1, 16, 16, config.clip_input_channel), dtype=jnp.float32)
        elif config.encoder.model == 'clip':
            pixel_values = jnp.ones((1, 224, 224, config.clip_input_channel), dtype=jnp.float32)
        else:
            raise NotImplementedError(f"Model not implemented: {config.encoder.model}")

        variables = module.init(
            init_rng, input_ids, attention_mask, pixel_values,
            mode=config.encoder.mode, training=False,
        )
        # variables = {"params": {...}, "norm_stats": {...}}

        if pretrained_params is not None:
            for key in pretrained_params:
                replace_params(variables, key, pretrained_params[key])

        # norm_stats는 optimizer 업데이트에서 제외 (학습 불가 상수)
        def _create_mask(variables):
            """params 내부는 True (학습), norm_stats 내부는 False (동결)."""
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

    # ContrastiveDecoderModule 생성
    module, pretrained_params = get_cnnclip_decoder_encoder(
        config.encoder,
        decoder_config=config.decoder,
        cond_norm_min=cond_norm_min,
        cond_norm_max=cond_norm_max,
        RL_training=False,
    )

    state = create_train_state(module, rng_key=rng_key, pretrained_params=pretrained_params)
    return state, lr_schedular


def save_checkpoint(config, state, step):
    ckpt_dir = get_ckpt_dir(config)
    ckpt_dir = os.path.abspath(ckpt_dir)
    checkpoints.save_checkpoint(ckpt_dir, target=state, prefix="", step=step, overwrite=True, keep=3)
    logger.info(f"Checkpoint saved at step {step}")


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

@hydra.main(version_base=None, config_path='./conf', config_name='train_clip_decoder')
def main(config: CLIPDecoderTrainConfig):
    if config.encoder.model is None:
        config.encoder.model = 'cnnclip'
        logger.warning("encoder.model is None, using default value: cnnclip")

    config = init_config(config)

    rng_key = jax.random.PRNGKey(config.seed)
    np.random.seed(config.seed)

    from instruct_rl.utils.env_loader import get_wandb_key
    wandb_key = get_wandb_key()
    if wandb_key:
        dt = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        wandb_id = f'{get_wandb_name(config)}-{dt}'
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
    logger.info(f'jax devices: {jax.devices()}')
    logger.info(f'running experiment at {exp_dir}')

    if config.overwrite and os.path.exists(exp_dir):
        shutil.rmtree(exp_dir)

    os.makedirs(exp_dir, exist_ok=True)

    make_train(config)(rng_key)


def _create_condition_scatter_plots(
    true_raw: np.ndarray,
    pred_raw: np.ndarray,
    reward_enum_1based: np.ndarray,
    game_names: list,
    num_reward_classes: int,
    epoch: int,
    out_dir: str,
):
    os.makedirs(out_dir, exist_ok=True)
    plot_paths = []

    uniq_games = sorted(set(game_names))
    game2color = palette_for_games(game_names)

    for enum_id in range(1, num_reward_classes + 1):
        mask = reward_enum_1based == enum_id
        x = true_raw[mask]
        y = pred_raw[mask]
        g = np.array(game_names)[mask] if len(game_names) > 0 else np.array([])

        fig, ax = plt.subplots(figsize=(5.2, 4.2))
        title = f"reward_enum={enum_id} ({_REWARD_ENUM_NAMES.get(enum_id, 'unknown')})"

        if len(x) == 0:
            ax.text(0.5, 0.5, "No samples", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(title)
        else:
            for game in uniq_games:
                gm = g == game
                if np.any(gm):
                    ax.scatter(x[gm], y[gm], s=12, alpha=0.65, color=game2color[game], label=game)

            lo = float(min(np.min(x), np.min(y)))
            hi = float(max(np.max(x), np.max(y)))
            ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.0, color="gray", label="y=x")

            if len(x) >= 2 and np.std(x) > 1e-8:
                m, b = np.polyfit(x, y, 1)
                xs = np.linspace(np.min(x), np.max(x), 100)
                ax.plot(xs, m * xs + b, color="black", linewidth=1.4, label=f"trend: y={m:.2f}x+{b:.2f}")

            if len(x) >= 2 and np.std(x) > 1e-8 and np.std(y) > 1e-8:
                r = float(np.corrcoef(x, y)[0, 1])
            else:
                r = float("nan")
            ax.text(0.03, 0.97, f"r = {r:.3f}\\nn = {len(x)}", va="top", ha="left", transform=ax.transAxes)

            ax.set_xlabel("True condition (raw)")
            ax.set_ylabel("Predicted condition (raw)")
            ax.set_title(title)
            ax.grid(alpha=0.2)
            ax.legend(loc="best", fontsize=8)

        path = os.path.join(out_dir, f"scatter_enum-{enum_id}_epoch-{epoch+1}.png")
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)
        plot_paths.append(path)

    return plot_paths

if __name__ == '__main__':
    main()

