from __future__ import annotations

import json
import logging
import math
import os
from collections import deque
from os.path import basename
from typing import Dict, List, Optional, Set, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import wandb
from tqdm import tqdm
from transformers import CLIPProcessor

from conf.config import CLIPDecoderTrainConfig
from encoder.data.clip_batch import CLIPDataset, CLIPDatasetBuilder, create_clip_decoder_batch
from encoder.data.split import (
    build_train_indices_for_ratio,
    parse_unseen_game_names,
    split_dataset_by_game,
    subset_clip_dataset,
)
from .common import _REWARD_ENUM_NAMES, _log_reward_condition_summary
from .metrics import (
    _build_scatter_data_from_arrays,
    _compute_train_set_metrics_from_arrays,
    evaluate_per_game,
)
from .state import get_train_state
from .step import train_step
from .visualization import (
    collect_tsne_embeddings,
    create_and_upload_tsne,
    create_regression_scatter_plots_per_enum,
)
from encoder.utils.training import build_multigame_dataset, save_encoder_checkpoint, save_norm_stats
from instruct_rl.utils.format_utils import simple_table

log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logger = logging.getLogger(basename(__file__))
logger.setLevel(getattr(logging, log_level, logging.INFO))


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

    # ── t-SNE 설정 ──────────────────────────────────────────────────────────
    tsne_freq:     int  = int(getattr(config, "tsne_freq", 0))        # 0 이면 비활성화
    tsne_samples:  int  = int(getattr(config, "tsne_samples", 1000))  # 샘플 개수
    if tsne_freq > 0:
        logger.info("  t-SNE enabled: freq=%d, samples=%d", tsne_freq, tsne_samples)

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

        # ── t-SNE 시각화 (tsne_freq 에폭마다) ───────────────────────────────
        if tsne_freq > 0 and (epoch + 1) % tsne_freq == 0:
            try:
                logger.info("  t-SNE 시작 (epoch %d)...", epoch + 1)
                # 임베딩 추출 (JAX JIT, 빠름)
                embed_data = collect_tsne_embeddings(
                    train_state=train_state,
                    dataset=train_ds,
                    game_names=train_game_names,
                    mode=mode,
                    n_samples=tsne_samples,
                    batch_size=config.batch_size,
                    seed=config.seed + epoch,
                )
                
                # t-SNE 계산 및 업로드 (동기 실행)
                has_state_mode = config.encoder.state
                create_and_upload_tsne(
                    text_embeds=embed_data["text_embed"],
                    state_embeds=embed_data["state_embed"],
                    game_names=embed_data["game_names"],
                    epoch=epoch + 1,
                    out_dir=config.exp_dir,
                    tag="train",
                    seed=config.seed + epoch,
                    has_state=has_state_mode,
                )
            except Exception as exc:
                logger.warning("  t-SNE 실패 (epoch %d): %s", epoch + 1, exc)

    per_game_acc = epoch_per_game_acc
    per_game_reg = epoch_per_game_reg
    per_game_enum_diff = epoch_per_game_enum_diff
    scatter_data = epoch_scatter_data
    per_enum_reg_loss = epoch_per_enum_reg_loss


    return per_game_acc, per_game_reg, per_game_enum_diff, scatter_data, per_enum_reg_loss


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
        per_game_acc, per_game_reg, per_game_enum_diff, scatter_data, per_enum_reg_loss = train_and_evaluate_ratio(
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
        )

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
