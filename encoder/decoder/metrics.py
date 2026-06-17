from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from flax.training.train_state import TrainState

from encoder.data.clip_batch import CLIPDataset, CLIPDecoderBatch
from .step import train_step


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
            regression_loss=config.regression_loss,
            norm_min_arr=norm_min_arr,
            norm_max_arr=norm_max_arr,
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
