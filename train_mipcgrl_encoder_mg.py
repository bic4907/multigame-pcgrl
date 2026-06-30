"""
train_mipcgrl_encoder_mg.py
===========================
Annotation 형식 멀티게임 데이터 기반 MIPCGRL MLP 인코더 사전학습.

train_ipcgrl_encoder_mg.py 와의 차이점:
  - 인코더 latent z 로부터 task(reward_enum) 분류 head 를 추가 학습.
  - 손실: Loss = MSE(condition) + classifier_weight * CrossEntropy(reward_enum)
  - 분류 head 는 ``apply_model`` 과 sibling 모듈로 두어 RL 측 encoder loader
    (``get_encoder_params_recursive(params, "encoder")``) 가 영향받지 않도록 한다.

데이터 파이프라인 / 체크포인트 포맷은 train_ipcgrl_encoder_mg 와 동일하므로
train_mipcgrl.py(또는 train_ipcgrl.py) 로 그대로 RL fine-tune 할 수 있다.

Usage:
    python train_mipcgrl_encoder_mg.py game=all
    python train_mipcgrl_encoder_mg.py game=all unseen_games=zd
    python train_mipcgrl_encoder_mg.py game=all classifier_weight=0.5 n_epochs=200
"""

from __future__ import annotations

import json
import logging
import math
import os
import shutil
from functools import partial
from os.path import basename

import flax.linen as nn
import hydra
import jax
import jax.numpy as jnp
import numpy as np
import optax
import wandb
from flax.training.train_state import TrainState
from jax import jit
from transformers import CLIPProcessor

from conf.config import MIPCGRLEncoderMGConfig
from conf.game_utils import parse_unseen_game_names
from encoder.data.mlp_batch import MLPDatasetBuilder, create_mlp_batches
from encoder.model import MLP, apply_model
from encoder.schedular import create_learning_rate_fn
from encoder.utils.path import init_config
from encoder.utils.training import build_multigame_dataset, save_encoder_checkpoint, setup_wandb

log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logger = logging.getLogger(basename(__file__))
logger.setLevel(getattr(logging, log_level, logging.INFO))
logging.getLogger("absl").setLevel(logging.ERROR)


# ── Model wrapper: apply_model + task classifier head ─────────────────────────

class MIPCGRLModel(nn.Module):
    """IPCGRL apply_model 을 감싸고, latent z 로부터 task 분류 head 를 덧붙인다.

    구조:
        - self.base       : 기존 apply_model (encoder + regression decoder)
        - self.classifier : latent z → task class logits 의 MLP head

    RL 측 encoder loader 는 params 트리에서 "encoder" key 를 재귀적으로 찾아
    self.base.encoder 만 추출하므로, classifier 파라미터는 자연히 무시된다.
    """

    config: MIPCGRLEncoderMGConfig
    num_classes: int
    classifier_num_layers: int = 2
    classifier_hidden_dim: int = 128
    classifier_dropout_rate: float = 0.0

    def setup(self) -> None:
        self.base = apply_model(config=self.config)
        self.classifier = MLP(
            num_layers=self.classifier_num_layers,
            hidden_size=self.classifier_hidden_dim,
            output_size=self.num_classes,
            dropout_rate=self.classifier_dropout_rate,
        )

    def __call__(self, x, rng, sampled_buffer=None, is_train=True):
        outputs = self.base(x, rng, sampled_buffer, is_train)
        outputs["class_logits"] = self.classifier(outputs["z"], train=is_train)
        return outputs


# ── Train State ───────────────────────────────────────────────────────────────

def get_train_state(config: MIPCGRLEncoderMGConfig, num_classes: int, rng: jax.random.PRNGKey):
    lr_fn = create_learning_rate_fn(config, config.lr, config.steps_per_epoch)

    model = MIPCGRLModel(
        config=config,
        num_classes=num_classes,
        classifier_num_layers=config.classifier_num_layers,
        classifier_hidden_dim=config.classifier_hidden_dim,
        classifier_dropout_rate=config.classifier_dropout_rate,
    )

    dummy_embed = jnp.ones((1, config.nlp_input_dim), dtype=jnp.float32)
    params = model.init(rng, dummy_embed, rng, None)

    tx = optax.adamw(learning_rate=lr_fn, weight_decay=config.weight_decay)
    state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)
    return state, lr_fn


# ── Train / Eval Step ─────────────────────────────────────────────────────────

def make_train_step(num_classes: int, cls_weight: float):
    cls_w = jnp.float32(cls_weight)

    @partial(jit, static_argnums=(5,))
    def train_step(
        state: TrainState,
        bert_embeds: jnp.ndarray,    # (B, nlp_input_dim)
        cond_targets: jnp.ndarray,   # (B,)
        class_targets: jnp.ndarray,  # (B,) int
        rng: jax.random.PRNGKey,
        is_train: bool,
    ):
        def loss_fn(params):
            outputs = state.apply_fn(
                params, bert_embeds, rng,
                None,
                is_train,
                rngs={"dropout": rng},
            )
            preds = outputs["logits"].squeeze(-1)
            mse = jnp.mean((preds - cond_targets) ** 2)

            class_logits = outputs["class_logits"]
            one_hot = jax.nn.one_hot(class_targets, num_classes)
            log_probs = jax.nn.log_softmax(class_logits, axis=-1)
            ce = -jnp.mean(jnp.sum(one_hot * log_probs, axis=-1))

            total = mse + cls_w * ce
            return total, (mse, ce, preds, class_logits)

        (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        mse_loss, ce_loss, preds, class_logits = aux

        if is_train:
            state = state.apply_gradients(grads=grads)

        return state, loss, mse_loss, ce_loss, preds, class_logits

    return train_step


# ── 메인 학습 루프 ─────────────────────────────────────────────────────────────

def make_train(config: MIPCGRLEncoderMGConfig):
    def train(rng: jax.random.PRNGKey):
        # 1. MultiGameDataset 로드
        multigame_ds = build_multigame_dataset(config)

        # 2. Unseen 게임 파싱
        unseen_game_set = parse_unseen_game_names(config.unseen_games) if config.unseen_games else set()
        logger.info("Unseen games (excluded from training): %s", unseen_game_set or "none")

        # 3. MLPDataset 빌드
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        rng, ds_key = jax.random.split(rng)
        builder = MLPDatasetBuilder(
            processor=processor,
            paired_data=multigame_ds,
            rng_key=ds_key,
            train_ratio=config.train_ratio,
            max_len=getattr(config.encoder, "token_max_len", 77),
            max_samples=config.max_samples,
            instruction_prefix=getattr(config, "instruction_prefix", "name"),
            longtail_cut=config.longtail_cut,
            tile_offset=getattr(config.encoder, "tile_offset", 0),
            exclude_games=unseen_game_set,
            nlp_input_dim=config.nlp_input_dim,
            unseen_ratio=config.unseen_ratio,
            seen_ratio=config.seen_ratio,
        )
        mlp_ds = builder.get_dataset()

        # ── num_classes 결정 ──
        unique_enums = sorted(set(int(v) for v in np.asarray(mlp_ds.reward_enum_targets).tolist()))
        if config.num_classes is None:
            # 0..max_enum 까지 모든 인덱스를 커버할 수 있도록 max+1 로 설정 (sparse 안전)
            num_classes = int(max(unique_enums) + 1)
        else:
            num_classes = int(config.num_classes)
        logger.info(
            "Task classifier: num_classes=%d (unique reward_enum=%s, classifier_weight=%.4f)",
            num_classes, unique_enums, config.classifier_weight,
        )

        # ── dataset_setting.json 저장 (IPCGRL/MIPCGRL RL 측 자동 주입에 사용) ──
        all_games = sorted(multigame_ds.count_by_game().keys())
        seen_games = [g for g in all_games if g not in unseen_game_set]
        unseen_games = [g for g in all_games if g in unseen_game_set]

        logger.info("=" * 70)
        logger.info("  MIPCGRL Seen/Unseen Split")
        logger.info("  Seen games   : %s", seen_games)
        logger.info("  Unseen games : %s", unseen_games)
        logger.info("  Seen ratio   : %.4f", config.seen_ratio)
        logger.info("  Unseen ratio : %.4f", config.unseen_ratio)
        logger.info("=" * 70)

        os.makedirs(config.exp_dir, exist_ok=True)
        dataset_setting = {
            "all_games": all_games,
            "seen_games": seen_games,
            "unseen_games": unseen_games,
            "unseen_ratio": config.unseen_ratio,
            "seen_ratio": config.seen_ratio,
            "num_classes": num_classes,
            "classifier_weight": config.classifier_weight,
        }
        with open(os.path.join(config.exp_dir, "dataset_setting.json"), "w") as f:
            json.dump(dataset_setting, f, indent=2, ensure_ascii=False)

        n_train = int(mlp_ds.is_train.sum())
        n_val = int((~mlp_ds.is_train).sum())
        logger.info("Train samples: %d, Val samples: %d", n_train, n_val)

        if n_train == 0:
            logger.error("No training samples — check game/unseen_games config.")
            return

        n_train_batch = max(1, math.ceil(n_train / config.batch_size))
        n_val_batch = max(1, math.ceil(n_val / config.batch_size)) if n_val > 0 else 0
        config.steps_per_epoch = n_train_batch

        # 4. Train state 초기화
        rng, init_key = jax.random.split(rng)
        state, lr_fn = get_train_state(config, num_classes, init_key)
        train_step = make_train_step(num_classes, float(config.classifier_weight))

        # 5. 학습 루프
        for epoch in range(config.n_epochs):
            rng, epoch_key = jax.random.split(rng)
            train_key, val_key = jax.random.split(epoch_key)

            train_losses, train_mses, train_ces, train_accs = [], [], [], []
            train_games_all, train_enums_all = [], []
            train_preds_all, train_targets_all = [], []

            for bert_emb, _, cond_t, g_names, re_t in create_mlp_batches(
                mlp_ds, config.batch_size, train=True, rng=train_key
            ):
                rng, step_key = jax.random.split(rng)
                state, loss, mse, ce, preds, class_logits = train_step(
                    state,
                    jax.device_put(bert_emb),
                    jax.device_put(cond_t),
                    jax.device_put(re_t.astype(np.int32)),
                    step_key,
                    True,
                )
                train_losses.append(float(loss))
                train_mses.append(float(mse))
                train_ces.append(float(ce))
                pred_cls = np.asarray(jnp.argmax(class_logits, axis=-1))
                train_accs.append(float((pred_cls == re_t).mean()))
                train_preds_all.append(np.array(preds))
                train_targets_all.append(cond_t)
                train_games_all.append(g_names)
                train_enums_all.append(re_t)

            val_losses, val_mses, val_ces, val_accs = [], [], [], []
            val_games_all, val_enums_all = [], []
            val_preds_all, val_targets_all = [], []

            for bert_emb, _, cond_t, g_names, re_t in create_mlp_batches(
                mlp_ds, config.batch_size, train=False, rng=val_key
            ):
                rng, step_key = jax.random.split(rng)
                _, loss, mse, ce, preds, class_logits = train_step(
                    state,
                    jax.device_put(bert_emb),
                    jax.device_put(cond_t),
                    jax.device_put(re_t.astype(np.int32)),
                    step_key,
                    False,
                )
                val_losses.append(float(loss))
                val_mses.append(float(mse))
                val_ces.append(float(ce))
                pred_cls = np.asarray(jnp.argmax(class_logits, axis=-1))
                val_accs.append(float((pred_cls == re_t).mean()))
                val_preds_all.append(np.array(preds))
                val_targets_all.append(cond_t)
                val_games_all.append(g_names)
                val_enums_all.append(re_t)

            train_loss = float(np.mean(train_losses)) if train_losses else 0.0
            train_mse = float(np.mean(train_mses)) if train_mses else 0.0
            train_ce = float(np.mean(train_ces)) if train_ces else 0.0
            train_acc = float(np.mean(train_accs)) if train_accs else 0.0
            val_loss = float(np.mean(val_losses)) if val_losses else float("nan")
            val_mse = float(np.mean(val_mses)) if val_mses else float("nan")
            val_ce = float(np.mean(val_ces)) if val_ces else float("nan")
            val_acc = float(np.mean(val_accs)) if val_accs else float("nan")

            per_game_val_mse = _per_game_mse(val_preds_all, val_targets_all, val_games_all)
            per_game_seen_mse, per_game_unseen_mse = _split_by_seen(per_game_val_mse, unseen_game_set)

            if (epoch + 1) % 10 == 0 or epoch == 0:
                logger.info(
                    "Epoch %3d/%d | train: total=%.4f mse=%.4f ce=%.4f acc=%.3f | "
                    "val: total=%.4f mse=%.4f ce=%.4f acc=%.3f | lr=%.2e",
                    epoch + 1, config.n_epochs,
                    train_loss, train_mse, train_ce, train_acc,
                    val_loss, val_mse, val_ce, val_acc,
                    float(lr_fn(state.step)),
                )
                for g, mse in sorted(per_game_val_mse.items()):
                    tag = "(unseen)" if g in unseen_game_set else "(seen)"
                    logger.info("  %-12s %s  val_mse=%.4f", g, tag, mse)

            if wandb.run is not None:
                log_dict = {
                    "train/loss": train_loss,
                    "train/mse": train_mse,
                    "train/ce": train_ce,
                    "train/acc": train_acc,
                    "val/loss": val_loss,
                    "val/mse": val_mse,
                    "val/ce": val_ce,
                    "val/acc": val_acc,
                    "train/lr": float(lr_fn(state.step)),
                    "epoch": epoch,
                }
                for g, mse in per_game_val_mse.items():
                    log_dict[f"val/mse_{g}"] = mse
                if per_game_seen_mse:
                    log_dict["val/mse_seen_overall"] = float(np.mean(list(per_game_seen_mse.values())))
                if per_game_unseen_mse:
                    log_dict["val/mse_unseen_overall"] = float(np.mean(list(per_game_unseen_mse.values())))
                wandb.log(log_dict)

            if (epoch + 1) % config.ckpt_freq == 0:
                save_encoder_checkpoint(config, state, step=epoch + 1)

        save_encoder_checkpoint(config, state, step=config.n_epochs)
        logger.info("Training complete. Checkpoint saved.")

    return lambda rng: train(rng)


# ── 보조 함수 ─────────────────────────────────────────────────────────────────

def _per_game_mse(preds_list, targets_list, games_list):
    if not preds_list:
        return {}
    all_preds = np.concatenate(preds_list)
    all_targets = np.concatenate(targets_list)
    all_games = np.concatenate(games_list)
    return {
        g: float(np.mean((all_preds[all_games == g] - all_targets[all_games == g]) ** 2))
        for g in sorted(set(all_games))
    }


def _split_by_seen(per_game, unseen):
    seen = {g: v for g, v in per_game.items() if g not in unseen}
    unseen_d = {g: v for g, v in per_game.items() if g in unseen}
    return seen, unseen_d


# ── Entry Point ───────────────────────────────────────────────────────────────

@hydra.main(version_base=None, config_path="./conf", config_name="train_mipcgrl_encoder_mg")
def main(config: MIPCGRLEncoderMGConfig):
    config = init_config(config)

    rng = jax.random.PRNGKey(config.seed)
    np.random.seed(config.seed)

    setup_wandb(config)

    exp_dir = config.exp_dir
    logger.info("jax devices: %s", jax.devices())
    logger.info("running experiment at %s", exp_dir)

    if config.overwrite and os.path.exists(exp_dir):
        shutil.rmtree(exp_dir)
    os.makedirs(exp_dir, exist_ok=True)

    make_train(config)(rng)

    if wandb.run:
        wandb.finish()


if __name__ == "__main__":
    main()
