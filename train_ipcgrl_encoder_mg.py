"""
train_ipcgrl_encoder_mg.py
===========================
Pretrain the IPCGRL MLP encoder on annotation-format multigame data.

Differences from train_ipcgrl_encoder.py:
  - Data source: legacy NPZ buffer -> annotation CSV (MultiGameDataset)
  - Unified multigame encoder: one MLP regresses instruction -> condition for all games
  - unseen_games: exclude selected games for zero-shot evaluation

data pipeline:
    MultiGameDataset (annotation format)
        ↓ BERTDatasetBuilder
            - instruction validity filtering and long-tail cutoff
            - BERT CLS embedding compute (bert-base-uncased)
            - log1p + per-enum min-max condition normalize
            - stratified train/validation split (unseen games excluded from training)
        ↓ create_mlp_batches
    apply_model (MLP encoder + MLP decoder)
        - encoder: BERT_embed (768) → latent z (output_dim)
        - decoder: z → condition_value (1)
    MSE Loss

Usage:
    python train_ipcgrl_encoder_mg.py game=all
    python train_ipcgrl_encoder_mg.py game=all unseen_games=zd
    python train_ipcgrl_encoder_mg.py game=all unseen_games=pkzd n_epochs=200
"""

from __future__ import annotations

import datetime
import json
import logging
import math
import os
import shutil
from collections import defaultdict
from functools import partial
from os.path import basename

import hydra
import jax
import jax.numpy as jnp
import numpy as np
import optax
import wandb
from flax.training.train_state import TrainState
from jax import jit
from transformers import CLIPProcessor

from conf.config import IPCGRLEncoderMGConfig
from encoder.data.mlp_batch import MLPDatasetBuilder, create_mlp_batches
from encoder.model import apply_model
from encoder.schedular import create_learning_rate_fn
from encoder.utils.path import get_ckpt_dir, init_config
from encoder.utils.training import build_multigame_dataset, save_encoder_checkpoint, setup_wandb
from instruct_rl.utils.logger import get_wandb_name
from conf.game_utils import parse_unseen_game_names

log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logger = logging.getLogger(basename(__file__))
logger.setLevel(getattr(logging, log_level, logging.INFO))
logging.getLogger("absl").setLevel(logging.ERROR)


# ── Train State ───────────────────────────────────────────────────────────────

def get_train_state(config: IPCGRLEncoderMGConfig, rng: jax.random.PRNGKey):
    """apply_model (MLP encoder + decoder) TrainState initialize."""
    lr_fn = create_learning_rate_fn(config, config.lr, config.steps_per_epoch)

    model = apply_model(config=config)

    # sampled_buffer=None: use only BERT embeddings without level maps
    dummy_embed = jnp.ones((1, config.nlp_input_dim), dtype=jnp.float32)
    params = model.init(rng, dummy_embed, rng, None)

    tx = optax.adamw(learning_rate=lr_fn, weight_decay=config.weight_decay)
    state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)
    return state, lr_fn


# ── Train / Eval Step ─────────────────────────────────────────────────────────

@partial(jit, static_argnums=(4,))
def train_step(
    state: TrainState,
    bert_embeds: jnp.ndarray,   # (B, nlp_input_dim)
    cond_targets: jnp.ndarray,  # (B,)
    rng: jax.random.PRNGKey,
    is_train: bool,
):
    def loss_fn(params):
        outputs = state.apply_fn(
            params, bert_embeds, rng,
            None,       # sampled_buffer=None (level maps unused)
            is_train,
            rngs={"dropout": rng},
        )
        preds = outputs["logits"].squeeze(-1)  # (B,)
        mse = jnp.mean((preds - cond_targets) ** 2)
        return mse, preds

    (loss, preds), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)

    if is_train:
        state = state.apply_gradients(grads=grads)

    return state, loss, preds


# ── Main training loop ───────────────────────────────────────────────────────

def make_train(config: IPCGRLEncoderMGConfig):
    def train(rng: jax.random.PRNGKey):
        # 1. MultiGameDataset load
        multigame_ds = build_multigame_dataset(config)

        # 2. Unseen game parsing
        unseen_game_set = parse_unseen_game_names(config.unseen_games) if config.unseen_games else set()
        logger.info("Unseen games (excluded from training): %s", unseen_game_set or "none")

        # 3. BERTDataset build
        #    CLIPProcessor: CLIPDatasetBuilder tokenizer for preprocessing, unrelated to BERT
        #    BERT embeddings are computed separately by _compute_bert_embeddings.
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
            tile_offset=getattr(config.encoder, 'tile_offset', 0),
            exclude_games=unseen_game_set,
            nlp_input_dim=config.nlp_input_dim,
            unseen_ratio=config.unseen_ratio,
            seen_ratio=config.seen_ratio,
        )
        mlp_ds = builder.get_dataset()
        cond_norm_min, cond_norm_max = builder.get_condition_norm_stats()

        # ── Save dataset_setting.json for automatic injection by train_ipcgrl.py ──
        all_games = sorted(multigame_ds.count_by_game().keys())
        seen_games = [g for g in all_games if g not in unseen_game_set]
        unseen_games = [g for g in all_games if g in unseen_game_set]

        logger.info("=" * 70)
        logger.info("  Seen/Unseen Split (train_ipcgrl_encoder_mg)")
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
        }
        dataset_setting_path = os.path.join(config.exp_dir, "dataset_setting.json")
        with open(dataset_setting_path, "w") as f:
            json.dump(dataset_setting, f, indent=2, ensure_ascii=False)
        logger.info("Dataset setting saved: %s", dataset_setting_path)

        n_train = int(mlp_ds.is_train.sum())
        n_val = int((~mlp_ds.is_train).sum())
        logger.info("Train samples: %d, Val samples: %d", n_train, n_val)

        if n_train == 0:
            logger.error("No training samples — check game/unseen_games config.")
            return

        n_train_batch = max(1, math.ceil(n_train / config.batch_size))
        n_val_batch = max(1, math.ceil(n_val / config.batch_size)) if n_val > 0 else 0
        config.steps_per_epoch = n_train_batch

        # 4. Train state initialize
        rng, init_key = jax.random.split(rng)
        state, lr_fn = get_train_state(config, init_key)

        # 5. Training loop
        for epoch in range(config.n_epochs):
            rng, epoch_key = jax.random.split(rng)
            train_key, val_key = jax.random.split(epoch_key)

            # ── Train ──
            train_losses: list[float] = []
            train_preds_all: list[np.ndarray] = []
            train_targets_all: list[np.ndarray] = []
            train_games_all: list[np.ndarray] = []
            train_enums_all: list[np.ndarray] = []

            for bert_emb, _, cond_t, g_names, re_t in create_mlp_batches(
                mlp_ds, config.batch_size, train=True, rng=train_key
            ):
                rng, step_key = jax.random.split(rng)
                state, loss, preds = train_step(
                    state,
                    jax.device_put(bert_emb),
                    jax.device_put(cond_t),
                    step_key,
                    True,
                )
                train_losses.append(float(loss))
                train_preds_all.append(np.array(preds))
                train_targets_all.append(cond_t)
                train_games_all.append(g_names)
                train_enums_all.append(re_t)

            # ── Val ──
            val_losses: list[float] = []
            val_preds_all: list[np.ndarray] = []
            val_targets_all: list[np.ndarray] = []
            val_games_all: list[np.ndarray] = []
            val_enums_all: list[np.ndarray] = []

            for bert_emb, _, cond_t, g_names, re_t in create_mlp_batches(
                mlp_ds, config.batch_size, train=False, rng=val_key
            ):
                rng, step_key = jax.random.split(rng)
                _, loss, preds = train_step(
                    state,
                    jax.device_put(bert_emb),
                    jax.device_put(cond_t),
                    step_key,
                    False,
                )
                val_losses.append(float(loss))
                val_preds_all.append(np.array(preds))
                val_targets_all.append(cond_t)
                val_games_all.append(g_names)
                val_enums_all.append(re_t)

            # ── Aggregation ──
            train_loss = float(np.mean(train_losses)) if train_losses else 0.0
            val_loss = float(np.mean(val_losses)) if val_losses else float("nan")

            # Per-game validation MSE
            per_game_val_mse = _per_game_mse(
                val_preds_all, val_targets_all, val_games_all
            )
            per_game_seen_mse, per_game_unseen_mse = _split_by_seen(
                per_game_val_mse, unseen_game_set
            )

            # Log output
            if (epoch + 1) % 10 == 0 or epoch == 0:
                logger.info(
                    "Epoch %3d/%d | train_mse=%.4f | val_mse=%.4f | lr=%.2e",
                    epoch + 1, config.n_epochs,
                    train_loss, val_loss,
                    float(lr_fn(state.step)),
                )
                for g, mse in sorted(per_game_val_mse.items()):
                    tag = "(unseen)" if g in unseen_game_set else "(seen)"
                    logger.info("  %-12s %s  val_mse=%.4f", g, tag, mse)

            # W&B logging
            if wandb.run is not None:
                log_dict = {
                    "train/mse": train_loss,
                    "val/mse": val_loss,
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

            # checkpoint save
            if (epoch + 1) % config.ckpt_freq == 0:
                save_encoder_checkpoint(config, state, step=epoch + 1)

        # Final checkpoint after training
        save_encoder_checkpoint(config, state, step=config.n_epochs)
        logger.info("Training complete. Checkpoint saved.")

    return lambda rng: train(rng)


# ── Helper functions ─────────────────────────────────────────────────────────

def _per_game_mse(
    preds_list: list[np.ndarray],
    targets_list: list[np.ndarray],
    games_list: list[np.ndarray],
) -> dict[str, float]:
    """Convert batched prediction/target/game_name lists to per-game MSE values."""
    if not preds_list:
        return {}
    all_preds = np.concatenate(preds_list)
    all_targets = np.concatenate(targets_list)
    all_games = np.concatenate(games_list)
    result = {}
    for g in sorted(set(all_games)):
        mask = all_games == g
        result[g] = float(np.mean((all_preds[mask] - all_targets[mask]) ** 2))
    return result


def _split_by_seen(
    per_game: dict[str, float],
    unseen: set[str],
) -> tuple[dict[str, float], dict[str, float]]:
    seen = {g: v for g, v in per_game.items() if g not in unseen}
    unseen_d = {g: v for g, v in per_game.items() if g in unseen}
    return seen, unseen_d


# ── Entry Point ───────────────────────────────────────────────────────────────

@hydra.main(version_base=None, config_path="./conf", config_name="train_ipcgrl_encoder_mg")
def main(config: IPCGRLEncoderMGConfig):
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
