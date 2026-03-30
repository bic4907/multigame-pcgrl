import datetime
import math
from os.path import basename
from collections import deque
import jax.debug
import pandas as pd
from copy import deepcopy
from tqdm import tqdm
import wandb
import hydra
import logging
import shutil
import numpy as np
from functools import partial
from typing import Any
import os
from flax.training import checkpoints
from flax.training.train_state import TrainState
from flax.core import freeze
from jax import jit
import jax.numpy as jnp
from jax.experimental.array_serialization.serialization import logger
import optax

from dataset.multigame import MultiGameDataset
from encoder.schedular import create_learning_rate_fn
from instruct_rl.utils.logger import get_wandb_name
from encoder.utils.path import (get_ckpt_dir, init_config)
from encoder.data import create_dataset, split_dataset, create_batches, EmbedData, create_embedding_table

from conf.config import BertTrainConfig

from encoder.model import apply_model
from encoder.utils.visualize import create_scatter_plot, create_embedding_figure

log_level = os.getenv('LOG_LEVEL', 'INFO').upper()  # Add the environment variable ;LOG_LEVEL=DEBUG
logger = logging.getLogger(basename(__file__))
logger.setLevel(getattr(logging, log_level, logging.INFO))
logging.getLogger('absl').setLevel(logging.ERROR)


@partial(jit, static_argnums=(4, 5, 6))
def train_step(train_state: TrainState, X_batch, y_batch, rng, lr_rate_fn, is_train=True, use_kl_loss=False):
    # Define compute_loss as a closure to include fine_tune
    def compute_loss(params, X_batch, reward):
        def compute_kl_loss(mu, log_var):
            return -0.5 * jnp.sum(1 + log_var - jnp.square(mu) - jnp.exp(log_var))

        X_prev_env_map, X_curr_env_map, X_embedding = X_batch

        X_prev_env_map = jnp.expand_dims(X_prev_env_map, axis=0)
        X_curr_env_map = jnp.expand_dims(X_curr_env_map, axis=0)

        # Stack two maps
        X_env_map = jnp.concatenate([X_prev_env_map, X_curr_env_map], axis=3)

        X_embedding = jnp.expand_dims(X_embedding, axis=0)

        outputs = train_state.apply_fn(params, X_embedding, rng=rng,
                                       sampled_buffer=X_env_map, is_train=is_train,
                                       rngs={'dropout': rng})

        mse_loss = jnp.mean((outputs['logits'].squeeze() - reward) ** 2, dtype=jnp.float32)
        kl_loss = jax.lax.cond(
            use_kl_loss,
            lambda _: compute_kl_loss(outputs["mu"], outputs["log_var"]),
            lambda _: 0.0,
            operand=None,
        )
        total_loss = mse_loss + kl_loss * 0.1
        return total_loss, mse_loss, kl_loss, outputs['logits'].squeeze(), outputs['z'].squeeze()

    # Use vmap to vectorize over the batch dimension
    compute_loss_vectorized = jax.vmap(
        compute_loss,
        in_axes=(None, 0, 0),  # Vectorize over X_batch and y_batch
        out_axes=(0, 0, 0, 0, 0)  # Output three loss per batch
    )

    def compute_loss(params):
        total_loss, mse_loss, kl_loss, predictions, embed = compute_loss_vectorized(params, X_batch, y_batch)

        mean_total_loss = jnp.mean(total_loss)
        mean_mse_loss = jnp.mean(mse_loss)
        mean_kl_loss = jnp.mean(kl_loss)

        return mean_total_loss, (mean_mse_loss, mean_kl_loss, predictions, embed)

    (loss, aux), grads = jax.value_and_grad(compute_loss, has_aux=True)(
        train_state.params
    )
    mean_mse_loss, mean_kl_loss, predictions, embed = aux

    # Conditionally update the model parameters
    train_state = jax.lax.cond(
        is_train,
        lambda _: train_state.apply_gradients(grads=grads),
        lambda _: train_state,
        operand=None,
    )

    return train_state, (loss, mean_mse_loss, mean_kl_loss), predictions, embed


def make_train(config: BertTrainConfig):
    def train(rng):
        dataset = MultiGameDataset(include_dungeon=True,
                                   include_sokoban=False,
                                   include_pokemon=False,
                                   include_zelda=False,
                                   include_doom=False,
                                   include_doom2=False,
                                   include_boxoban=False)

        database = create_dataset(config.buffer_dir, dataset, config=config)
        reward_max, reward_min = np.max(database.reward), np.min(database.reward)
        logger.info(f"reward max: {reward_max}, reward min: {reward_min}")

        train_set, test_set = split_dataset(database, train_ratio=config.train_ratio)
        n_train = len(train_set.curr_map_obs)
        n_test = len(test_set.curr_map_obs)

        # Training loops
        n_train_batch = math.ceil(n_train / config.batch_size)
        n_test_batch = math.ceil(n_test / config.batch_size)

        config.steps_per_epoch = n_train_batch
        train_state, lr_schedular = get_train_state(config, rng)

        key = rng
        use_kl_loss = config.encoder.model == 'mlp_vae'

        logger.info("Start training model")

        train_embed_queue = deque(maxlen=config.n_max_points)
        val_embed_queue = deque(maxlen=config.n_max_points)

        for epoch in range(config.n_epochs):
            train_losses = {"total": jnp.zeros(()), "mse": jnp.zeros(()), "kl": jnp.zeros(())}
            val_losses = deepcopy(train_losses)

            i = 1

            with tqdm(total=n_train_batch + n_test_batch, desc=f"Epoch {epoch + 1}") as pbar:
                key, subkey = jax.random.split(key)
                train_y_gt, train_y_pd, train_reward_id = list(), list(), list()

                # Training Loop
                for X_batch, y_batch, reward_id in create_batches(train_set, config.batch_size, augment=True):
                    X_batch = jax.device_put(X_batch)
                    y_batch = jax.device_put(y_batch)

                    train_state, (batch_total_loss, batch_mse_loss, batch_kl_loss), predictions, embed = train_step(
                        train_state, X_batch, y_batch, rng=subkey, lr_rate_fn=lr_schedular, is_train=True,
                        use_kl_loss=use_kl_loss
                    )

                    embeddings = [EmbedData(reward_id=r, embedding=e) for r, e in zip(reward_id, embed)]
                    train_embed_queue.extend(embeddings)

                    train_losses["total"] += batch_total_loss
                    train_losses["mse"] += batch_mse_loss
                    train_losses["kl"] += batch_kl_loss

                    train_y_gt.extend(y_batch), train_y_pd.extend(predictions), train_reward_id.extend(reward_id)

                    pbar.update(1)  # Progress bar 업데이트
                    pbar.set_postfix({"Train Loss": train_losses['total'] / i, "Val Loss": val_losses['total']})
                    i += 1

                train_losses = {k: float(v / n_train_batch) for k, v in train_losses.items()}  # Training Loss 평균 계산

                # Validation Loop
                i = 1

                val_y_gt, val_y_pd, val_reward_id = list(), list(), list()
                for X_batch, y_batch, reward_id in create_batches(test_set, config.batch_size):
                    X_batch = jax.device_put(X_batch)
                    y_batch = jax.device_put(y_batch)

                    _, (batch_total_loss, batch_mse_loss, batch_kl_loss), predictions, embed = train_step(
                        train_state, X_batch, y_batch, rng=subkey, lr_rate_fn=lr_schedular, is_train=False,
                        use_kl_loss=use_kl_loss
                    )

                    embeddings = [EmbedData(reward_id=r, embedding=e) for r, e in zip(reward_id, embed)]
                    val_embed_queue.extend(embeddings)

                    val_losses["total"] += batch_total_loss
                    val_losses["mse"] += batch_mse_loss
                    val_losses["kl"] += batch_kl_loss

                    val_y_gt.extend(y_batch), val_y_pd.extend(predictions), val_reward_id.extend(reward_id)

                    pbar.update(1)  # Progress bar 업데이트
                    pbar.set_postfix({"Train Loss": train_losses['total'], "Val Loss": val_losses['total'] / i})
                    i += 1

                val_losses = {k: float(v / n_test_batch) for k, v in val_losses.items()}  # Validation Loss 평균 계산

            if (epoch + 1) % config.ckpt_freq == 0:
                save_checkpoint(config, train_state, step=epoch + 1)

            rand_train_idx = np.random.choice(len(train_y_gt), min(len(train_y_gt), 1000), replace=False)
            rand_val_idx = np.random.choice(len(val_y_gt), min(len(val_y_gt), 1000), replace=False)

            train_reward_ids = [e.reward_enum for e in train_embed_queue]
            val_reward_ids = [e.reward_enum for e in val_embed_queue]

            # Pandas DataFrame 생성
            df_train = pd.DataFrame({
                "epoch": epoch,
                "reward_id": train_reward_ids,
                "ground_truth": [float(train_y_gt[j]) for j in rand_train_idx],
                "prediction": [float(train_y_pd[j]) for j in rand_train_idx]
            })

            df_val = pd.DataFrame({
                "epoch": epoch,
                "reward_id": val_reward_ids,
                "ground_truth": [float(val_y_gt[j]) for j in rand_val_idx],
                "prediction": [float(val_y_pd[j]) for j in rand_val_idx]
            })

            settings = {'epoch': epoch, 'config': config, 'min_val': 0, 'max_val': config.n_epochs,
                        'xlim': (reward_min - 0.2, reward_max + 0.2), 'ylim': (reward_min - 0.2, reward_max + 0.2)
                        }
            train_fig_path = create_scatter_plot(df_train, postfix='_train', **settings)
            val_fig_path = create_scatter_plot(df_val, postfix='_val', **settings)

            train_table = wandb.Table(dataframe=df_train)
            val_table = wandb.Table(dataframe=df_val)

            if (epoch + 1) % config.embed_visualize_freq == 0:
                train_embed_path = create_embedding_figure(train_embed_queue, epoch, config,
                                                           postfix='_train')
                val_embed_path = create_embedding_figure(val_embed_queue, epoch, config, postfix='_val')

                aux_dict = {"train/embed": wandb.Image(train_embed_path),
                            "val/embed": wandb.Image(val_embed_path)}
            else:
                aux_dict = dict()

            if wandb.run is not None:
                wandb.log({
                    "train/loss": train_losses["total"],
                    "train/mse_loss": train_losses["mse"],
                    "train/kl_loss": train_losses["kl"],
                    "val/loss": val_losses["total"],
                    "val/mse_loss": val_losses["mse"],
                    "val/kl_loss": val_losses["kl"],
                    "train/prediction": train_table,
                    "val/prediction": val_table,
                    "train/result": wandb.Image(train_fig_path),
                    "val/result": wandb.Image(val_fig_path),
                    "epoch": epoch,
                    "train/lr": lr_schedular(train_state.step),
                    **aux_dict
                })

        train_table = create_embedding_table(train_embed_queue)
        val_table = create_embedding_table(val_embed_queue)

        if wandb.run is not None:
            wandb.log({
                "train/last_embed": train_table,
                "val/last_embed": val_table,
                "epoch": epoch
            })

    return lambda rng: train(rng)


def get_train_state(config: BertTrainConfig, rng: jax.random.PRNGKey):
    lr_schedular = create_learning_rate_fn(config, config.lr, config.steps_per_epoch)

    def create_train_state(model, rng, num_samples, buffer=None):
        params = model.init(rng, jnp.ones((num_samples, 768), dtype=jnp.float32), rng, buffer)
        tx = optax.adamw(learning_rate=lr_schedular, weight_decay=config.weight_decay)
        return TrainState.create(apply_fn=model.apply, params=params, tx=tx)

    model = apply_model(config=config)

    state = create_train_state(model,
                               rng=rng,
                               buffer=np.zeros((1, 31, 31, 5 * 2), dtype=np.float32),
                               num_samples=1,
                               )
    return state, lr_schedular


def save_checkpoint(config, state, step):
    ckpt_dir = get_ckpt_dir(config)
    ckpt_dir = os.path.abspath(ckpt_dir)
    checkpoints.save_checkpoint(ckpt_dir, target=state, prefix="", step=step, overwrite=True, keep=3)
    logger.info(f"Checkpoint saved at step {step}")


@hydra.main(version_base=None, config_path='./conf', config_name='train_reward')
def main(config: BertTrainConfig):
    if config.encoder.model is None:
        config.encoder.model = 'mlp'

    config.aug_type = 'test' if config.aug_type is None else config.aug_type
    config.embed_type = 'bert' if config.embed_type is None else config.embed_type
    config.instruct = 'scn-1_se-whole' if config.instruct is None else config.instruct

    config = init_config(config)

    rng = jax.random.PRNGKey(config.seed)
    np.random.seed(config.seed)

    if config.wandb_key:
        dt = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        wandb_id = f'{get_wandb_name(config)}-{dt}'
        wandb.login(key=config.wandb_key)
        wandb.init(
            project=config.wandb_project,
            group=config.instruct,
            entity=config.wandb_entity,
            name=get_wandb_name(config),
            id=wandb_id,
            save_code=True)
        wandb.config.update(dict(config), allow_val_change=True)

    exp_dir = config.exp_dir
    logger.info(f'jax devices: {jax.devices()}')
    logger.info(f'running experiment at {exp_dir}')

    # Need to do this before setting up checkpoint manager so that it doesn't refer to old checkpoints.
    if config.overwrite and os.path.exists(exp_dir):
        shutil.rmtree(exp_dir)

    os.makedirs(exp_dir, exist_ok=True)

    make_train(config)(rng)


if __name__ == '__main__':
    main()
    # wandb.finish()
