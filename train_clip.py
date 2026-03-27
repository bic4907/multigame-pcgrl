import datetime
import math
from os.path import basename, abspath
from collections import deque
# from typing import Dict
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
from encoder.data import CLIPDatasetBuilder, create_clip_batch, CLIPContrastiveBatch, CLIPEmbedData

from conf.config import CLIPTrainConfig

from encoder.clip_model import get_clip_encoder, get_cnnclip_encoder
from encoder.utils.visualize import create_clip_embedding_figures


from transformers import CLIPProcessor, FlaxCLIPModel

log_level = os.getenv('LOG_LEVEL', 'INFO').upper()  # Add the environment variable ;LOG_LEVEL=DEBUG
logger = logging.getLogger(basename(__file__))
logger.setLevel(getattr(logging, log_level, logging.INFO))
logging.getLogger('absl').setLevel(logging.ERROR)

@partial(jit, static_argnums=(3,4))
def train_step(train_state: TrainState, batch: CLIPContrastiveBatch, rng_key:jax.random.PRNGKey, is_train: bool=True, mode: str="text_state"):
    rng_key, dropout_rng = jax.random.split(rng_key)

    def pairwise_contrastive_loss_accuracy(a, b, temperature):
        logits = jnp.matmul(a, b.T) / jnp.exp(temperature)

        # Compute the log probabilities
        a2b_logps = jax.nn.log_softmax(logits, axis=1)
        b2a_logps = jax.nn.log_softmax(logits, axis=0)

        # collect the logprobs of the positve state-text pairs (semantic class)
        a2b_pos_logps = a2b_logps - 1e9 * (1 - batch.duplicate_matrix)
        b2a_pos_logps = b2a_logps - 1e9 * (1 - batch.duplicate_matrix)

        # Compute the loss
        a2b_loss = -jnp.mean(jax.scipy.special.logsumexp(a2b_pos_logps, axis=1))
        b2a_loss = -jnp.mean(jax.scipy.special.logsumexp(b2a_pos_logps, axis=0))

        # Average probability mass assigned to the correct pairs in the softmax distribution"
        a2b_correct_pr = jnp.mean(
            jnp.sum(jnp.exp(a2b_logps) * batch.duplicate_matrix, axis=1)
        )
        b2a_correct_pr = jnp.mean(
            jnp.sum(jnp.exp(b2a_logps) * batch.duplicate_matrix, axis=0)
        )

        # accuracy of retrieving one of the correct labels(top1 accuracy)
        a2b_top1_accuracy = jnp.mean(
            jnp.max(a2b_logps, axis=1) == jnp.max(a2b_pos_logps, axis=1)
        )
        b2a_top1_accuracy = jnp.mean(
            jnp.max(b2a_logps, axis=0) == jnp.max(b2a_pos_logps, axis=0)
        )

        return a2b_loss, b2a_loss, a2b_correct_pr, b2a_correct_pr, a2b_top1_accuracy, b2a_top1_accuracy

    def loss_fn(params):
        #Get the model outputs
        input_ids = batch.input_ids
        attention_mask = batch.attention_mask
        pixel_values = batch.pixel_values

        outputs = train_state.apply_fn(
            params,
            input_ids,
            attention_mask,
            pixel_values,
            mode=mode,
            training=is_train,
            rngs={"dropout": dropout_rng},
        )

        text_embed = outputs["text_embed"]
        
        text_state_temperature = outputs["text_state_temperature"]

        state_embed = outputs.get("state_embed", jnp.zeros_like(text_embed))
        state_mask = jnp.any(state_embed != 0).astype(jnp.float32)    # 0.0 or 1.0
        
        embed_pairs = {
            "state2text": (state_embed, text_embed, state_mask, text_state_temperature),
        }

        total_loss = 0.0
        total_pair_count = 0
        metrics = {
            "state_embed": state_embed,
            "text_embed": text_embed,
            "text_state_temperature": text_state_temperature,
        }

        # compute losses and metrics for all directions
        for name, (a, b, masking, temperature) in embed_pairs.items():
            temperature = jnp.clip(temperature, jnp.log(0.01), jnp.log(100))
            a2b_loss, b2a_loss, a2b_pr, b2a_pr, a2b_top1, b2a_top1 = pairwise_contrastive_loss_accuracy(
                a, b, temperature)

            total_loss += masking * (a2b_loss + b2a_loss)
            total_pair_count += 2 * masking  # static count

            name_a, name_b = name.split("2")
            metrics[f"{name}_loss"] = a2b_loss * masking
            metrics[f"{name_b}2{name_a}_loss"] = b2a_loss * masking

            metrics[f"{name}_correct_pr"] = a2b_pr * masking
            metrics[f"{name_b}2{name_a}_correct_pr"] = b2a_pr * masking

            metrics[f"{name}_top1_accuracy"] = a2b_top1 * masking
            metrics[f"{name_b}2{name_a}_top1_accuracy"] = b2a_top1 * masking

        # final average loss over 6 directions
        loss = total_loss / total_pair_count
        metrics["total_loss"] = loss

        return loss, metrics
    
    (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(train_state.params)
    train_state = jax.lax.cond(
        is_train,
        lambda _: train_state.apply_gradients(grads=grads),
        lambda _: train_state,
        operand=None,
    )
    return train_state, loss, metrics, rng_key


def make_train(config: CLIPTrainConfig):
    def train(rng_key):
        rng_key, subkey = jax.random.split(rng_key)
        dataset = MultiGameDataset()

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

        n_train = len(train_clip_dataset.class_ids)
        n_test = len(test_clip_dataset.class_ids)

        # Training loops
        n_train_batch = math.ceil(n_train / config.batch_size)
        n_test_batch = math.ceil(n_test / config.batch_size)

        config.steps_per_epoch = n_train_batch

        mode = "text"

        if config.encoder.state:
            mode += "_state"

        config.encoder.mode = mode

        train_state, lr_schedular = get_train_state(config, subkey)


        logger.info("Start training model")

        train_embed_queue = deque(maxlen=config.n_max_points)
        val_embed_queue = deque(maxlen=config.n_max_points)

        for epoch in range(config.n_epochs):
            train_losses = {
                "total": jnp.zeros(()),
                "state2text": jnp.zeros(()),
                "text2state": jnp.zeros(()),
            }
            train_metrics = {
                "state2text_correct_pr": jnp.zeros(()),
                "text2state_correct_pr": jnp.zeros(()),
                "state2text_top1_accuracy": jnp.zeros(()),
                "text2state_top1_accuracy": jnp.zeros(()),
                "text_state_temperature": jnp.zeros(()),
            }
            val_losses = deepcopy(train_losses)
            val_metrics = deepcopy(train_metrics)
            
            i = 1

            with tqdm(total=n_train_batch + n_test_batch, desc=f"Epoch {epoch + 1}") as pbar:
                rng_key, subkey = jax.random.split(rng_key)
                
                # Training Loop
                for clip_batch_data in create_clip_batch(train_clip_dataset, config.batch_size, rng_key=subkey):
                    class_ids = clip_batch_data.class_ids
                    clip_batch_data = jax.device_put(clip_batch_data)
                    
                    train_state, loss, metrics, rng_key = train_step(
                        train_state, 
                        clip_batch_data,
                        rng_key=subkey,
                        is_train=True,
                        mode=config.encoder.mode
                    )
                    
                    state_embed = metrics["state_embed"]
                    text_embed = metrics["text_embed"]

                    embeddings = [CLIPEmbedData(class_ids=c, state_embeddings=st, text_embeddings=t) for c, st, t in zip(class_ids, state_embed, text_embed)]
                    train_embed_queue.extend(embeddings)

                    train_losses["total"] += loss
                    train_losses["state2text"] += metrics["state2text_loss"]
                    train_losses["text2state"] += metrics["text2state_loss"]
                    
                    train_metrics["state2text_correct_pr"] += metrics["state2text_correct_pr"]
                    train_metrics["text2state_correct_pr"] += metrics["text2state_correct_pr"]
                    train_metrics["state2text_top1_accuracy"] += metrics["state2text_top1_accuracy"]
                    train_metrics["text2state_top1_accuracy"] += metrics["text2state_top1_accuracy"]

                    train_metrics["text_state_temperature"] += metrics["text_state_temperature"]

                    pbar.update(1)  # update Progress bar
                    pbar.set_postfix({"Train Loss": train_losses['total'] / i, "Val Loss": val_losses['total']})
                    i += 1

                train_losses = {k: float(v / n_train_batch) for k, v in train_losses.items()}  # calculate average Training Loss
                train_metrics = {k: float(v / n_train_batch) for k, v in train_metrics.items()}  # calculate average Training Loss
                
                # Validation Loop
                i = 1

                for clip_batch_data in create_clip_batch(test_clip_dataset, config.batch_size, rng_key=subkey):
                    class_ids = clip_batch_data.class_ids
                    clip_batch_data = jax.device_put(clip_batch_data)
                    
                    _, loss, metrics, rng_key = train_step(
                        train_state, 
                        clip_batch_data,
                        is_train=False,
                        rng_key=subkey,
                        mode=config.encoder.mode
                    )

                    state_embed = metrics["state_embed"]
                    text_embed = metrics["text_embed"]

                    embeddings = [CLIPEmbedData(class_ids=c, state_embeddings=st, text_embeddings=t) for c, st, t in zip(class_ids, state_embed, text_embed)]
                    val_embed_queue.extend(embeddings)

                    val_losses["total"] += loss
                    val_losses["state2text"] += metrics["state2text_loss"]
                    val_losses["text2state"] += metrics["text2state_loss"]
                    
                    val_metrics["state2text_correct_pr"] += metrics["state2text_correct_pr"]
                    val_metrics["text2state_correct_pr"] += metrics["text2state_correct_pr"]
                    val_metrics["state2text_top1_accuracy"] += metrics["state2text_top1_accuracy"]
                    val_metrics["text2state_top1_accuracy"] += metrics["text2state_top1_accuracy"]

                    pbar.update(1)  # update Progress bar
                    pbar.set_postfix({"Train Loss": train_losses['total'], "Val Loss": val_losses['total'] / i})
                    i += 1

                val_losses = {k: float(v / n_test_batch) for k, v in val_losses.items()} # calculate Validation Loss
                val_metrics = {k: float(v / n_test_batch) for k, v in val_metrics.items()} # calculate Validation Loss

            if (epoch + 1) % config.ckpt_freq == 0:
                save_checkpoint(config, train_state, step=epoch + 1)
 
            if (epoch + 1) % config.embed_visualize_freq == 0:

                task_train_embed_paths = create_clip_embedding_figures(train_embed_queue, class_id2reward_cond, epoch, config, postfix='_train')
                task_val_embed_paths = create_clip_embedding_figures(val_embed_queue, class_id2reward_cond, epoch, config, postfix='_val')

                aux_dict = {
                    "train_tsne/embed_all": wandb.Image(task_train_embed_paths),
                    "val_tsne/embed_all": wandb.Image(task_val_embed_paths),
                }
            else:
                aux_dict = dict()

            if wandb.run is not None:
                wandb.log({
                    "total/train_loss": train_losses["total"], 
                    
                    "train(text-state)/state-text_temperature": train_metrics["text_state_temperature"],
                    "train(text-state)/state2text_loss": train_losses["state2text"],
                    "train(text-state)/text2state_loss": train_losses["text2state"],
                    "train(text-state)/state2text_correct_pr": train_metrics["state2text_correct_pr"],
                    "train(text-state)/text2state_correct_pr": train_metrics["text2state_correct_pr"],
                    "train(text-state)/state2text_top1_accuracy": train_metrics["state2text_top1_accuracy"],
                    "train(text-state)/text2state_top1_accuracy": train_metrics["text2state_top1_accuracy"],
                    
                    "total/val_loss": val_losses["total"],
                    
                    "val(text-state)/state2text_loss": val_losses["state2text"],
                    "val(text-state)/text2state_loss": val_losses["text2state"],
                    "val(text-state)/state2text_correct_pr": val_metrics["state2text_correct_pr"],
                    "val(text-state)/text2state_correct_pr": val_metrics["text2state_correct_pr"],
                    "val(text-state)/state2text_top1_accuracy": val_metrics["state2text_top1_accuracy"],
                    "val(text-state)/text2state_top1_accuracy": val_metrics["text2state_top1_accuracy"],
                    
                    "total/epoch": epoch,
                    "total/lr": lr_schedular(train_state.step),
                    **aux_dict
                })


    return lambda rng_key: train(rng_key)

def get_train_state(config: CLIPTrainConfig, rng_key: jax.random.PRNGKey):
    lr_schedular = create_learning_rate_fn(config, config.lr, config.steps_per_epoch)
    def create_train_state(encoder, rng_key, pretrained_params):
        def replace_params(params, key, replacement):
            for k in params.keys():
                if k == key:
                    params[k] = replacement
                    logging.info(f"replaced {key} in params")
                    return
                if isinstance(params[k], type(params)):
                    replace_params(params[k], key, replacement)

        rng_key, init_rng = jax.random.split(rng_key)
        input_ids=jnp.ones((1, config.encoder.token_max_len), dtype=jnp.int32)
        attention_mask=jnp.ones((1, config.encoder.token_max_len), dtype=jnp.int32)
        if config.encoder.model == 'clip':
            pixel_values=jnp.ones((1, 224, 224, config.clip_input_channel), dtype=jnp.float32)
        elif config.encoder.model == 'cnnclip':
            pixel_values=jnp.ones((1, 16, 16, config.clip_input_channel), dtype=jnp.float32)
        else:
            raise NotImplementedError("Model not implemented")

        params = encoder.init(init_rng, input_ids, attention_mask, pixel_values, mode=config.encoder.mode, training=False)

        for key in pretrained_params:
            replace_params(params, key, pretrained_params[key])

        # params = model.init(rng_key, input_ids, attention_mask, pixel_values)
        tx = optax.adamw(learning_rate=lr_schedular, weight_decay=config.weight_decay)
        return TrainState.create(apply_fn=encoder.apply, params=params, tx=tx)

    encoders, pretrained_params = None, None
    if config.encoder.model == 'clip':
        encoder, pretrained_params = get_clip_encoder(config.encoder, RL_training=False)
    elif config.encoder.model == 'cnnclip':
        encoder, pretrained_params = get_cnnclip_encoder(config.encoder, RL_training=False)
    else:
        NotImplementedError("Model not implemented")

    state = create_train_state(encoder,
                               rng_key=rng_key,
                               pretrained_params=pretrained_params
                               )
    return state, lr_schedular


def save_checkpoint(config, state, step):
    ckpt_dir = get_ckpt_dir(config)
    ckpt_dir = os.path.abspath(ckpt_dir)
    checkpoints.save_checkpoint(ckpt_dir, target=state, prefix="", step=step, overwrite=True, keep=3)
    logger.info(f"Checkpoint saved at step {step}")



@hydra.main(version_base=None, config_path='./conf', config_name='train_clip')
def main(config: CLIPTrainConfig):
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
            save_code=True)
        wandb.config.update(dict(config), allow_val_change=True)

    exp_dir = config.exp_dir
    logger.info(f'jax devices: {jax.devices()}')
    logger.info(f'running experiment at {exp_dir}')

    # Need to do this before setting up checkpoint manager so that it doesn't refer to old checkpoints.
    if config.overwrite and os.path.exists(exp_dir):
        shutil.rmtree(exp_dir)

    os.makedirs(exp_dir, exist_ok=True)

    make_train(config)(rng_key)



if __name__ == '__main__':
    main()
    # wandb.finish()
