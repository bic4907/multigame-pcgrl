"""
CLIP decoder training entrypoint.

This file keeps the runnable training flow visible:
configuration setup, W&B initialization, experiment directory handling, and the
call into the decoder trainer. Heavier implementation details live under
``encoder/decoder`` so the training components stay easier to navigate.
"""

from __future__ import annotations

import datetime
import logging
import os
import shutil
from os.path import basename

import hydra
import jax
import numpy as np
import wandb

from conf.config import CLIPDecoderTrainConfig
from encoder.utils.path import init_config
from instruct_rl.utils.logger import get_wandb_name


log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logger = logging.getLogger(basename(__file__))
logger.setLevel(getattr(logging, log_level, logging.INFO))
logging.getLogger("absl").setLevel(logging.ERROR)


def init_decoder_training(config: CLIPDecoderTrainConfig):
    """Initialize config, RNG, numpy seed, W&B, and experiment directory."""
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
    logger.info("jax devices: %s", jax.devices())
    logger.info("running experiment at %s", exp_dir)

    if config.overwrite and os.path.exists(exp_dir):
        shutil.rmtree(exp_dir)

    os.makedirs(exp_dir, exist_ok=True)
    return config, rng_key


def run_decoder_training(config: CLIPDecoderTrainConfig) -> None:
    """Run the configured seen/unseen CLIP decoder training job."""
    from encoder.decoder.trainer import make_train_unseen

    config, rng_key = init_decoder_training(config)
    make_train_unseen(config)(rng_key)

    if wandb.run:
        wandb.finish()


@hydra.main(version_base=None, config_path="./conf", config_name="train_clip_decoder")
def main(config: CLIPDecoderTrainConfig):
    run_decoder_training(config)


if __name__ == "__main__":
    main()
