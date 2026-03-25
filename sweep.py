from copy import deepcopy

import hydra
import os
from os.path import basename

# set environment for jax full prealloc
XLA_PYTHON_CLIENT_MEM_FRACTION=.95
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = str(XLA_PYTHON_CLIENT_MEM_FRACTION)

import logging

import jax
import wandb

from conf.config import SweepConfig
from instruct_rl.utils.cuda import get_gpu_memory

from train import main as train_main
from eval import main as evaluate_main

log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
logger = logging.getLogger(basename(__file__))
logger.setLevel(getattr(logging, log_level, logging.INFO))



@hydra.main(version_base=None, config_path="./conf", config_name="sweep_pcgrl")
def main(config: SweepConfig):

    gpu_memory = get_gpu_memory()['total']

    max_n_envs = 600
    if gpu_memory > 0: # if the gpu is available
        if config.encoder.model in ['cnnclip']:
            if gpu_memory >= 48000:
                max_n_envs = 600
            elif gpu_memory >= 32000:
                max_n_envs = 500
            else:
                max_n_envs = 400
        elif config.encoder.model in ['mlp']:
            if gpu_memory >= 48000:
                max_n_envs = 600
            elif gpu_memory >= 32000:
                max_n_envs = 600
            else:
                max_n_envs = 550

    if config.n_envs > max_n_envs:
        logger.warning(f'config.n_envs: {config.n_envs}, changed to {min(config.n_envs, max_n_envs)} based on GPU memory: {gpu_memory}MB')
        config.n_envs = min(config.n_envs, max_n_envs)

    logger.info(f'Starting training with config: {config}...')
    train_config = deepcopy(config)
    train_main(train_config)
    logger.info('Training finished.')

    if config.wandb_key:
        wandb.finish()

    jax.clear_caches()

    config.n_envs = 100

    eval_config = deepcopy(config)
    eval_config.wandb_project = f'eval_{config.wandb_project}'

    if config.encoder.model not in ['cnnclip']:
        config = deepcopy(eval_config)

        logger.info(f'Starting evaluation with config: {config}...')
        evaluate_main(config)

        if config.wandb_key:
            wandb.finish()

        logger.info('Evaluation finished.')
    else:
        for eval_modality in ['text', 'state']:
            config = deepcopy(eval_config)

            config.eval_modality = eval_modality
            logger.info(f'Starting evaluation with config: {config}...')
            evaluate_main(config)

            if config.wandb_key:
                wandb.finish()

            logger.info('Evaluation finished.')

if __name__ == "__main__":
    main()
