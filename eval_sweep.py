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
from eval import main as evaluate_main

log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
logger = logging.getLogger(basename(__file__))
logger.setLevel(getattr(logging, log_level, logging.INFO))



@hydra.main(version_base=None, config_path="./conf", config_name="sweep_pcgrl")
def main(config: SweepConfig):

    jax.clear_caches()

    config.n_envs = 100


    train_config = deepcopy(config)
    train_config.wandb_project = f'eval_{config.wandb_project}'

    if config.encoder.model not in ['cnnclip']:
        config = deepcopy(train_config)

        logger.info(f'Starting evaluation with config: {config}...')
        evaluate_main(config)

        if config.wandb_key:
            wandb.finish()

        logger.info('Evaluation finished.')
    else:
        for eval_modality in ['text', 'state']:
            config = deepcopy(train_config)

            config.eval_modality = eval_modality
            logger.info(f'Starting evaluation with config: {config}...')
            evaluate_main(config)

            if config.wandb_key:
                wandb.finish()

            logger.info('Evaluation finished.')

if __name__ == "__main__":
    main()
