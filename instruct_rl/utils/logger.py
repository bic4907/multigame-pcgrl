import logging
from os import path
import numpy as np
import wandb
from glob import glob

from conf.config import Config


def get_group_name(config):
    group_name = f"rep-{config.representation}_model-{config.model}"
    if config.embed_type != "test":
        group_name = group_name + f"_embed-{config.embed_type}"
    if config.encoder.model is not None:
        group_name = group_name + f"_enc-{config.encoder.model}_enctr-{str(config.encoder.trainable).lower()}"
    if config.instruct is not None:
        group_name = group_name + f"_inst-{config.instruct}"

    # RQ4 parameters
    if config.encoder.buffer_ratio != 1.0:
        group_name = group_name + f"_br-{config.encoder.buffer_ratio}"
    if config.encoder.output_dim != 64:
        group_name = group_name + f"_es-{config.encoder.output_dim}"

    return group_name


def get_wandb_name(config: Config):
    exp_dir_path = config.exp_dir.replace('\\', '/')
    exp_dirs = exp_dir_path.split('/')
    return exp_dirs[-1]


def get_wandb_name_eval(config: Config):
    wandb_name = get_wandb_name(config)

    wandb_eval_name = f"{wandb_name}"

    if config.eval_exp_name is not None:
        wandb_eval_name += f"-{config.eval_exp_name}"

    return wandb_eval_name

