"""
Export t-SNE plots from a trained CLIP decoder checkpoint.

This entrypoint uses the same Hydra config as train_clip_decoder.py so model
shape, dataset flags, instruction prefix, and input channels stay aligned with
training.

Examples
--------
python encoder/export_tsne.py game=all \
    encoder.ckpt_path=pretrained_encoders/reward/clipdec-game-all_exp-def_0/ckpts

python encoder/export_tsne.py game=all \
    encoder.ckpt_dir=pretrained_encoders/reward \
    encoder.ckpt_name=clipdec-game-all_exp-def_0 \
    +tsne.samples_per_game=1000

Useful overrides
----------------
+tsne.out_dir=saves
+tsne.games=[dungeon,pokemon,sokoban,doom,zelda]  # default
+tsne.include_text=true  # default, exports state and text embeddings
+tsne.samples_per_game=1000  # default, use 0 for all usable samples per game
+tsne.batch_size=256
+tsne.perplexity=30
+tsne.max_iter=1000
+tsne.n_jobs=-1  # default, use all CPU cores when sklearn supports it
+tsne.allow_replacement=false
"""

from __future__ import annotations

import logging
import os
from os.path import basename

import hydra
from omegaconf import DictConfig

from conf.config import CLIPDecoderTrainConfig
from encoder.utils.path import init_config
from encoder.visualization.tsne_export import export_tsne_from_config


log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logger = logging.getLogger(basename(__file__))
logger.setLevel(getattr(logging, log_level, logging.INFO))
logging.getLogger("absl").setLevel(logging.ERROR)


@hydra.main(version_base=None, config_path="../conf", config_name="train_clip_decoder")
def main(config: CLIPDecoderTrainConfig | DictConfig) -> None:
    config = init_config(config)

    if config.encoder.model is None:
        config.encoder.model = "cnnclip"
        logger.warning("encoder.model is None, using default value: cnnclip")

    if config.encoder.model != "cnnclip":
        raise NotImplementedError(
            f"export_tsne currently supports cnnclip decoder checkpoints, got {config.encoder.model}"
        )

    export_tsne_from_config(config)


if __name__ == "__main__":
    main()
