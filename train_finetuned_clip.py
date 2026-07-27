"""
train_finetuned_clip.py
=======================
Fine-tuned CLIP based PCGRL training entrypoint.

Uses the same observation/reward injection as `train_pretrained_clip.py`, but
replaces RL encoder parameters with the fine-tuned CLIP checkpoint specified by
`encoder.ckpt_name` (or ckpt_path), reusing `apply_encoder_params`.

Usage:
    python -m train_finetuned_clip encoder.ckpt_name=finetuned-clip-...
"""
import json
import logging
import os

import hydra

from conf.config import FinetunedCLIPPCGRLConfig
from conf.game_utils import GAME_ABBR, GAME_ABBR_INV
from instruct_rl.utils.log_utils import suppress_jax_debug_logs
from instruct_rl.utils.train_utils import main_entry
from train_pretrained_clip import (inject_pretrained_clip_pcgrl_obs,
                                   inject_pretrained_clip_reward)

suppress_jax_debug_logs()

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="./conf", config_name="finetuned_clip_pcgrl")
def main(config: FinetunedCLIPPCGRLConfig):

    if not config.encoder.ckpt_dir or not config.encoder.ckpt_name:
        raise ValueError("Both encoder.ckpt_dir and encoder.ckpt_name must be set in the configuration.")

    # ── encoder of  dataset_setting.json in  seen_ratio / seen_games inject ──
    dataset_setting_path = os.path.join(config.encoder.ckpt_dir, config.encoder.ckpt_name, "dataset_setting.json")
    if os.path.exists(dataset_setting_path):
        with open(dataset_setting_path, "r") as f:
            dataset_setting = json.load(f)

        # ── Reuse the seen-game ratio from encoder training ──
        seen_ratio = dataset_setting.get("seen_ratio", 1.0)
        if seen_ratio != config.dataset_seen_ratio:
            logger.info(
                "Auto-setting dataset_seen_ratio=%.4f from encoder dataset_setting.json",
                seen_ratio,
            )
            config.dataset_seen_ratio = seen_ratio

        # ── game_setting_mode=encoder_seen: configure seen games as training targets ──
        game_setting_mode = getattr(config, "game_setting_mode", "all")
        dataset_unseen_ratio = dataset_setting.get("unseen_ratio", 0.0)
        if hasattr(config, "dataset_unseen_ratio") and dataset_unseen_ratio != config.dataset_unseen_ratio:
            logger.info(
                "Auto-setting dataset_unseen_ratio=%.4f from encoder dataset_setting.json",
                dataset_unseen_ratio,
            )
            config.dataset_unseen_ratio = dataset_unseen_ratio

        if game_setting_mode == "encoder_seen":
            seen_games = dataset_setting.get("seen_games", [])
            if seen_games:
                seen_abbrs = dict.fromkeys(GAME_ABBR_INV[g] for g in seen_games if g in GAME_ABBR_INV)
                if seen_abbrs.keys() == GAME_ABBR.keys():
                    game_str = "all"
                else:
                    game_str = "".join(seen_abbrs)
                if dataset_unseen_ratio > 0.0:
                    logger.info(
                        "game_setting_mode=encoder_seen + unseen_ratio=%.4f > 0 -> expanding game='all'",
                        dataset_unseen_ratio,
                    )
                    config.game = "all"
                else:
                    logger.info(
                        "game_setting_mode=encoder_seen → setting game='%s' (seen_games=%s)",
                        game_str, seen_games,
                    )
                    config.game = game_str
            else:
                logger.warning(
                    "game_setting_mode=%s but dataset_setting.json has empty seen_games — keeping config.game='%s'",
                    game_setting_mode, config.game,
                )

        # ── Always inject reward_seen_games from encoder's dataset_setting.json ──
        seen_games = dataset_setting.get("seen_games", [])
        if seen_games:
            config.reward_seen_games = list(seen_games)
            logger.info(
                "Auto-setting reward_seen_games=%s from encoder dataset_setting.json",
                seen_games,
            )
        else:
            logger.warning(
                "dataset_setting.json has empty seen_games — "
                "train_setting.json seen/unseen will also be empty"
            )
    else:
        logger.warning("dataset_setting.json not found at %s", dataset_setting_path)

    main_entry(
        config,
        inject_obs_fn=inject_pretrained_clip_pcgrl_obs,
        inject_reward_fn=inject_pretrained_clip_reward,
    )


if __name__ == "__main__":
    main()
