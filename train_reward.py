"""
train_reward.py
=================
ReWARD (MultiGame PCGRL) uses pretrained CLIP embeddings as input features.

Based on the VIPCGRL pipeline, with experiment entries and configuration
separated under the ReWARD name.

Usage:
    python -m train_reward [overrides]
    python -m train_reward dataset_game=dungeon dataset_reward_enum=1 SIM_COEF=3.5
"""
import json
import logging
import os
import shutil

import hydra

from conf.config import ReWARDConfig
from conf.game_utils import GAME_ABBR, GAME_ABBR_INV
from instruct_rl.utils.log_utils import suppress_jax_debug_logs
from instruct_rl.utils.train_utils import main_entry

logger = logging.getLogger(__name__)

suppress_jax_debug_logs()


# ── VIPCGRL obs inject: CLIP embedding → nlp_obs ───────────────────────────────

def inject_vipcgrl_obs(last_obs, env_state, instruct_sample, config, env):
    """Inject embeddings computed by the pretrained CLIP encoder into nlp_obs."""
    return last_obs.replace(nlp_obs=instruct_sample.embedding)


# ── Hydra entrypoint ──────────────────────────────────────────────────────────

@hydra.main(version_base=None, config_path="./conf", config_name="train_reward")
def main(config: ReWARDConfig):

    if not config.encoder.ckpt_dir or not config.encoder.ckpt_name:
        raise ValueError("Both encoder.ckpt_dir and encoder.ckpt_name must be set in the configuration.")

    # ── Read delta_weight from encoder_config.json into config for wandb logging ──
    encoder_config_path = os.path.join(config.encoder.ckpt_dir, config.encoder.ckpt_name, "encoder_config.json")
    encoder_config_src = None  # Keep locally rather than adding it to config

    if os.path.exists(encoder_config_path):
        with open(encoder_config_path, "r") as f:
            encoder_training_config = json.load(f)
        # Store delta_weight in config
        config.encoder_delta_weight = encoder_training_config.get('delta_weight', 0.0)
        logger.info("Loaded encoder delta_weight=%.4f from: %s",
                    config.encoder_delta_weight, encoder_config_path)
        encoder_config_src = encoder_config_path  # Save the source path for copying
    else:
        logger.warning("encoder_config.json not found at %s", encoder_config_path)
        config.encoder_delta_weight = 0.0

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

        # ── unseen_ratio inject ──────────────────────────────────────────────────
        # dataset_unseen_ratio default value  1.0 (ReWARDConfig in  fixed).
        # Preserve values explicitly supplied through the CLI.
        # Inject reward_unseen_ratio separately from the encoder's unseen_ratio.
        unseen_ratio = dataset_setting.get("unseen_ratio", 0.0)

        # ── reward_unseen_ratio: metadata/decoder boundary within unseen samples ──
        # each unseen game of  sample  order basis as  split:
        #   front (reward_unseen_ratio ratio) → metadata (GT, encoder training subset)
        #   remaining samples -> predict conditions with the reward decoder
        if unseen_ratio != config.reward_unseen_ratio:
            logger.info(
                "Auto-setting reward_unseen_ratio=%.4f from encoder dataset_setting.json",
                unseen_ratio,
            )
            config.reward_unseen_ratio = unseen_ratio

        # ── game_setting_mode=encoder_seen: configure seen games as training targets ──
        if config.game_setting_mode == "encoder_seen":
            seen_games = dataset_setting.get("seen_games", [])
            if seen_games:
                seen_abbrs = dict.fromkeys(GAME_ABBR_INV[g] for g in seen_games if g in GAME_ABBR_INV)
                if seen_abbrs.keys() == GAME_ABBR.keys():
                    game_str = "all"
                else:
                    game_str = "".join(seen_abbrs)
                logger.info(
                    "game_setting_mode=encoder_seen → setting game='%s' (seen_games=%s)",
                    game_str, seen_games,
                )
                config.game = game_str
            else:
                logger.warning(
                    "game_setting_mode=encoder_seen but dataset_setting.json has empty seen_games "
                    "— keeping config.game='%s'", config.game,
                )

        # ── Always inject reward_seen_games from encoder's dataset_setting.json ──
        # The seen/unseen split reflects the encoder training distribution and is
        # independent of reward_decoder_mode (noop / all / unseen). rdm only
        # controls how reward annotations are computed; seen/unseen lists must
        # always come from the encoder.
        seen_games = dataset_setting.get("seen_games", [])
        unseen_games = dataset_setting.get("unseen_games", [])
        if seen_games:
            config.reward_seen_games = list(seen_games)
            logger.info(
                "Auto-setting reward_seen_games=%s from encoder dataset_setting.json "
                "(reward_decoder_mode=%s, reward_unseen_ratio=%.4f)",
                seen_games, config.reward_decoder_mode, config.reward_unseen_ratio,
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
        inject_obs_fn=inject_vipcgrl_obs,
    )

    # ── Copy encoder_config.json to the PCGRL training directory for reference ──
    # main_entry() creates exp_dir; copy the encoder config there afterward
    if encoder_config_src and hasattr(config, 'exp_dir') and config.exp_dir:
        dst_path = os.path.join(config.exp_dir, "encoder_config.json")
        try:
            shutil.copy2(encoder_config_src, dst_path)
            logger.info("Copied encoder_config.json to: %s", dst_path)
        except Exception as e:
            logger.warning("Failed to copy encoder_config.json: %s", e)


if __name__ == "__main__":
    main()
