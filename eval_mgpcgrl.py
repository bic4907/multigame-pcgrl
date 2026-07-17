"""
eval_cpcgrl.py
==============
CPCGRL (Conditional PCGRL) evaluation entry point.
raw condition text  nlp_obs  in  injecttext evaluationtext.

Usage:
    python -m eval_cpcgrl [overrides]
"""
import json
import logging
import os
import shutil

import hydra

from conf.config import MGPCGRLEvalConfig
from conf.game_utils import (
    compute_seen_unseen_split,
    infer_seen_games_from_ckpt_name,
    unseen_abbr_from_seen_games,
)
from instruct_rl.utils.log_utils import suppress_jax_debug_logs
from instruct_rl.utils.eval_utils import main_eval_entry
from train_mgpcgrl import inject_vipcgrl_obs


logger = logging.getLogger(__name__)

suppress_jax_debug_logs()



# ── Hydra entrypoint ──────────────────────────────────────────────────────────

@hydra.main(version_base=None, config_path="./conf", config_name="eval_mgpcgrl")
def main(config: MGPCGRLEvalConfig):

    if not config.encoder.ckpt_dir or not config.encoder.ckpt_name:
        raise ValueError("Both encoder.ckpt_dir and encoder.ckpt_name must be set in the configuration.")

    # ── encoder_config.json in  delta_weight text config in  inject (wandb  to text for ) ──
    encoder_config_path = os.path.join(config.encoder.ckpt_dir, config.encoder.ckpt_name, "encoder_config.json")
    encoder_config_src = None  # local variable to  save (config in  text text)

    if os.path.exists(encoder_config_path):
        with open(encoder_config_path, "r") as f:
            encoder_training_config = json.load(f)
        # delta_weighttext config in  save
        config.encoder_delta_weight = encoder_training_config.get('delta_weight', 0.0)
        logger.info("Loaded encoder delta_weight=%.4f from: %s",
                    config.encoder_delta_weight, encoder_config_path)
        encoder_config_src = encoder_config_path  # copy  abovetext path save
    else:
        logger.warning("encoder_config.json not found at %s", encoder_config_path)
        config.encoder_delta_weight = 0.0

    # ── Read encoder's dataset_setting.json → inject train_seen_ratio + seen/unseen ──
    dataset_setting_path = os.path.join(config.encoder.ckpt_dir, config.encoder.ckpt_name, "dataset_setting.json")
    if os.path.exists(dataset_setting_path):
        with open(dataset_setting_path, "r") as f:
            dataset_setting = json.load(f)

        # ── seen_ratio: analysis only (dataset filtering is not applied) ──
        seen_ratio = dataset_setting.get("seen_ratio", 1.0)
        if seen_ratio != config.train_seen_ratio:
            logger.info(
                "Auto-setting train_seen_ratio=%.4f from encoder dataset_setting.json (analysis only)",
                seen_ratio,
            )
            config.train_seen_ratio = seen_ratio

        # ── seen/unseen games (canonical 5-game split) → injected into config
        #    so they appear in WandB regardless of train_setting.json state. ──
        seen_raw = dataset_setting.get("seen_games", [])
        if seen_raw:
            _seen, _unseen = compute_seen_unseen_split(seen_raw)
            config.seen_games = list(_seen)
            config.unseen_games = list(_unseen)
            config.train_unseen_abbr = unseen_abbr_from_seen_games(_seen)
            logger.info(
                "Auto-setting seen_games=%s, unseen_games=%s from encoder dataset_setting.json",
                _seen, _unseen,
            )
    else:
        logger.warning("dataset_setting.json not found at %s", dataset_setting_path)

    # Ensure eval exp_dir is resolved to the matching trained MGPCGRL run before
    # main_eval_entry() calls init_config(). Older full-shot subset encoder names
    # only encode seen games as ``clipdec-game-dgpk...`` rather than
    # ``_unseen-...``; make the derived unseen suffix explicit for eval names.
    if not config.train_unseen_abbr:
        inferred_seen = infer_seen_games_from_ckpt_name(config.encoder.ckpt_name)
        inferred_unseen_abbr = unseen_abbr_from_seen_games(inferred_seen)
        if inferred_unseen_abbr:
            config.train_unseen_abbr = inferred_unseen_abbr
            logger.info(
                "Auto-setting train_unseen_abbr=%s from encoder ckpt_name=%s",
                inferred_unseen_abbr,
                config.encoder.ckpt_name,
            )

    main_eval_entry(config, inject_obs_fn=inject_vipcgrl_obs)

    # ── encoder_config.json  PCGRL evaluation folder to  copy (text for ) ──
    # main_eval_entry() call  after  exp_dir  createtext to  encoder ckpt path in  copy
    if encoder_config_src and hasattr(config, 'exp_dir') and config.exp_dir:
        dst_path = os.path.join(config.exp_dir, "encoder_config.json")
        try:
            shutil.copy2(encoder_config_src, dst_path)
            logger.info("Copied encoder_config.json to: %s", dst_path)
        except Exception as e:
            logger.warning("Failed to copy encoder_config.json: %s", e)


if __name__ == "__main__":
    main()
