"""
eval_vipcgrl.py
===============
VIPCGRL (Vision-Instructed PCGRL) evaluation entry point.
pretrained CLIP embedding  nlp_obs  in  injecttext evaluationtext.

Usage:
    python -m eval_vipcgrl [overrides]
"""
import json
import logging
import os

import hydra

from conf.config import VIPCGRLEvalConfig
from conf.game_utils import (
    GAME_ABBR,
    GAME_ABBR_INV,
    compute_seen_unseen_split,
    infer_seen_games_from_ckpt_name,
)
from instruct_rl.utils.log_utils import suppress_jax_debug_logs
from instruct_rl.utils.eval_utils import main_eval_entry
from train_vipcgrl import inject_vipcgrl_obs

logger = logging.getLogger(__name__)

suppress_jax_debug_logs()


def _apply_seen_games_to_config(config: VIPCGRLEvalConfig, seen_games, unseen_ratio: float) -> bool:
    """Inject encoder seen-game metadata into a VIPCGRL eval config."""
    if not seen_games:
        return False

    if config.game_setting_mode == "encoder_seen":
        seen_abbrs = dict.fromkeys(
            GAME_ABBR_INV[g] for g in seen_games if g in GAME_ABBR_INV
        )
        game_str = "all" if seen_abbrs.keys() == GAME_ABBR.keys() else "".join(seen_abbrs)

        if unseen_ratio > 0.0:
            logger.info(
                "game_setting_mode=encoder_seen + unseen_ratio=%.4f > 0 "
                "→ expanding game to 'all' for exp_dir matching",
                unseen_ratio,
            )
            config.game = "all"
        elif game_str != config.game:
            logger.info(
                "game_setting_mode=encoder_seen → overriding config.game "
                "'%s' → '%s' (seen_games=%s) for exp_dir matching",
                config.game, game_str, seen_games,
            )
            config.game = game_str

    _seen, _unseen = compute_seen_unseen_split(seen_games)
    config.seen_games = list(_seen)
    config.unseen_games = list(_unseen)
    logger.info(
        "Auto-setting seen_games=%s, unseen_games=%s from encoder metadata",
        _seen, _unseen,
    )
    return True



# ── Hydra entrypoint ──────────────────────────────────────────────────────────

@hydra.main(version_base=None, config_path="./conf", config_name="eval_vipcgrl")
def main(config: VIPCGRLEvalConfig):

    if not config.encoder.ckpt_dir or not config.encoder.ckpt_name:
        raise ValueError("Both encoder.ckpt_dir and encoder.ckpt_name must be set in the configuration.")

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

        # ── unseen_ratio: train_vipcgrl.py  and  sametext game expand basis in  text for  ──
        unseen_ratio = dataset_setting.get("unseen_ratio", 0.0)

        # ── seen/unseen games (canonical 5-game split) → injected into config
        #    so they appear in WandB regardless of train_setting.json state. ──
        seen_raw = dataset_setting.get("seen_games", [])
        if not _apply_seen_games_to_config(config, seen_raw, unseen_ratio):
            logger.warning(
                "dataset_setting.json has empty seen_games — keeping config.game='%s'",
                config.game,
            )
    else:
        logger.warning("dataset_setting.json not found at %s", dataset_setting_path)
        inferred_seen_games = infer_seen_games_from_ckpt_name(config.encoder.ckpt_name)
        if inferred_seen_games:
            logger.info(
                "Inferred seen_games=%s from encoder.ckpt_name='%s'",
                inferred_seen_games, config.encoder.ckpt_name,
            )
            _apply_seen_games_to_config(config, inferred_seen_games, 0.0)

    main_eval_entry(config, inject_obs_fn=inject_vipcgrl_obs)


if __name__ == "__main__":
    main()
