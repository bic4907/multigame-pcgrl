"""
eval_vipcgrl.py
===============
VIPCGRL (Vision-Instructed PCGRL) 평가 엔트리포인트.
pretrained CLIP 임베딩을 nlp_obs 에 주입하여 평가한다.

실행:
    python -m eval_vipcgrl [overrides]
"""
import json
import logging
import os

import hydra

from conf.config import VIPCGRLEvalConfig
from conf.game_utils import compute_seen_unseen_split
from instruct_rl.utils.log_utils import suppress_jax_debug_logs
from instruct_rl.utils.eval_utils import main_eval_entry
from train_vipcgrl import inject_vipcgrl_obs

logger = logging.getLogger(__name__)

suppress_jax_debug_logs()



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

        # ── seen/unseen games (canonical 5-game split) → injected into config
        #    so they appear in WandB regardless of train_setting.json state. ──
        seen_raw = dataset_setting.get("seen_games", [])
        if seen_raw:
            _seen, _unseen = compute_seen_unseen_split(seen_raw)
            config.seen_games = list(_seen)
            config.unseen_games = list(_unseen)
            logger.info(
                "Auto-setting seen_games=%s, unseen_games=%s from encoder dataset_setting.json",
                _seen, _unseen,
            )
    else:
        logger.warning("dataset_setting.json not found at %s", dataset_setting_path)

    main_eval_entry(config, inject_obs_fn=inject_vipcgrl_obs)


if __name__ == "__main__":
    main()

