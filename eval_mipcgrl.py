"""
eval_mipcgrl.py
===============
MIPCGRL 평가 엔트리포인트. eval_ipcgrl 와 동일하지만 MIPCGRLEvalConfig 를
사용해 exp_dir / wandb name 이 ``mipcgrl_...`` prefix 로 분리되도록 한다.
"""
import json
import logging
import os

import hydra

from conf.config import MIPCGRLEvalConfig
from conf.game_utils import compute_seen_unseen_split
from instruct_rl.utils.log_utils import suppress_jax_debug_logs
from instruct_rl.utils.eval_utils import main_eval_entry
from train_mipcgrl import inject_mipcgrl_obs

suppress_jax_debug_logs()

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="./conf", config_name="eval_mipcgrl")
def main(config: MIPCGRLEvalConfig):
    if config.encoder.ckpt_dir and config.encoder.ckpt_name:
        dataset_setting_path = os.path.join(
            config.encoder.ckpt_dir, config.encoder.ckpt_name, "dataset_setting.json"
        )
        if os.path.exists(dataset_setting_path):
            with open(dataset_setting_path, "r") as f:
                dataset_setting = json.load(f)

            seen_ratio = dataset_setting.get("seen_ratio", 1.0)
            if hasattr(config, "train_seen_ratio") and seen_ratio != config.train_seen_ratio:
                logger.info(
                    "Auto-setting train_seen_ratio=%.4f from encoder dataset_setting.json (analysis only)",
                    seen_ratio,
                )
                config.train_seen_ratio = seen_ratio

            seen_raw = dataset_setting.get("seen_games", [])
            if seen_raw:
                _seen, _unseen = compute_seen_unseen_split(seen_raw)
                if hasattr(config, "seen_games"):
                    config.seen_games = list(_seen)
                if hasattr(config, "unseen_games"):
                    config.unseen_games = list(_unseen)
                logger.info(
                    "Auto-setting seen_games=%s, unseen_games=%s from encoder dataset_setting.json",
                    _seen, _unseen,
                )
        else:
            logger.warning("dataset_setting.json not found at %s", dataset_setting_path)

    main_eval_entry(config, inject_obs_fn=inject_mipcgrl_obs)


if __name__ == "__main__":
    main()
