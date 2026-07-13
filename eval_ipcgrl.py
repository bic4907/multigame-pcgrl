"""
eval_ipcgrl.py
==============
IPCGRL (Instructed PCGRL) 평가 엔트리포인트.
BERT 임베딩을 nlp_obs 에 주입하여 평가한다.

실행:
    python -m eval_ipcgrl [overrides]
"""
import json
import logging
import os

import hydra

from conf.config import IPCGRLEvalConfig
from conf.game_utils import compute_seen_unseen_split
from instruct_rl.utils.log_utils import suppress_jax_debug_logs
from instruct_rl.utils.eval_utils import main_eval_entry
from train_ipcgrl import inject_ipcgrl_obs

suppress_jax_debug_logs()

logger = logging.getLogger(__name__)


# ── Hydra entrypoint ──────────────────────────────────────────────────────────

@hydra.main(version_base=None, config_path="./conf", config_name="eval_ipcgrl")
def main(config: IPCGRLEvalConfig):
    # ── MGPCGRL과 동일한 로직: encoder의 dataset_setting.json에서 seen/unseen 게임 정보 주입 ──
    if config.encoder.ckpt_dir and config.encoder.ckpt_name:
        dataset_setting_path = os.path.join(config.encoder.ckpt_dir, config.encoder.ckpt_name, "dataset_setting.json")
        if os.path.exists(dataset_setting_path):
            with open(dataset_setting_path, "r") as f:
                dataset_setting = json.load(f)

            # ── seen_ratio: analysis only (dataset filtering is not applied) ──
            seen_ratio = dataset_setting.get("seen_ratio", 1.0)
            if hasattr(config, "train_seen_ratio") and seen_ratio != config.train_seen_ratio:
                logger.info(
                    "Auto-setting train_seen_ratio=%.4f from encoder dataset_setting.json (analysis only)",
                    seen_ratio,
                )
                config.train_seen_ratio = seen_ratio

            # ── seen/unseen games (canonical 5-game split) → injected into config
            #    so they appear in WandB regardless of dataset_setting.json state. ──
            seen_raw = dataset_setting.get("seen_games", [])
            if seen_raw:
                _seen, _unseen = compute_seen_unseen_split(seen_raw)
                if hasattr(config, "seen_games"):
                    config.seen_games = list(_seen)
                if hasattr(config, "reward_seen_games"):
                    config.reward_seen_games = list(_seen)
                if hasattr(config, "unseen_games"):
                    config.unseen_games = list(_unseen)
                logger.info(
                    "Auto-setting seen_games=%s, unseen_games=%s from encoder dataset_setting.json",
                    _seen, _unseen,
                )
        else:
            logger.warning("dataset_setting.json not found at %s", dataset_setting_path)

    main_eval_entry(config, inject_obs_fn=inject_ipcgrl_obs)


if __name__ == "__main__":
    main()

