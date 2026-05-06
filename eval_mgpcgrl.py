"""
eval_cpcgrl.py
==============
CPCGRL (Conditional PCGRL) 평가 엔트리포인트.
raw condition 벡터를 nlp_obs 에 주입하여 평가한다.

실행:
    python -m eval_cpcgrl [overrides]
"""
import json
import logging
import os

import hydra

from conf.config import MGPCGRLEvalConfig
from conf.game_utils import GAME_ABBR, GAME_ABBR_INV
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

    # ── encoder의 dataset_setting.json에서 seen_games를 읽어 game 자동 설정 ──
    dataset_setting_path = os.path.join(config.encoder.ckpt_dir, config.encoder.ckpt_name, "dataset_setting.json")
    if os.path.exists(dataset_setting_path):
        with open(dataset_setting_path, "r") as f:
            dataset_setting = json.load(f)
        seen_games = dataset_setting.get("seen_games", [])
        if seen_games:
            seen_abbrs = dict.fromkeys(GAME_ABBR_INV[g] for g in seen_games if g in GAME_ABBR_INV)
            if seen_abbrs.keys() == GAME_ABBR.keys():  # 모든 약어가 포함되면 "all"
                game_str = "all"
            else:
                game_str = "".join(seen_abbrs)  # 순서 유지, 중복 제거 (doom+doom2 → dm 한 번)
            logger.info(
                "Auto-setting game='%s' from encoder dataset_setting.json (seen_games=%s)",
                game_str, seen_games,
            )
            config.game = game_str
        else:
            logger.warning("dataset_setting.json has empty seen_games — keeping config.game='%s'", config.game)
    else:
        logger.warning("dataset_setting.json not found at %s — keeping config.game='%s'", dataset_setting_path, config.game)

    main_eval_entry(config, inject_obs_fn=inject_vipcgrl_obs)


if __name__ == "__main__":
    main()

