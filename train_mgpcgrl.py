"""
train_mg_pcgrl.py
=================
MGPCGRL (MultiGame PCGRL) — pretrained CLIP 임베딩을 입력 피처로 사용.

기존 VIPCGRL 파이프라인을 기반으로 하되,
실험 엔트리/설정을 mgpcgrl 이름으로 분리한 실행 스크립트.

실행:
    python -m train_mg_pcgrl [overrides]
    python -m train_mg_pcgrl dataset_game=dungeon dataset_reward_enum=1 SIM_COEF=3.5
"""
import json
import logging
import os

import hydra

from conf.config import MGPCGRLConfig
from conf.game_utils import GAME_ABBR_INV
from instruct_rl.utils.log_utils import suppress_jax_debug_logs
from instruct_rl.utils.train_utils import main_entry

logger = logging.getLogger(__name__)

suppress_jax_debug_logs()


# ── VIPCGRL obs 주입: CLIP embedding → nlp_obs ───────────────────────────────

def inject_vipcgrl_obs(last_obs, env_state, instruct_sample, config, env):
    """pretrained CLIP 인코더로 계산된 임베딩을 nlp_obs 에 주입."""
    return last_obs.replace(nlp_obs=instruct_sample.embedding)


# ── Hydra entrypoint ──────────────────────────────────────────────────────────

@hydra.main(version_base=None, config_path="./conf", config_name="train_mgpcgrl")
def main(config: MGPCGRLConfig):

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

    main_entry(
        config,
        inject_obs_fn=inject_vipcgrl_obs,
    )


if __name__ == "__main__":
    main()
