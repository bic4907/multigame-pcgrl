"""
train_mipcgrl.py
================
MIPCGRL (Multi-task Instructed PCGRL) — IPCGRL과 동일한 RL 학습 파이프라인.

차이점은 사용하는 인코더 체크포인트뿐이다. MIPCGRL 인코더는 condition value
회귀에 더해 task(reward_enum) 분류를 함께 학습한 가중치이지만, RL 단계에서는
인코더 forward(latent z 추출)만 사용하므로 train_ipcgrl 과 동일한 inject 함수를
재사용한다.

실행:
    python -m train_mipcgrl encoder.ckpt_dir=... encoder.ckpt_name=...
"""
import json
import logging
import os

import hydra

from conf.config import MIPCGRLConfig
from instruct_rl.utils.log_utils import suppress_jax_debug_logs
from instruct_rl.utils.train_utils import main_entry

suppress_jax_debug_logs()

logger = logging.getLogger(__name__)


def inject_mipcgrl_obs(last_obs, env_state, instruct_sample, config, env):
    """BERT 임베딩을 nlp_obs 에 주입. 이후 네트워크 내부의 MLP 인코더가 처리."""
    return last_obs.replace(nlp_obs=instruct_sample.embedding)


@hydra.main(version_base=None, config_path="./conf", config_name="train_mipcgrl")
def main(config: MIPCGRLConfig):
    # ── encoder 의 dataset_setting.json 에서 seen/unseen 게임 정보 자동 주입 ──
    if config.encoder.ckpt_dir and config.encoder.ckpt_name:
        dataset_setting_path = os.path.join(
            config.encoder.ckpt_dir, config.encoder.ckpt_name, "dataset_setting.json"
        )
        if os.path.exists(dataset_setting_path):
            with open(dataset_setting_path, "r") as f:
                dataset_setting = json.load(f)

            seen_ratio = dataset_setting.get("seen_ratio", 1.0)
            if hasattr(config, "dataset_seen_ratio") and seen_ratio != config.dataset_seen_ratio:
                logger.info(
                    "Auto-setting dataset_seen_ratio=%.4f from encoder dataset_setting.json",
                    seen_ratio,
                )
                config.dataset_seen_ratio = seen_ratio

            unseen_ratio = dataset_setting.get("unseen_ratio", 0.0)
            if hasattr(config, "dataset_unseen_ratio") and unseen_ratio != config.dataset_unseen_ratio:
                logger.info(
                    "Auto-setting dataset_unseen_ratio=%.4f from encoder dataset_setting.json",
                    unseen_ratio,
                )
                config.dataset_unseen_ratio = unseen_ratio

            seen_games = dataset_setting.get("seen_games", [])
            if seen_games and hasattr(config, "reward_seen_games"):
                config.reward_seen_games = list(seen_games)
                logger.info(
                    "Auto-setting reward_seen_games=%s from encoder dataset_setting.json",
                    seen_games,
                )
            elif not seen_games:
                logger.warning("dataset_setting.json has empty seen_games")
        else:
            logger.warning("dataset_setting.json not found at %s", dataset_setting_path)

    main_entry(config, inject_obs_fn=inject_mipcgrl_obs)


if __name__ == "__main__":
    main()
