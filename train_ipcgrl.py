"""
train_ipcgrl.py
================
IPCGRL (Instructed PCGRL) — BERT 임베딩 → MLP 인코더를 거친 피처를 입력으로 사용.

기존 train.py 의 `encoder.model='mlp'` 모드에 해당하며,
dataset 기반 파이프라인(MultiGameDataset)으로 동작한다.

실행:
    python -m train_ipcgrl [overrides]
"""
import json
import logging
import os

import hydra

from conf.config import IPCGRLConfig
from instruct_rl.utils.log_utils import suppress_jax_debug_logs
from instruct_rl.utils.train_utils import main_entry

suppress_jax_debug_logs()

logger = logging.getLogger(__name__)


# ── IPCGRL obs 주입: embedding → nlp_obs ─────────────────────────────────────

def inject_ipcgrl_obs(last_obs, env_state, instruct_sample, config, env):
    """BERT 임베딩을 nlp_obs 에 주입. 이후 네트워크 내부의 MLP 인코더가 처리."""
    return last_obs.replace(nlp_obs=instruct_sample.embedding)


# ── Hydra entrypoint ──────────────────────────────────────────────────────────

@hydra.main(version_base=None, config_path="./conf", config_name="train_ipcgrl")
def main(config: IPCGRLConfig):
    # ── MGPCGRL과 동일한 로직: encoder의 dataset_setting.json에서 seen/unseen 게임 정보 주입 ──
    if config.encoder.ckpt_dir and config.encoder.ckpt_name:
        dataset_setting_path = os.path.join(config.encoder.ckpt_dir, config.encoder.ckpt_name, "dataset_setting.json")
        if os.path.exists(dataset_setting_path):
            with open(dataset_setting_path, "r") as f:
                dataset_setting = json.load(f)

            # ── seen_ratio 주입: encoder 학습 때 쓴 seen 게임 데이터 비율을 그대로 사용 ──
            seen_ratio = dataset_setting.get("seen_ratio", 1.0)
            if hasattr(config, "dataset_seen_ratio") and seen_ratio != config.dataset_seen_ratio:
                logger.info(
                    "Auto-setting dataset_seen_ratio=%.4f from encoder dataset_setting.json",
                    seen_ratio,
                )
                config.dataset_seen_ratio = seen_ratio

            # ── unseen_ratio 주입 ──
            unseen_ratio = dataset_setting.get("unseen_ratio", 0.0)
            if hasattr(config, "dataset_unseen_ratio") and unseen_ratio != config.dataset_unseen_ratio:
                logger.info(
                    "Auto-setting dataset_unseen_ratio=%.4f from encoder dataset_setting.json",
                    unseen_ratio,
                )
                config.dataset_unseen_ratio = unseen_ratio

            # ── seen_games 주입 ──
            seen_games = dataset_setting.get("seen_games", [])
            unseen_games = dataset_setting.get("unseen_games", [])
            if seen_games and hasattr(config, "reward_seen_games"):
                config.reward_seen_games = list(seen_games)
                logger.info(
                    "Auto-setting reward_seen_games=%s from encoder dataset_setting.json",
                    seen_games,
                )
            else:
                if not seen_games:
                    logger.warning(
                        "dataset_setting.json has empty seen_games"
                    )
        else:
            logger.warning("dataset_setting.json not found at %s", dataset_setting_path)

    main_entry(config, inject_obs_fn=inject_ipcgrl_obs)


if __name__ == "__main__":
    main()
