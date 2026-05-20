"""
train_finetuned_clip.py
=======================
Fine-tuned CLIP 기반 PCGRL 학습 entrypoint.

`train_pretrained_clip.py` 와 obs/reward 주입 로직이 동일하지만, RL 인코더의
파라미터를 `encoder.ckpt_name` (또는 ckpt_path) 으로 지정된 fine-tuned CLIP
체크포인트로 덮어쓴다 (기존 `apply_encoder_params` 메커니즘 그대로 활용).

실행:
    python -m train_finetuned_clip encoder.ckpt_name=finetuned-clip-...
"""
import json
import logging
import os

import hydra

from conf.config import FinetunedCLIPPCGRLConfig
from conf.game_utils import GAME_ABBR, GAME_ABBR_INV
from instruct_rl.utils.log_utils import suppress_jax_debug_logs
from instruct_rl.utils.train_utils import main_entry
from train_pretrained_clip import (inject_pretrained_clip_pcgrl_obs,
                                   inject_pretrained_clip_reward)

suppress_jax_debug_logs()

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="./conf", config_name="finetuned_clip_pcgrl")
def main(config: FinetunedCLIPPCGRLConfig):

    if not config.encoder.ckpt_dir or not config.encoder.ckpt_name:
        raise ValueError("Both encoder.ckpt_dir and encoder.ckpt_name must be set in the configuration.")

    # ── encoder의 dataset_setting.json에서 seen_ratio / seen_games 주입 ──
    dataset_setting_path = os.path.join(config.encoder.ckpt_dir, config.encoder.ckpt_name, "dataset_setting.json")
    if os.path.exists(dataset_setting_path):
        with open(dataset_setting_path, "r") as f:
            dataset_setting = json.load(f)

        # ── seen_ratio 주입: encoder 학습 때 쓴 seen 게임 데이터 비율을 그대로 사용 ──
        seen_ratio = dataset_setting.get("seen_ratio", 1.0)
        if seen_ratio != config.dataset_seen_ratio:
            logger.info(
                "Auto-setting dataset_seen_ratio=%.4f from encoder dataset_setting.json",
                seen_ratio,
            )
            config.dataset_seen_ratio = seen_ratio

        # ── game_setting_mode=encoder_seen: seen 게임만 학습 대상으로 설정 ──
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
        seen_games = dataset_setting.get("seen_games", [])
        if seen_games:
            config.reward_seen_games = list(seen_games)
            logger.info(
                "Auto-setting reward_seen_games=%s from encoder dataset_setting.json",
                seen_games,
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
        inject_obs_fn=inject_pretrained_clip_pcgrl_obs,
        inject_reward_fn=inject_pretrained_clip_reward,
    )


if __name__ == "__main__":
    main()
