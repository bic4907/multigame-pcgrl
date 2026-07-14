"""
train_vipcgrl.py
=================
VIPCGRL (Vision-Instructed PCGRL) — pretrained CLIP 임베딩을 입력 피처로 사용.

기존 train.py 의 `encoder.model='cnnclip'` 모드에 해당하며,
dataset 기반 파이프라인(MultiGameDataset)으로 동작한다.

실행:
    python -m train_vipcgrl [overrides]
    python -m train_vipcgrl dataset_game=dungeon dataset_reward_enum=1 SIM_COEF=3.5
"""
import json
import logging
import os

import hydra

from conf.config import VIPCGRLConfig
from conf.game_utils import GAME_ABBR, GAME_ABBR_INV, infer_seen_games_from_ckpt_name
from instruct_rl.utils.log_utils import suppress_jax_debug_logs
from instruct_rl.utils.train_utils import main_entry

logger = logging.getLogger(__name__)

suppress_jax_debug_logs()


# ── VIPCGRL obs 주입: CLIP embedding → nlp_obs ───────────────────────────────

def inject_vipcgrl_obs(last_obs, env_state, instruct_sample, config, env):
    """pretrained CLIP 인코더로 계산된 임베딩을 nlp_obs 에 주입."""
    return last_obs.replace(nlp_obs=instruct_sample.embedding)


def _apply_seen_games_to_config(config: VIPCGRLConfig, seen_games, unseen_ratio: float) -> bool:
    """Inject encoder seen-game metadata into a VIPCGRL train config."""
    if not seen_games:
        return False

    if config.game_setting_mode == "encoder_seen":
        seen_abbrs = dict.fromkeys(GAME_ABBR_INV[g] for g in seen_games if g in GAME_ABBR_INV)
        game_str = "all" if seen_abbrs.keys() == GAME_ABBR.keys() else "".join(seen_abbrs)

        if unseen_ratio > 0.0:
            logger.info(
                "game_setting_mode=encoder_seen + unseen_ratio=%.4f > 0 "
                "→ expanding game to 'all' (dataset_unseen_ratio=%.4f)",
                unseen_ratio, unseen_ratio,
            )
            config.game = "all"
        else:
            logger.info(
                "game_setting_mode=encoder_seen → setting game='%s' (seen_games=%s)",
                game_str, seen_games,
            )
            config.game = game_str

    config.reward_seen_games = list(seen_games)
    logger.info(
        "Auto-setting reward_seen_games=%s from encoder metadata",
        seen_games,
    )
    return True


# ── Hydra entrypoint ──────────────────────────────────────────────────────────

@hydra.main(version_base=None, config_path="./conf", config_name="train_vipcgrl")
def main(config: VIPCGRLConfig):

    if not config.encoder.ckpt_dir or not config.encoder.ckpt_name:
        raise ValueError("Both encoder.ckpt_dir and encoder.ckpt_name must be set in the configuration.")

    # ── encoder의 dataset_setting.json에서 seen_ratio / seen_games 자동 주입 ──
    # (mgpcgrl 의 동일 로직을 포팅. VIPCGRL은 decoder가 없으므로 reward_decoder_mode 는 다루지 않음.)
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

        # ── unseen_ratio 주입: encoder 학습 때 쓴 unseen 게임 데이터 비율을 그대로 사용 ──
        unseen_ratio = dataset_setting.get("unseen_ratio", 0.0)
        if unseen_ratio != config.dataset_unseen_ratio:
            logger.info(
                "Auto-setting dataset_unseen_ratio=%.4f from encoder dataset_setting.json",
                unseen_ratio,
            )
            config.dataset_unseen_ratio = unseen_ratio

        # ── game 범위 결정 + reward_seen_games 주입 ─────────────────────────────
        # The seen/unseen split reflects the encoder training distribution and is
        # used to write train_setting.json for downstream WandB analysis.
        seen_games = dataset_setting.get("seen_games", [])
        if not _apply_seen_games_to_config(config, seen_games, unseen_ratio):
            logger.warning(
                "dataset_setting.json has empty seen_games — "
                "train_setting.json seen/unseen will also be empty"
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

    main_entry(config, inject_obs_fn=inject_vipcgrl_obs)



if __name__ == "__main__":
    main()
