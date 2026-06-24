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
import shutil

import hydra

from conf.config import MGPCGRLConfig
from conf.game_utils import GAME_ABBR, GAME_ABBR_INV
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

    # ── encoder_config.json에서 delta_weight 읽어서 config에 주입 (wandb 로깅용) ──
    encoder_config_path = os.path.join(config.encoder.ckpt_dir, config.encoder.ckpt_name, "encoder_config.json")
    if os.path.exists(encoder_config_path):
        with open(encoder_config_path, "r") as f:
            encoder_training_config = json.load(f)
        # delta_weight만 config에 저장
        config.encoder_delta_weight = encoder_training_config.get('delta_weight', 0.0)
        logger.info("Loaded encoder delta_weight=%.4f from: %s", 
                    config.encoder_delta_weight, encoder_config_path)
        
        # encoder_config.json을 PCGRL 학습 폴더로 복사 (나중에 참조 가능하도록)
        # exp_dir는 init_config() 이후에 설정되므로, main_entry 안에서 복사해야 함.
        # 여기서는 경로만 저장해두고, main_entry 안에서 복사하도록 표시
        config._encoder_config_src = encoder_config_path
    else:
        logger.warning("encoder_config.json not found at %s", encoder_config_path)
        config.encoder_delta_weight = 0.0
        config._encoder_config_src = None

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

        # ── unseen_ratio 주입 ──────────────────────────────────────────────────
        # MGPCGRL: unseen 게임 데이터는 항상 전체(1.0)를 로드한다.
        #   - reward decoder가 있으므로 encoder가 보지 못한 샘플도 condition 예측 가능.
        #   - dataset_unseen_ratio=1.0 으로 고정하여 load_dataset_instruct 의
        #     per-game ratio 필터링이 unseen 게임을 100% 포함하도록 한다.
        unseen_ratio = dataset_setting.get("unseen_ratio", 0.0)
        if config.dataset_unseen_ratio != 1.0:
            logger.info(
                "MGPCGRL: fixing dataset_unseen_ratio=1.0 (full load; "
                "encoder unseen_ratio=%.4f is used for reward_unseen_ratio only)",
                unseen_ratio,
            )
            config.dataset_unseen_ratio = 1.0

        # ── reward_unseen_ratio: unseen 샘플 내 metadata/decoder 경계 ──────────
        # 각 unseen 게임의 샘플을 순서 기준으로 분할:
        #   앞쪽 (reward_unseen_ratio 비율) → metadata (GT, encoder 학습분)
        #   나머지                          → reward decoder 로 condition 예측
        if unseen_ratio != config.reward_unseen_ratio:
            logger.info(
                "Auto-setting reward_unseen_ratio=%.4f from encoder dataset_setting.json",
                unseen_ratio,
            )
            config.reward_unseen_ratio = unseen_ratio

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
        # The seen/unseen split reflects the encoder training distribution and is
        # independent of reward_decoder_mode (noop / all / unseen). rdm only
        # controls how reward annotations are computed; seen/unseen lists must
        # always come from the encoder.
        seen_games = dataset_setting.get("seen_games", [])
        unseen_games = dataset_setting.get("unseen_games", [])
        if seen_games:
            config.reward_seen_games = list(seen_games)
            logger.info(
                "Auto-setting reward_seen_games=%s from encoder dataset_setting.json "
                "(reward_decoder_mode=%s, reward_unseen_ratio=%.4f)",
                seen_games, config.reward_decoder_mode, config.reward_unseen_ratio,
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
        inject_obs_fn=inject_vipcgrl_obs,
    )
    
    # ── encoder_config.json을 PCGRL 학습 폴더로 복사 (참조용) ──
    if hasattr(config, '_encoder_config_src') and config._encoder_config_src:
        if hasattr(config, 'exp_dir') and config.exp_dir:
            dst_path = os.path.join(config.exp_dir, "encoder_config.json")
            try:
                shutil.copy2(config._encoder_config_src, dst_path)
                logger.info("Copied encoder_config.json to: %s", dst_path)
            except Exception as e:
                logger.warning("Failed to copy encoder_config.json: %s", e)


if __name__ == "__main__":
    main()
