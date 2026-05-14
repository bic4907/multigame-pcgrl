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

    # ── encoder의 dataset_setting.json에서 seen_ratio를 읽어 기록 ──
    dataset_setting_path = os.path.join(config.encoder.ckpt_dir, config.encoder.ckpt_name, "dataset_setting.json")
    if os.path.exists(dataset_setting_path):
        with open(dataset_setting_path, "r") as f:
            dataset_setting = json.load(f)

        # ── seen_ratio 기록: 분석용으로만 저장 (데이터셋 필터링에는 미적용) ──
        seen_ratio = dataset_setting.get("seen_ratio", 1.0)
        if seen_ratio != config.train_seen_ratio:
            logger.info(
                "Auto-setting train_seen_ratio=%.4f from encoder dataset_setting.json (analysis only)",
                seen_ratio,
            )
            config.train_seen_ratio = seen_ratio
    else:
        logger.warning("dataset_setting.json not found at %s", dataset_setting_path)

    main_eval_entry(config, inject_obs_fn=inject_vipcgrl_obs)


if __name__ == "__main__":
    main()

