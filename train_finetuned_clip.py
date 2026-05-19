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
import hydra

from conf.config import FinetunedCLIPPCGRLConfig
from instruct_rl.utils.log_utils import suppress_jax_debug_logs
from instruct_rl.utils.train_utils import main_entry
from train_pretrained_clip import (inject_pretrained_clip_pcgrl_obs,
                                   inject_pretrained_clip_reward)

suppress_jax_debug_logs()


@hydra.main(version_base=None, config_path="./conf", config_name="finetuned_clip_pcgrl")
def main(config: FinetunedCLIPPCGRLConfig):
    main_entry(
        config,
        inject_obs_fn=inject_pretrained_clip_pcgrl_obs,
        inject_reward_fn=inject_pretrained_clip_reward,
    )


if __name__ == "__main__":
    main()

