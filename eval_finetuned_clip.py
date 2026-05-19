"""
eval_finetuned_clip.py
======================
Fine-tuned CLIP PCGRL 평가 entrypoint.

`train_finetuned_clip.py` 로 학습한 체크포인트를 평가한다. inject_obs_fn 은
`train_pretrained_clip.py` 의 것을 그대로 재사용한다.

실행:
    python -m eval_finetuned_clip encoder.ckpt_name=finetuned-clip-...
"""
import hydra

from conf.config import FinetunedCLIPEvalConfig
from instruct_rl.utils.eval_utils import main_eval_entry
from instruct_rl.utils.log_utils import suppress_jax_debug_logs
from train_pretrained_clip import inject_pretrained_clip_pcgrl_obs

suppress_jax_debug_logs()


@hydra.main(version_base=None, config_path="./conf", config_name="eval_finetuned_clip")
def main(config: FinetunedCLIPEvalConfig):
    main_eval_entry(config, inject_obs_fn=inject_pretrained_clip_pcgrl_obs)


if __name__ == "__main__":
    main()

