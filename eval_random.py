"""
eval_random.py
==============
완전 랜덤 정책(Pure Random Policy)으로 PCGRL 환경을 평가하는 엔트리포인트.

NN을 사용하지 않고 매 스텝마다 action space에서 uniform random sampling을 수행한다.
(초기화된 NN policy가 아닌 진짜 random임을 주의)

실행:
    python -m eval_random [overrides]
"""
import hydra

from conf.config import RandomEvalConfig
from instruct_rl.utils.log_utils import suppress_jax_debug_logs
from instruct_rl.utils.eval_utils import main_eval_entry

suppress_jax_debug_logs()


# ── Hydra entrypoint ──────────────────────────────────────────────────────────

@hydra.main(version_base=None, config_path="./conf", config_name="eval_random")
def main(config: RandomEvalConfig):
    # ── 완전 random policy 보장 ──────────────────────────────────────────────
    # random_agent=True → runner.py에서 NN forward pass 없이
    # action space에서 uniform random sampling 수행 (진짜 랜덤, initialized policy 아님)
    config.random_agent = True

    # inject_obs_fn=None: obs 변환 없이 그대로 사용 (어차피 NN 미사용이므로 무관)
    main_eval_entry(config, inject_obs_fn=None)


if __name__ == "__main__":
    main()
