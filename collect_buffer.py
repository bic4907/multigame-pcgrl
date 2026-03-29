"""
collect_buffer.py
==================
RL 에이전트 학습과 동시에 trajectory 버퍼를 수집하는 엔트리포인트.

학습 구간의 일부(기본 50%~100%)에서 첫 번째 환경(env_idx=0) 기준으로
일정 주기로 (obs, action, reward, done, env_map) 데이터를 수집하여
실험 폴더의 buffer/ 디렉토리에 .npz 파일로 저장한다.

수집 간격은 buffer_max_samples / num_steps 로 자동 계산된다.

실행:
    python -m collect_buffer [overrides]

주요 파라미터:
    buffer_max_samples    : 수집할 최대 transition 수 (기본 10,000)
    collect_start_ratio   : 수집 시작 시점 (기본 0.5 = 학습 50%)
    collect_end_ratio     : 수집 종료 시점 (기본 1.0 = 학습 100%)
    buffer_save_dir       : 저장 경로 (기본 None → exp_dir/buffer)
"""
import jax
import hydra

from conf.config import CollectBufferConfig
from instruct_rl.utils.log_utils import suppress_jax_debug_logs
from instruct_rl.utils.train_utils import main_entry

suppress_jax_debug_logs()


# ── CPCGRL obs 주입: get_cont_obs → nlp_obs ──────────────────────────────────

def inject_cpcgrl_obs(last_obs, env_state, instruct_sample, config, env):
    """env_map + condition 으로 continuous observation 을 계산하여 nlp_obs 에 주입."""
    vmap_state_fn = jax.vmap(env.prob.get_cont_obs, in_axes=(0, 0, None))
    cont_obs = vmap_state_fn(
        env_state.env_state.env_map,
        instruct_sample.condition,
        config.raw_obs,
    )
    return last_obs.replace(nlp_obs=cont_obs)


# ── Hydra entrypoint ──────────────────────────────────────────────────────────

@hydra.main(version_base=None, config_path="./conf", config_name="collect_buffer")
def main(config: CollectBufferConfig):
    main_entry(config, inject_obs_fn=inject_cpcgrl_obs)


if __name__ == "__main__":
    main()
