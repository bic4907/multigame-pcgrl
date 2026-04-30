"""
train_ipcgrl.py
================
IPCGRL (Instructed PCGRL) — BERT 임베딩 → MLP 인코더를 거친 피처를 입력으로 사용.

기존 train.py 의 `encoder.model='mlp'` 모드에 해당하며,
dataset 기반 파이프라인(MultiGameDataset)으로 동작한다.

실행:
    python -m train_ipcgrl [overrides]
"""
import hydra

from conf.config import IPCGRLConfig
from instruct_rl.utils.log_utils import suppress_jax_debug_logs
from instruct_rl.utils.train_utils import main_entry

suppress_jax_debug_logs()


# ── IPCGRL obs 주입: embedding → nlp_obs ─────────────────────────────────────

def inject_ipcgrl_obs(last_obs, env_state, instruct_sample, config, env):
    """BERT 임베딩을 nlp_obs 에 주입. 이후 네트워크 내부의 MLP 인코더가 처리."""
    return last_obs.replace(nlp_obs=instruct_sample.embedding)


# ── Hydra entrypoint ──────────────────────────────────────────────────────────

@hydra.main(version_base=None, config_path="./conf", config_name="train_ipcgrl")
def main(config: IPCGRLConfig):
    main_entry(config, inject_obs_fn=inject_ipcgrl_obs)


if __name__ == "__main__":
    main()
