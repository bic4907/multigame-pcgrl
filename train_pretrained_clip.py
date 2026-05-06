"""
train_pretrained_clip.py
========================
Pretrained CLIP 기반 PCGRL 학습.

맵 이미지와 텍스트 지시어(instruct_sample.embedding) 간의
코사인 유사도 delta를 보상으로 사용한다.

실행:
    python -m train_pretrained_clip [overrides]
    python -m train_pretrained_clip dataset_game=dungeon dataset_reward_enum=1
"""
import jax.numpy as jnp
import hydra

from conf.config import PretrainedCLIPPCGRLConfig
from instruct_rl.utils.log_utils import suppress_jax_debug_logs
from instruct_rl.utils.train_utils import _cosine_similarity, main_entry

suppress_jax_debug_logs()


# ── obs 주입: 사전 계산된 CLIP 텍스트 임베딩 → nlp_obs ──────────────────────

def inject_pretrained_clip_pcgrl_obs(last_obs, env_state, instruct_sample, config, env):
    """pretrained CLIP 으로 사전 계산된 텍스트 임베딩을 nlp_obs 에 주입."""
    return last_obs.replace(nlp_obs=instruct_sample.embedding)


# ── 보상 주입: 맵 이미지 ↔ 텍스트 임베딩 코사인 유사도 delta ─────────────────

def inject_pretrained_clip_reward(
    prev_env_state,
    curr_env_state,
    instruct_sample,
    prev_state_embed: jnp.ndarray,
    curr_state_embed: jnp.ndarray,
) -> jnp.ndarray:
    """Pretrained CLIP 코사인 유사도 delta 보상.

    Parameters
    ----------
    prev_env_state : LogWrapper state (이전 스텝)
    curr_env_state : LogWrapper state (현재 스텝)
    instruct_sample : Instruct
        ``instruct_sample.embedding`` — 사전 계산된 CLIP 텍스트 임베딩 (B, D).
    prev_state_embed : jnp.ndarray (B, D)
        이전 맵 이미지를 pretrained CLIP 비전 인코더로 인코딩한 state embedding.
    curr_state_embed : jnp.ndarray (B, D)
        현재 맵 이미지를 pretrained CLIP 비전 인코더로 인코딩한 state embedding.

    Returns
    -------
    reward : jnp.ndarray (B,)
        cos_sim(text, curr_map) - cos_sim(text, prev_map)
        → 텍스트 지시에 가까워질수록 양수 보상.
    """
    # 사전 계산된 텍스트 임베딩 (B, D) — 학습 전 CSV 로더가 생성
    text_embed = instruct_sample.embedding  # already L2-normalized

    curr_sim = _cosine_similarity(text_embed, curr_state_embed)  # (B,)
    prev_sim = _cosine_similarity(text_embed, prev_state_embed)  # (B,)

    # 유사도가 증가한 만큼 보상
    return curr_sim - prev_sim


# ── Hydra entrypoint ──────────────────────────────────────────────────────────

@hydra.main(version_base=None, config_path="./conf", config_name="pretrained_clip_pcgrl")
def main(config: PretrainedCLIPPCGRLConfig):
    main_entry(
        config,
        inject_obs_fn=inject_pretrained_clip_pcgrl_obs,
        inject_reward_fn=inject_pretrained_clip_reward,
    )


if __name__ == "__main__":
    main()



