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
import jax
import jax.numpy as jnp
import hydra

from conf.config import PretrainedCLIPPCGRLConfig
from instruct_rl.utils.log_utils import suppress_jax_debug_logs
from instruct_rl.utils.train_utils import _cosine_similarity, main_entry
from instruct_rl.utils.img_preprocess import render_level_from_arr, clip_batch_preprocess

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
    network_apply_fn,
    params,
    last_obs,
) -> jnp.ndarray:
    """Pretrained CLIP 코사인 유사도 delta 보상.

    맵 이미지를 직접 렌더링 → CLIP 전처리 → 비전 인코더로 state embedding 계산 후
    텍스트 임베딩과의 코사인 유사도 delta를 보상으로 반환한다.

    Parameters
    ----------
    prev_env_state : LogWrapper state (이전 스텝)
    curr_env_state : LogWrapper state (현재 스텝)
    instruct_sample : Instruct
        ``instruct_sample.embedding`` — 사전 계산된 CLIP 텍스트 임베딩 (B, D).
    network_apply_fn : Callable
        ``network.apply`` — CLIP 네트워크의 apply 함수.
    params : PyTree
        네트워크 파라미터 (train_state.params).
    last_obs : Observation
        현재 관측 (pixel_values 슬롯을 임시로 대체하는 데 사용).

    Returns
    -------
    reward : jnp.ndarray (B,)
        cos_sim(text, curr_map) - cos_sim(text, prev_map)
        → 텍스트 지시에 가까워질수록 양수 보상.
    """
    # ── 렌더링: env_map → 타일 픽셀 이미지 ──────────────────────────────────
    curr_rendered = jax.vmap(render_level_from_arr)(
        curr_env_state.env_state.env_map
    )  # (B, H_px, W_px, C)
    prev_rendered = jax.vmap(render_level_from_arr)(
        prev_env_state.env_state.env_map
    )

    # ── CLIP 전처리: float32 변환 + 224×224 리사이즈 + 정규화 ─────────────────
    curr_clip_pixels = clip_batch_preprocess(curr_rendered.astype(jnp.float32))
    prev_clip_pixels = clip_batch_preprocess(prev_rendered.astype(jnp.float32))

    # ── Pretrained CLIP 비전 인코더로 state embedding 획득 ───────────────────
    _, _, _, _, curr_state_embed = network_apply_fn(
        params,
        last_obs.replace(pixel_values=curr_clip_pixels),
        return_text_embed=False,
        return_state_embed=True,
    )
    _, _, _, _, prev_state_embed = network_apply_fn(
        params,
        last_obs.replace(pixel_values=prev_clip_pixels),
        return_text_embed=False,
        return_state_embed=True,
    )

    # ── 코사인 유사도 delta 보상 ──────────────────────────────────────────────
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



