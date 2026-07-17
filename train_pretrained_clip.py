"""
train_pretrained_clip.py
========================
Pretrained CLIP based PCGRL training.

map image and  text instruction(instruct_sample.embedding) text of
text text also  delta  reward as  text for text.

Usage:
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


# ── obs inject: precomputed CLIP text embedding → nlp_obs ──────────────────────

def inject_pretrained_clip_pcgrl_obs(last_obs, env_state, instruct_sample, config, env):
    """pretrained CLIP  as  precomputed text embedding  nlp_obs  in  inject."""
    return last_obs.replace(nlp_obs=instruct_sample.embedding)


# ── reward inject: map image ↔ text embedding text text also  delta ─────────────────

def inject_pretrained_clip_reward(
    prev_env_state,
    curr_env_state,
    instruct_sample,
    network_apply_fn,
    params,
    last_obs,
) -> jnp.ndarray:
    """Pretrained CLIP text text also  delta reward.

    map image  direct rendering → CLIP preprocessing → vision text to  state embedding compute  after
    text embedding and  of  text text also  delta  reward as  returntext.

    Parameters
    ----------
    prev_env_state : LogWrapper state (previous text)
    curr_env_state : LogWrapper state (current text)
    instruct_sample : Instruct
        ``instruct_sample.embedding`` — precomputed CLIP text embedding (B, D).
    network_apply_fn : Callable
        ``network.apply`` — CLIP network of  apply function.
    params : PyTree
        network parameter (train_state.params).
    last_obs : Observation
        current observation (pixel_values slot  text to  text  text text for ).

    Returns
    -------
    reward : jnp.ndarray (B,)
        cos_sim(text, curr_map) - cos_sim(text, prev_map)
        → text text in   text text reward.
    """
    # ── rendering: env_map → tile textcell image ──────────────────────────────────
    curr_rendered = jax.vmap(render_level_from_arr)(
        curr_env_state.env_state.env_map
    )  # (B, H_px, W_px, C)
    prev_rendered = jax.vmap(render_level_from_arr)(
        prev_env_state.env_state.env_map
    )

    # ── CLIP preprocessing: float32 convert + 224×224 text text + normalize ─────────────────
    curr_clip_pixels = clip_batch_preprocess(curr_rendered.astype(jnp.float32))
    prev_clip_pixels = clip_batch_preprocess(prev_rendered.astype(jnp.float32))

    # ── Pretrained CLIP vision text to  state embedding text ───────────────────
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

    # ── text text also  delta reward ──────────────────────────────────────────────
    # precomputed text embedding (B, D) — training  before  CSV  to text  create
    text_embed = instruct_sample.embedding  # already L2-normalized

    curr_sim = _cosine_similarity(text_embed, curr_state_embed)  # (B,)
    prev_sim = _cosine_similarity(text_embed, prev_state_embed)  # (B,)

    # text also   text text text reward
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



