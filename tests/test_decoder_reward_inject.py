import types

import jax.numpy as jnp

from encoder.utils.decoder_reward import (
    build_decoder_reward_inject_fn,
    _extract_instruction_embedding,
)


def test_extract_instruction_embedding_prefers_instruct_sample():
    instruct_sample = types.SimpleNamespace(embedding=jnp.ones((2, 64), dtype=jnp.float32))
    curr_obs = types.SimpleNamespace(nlp_obs=jnp.zeros((2, 64), dtype=jnp.float32))

    emb = _extract_instruction_embedding(instruct_sample, curr_obs)
    assert emb.shape == (2, 64)
    assert jnp.all(emb == 1.0)


def test_extract_instruction_embedding_fallback_curr_obs():
    curr_obs = types.SimpleNamespace(nlp_obs=jnp.ones((3, 32), dtype=jnp.float32) * 5)
    emb = _extract_instruction_embedding(None, curr_obs)
    assert emb.shape == (3, 32)
    assert jnp.all(emb == 5.0)


def test_build_decoder_reward_inject_fn_with_monkeypatch(monkeypatch):
    # load_decoder / predict_from_instruction 을 가짜로 바꿔서 콜백 wiring만 검증
    def fake_load_decoder(**kwargs):
        return object(), {"params": {}}

    def fake_predict_from_instruction(apply_fn, variables, instruction_embedding, **kwargs):
        n = instruction_embedding.shape[0]
        reward_i = jnp.ones((n, 1), dtype=jnp.int32) * 3
        condition = jnp.full((n, 9), -1.0, dtype=jnp.float32)
        condition = condition.at[:, 2].set(42.0)
        return reward_i, condition

    monkeypatch.setattr("encoder.utils.decoder_reward.load_decoder", fake_load_decoder)
    monkeypatch.setattr("encoder.utils.decoder_reward.predict_from_instruction", fake_predict_from_instruction)

    config = types.SimpleNamespace(
        decoder_ckpt_path="/tmp/fake",
        decoder_reward_classes=5,
        encoder=types.SimpleNamespace(token_max_len=32),
    )

    fn = build_decoder_reward_inject_fn(config)

    prev_env_state = types.SimpleNamespace(env_state=types.SimpleNamespace(env_map=jnp.zeros((4, 16, 16), dtype=jnp.int32)))
    curr_env_state = types.SimpleNamespace(env_state=types.SimpleNamespace(env_map=jnp.zeros((4, 16, 16), dtype=jnp.int32)))
    instruct_sample = types.SimpleNamespace(embedding=jnp.ones((4, 64), dtype=jnp.float32))

    reward_i, condition = fn(prev_env_state, curr_env_state, None, None, instruct_sample, config, None)

    assert reward_i.shape == (4, 1)
    assert condition.shape == (4, 9)
    assert jnp.all(reward_i == 3)
    assert jnp.all(condition[:, 2] == 42.0)


def test_build_decoder_reward_inject_fn_passes_dummy_decoder(monkeypatch):
    called = {}

    def fake_load_decoder(**kwargs):
        called.update(kwargs)
        return object(), {"params": {}}

    def fake_predict_from_instruction(apply_fn, variables, instruction_embedding, **kwargs):
        n = instruction_embedding.shape[0]
        return jnp.ones((n, 1), dtype=jnp.int32), jnp.full((n, 9), -1.0, dtype=jnp.float32)

    monkeypatch.setattr("encoder.utils.decoder_reward.load_decoder", fake_load_decoder)
    monkeypatch.setattr("encoder.utils.decoder_reward.predict_from_instruction", fake_predict_from_instruction)

    config = types.SimpleNamespace(
        decoder_ckpt_path=None,
        decoder_reward_classes=5,
        dummy_decoder=True,
        encoder=types.SimpleNamespace(token_max_len=32),
    )

    fn = build_decoder_reward_inject_fn(config)
    _ = fn(
        types.SimpleNamespace(env_state=types.SimpleNamespace(env_map=jnp.zeros((2, 16, 16), dtype=jnp.int32))),
        types.SimpleNamespace(env_state=types.SimpleNamespace(env_map=jnp.zeros((2, 16, 16), dtype=jnp.int32))),
        None,
        types.SimpleNamespace(nlp_obs=jnp.ones((2, 64), dtype=jnp.float32)),
        None,
        config,
        None,
    )

    assert called.get("dummy_decoder") is True
