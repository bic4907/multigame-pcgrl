"""Transition reward model utility tests."""
import tempfile

import jax
import jax.numpy as jnp
import pytest


def _make_dummy_configs(num_reward_classes: int = 5):
    from conf.config import DecoderConfig, EncoderConfig

    enc_cfg = EncoderConfig()
    enc_cfg.model = "cnnclip"
    enc_cfg.state = True
    enc_cfg.output_dim = 64
    enc_cfg.token_max_len = 32
    enc_cfg.dropout_rate = 0.0
    enc_cfg.freeze_text_enc = True

    dec_cfg = DecoderConfig()
    dec_cfg.num_reward_classes = num_reward_classes
    dec_cfg.hidden_dim = 64
    dec_cfg.num_layers = 1
    dec_cfg.cnn_reward_enum_onehot = True
    return enc_cfg, dec_cfg


def _init_random_transition_model():
    from encoder.clip_model import get_cnnclip_decoder_encoder

    enc_cfg, dec_cfg = _make_dummy_configs()
    module, _ = get_cnnclip_decoder_encoder(enc_cfg, decoder_config=dec_cfg, RL_training=True)

    rng = jax.random.PRNGKey(42)
    dummy_ids = jnp.ones((1, enc_cfg.token_max_len), dtype=jnp.int32)
    dummy_mask = jnp.ones((1, enc_cfg.token_max_len), dtype=jnp.int32)
    dummy_pix = jnp.ones((1, 16, 16, 7), dtype=jnp.float32)
    variables = module.init(
        rng,
        dummy_ids,
        dummy_mask,
        dummy_pix,
        prev_pixel_values=dummy_pix,
        curr_pixel_values=dummy_pix,
        mode="text_state",
        training=False,
    )
    return module.apply, variables


class TestTransitionReward:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.apply_fn, self.variables = _init_random_transition_model()

    def test_predict_transition_reward_shape(self):
        from encoder.utils.decoder_reward import predict_transition_reward

        n_envs = 4
        instruction_embedding = jax.random.normal(jax.random.PRNGKey(0), (n_envs, 64))
        prev_map = jax.random.randint(jax.random.PRNGKey(1), (n_envs, 16, 16), 1, 6)
        curr_map = jax.random.randint(jax.random.PRNGKey(2), (n_envs, 16, 16), 1, 6)

        rewards = predict_transition_reward(
            self.apply_fn,
            self.variables,
            instruction_embedding,
            prev_map,
            curr_map,
            num_classes=5,
        )

        assert rewards.shape == (n_envs,)
        assert rewards.dtype == jnp.float32
        assert jnp.all(jnp.isfinite(rewards))
        assert jnp.all(rewards >= -2.0)
        assert jnp.all(rewards <= 2.0)

    def test_module_forward_outputs_reward_pred(self):
        n_envs = 2
        ids = jnp.ones((n_envs, 32), dtype=jnp.int32)
        mask = jnp.ones((n_envs, 32), dtype=jnp.int32)
        pix = jnp.ones((n_envs, 16, 16, 7), dtype=jnp.float32)

        out = self.apply_fn(
            self.variables,
            ids,
            mask,
            pix,
            prev_pixel_values=pix,
            curr_pixel_values=pix,
            mode="text_state",
            training=False,
        )

        assert "reward_pred" in out
        assert out["reward_pred"].shape == (n_envs,)
        assert "reward_logits" not in out
        assert "condition_pred_raw" not in out


class TestLoadDecoder:
    def test_load_nonexistent_dir_returns_template(self):
        from encoder.utils.decoder_reward import load_decoder

        enc_cfg, dec_cfg = _make_dummy_configs()
        with tempfile.TemporaryDirectory() as tmpdir:
            apply_fn, variables = load_decoder(tmpdir, enc_cfg, dec_cfg)

        ids = jnp.ones((2, 32), dtype=jnp.int32)
        mask = jnp.ones((2, 32), dtype=jnp.int32)
        pix = jnp.ones((2, 16, 16, 7), dtype=jnp.float32)
        out = apply_fn(
            variables,
            ids,
            mask,
            pix,
            prev_pixel_values=pix,
            curr_pixel_values=pix,
            mode="text_state",
            training=False,
        )
        assert out["reward_pred"].shape == (2,)
