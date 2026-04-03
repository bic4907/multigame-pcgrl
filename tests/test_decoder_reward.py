"""
tests/test_decoder_reward.py
=============================
CLIP Decoder → reward_enum / condition 예측 유틸 단위 테스트.

체크포인트 없이 **랜덤 가중치**로 초기화된 디코더를 사용하여
`predict_reward_condition` 의 출력 shape / dtype / 범위를 검증한다.

실행:
    pytest tests/test_decoder_reward.py -v
"""
import pytest
import jax
import jax.numpy as jnp


# ── 더미 config 생성 ──────────────────────────────────────────────────────────

def _make_dummy_configs(num_reward_classes: int = 5):
    """encoder / decoder config를 최소 설정으로 생성한다."""
    from conf.config import EncoderConfig, DecoderConfig

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

    return enc_cfg, dec_cfg


# ── 랜덤 초기화된 디코더를 반환 (체크포인트 불필요) ──────────────────────────

def _init_random_decoder(num_reward_classes: int = 5):
    """랜덤 가중치로 ContrastiveDecoderModule 을 초기화하여
    (apply_fn, variables) 를 반환한다.
    """
    from encoder.clip_model import get_cnnclip_decoder_encoder

    enc_cfg, dec_cfg = _make_dummy_configs(num_reward_classes)

    norm_min = jnp.array([0.0] * num_reward_classes, dtype=jnp.float32)
    norm_max = jnp.array([100.0] * num_reward_classes, dtype=jnp.float32)

    module, _ = get_cnnclip_decoder_encoder(
        enc_cfg,
        decoder_config=dec_cfg,
        cond_norm_min=norm_min,
        cond_norm_max=norm_max,
        RL_training=True,
    )

    rng = jax.random.PRNGKey(42)
    dummy_ids = jnp.ones((1, enc_cfg.token_max_len), dtype=jnp.int32)
    dummy_mask = jnp.ones((1, enc_cfg.token_max_len), dtype=jnp.int32)
    dummy_pix = jnp.ones((1, 16, 16, 6), dtype=jnp.float32)

    variables = module.init(
        rng, dummy_ids, dummy_mask, dummy_pix,
        mode="text_state", training=False,
    )
    return module.apply, variables


# ═══════════════════════════════════════════════════════════════════════════════
#  Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestPredictRewardCondition:
    """predict_reward_condition 의 출력 검증."""

    NUM_CLASSES = 5

    @pytest.fixture(autouse=True)
    def setup(self):
        self.apply_fn, self.variables = _init_random_decoder(self.NUM_CLASSES)

    def test_output_shapes(self):
        """reward_i: (n_envs, 1), condition: (n_envs, 9)."""
        from encoder.utils.decoder_reward import predict_reward_condition

        n_envs = 4
        pixel_values = jax.random.normal(
            jax.random.PRNGKey(0), (n_envs, 16, 16, 6)
        )

        reward_i, condition = predict_reward_condition(
            self.apply_fn, self.variables,
            pixel_values, num_reward_classes=self.NUM_CLASSES,
        )

        assert reward_i.shape == (n_envs, 1), f"reward_i shape: {reward_i.shape}"
        assert condition.shape == (n_envs, 9), f"condition shape: {condition.shape}"

    def test_reward_i_dtype_and_range(self):
        """reward_i 는 int32 이고 1-based (1 ~ num_classes)."""
        from encoder.utils.decoder_reward import predict_reward_condition

        n_envs = 8
        pixel_values = jax.random.normal(
            jax.random.PRNGKey(1), (n_envs, 16, 16, 6)
        )

        reward_i, _ = predict_reward_condition(
            self.apply_fn, self.variables,
            pixel_values, num_reward_classes=self.NUM_CLASSES,
        )

        assert reward_i.dtype == jnp.int32
        assert jnp.all(reward_i >= 1)
        assert jnp.all(reward_i <= self.NUM_CLASSES)

    def test_condition_has_predicted_value(self):
        """예측된 enum 슬롯에는 실수 값이, 나머지는 -1 이어야 한다."""
        from encoder.utils.decoder_reward import predict_reward_condition

        n_envs = 4
        pixel_values = jax.random.normal(
            jax.random.PRNGKey(2), (n_envs, 16, 16, 6)
        )

        reward_i, condition = predict_reward_condition(
            self.apply_fn, self.variables,
            pixel_values, num_reward_classes=self.NUM_CLASSES,
        )

        for i in range(n_envs):
            enum_0based = int(reward_i[i, 0]) - 1
            # 예측 슬롯: -1 이 아닌 값이어야 함
            assert float(condition[i, enum_0based]) != -1.0, (
                f"env {i}: predicted enum slot {enum_0based} should not be -1"
            )

    def test_condition_unused_slots_are_negative_one(self):
        """예측 enum 외의 condition 슬롯은 -1."""
        from encoder.utils.decoder_reward import predict_reward_condition

        n_envs = 4
        pixel_values = jax.random.normal(
            jax.random.PRNGKey(3), (n_envs, 16, 16, 6)
        )

        reward_i, condition = predict_reward_condition(
            self.apply_fn, self.variables,
            pixel_values, num_reward_classes=self.NUM_CLASSES,
        )

        for i in range(n_envs):
            enum_0based = int(reward_i[i, 0]) - 1
            for j in range(9):
                if j != enum_0based:
                    assert float(condition[i, j]) == -1.0, (
                        f"env {i}: slot {j} should be -1, got {condition[i, j]}"
                    )

    def test_get_reward_batch_compatible(self):
        """예측 결과를 get_reward_batch 에 넘겨도 에러 없이 실행되는지 검증."""
        from encoder.utils.decoder_reward import predict_reward_condition
        from evaluator import get_reward_batch

        n_envs = 4
        pixel_values = jax.random.normal(
            jax.random.PRNGKey(4), (n_envs, 16, 16, 6)
        )

        reward_i, condition = predict_reward_condition(
            self.apply_fn, self.variables,
            pixel_values, num_reward_classes=self.NUM_CLASSES,
        )

        # 더미 env_map
        prev_map = jax.random.randint(
            jax.random.PRNGKey(5), (n_envs, 16, 16), 0, 7
        )
        curr_map = jax.random.randint(
            jax.random.PRNGKey(6), (n_envs, 16, 16), 0, 7
        )

        rewards = get_reward_batch(
            reward_i, condition, prev_map, curr_map, map_size=16,
        )

        assert rewards.shape == (n_envs,), f"rewards shape: {rewards.shape}"
        assert jnp.all(jnp.isfinite(rewards)), "rewards contain NaN/Inf"

    def test_batch_size_one(self):
        """n_envs=1 에서도 정상 동작."""
        from encoder.utils.decoder_reward import predict_reward_condition

        pixel_values = jax.random.normal(
            jax.random.PRNGKey(7), (1, 16, 16, 6)
        )

        reward_i, condition = predict_reward_condition(
            self.apply_fn, self.variables,
            pixel_values, num_reward_classes=self.NUM_CLASSES,
        )

        assert reward_i.shape == (1, 1)
        assert condition.shape == (1, 9)


class TestLoadDecoder:
    """load_decoder 체크포인트 로딩 (파일 없을 때 graceful fail)."""

    def test_load_nonexistent_dir_returns_template(self):
        """존재하지 않는 ckpt_dir 을 전달하면 restore_checkpoint 가
        template 을 그대로 반환 → apply_fn 은 여전히 호출 가능."""
        from encoder.utils.decoder_reward import load_decoder
        import tempfile

        enc_cfg, dec_cfg = _make_dummy_configs(5)

        with tempfile.TemporaryDirectory() as tmpdir:
            apply_fn, variables = load_decoder(
                tmpdir, enc_cfg, dec_cfg,
                cond_norm_min=jnp.zeros(5),
                cond_norm_max=jnp.ones(5) * 100,
            )

        # 호출 가능 확인
        dummy_pix = jnp.ones((2, 16, 16, 6), dtype=jnp.float32)
        dummy_ids = jnp.ones((2, 32), dtype=jnp.int32)
        dummy_mask = jnp.ones((2, 32), dtype=jnp.int32)

        out = apply_fn(
            variables, dummy_ids, dummy_mask, dummy_pix,
            mode="text_state", training=False,
        )
        assert "reward_logits" in out
        assert "condition_pred_raw" in out

