"""tests/test_ipcgrl_baseline.py

IPCGRL (Instructed PCGRL) baseline 통합 테스트.
- IPCGRLConfig 설정 검증 (use_nlp=True, model=nlpconv, encoder.model=mlp)
- 데이터셋 로딩 시 BERT 임베딩 생성 검증
- 짧은 학습 실행 후 정상 종료(exit 0) 검증

실행
----
    python -m pytest tests/test_ipcgrl_baseline.py -v -s
"""

from __future__ import annotations

import os
import sys

import pytest

# ── 프로젝트 루트를 PYTHONPATH 에 추가 ────────────────────────────────────────
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from instruct_rl.utils.log_utils import get_logger, suppress_jax_debug_logs

suppress_jax_debug_logs()
logger = get_logger(__file__)


# ── Helper ─────────────────────────────────────────────────────────────────────

def _make_ipcgrl_config(**overrides):
    """IPCGRL 테스트용 최소 config 를 생성한다."""
    from conf.config import CPCGRLConfig
    from instruct_rl.utils.path_utils import init_config

    defaults = dict(
        n_envs=4,
        num_steps=4,
        total_timesteps=256,
        update_epochs=1,
        NUM_MINIBATCHES=1,
        seed=42,
        overwrite=True,
        ckpt_freq=1,
        render_freq=-1,
        dataset_game="dungeon",
        dataset_reward_enum=1,
        exp_name="test_ipcgrl",
        use_nlp=True,
    )
    defaults.update(overrides)

    config = CPCGRLConfig(**defaults)
    config = init_config(config)
    return config


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture()
def ipcgrl_config():
    return _make_ipcgrl_config()


# ═══════════════════════════════════════════════════════════════════════════════
#  1. IPCGRLConfig 설정 테스트
# ═══════════════════════════════════════════════════════════════════════════════

class TestIPCGRLConfig:
    """IPCGRL 모드 config 검증."""

    def test_ipcgrl_sets_use_nlp(self):
        c = _make_ipcgrl_config()
        assert c.use_nlp is True
        assert c.vec_cont is False
        assert c.use_clip is False

    def test_ipcgrl_model_is_nlpconv(self):
        c = _make_ipcgrl_config()
        assert c.model == "nlpconv"

    def test_ipcgrl_encoder_is_mlp(self):
        """IPCGRL 은 MLP 인코더를 사용한다."""
        c = _make_ipcgrl_config()
        assert c.encoder.model == "mlp"

    def test_ipcgrl_nlp_input_dim(self):
        """BERT base 768 차원이 기본."""
        c = _make_ipcgrl_config()
        assert c.nlp_input_dim == 768
        assert c.vec_input_dim == 768

    def test_ipcgrl_exp_dir_contains_nlp(self):
        """exp_dir 에 _nlp 접미사가 붙어야 한다."""
        c = _make_ipcgrl_config(seed=99, exp_name="ci")
        assert c.exp_dir is not None
        assert "_nlp" in c.exp_dir
        assert "nlpconv" in c.exp_dir

    def test_ipcgrl_disables_vec_cont(self):
        """IPCGRL 은 vec_cont=False 여야 한다."""
        c = _make_ipcgrl_config()
        assert c.vec_cont is False
        assert c.raw_obs is True

    def test_mlp_compresses_bert_embedding(self):
        """MLP encoder.output_dim 이 BERT 임베딩(768)보다 작아야 한다 (압축)."""
        c = _make_ipcgrl_config()
        assert c.encoder.output_dim < c.nlp_input_dim, (
            f"encoder.output_dim({c.encoder.output_dim}) should be < "
            f"nlp_input_dim({c.nlp_input_dim}) for MLP compression"
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  2. 데이터셋 로딩 + 임베딩 테스트
# ═══════════════════════════════════════════════════════════════════════════════

class TestIPCGRLDatasetLoading:
    """IPCGRL 데이터셋 로딩 시 BERT 임베딩이 생성되는지 검증."""

    def test_load_dataset_has_embedding(self, ipcgrl_config):
        """use_nlp=True 일 때 embedding 이 zeros 가 아니어야 한다."""
        from instruct_rl.utils.dataset_loader import load_dataset_instruct
        import jax.numpy as jnp

        train_inst, test_inst = load_dataset_instruct(ipcgrl_config)

        assert train_inst is not None
        assert test_inst is not None
        # embedding shape: (N, 768)
        assert train_inst.embedding.ndim == 2
        assert train_inst.embedding.shape[1] == 768
        # instruction 이 있는 dungeon 샘플이므로 일부 embedding 은 non-zero
        assert not jnp.allclose(train_inst.embedding, 0.0)

    def test_embedding_dtype(self, ipcgrl_config):
        """embedding 이 float32 여야 한다."""
        from instruct_rl.utils.dataset_loader import load_dataset_instruct
        import jax.numpy as jnp

        train_inst, _ = load_dataset_instruct(ipcgrl_config)
        assert train_inst.embedding.dtype == jnp.float32

    def test_condition_still_present(self, ipcgrl_config):
        """IPCGRL 에서도 condition, reward_i 는 여전히 존재해야 한다."""
        from instruct_rl.utils.dataset_loader import load_dataset_instruct

        train_inst, _ = load_dataset_instruct(ipcgrl_config)
        assert train_inst.reward_i.ndim == 2
        assert train_inst.condition.ndim == 2
        assert (train_inst.reward_i == 1).all()


# ═══════════════════════════════════════════════════════════════════════════════
#  3. 학습 E2E 테스트
# ═══════════════════════════════════════════════════════════════════════════════

class TestIPCGRLTraining:
    """IPCGRL 학습 E2E 테스트.

    subprocess 로 train_ipcgrl.py 를 실행하여 정상 종료(exit 0)를 검증한다.
    """

    def test_train_exits_zero(self, tmp_path):
        """train_ipcgrl.py 를 total_timesteps=200 으로 실행하고 exit 0 을 확인한다."""
        import subprocess

        result = subprocess.run(
            [
                sys.executable, "-m", "train_ipcgrl",
                "overwrite=true",
                "total_timesteps=200",
                "n_envs=4",
                "num_steps=4",
                "update_epochs=1",
                "NUM_MINIBATCHES=1",
                "seed=42",
                "ckpt_freq=1",
                "render_freq=-1",
                "eval_freq=-1",
                "exp_name=test_ipcgrl_e2e",
            ],
            cwd=_ROOT,
            capture_output=True,
            text=True,
            timeout=600,
        )

        logger.info(f"stdout (last 2000 chars):\n{result.stdout[-2000:]}")
        if result.returncode != 0:
            logger.error(f"stderr:\n{result.stderr[-3000:]}")

        assert result.returncode == 0, (
            f"train_ipcgrl.py exited with code {result.returncode}\n"
            f"stderr:\n{result.stderr[-3000:]}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

