"""tests/test_vipcgrl_baseline.py

VIPCGRL (Vision-Instructed PCGRL) baseline 통합 테스트.
- Config 설정 검증 (use_clip=True, encoder.model=cnnclip)
- 데이터셋 로딩 시 CLIP 임베딩 생성 검증
- 짧은 학습 실행 후 정상 종료(exit 0) 검증

실행
----
    python -m pytest tests/test_vipcgrl_baseline.py -v -s
"""
from __future__ import annotations

import os
import sys

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from instruct_rl.utils.log_utils import get_logger, suppress_jax_debug_logs

suppress_jax_debug_logs()
logger = get_logger(__file__)


# ── Helper ─────────────────────────────────────────────────────────────────────

def _make_vipcgrl_config(**overrides):
    """VIPCGRL 테스트용 최소 config 를 생성한다."""
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
        wandb_key=None,
        exp_name="test_vipcgrl",
        use_clip=True,
    )
    defaults.update(overrides)

    config = CPCGRLConfig(**defaults)
    config = init_config(config)
    return config


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture()
def vipcgrl_config():
    return _make_vipcgrl_config()


# ═══════════════════════════════════════════════════════════════════════════════
#  1. Config 설정 테스트
# ═══════════════════════════════════════════════════════════════════════════════

class TestVIPCGRLConfig:
    """VIPCGRL 모드 config 검증."""

    def test_vipcgrl_sets_use_clip(self):
        c = _make_vipcgrl_config()
        assert c.use_clip is True
        assert c.vec_cont is False
        assert c.use_nlp is False

    def test_vipcgrl_encoder_is_cnnclip(self):
        """VIPCGRL 은 cnnclip 인코더를 사용한다."""
        c = _make_vipcgrl_config()
        assert c.encoder.model == "cnnclip"

    def test_vipcgrl_nlp_input_dim(self):
        """CLIP text feature 512 차원이 기본."""
        c = _make_vipcgrl_config()
        assert c.nlp_input_dim == 512
        assert c.vec_input_dim == 512

    def test_vipcgrl_enables_sim_reward(self):
        """VIPCGRL 은 use_sim_reward=True 여야 한다."""
        c = _make_vipcgrl_config()
        assert c.use_sim_reward is True

    def test_vipcgrl_exp_dir_contains_clip(self):
        """exp_dir 에 _clip 접미사가 붙어야 한다."""
        c = _make_vipcgrl_config(seed=99, exp_name="ci")
        assert c.exp_dir is not None
        assert "_clip" in c.exp_dir

    def test_vipcgrl_disables_vec_cont_and_nlp(self):
        """VIPCGRL 은 vec_cont=False, use_nlp=False 여야 한다."""
        c = _make_vipcgrl_config()
        assert c.vec_cont is False
        assert c.use_nlp is False
        assert c.raw_obs is True


# ═══════════════════════════════════════════════════════════════════════════════
#  2. 데이터셋 로딩 + CLIP 임베딩 테스트
# ═══════════════════════════════════════════════════════════════════════════════

class TestVIPCGRLDatasetLoading:
    """VIPCGRL 데이터셋 로딩 시 CLIP 임베딩이 생성되는지 검증."""

    def test_load_dataset_has_embedding(self, vipcgrl_config):
        """use_clip=True 일 때 embedding 이 zeros 가 아니어야 한다."""
        from instruct_rl.utils.dataset_loader import load_dataset_instruct
        import jax.numpy as jnp

        train_inst, test_inst = load_dataset_instruct(vipcgrl_config)

        assert train_inst is not None
        assert test_inst is not None
        # embedding shape: (N, 512)
        assert train_inst.embedding.ndim == 2
        assert train_inst.embedding.shape[1] == 512
        # instruction 이 있는 dungeon 샘플이므로 일부 embedding 은 non-zero
        assert not jnp.allclose(train_inst.embedding, 0.0)

    def test_embedding_dtype(self, vipcgrl_config):
        """embedding 이 float32 여야 한다."""
        from instruct_rl.utils.dataset_loader import load_dataset_instruct
        import jax.numpy as jnp

        train_inst, _ = load_dataset_instruct(vipcgrl_config)
        assert train_inst.embedding.dtype == jnp.float32

    def test_clip_embedding_differs_from_bert(self, vipcgrl_config):
        """CLIP 임베딩 차원(512)이 BERT(768)와 다른지 확인."""
        from instruct_rl.utils.dataset_loader import load_dataset_instruct

        train_inst, _ = load_dataset_instruct(vipcgrl_config)
        assert train_inst.embedding.shape[1] == 512  # CLIP, not BERT 768

    def test_condition_still_present(self, vipcgrl_config):
        """VIPCGRL 에서도 condition, reward_i 는 여전히 존재해야 한다."""
        from instruct_rl.utils.dataset_loader import load_dataset_instruct

        train_inst, _ = load_dataset_instruct(vipcgrl_config)
        assert train_inst.reward_i.ndim == 2
        assert train_inst.condition.ndim == 2
        assert (train_inst.reward_i == 1).all()


# ═══════════════════════════════════════════════════════════════════════════════
#  3. 학습 E2E 테스트
# ═══════════════════════════════════════════════════════════════════════════════

class TestVIPCGRLTraining:
    """VIPCGRL 학습 E2E 테스트.

    subprocess 로 train_vipcgrl.py 를 실행하여 정상 종료(exit 0)를 검증한다.
    """

    def test_train_exits_zero(self, tmp_path):
        """train_vipcgrl.py 를 total_timesteps=200 으로 실행하고 exit 0 을 확인한다."""
        import subprocess

        result = subprocess.run(
            [
                sys.executable, "-m", "train_vipcgrl",
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
                "exp_name=test_vipcgrl_e2e",
                "wandb_key=null",
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
            f"train_vipcgrl.py exited with code {result.returncode}\n"
            f"stderr:\n{result.stderr[-3000:]}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

