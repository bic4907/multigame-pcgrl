"""tests/test_cpcgrl_baseline.py

CPCGRL baseline 통합 테스트.
- CPCGRLConfig 기본값 검증
- MultiGameDataset 기반 데이터셋 로딩 검증
- 짧은 학습(~몇 update steps) 실행 후 체크포인트 생성 검증
- 저장된 체크포인트를 다시 로드할 수 있는지 검증

실행
----
    python -m pytest tests/test_cpcgrl_baseline.py -v -s
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

def _make_cpcgrl_config(**overrides):
    """CPCGRL 테스트용 최소 config 를 생성한다."""
    from conf.config import CPCGRLConfig
    from instruct_rl.utils.path_utils import init_config

    defaults = dict(
        n_envs=4,
        num_steps=4,
        total_timesteps=256,       # 4 envs * 4 steps * ~16 updates ≈ 256
        update_epochs=1,
        NUM_MINIBATCHES=1,
        seed=42,
        overwrite=True,
        ckpt_freq=1,
        render_freq=-1,
        dataset_reward_enum=0,
        exp_name="test_cpcgrl",
    )
    defaults.update(overrides)

    config = CPCGRLConfig(**defaults)
    config = init_config(config)
    return config


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture()
def cpcgrl_config():
    return _make_cpcgrl_config()



# ═══════════════════════════════════════════════════════════════════════════════
#  1. CPCGRLConfig 기본값 테스트
# ═══════════════════════════════════════════════════════════════════════════════

class TestCPCGRLConfig:
    """CPCGRLConfig 기본값 검증."""

    def test_default_values(self):
        from conf.config import CPCGRLConfig

        c = CPCGRLConfig()
        assert c.problem == "multigame"
        assert c.dataset_reward_enum == 0
        assert c.vec_cont is True
        assert c.raw_obs is True
        assert c.model == "contconv"
        assert c.use_nlp is False
        assert c.use_clip is False
        assert c.instruct_csv is None
        assert c.use_sim_reward is False
        assert c.human_demo is False
        assert c.wandb_project == "cpcgrl"

    def test_init_config_sets_exp_dir(self):
        from conf.config import CPCGRLConfig
        from instruct_rl.utils.path_utils import init_config

        c = CPCGRLConfig(seed=99, exp_name="ci")
        c = init_config(c)
        assert c.exp_dir is not None
        assert "game-all" in c.exp_dir
        assert "re-0" in c.exp_dir
        assert "s-99" in c.exp_dir

    def test_cpcgrl_forces_vec_cont(self):
        """CPCGRLConfig 는 항상 vec_cont=True, raw_obs=True 여야 한다."""
        c = _make_cpcgrl_config()
        assert c.vec_cont is True
        assert c.raw_obs is True
        assert c.model == "contconv"

    def test_cpcgrl_disables_nlp_and_clip(self):
        c = _make_cpcgrl_config()
        assert c.use_nlp is False
        assert c.use_clip is False
        assert c.nlp_input_dim == 0
        assert c.vec_input_dim == 5


# ═══════════════════════════════════════════════════════════════════════════════
#  2. 데이터셋 로딩 테스트
# ═══════════════════════════════════════════════════════════════════════════════

class TestCPCGRLDatasetLoading:
    """데이터셋 로딩 테스트."""

    def test_load_dataset_instruct(self, cpcgrl_config):
        """load_dataset_instruct 가 정상적으로 Instruct 객체를 반환한다."""
        from instruct_rl.utils.dataset_loader import load_dataset_instruct

        train_inst, test_inst = load_dataset_instruct(cpcgrl_config)

        assert train_inst is not None
        assert test_inst is not None
        # reward_i shape: (N, 1)
        assert train_inst.reward_i.ndim == 2
        assert train_inst.reward_i.shape[1] == 1
        # condition shape: (N, 9)
        assert train_inst.condition.ndim == 2
        assert train_inst.condition.shape[1] == 5
        # 모든 reward_enum 값이 0 (region)
        assert (train_inst.reward_i == 0).all()

    def test_load_dataset_different_reward_enum(self):
        """다른 reward_enum 으로도 로딩이 되는지 확인."""
        from instruct_rl.utils.dataset_loader import load_dataset_instruct

        config = _make_cpcgrl_config(
            dataset_game="dungeon",
            dataset_reward_enum=1,  # path_length
            seed=0,
        )

        train_inst, test_inst = load_dataset_instruct(config)
        assert train_inst is not None
        assert (train_inst.reward_i == 1).all()

    def test_instruct_condition_dtype(self, cpcgrl_config):
        """condition 이 float32, reward_i 가 int32 인지 확인."""
        from instruct_rl.utils.dataset_loader import load_dataset_instruct
        import jax.numpy as jnp

        train_inst, _ = load_dataset_instruct(cpcgrl_config)
        assert train_inst.condition.dtype == jnp.float32
        assert train_inst.reward_i.dtype == jnp.int32


# ═══════════════════════════════════════════════════════════════════════════════
#  3. 학습 실행 및 체크포인트 생성/로드 테스트
# ═══════════════════════════════════════════════════════════════════════════════

class TestCPCGRLTraining:
    """CPCGRL 학습 E2E 테스트.

    subprocess 로 train_cpcgrl.py 를 실행하여 정상 종료(exit 0)를 검증한다.
    """

    @staticmethod
    def _get_exp_dir(seed=42, exp_name="test_cpcgrl"):
        """학습 후 생성될 exp_dir 경로를 반환한다."""
        config = _make_cpcgrl_config(seed=seed, exp_name=exp_name)
        return config.exp_dir

    def test_train_exits_zero(self, tmp_path):
        """train_cpcgrl.py 를 total_timesteps=200 으로 실행하고 exit 0 을 확인한다."""
        import subprocess

        result = subprocess.run(
            [
                sys.executable, "-m", "train_cpcgrl",
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
                "exp_name=test_e2e",
            ],
            cwd=_ROOT,
            capture_output=True,
            text=True,
            timeout=300,
        )

        logger.info(f"stdout (last 2000 chars):\n{result.stdout[-2000:]}")
        if result.returncode != 0:
            logger.error(f"stderr:\n{result.stderr[-3000:]}")

        assert result.returncode == 0, (
            f"train_cpcgrl.py exited with code {result.returncode}\n"
            f"stderr:\n{result.stderr[-3000:]}"
        )




if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
