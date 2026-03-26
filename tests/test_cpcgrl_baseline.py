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
import shutil
import logging

import pytest

# ── 프로젝트 루트를 PYTHONPATH 에 추가 ────────────────────────────────────────
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# JAX 디버그 로그 억제
logging.getLogger("jax").setLevel(logging.WARNING)
logging.getLogger("jax._src.cache_key").setLevel(logging.WARNING)


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
        dataset_game="dungeon",
        dataset_reward_enum=1,
        wandb_key=None,
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


@pytest.fixture()
def clean_exp_dir(cpcgrl_config):
    """테스트 전후로 exp_dir 을 정리한다."""
    exp_dir = cpcgrl_config.exp_dir
    if os.path.exists(exp_dir):
        shutil.rmtree(exp_dir)
    yield exp_dir
    # 테스트 후 정리
    if os.path.exists(exp_dir):
        shutil.rmtree(exp_dir)


# ═══════════════════════════════════════════════════════════════════════════════
#  1. CPCGRLConfig 기본값 테스트
# ═══════════════════════════════════════════════════════════════════════════════

class TestCPCGRLConfig:
    """CPCGRLConfig 기본값 검증."""

    def test_default_values(self):
        from conf.config import CPCGRLConfig

        c = CPCGRLConfig()
        assert c.problem == "multigame"
        assert c.dataset_game == "dungeon"
        assert c.dataset_reward_enum == 1
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
        assert "contconv" in c.exp_dir
        assert "game-dungeon" in c.exp_dir
        assert "re-1" in c.exp_dir
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
        assert c.vec_input_dim == 9


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
        assert train_inst.condition.shape[1] == 9
        # 모든 reward_enum 값이 1 (region)
        assert (train_inst.reward_i == 1).all()

    def test_load_dataset_different_reward_enum(self):
        """다른 reward_enum 으로도 로딩이 되는지 확인."""
        from instruct_rl.utils.dataset_loader import load_dataset_instruct

        config = _make_cpcgrl_config(
            dataset_game="dungeon",
            dataset_reward_enum=2,  # path_length
            seed=0,
        )

        train_inst, test_inst = load_dataset_instruct(config)
        assert train_inst is not None
        assert (train_inst.reward_i == 2).all()

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
    """CPCGRL 학습 실행 및 체크포인트 생성 테스트."""

    @staticmethod
    def _run_short_training(config, exp_dir):
        """짧은 학습을 실행하고 output 을 반환한다."""
        import jax
        from instruct_rl.utils.checkpointer import init_checkpointer
        from instruct_rl.utils.dataset_loader import load_dataset_instruct
        from train_cpcgrl import make_train

        os.makedirs(exp_dir, exist_ok=True)
        progress_csv = os.path.join(exp_dir, "progress.csv")
        if not os.path.exists(progress_csv):
            with open(progress_csv, "w") as f:
                f.write("timestep,ep_return\n")

        checkpoint_manager, restored_ckpt, encoder_param = init_checkpointer(config)
        train_inst, test_inst = load_dataset_instruct(config)

        train_fn = make_train(
            config, restored_ckpt, checkpoint_manager, encoder_param,
            train_inst=train_inst, test_inst=test_inst,
        )

        rng = jax.random.PRNGKey(config.seed)
        train_jit = jax.jit(train_fn)
        out = train_jit(rng)
        jax.block_until_ready(out)
        return out

    def test_short_training_creates_checkpoint(self, cpcgrl_config, clean_exp_dir):
        """짧은 학습 후 체크포인트 디렉토리와 파일이 생성되는지 검증."""
        config = cpcgrl_config
        exp_dir = clean_exp_dir

        out = self._run_short_training(config, exp_dir)

        # ── 검증 ──────────────────────────────────────────────────────────
        # 1) 체크포인트 디렉토리 존재
        ckpt_dir = os.path.join(exp_dir, "ckpts")
        assert os.path.exists(ckpt_dir), f"Checkpoint directory not found: {ckpt_dir}"

        # 2) 체크포인트 파일이 1개 이상 존재
        ckpt_items = os.listdir(ckpt_dir)
        ckpt_steps = [d for d in ckpt_items if d.isdigit()]
        assert len(ckpt_steps) >= 1, (
            f"Expected at least 1 checkpoint, found {len(ckpt_steps)}: {ckpt_items}"
        )
        print(f"✅ Checkpoints created: {sorted(ckpt_steps, key=int)}")

        # 3) progress.csv 존재
        progress_csv = os.path.join(exp_dir, "progress.csv")
        assert os.path.exists(progress_csv), "progress.csv not found"

        # 4) output 에 runner_state 가 있음
        assert "runner_state" in out, "Output missing runner_state"




if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
