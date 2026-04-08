"""tests/test_ipcgrl_checkpoint.py

train_ipcgrl_encoder → train_ipcgrl 체크포인트 파이프라인 통합 테스트.

1. train_ipcgrl_encoder.py 를 max_samples 로 dry-run 하여 MLP 인코더 체크포인트를 /tmp 에 저장
2. 저장된 체크포인트를 /tmp/pretrained_encoders 구조로 구성
3. train_ipcgrl.py 에서 encoder.ckpt_path 로 해당 체크포인트를 로드하여 학습

모든 파일은 /tmp 에 생성되며, 작업 디렉토리에는 아무것도 남지 않는다.

실행
----
    python -m pytest tests/test_ipcgrl_checkpoint.py -v -s
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from instruct_rl.utils.log_utils import get_logger, suppress_jax_debug_logs

suppress_jax_debug_logs()
logger = get_logger(__file__)


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def tmp_base():
    """모듈 전체에서 공유하는 /tmp 하위 임시 디렉토리. 테스트 종료 시 삭제."""
    base = tempfile.mkdtemp(prefix="ipcgrl_ckpt_test_")
    logger.info(f"Temporary base directory: {base}")
    yield base
    shutil.rmtree(base, ignore_errors=True)


@pytest.fixture(scope="module")
def mlp_ckpt_dir(tmp_base):
    """train_ipcgrl_encoder.py dry-run → MLP 인코더 체크포인트 경로를 반환한다.

    /tmp/.../mlp_exp/ckpts/<step>/ 구조로 저장된다.
    """
    mlp_exp_dir = os.path.join(tmp_base, "mlp_exp")
    os.makedirs(mlp_exp_dir, exist_ok=True)

    # Hydra 는 outputs/ 를 cwd 에 만들므로 /tmp 로 격리
    hydra_run_dir = os.path.join(tmp_base, "hydra_mlp")

    result = subprocess.run(
        [
            sys.executable, "-m", "train_ipcgrl_encoder",
            "overwrite=true",
            "n_epochs=1",
            "ckpt_freq=1",
            f"max_samples=32",
            "batch_size=8",
            "seed=0",
            "encoder.model=mlp",
            "encoder.state=true",
            "encoder.output_dim=64",
            f"hydra.run.dir={hydra_run_dir}",
        ],
        cwd=_ROOT,
        capture_output=True,
        text=True,
        timeout=600,
        env={**os.environ, "WANDB_MODE": "disabled"},
    )

    logger.info(f"[train_ipcgrl_encoder] stdout (last 3000):\n{result.stdout[-3000:]}")
    if result.returncode != 0:
        logger.error(f"[train_ipcgrl_encoder] stderr:\n{result.stderr[-3000:]}")
    assert result.returncode == 0, (
        f"train_ipcgrl_encoder.py exited with code {result.returncode}\n"
        f"stderr:\n{result.stderr[-3000:]}"
    )

    # train_ipcgrl_encoder 이 저장한 체크포인트 탐색 — saves/ 또는 hydra cwd 아래
    # checkpoints.save_checkpoint 은 config.exp_dir/ckpts/ 에 저장
    # hydra 가 cwd 를 바꾸므로 hydra_run_dir 아래에서도 탐색
    found_ckpts = None
    for search_root in [_ROOT, hydra_run_dir]:
        for dirpath, dirnames, filenames in os.walk(search_root):
            if os.path.basename(dirpath) == "ckpts":
                # flax checkpoint: <step> 이름의 숫자 파일이 존재
                steps = [d for d in os.listdir(dirpath)
                         if d.isdigit() or os.path.isfile(os.path.join(dirpath, d))]
                if steps:
                    found_ckpts = dirpath
                    break
        if found_ckpts:
            break

    assert found_ckpts is not None, (
        "train_ipcgrl_encoder.py 가 체크포인트를 저장하지 않았습니다.\n"
        f"searched: {_ROOT}, {hydra_run_dir}"
    )

    # /tmp/pretrained_encoders/test_mlp/ckpts 구조로 복사
    pretrained_ckpt_dir = os.path.join(tmp_base, "pretrained_encoders", "test_mlp", "ckpts")
    if os.path.exists(pretrained_ckpt_dir):
        shutil.rmtree(pretrained_ckpt_dir)
    shutil.copytree(found_ckpts, pretrained_ckpt_dir)

    logger.info(f"[mlp_ckpt_dir] source: {found_ckpts}")
    logger.info(f"[mlp_ckpt_dir] copied to: {pretrained_ckpt_dir}")
    logger.info(f"[mlp_ckpt_dir] contents: {os.listdir(pretrained_ckpt_dir)}")

    return pretrained_ckpt_dir


# ═══════════════════════════════════════════════════════════════════════════════
#  1. MLP 체크포인트 생성 테스트
# ═══════════════════════════════════════════════════════════════════════════════

class TestMLPCheckpoint:
    """train_ipcgrl_encoder dry-run 후 체크포인트가 존재하는지 검증."""

    def test_ckpt_dir_exists(self, mlp_ckpt_dir):
        assert os.path.isdir(mlp_ckpt_dir)

    def test_ckpt_has_steps(self, mlp_ckpt_dir):
        """체크포인트 디렉토리에 step 숫자 항목이 존재해야 한다."""
        entries = os.listdir(mlp_ckpt_dir)
        step_entries = [e for e in entries if e.isdigit()]
        logger.info(f"checkpoint entries: {entries}, step_entries: {step_entries}")
        assert len(step_entries) > 0, f"No step entries in {mlp_ckpt_dir}: {entries}"

    def test_ckpt_is_loadable(self, mlp_ckpt_dir):
        """flax checkpoints.restore_checkpoint 으로 로드 가능해야 한다."""
        from flax.training import checkpoints

        entries = os.listdir(mlp_ckpt_dir)
        step_entries = sorted([e for e in entries if e.isdigit()], key=int, reverse=True)
        assert len(step_entries) > 0

        ckpt_dir = os.path.join(mlp_ckpt_dir, step_entries[0])
        state = checkpoints.restore_checkpoint(ckpt_dir=ckpt_dir, target=None, prefix="")
        assert state is not None, f"Failed to restore checkpoint from {ckpt_dir}"
        assert "params" in state, f"Restored state has no 'params' key: {list(state.keys())}"
        logger.info(f"Restored checkpoint keys: {list(state.keys())}")
        logger.info(f"params keys: {list(state['params'].keys())}")


# ═══════════════════════════════════════════════════════════════════════════════
#  2. IPCGRL 이 MLP 체크포인트를 로드하여 학습하는 E2E 테스트
# ═══════════════════════════════════════════════════════════════════════════════

class TestIPCGRLWithMLPCheckpoint:
    """train_ipcgrl.py 가 MLP 체크포인트를 로드하여 정상 학습하는지 검증."""

    def test_ipcgrl_loads_mlp_ckpt_and_exits_zero(self, mlp_ckpt_dir, tmp_base):
        """train_ipcgrl.py 에서 encoder.ckpt_path 로 MLP 체크포인트를 지정하고
        정상 종료(exit 0)를 확인한다."""

        hydra_run_dir = os.path.join(tmp_base, "hydra_ipcgrl")

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
                "exp_name=test_ipcgrl_ckpt",
                f"encoder.ckpt_path={mlp_ckpt_dir}",
                f"hydra.run.dir={hydra_run_dir}",
            ],
            cwd=_ROOT,
            capture_output=True,
            text=True,
            timeout=600,
            env={**os.environ, "WANDB_MODE": "disabled"},
        )

        logger.info(f"[train_ipcgrl] stdout (last 3000):\n{result.stdout[-3000:]}")
        if result.returncode != 0:
            logger.error(f"[train_ipcgrl] stderr:\n{result.stderr[-3000:]}")

        assert result.returncode == 0, (
            f"train_ipcgrl.py exited with code {result.returncode}\n"
            f"stderr:\n{result.stderr[-3000:]}"
        )

    def test_ipcgrl_stdout_contains_encoder_loaded_log(self, mlp_ckpt_dir, tmp_base):
        """학습 로그에 인코더 체크포인트 로딩 성공 메시지가 포함되어야 한다."""

        hydra_run_dir = os.path.join(tmp_base, "hydra_ipcgrl_log")

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
                "exp_name=test_ipcgrl_ckpt_log",
                f"encoder.ckpt_path={mlp_ckpt_dir}",
                f"hydra.run.dir={hydra_run_dir}",
            ],
            cwd=_ROOT,
            capture_output=True,
            text=True,
            timeout=600,
            env={**os.environ, "WANDB_MODE": "disabled"},
        )

        combined_output = result.stdout + result.stderr
        assert "Encoder checkpoint" in combined_output or "encoder checkpoint" in combined_output.lower(), (
            "인코더 체크포인트 로딩 로그를 찾을 수 없습니다.\n"
            f"stdout (last 2000):\n{result.stdout[-2000:]}\n"
            f"stderr (last 2000):\n{result.stderr[-2000:]}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

