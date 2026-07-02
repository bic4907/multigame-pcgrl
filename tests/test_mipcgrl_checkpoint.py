"""tests/test_mipcgrl_checkpoint.py

train_mipcgrl_encoder_mg → train_mipcgrl 체크포인트 파이프라인 통합 테스트.

1. train_mipcgrl_encoder_mg.py 를 max_samples 로 dry-run 하여
   (MLP encoder + task classifier) 체크포인트를 /tmp 에 저장
2. train_mipcgrl.py 에서 encoder.ckpt_dir + encoder.ckpt_name 으로
   해당 체크포인트를 로드하여 학습
3. exp_dir prefix 가 ``mipcgrl_`` 인지 (IPCGRL 와 분리되는지) 검증

실행
----
    python -m pytest tests/test_mipcgrl_checkpoint.py -v -s
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
    base = tempfile.mkdtemp(prefix="mipcgrl_ckpt_test_")
    logger.info(f"Temporary base directory: {base}")
    yield base
    shutil.rmtree(base, ignore_errors=True)


@pytest.fixture(scope="module")
def mipcgrl_encoder_ckpt(tmp_base):
    """train_mipcgrl_encoder_mg.py dry-run → 체크포인트 (ckpt_dir, ckpt_name) 반환."""
    saves_dir = os.path.join(tmp_base, "mipcgrl_encoder")
    hydra_run_dir = os.path.join(tmp_base, "hydra_mipcgrl_encoder")

    result = subprocess.run(
        [
            sys.executable, "-m", "train_mipcgrl_encoder_mg",
            "game=dg",
            "overwrite=true",
            "n_epochs=1",
            "ckpt_freq=1",
            "max_samples=32",
            "batch_size=8",
            "seed=0",
            "classifier_weight=1.0",
            f"saves_dir={saves_dir}",
            f"hydra.run.dir={hydra_run_dir}",
        ],
        cwd=_ROOT,
        capture_output=True,
        text=True,
        timeout=1200,
        env={**os.environ, "WANDB_MODE": "disabled"},
    )

    logger.info(f"[train_mipcgrl_encoder_mg] stdout (last 3000):\n{result.stdout[-3000:]}")
    if result.returncode != 0:
        logger.error(f"[train_mipcgrl_encoder_mg] stderr:\n{result.stderr[-3000:]}")
    assert result.returncode == 0, (
        f"train_mipcgrl_encoder_mg.py exited with {result.returncode}\n"
        f"stderr:\n{result.stderr[-3000:]}"
    )

    # exp_dir 은 saves_dir/<dir_prefix><exp_group>_<seed> 구조
    entries = [
        e for e in os.listdir(saves_dir)
        if os.path.isdir(os.path.join(saves_dir, e)) and e.startswith("mipcgrl-enc-mg-")
    ]
    assert entries, (
        f"mipcgrl-enc-mg- 로 시작하는 exp_dir 가 saves_dir 에 없습니다: "
        f"{os.listdir(saves_dir)}"
    )
    ckpt_name = entries[0]
    ckpt_dir = saves_dir
    exp_dir = os.path.join(ckpt_dir, ckpt_name)

    # ckpts/<step>/ 가 생성됐는지 확인
    ckpts_path = os.path.join(exp_dir, "ckpts")
    assert os.path.isdir(ckpts_path), f"ckpts 디렉토리 없음: {ckpts_path}"
    step_dirs = [e for e in os.listdir(ckpts_path) if e.isdigit()]
    assert step_dirs, f"step 디렉토리 없음: {os.listdir(ckpts_path)}"

    # dataset_setting.json 이 저장되었는지 확인 (RL 측 자동 주입에 사용됨)
    assert os.path.isfile(os.path.join(exp_dir, "dataset_setting.json")), (
        "dataset_setting.json 이 저장되지 않았습니다."
    )

    logger.info(f"[mipcgrl_encoder_ckpt] ckpt_dir={ckpt_dir} ckpt_name={ckpt_name}")
    return ckpt_dir, ckpt_name


# ═══════════════════════════════════════════════════════════════════════════════
#  1. Encoder 체크포인트 생성 + dir_prefix 검증
# ═══════════════════════════════════════════════════════════════════════════════

class TestMIPCGRLEncoderCheckpoint:
    def test_ckpt_name_uses_mipcgrl_prefix(self, mipcgrl_encoder_ckpt):
        """인코더 체크포인트 폴더가 ``mipcgrl-enc-mg-`` prefix 를 사용해야 IPCGRL
        체크포인트(``ipcgrl-enc-mg-``)와 충돌하지 않는다."""
        _, ckpt_name = mipcgrl_encoder_ckpt
        assert ckpt_name.startswith("mipcgrl-enc-mg-"), (
            f"unexpected ckpt_name prefix: {ckpt_name}"
        )
        assert not ckpt_name.startswith("ipcgrl-enc-mg-")


# ═══════════════════════════════════════════════════════════════════════════════
#  2. train_mipcgrl 이 인코더 체크포인트를 로드해 학습되는지 E2E
# ═══════════════════════════════════════════════════════════════════════════════

class TestMIPCGRLWithEncoderCheckpoint:
    def test_train_mipcgrl_loads_encoder(self, mipcgrl_encoder_ckpt, tmp_base):
        ckpt_dir, ckpt_name = mipcgrl_encoder_ckpt
        rl_saves_dir = os.path.join(tmp_base, "mipcgrl_rl")
        hydra_run_dir = os.path.join(tmp_base, "hydra_mipcgrl_rl")

        result = subprocess.run(
            [
                sys.executable, "-m", "train_mipcgrl",
                "game=dg",
                "overwrite=true",
                "total_timesteps=100",
                "n_envs=4",
                "num_steps=4",
                "update_epochs=1",
                "NUM_MINIBATCHES=1",
                "seed=42",
                "ckpt_freq=1",
                "render_freq=-1",
                "eval_freq=-1",
                "exp_name=test_mipcgrl_ckpt",
                f"encoder.ckpt_dir={ckpt_dir}",
                f"encoder.ckpt_name={ckpt_name}",
                f"saves_dir={rl_saves_dir}",
                f"hydra.run.dir={hydra_run_dir}",
            ],
            cwd=_ROOT,
            capture_output=True,
            text=True,
            timeout=1200,
            env={**os.environ, "WANDB_MODE": "disabled"},
        )

        logger.info(f"[train_mipcgrl] stdout (last 3000):\n{result.stdout[-3000:]}")
        if result.returncode != 0:
            logger.error(f"[train_mipcgrl] stderr:\n{result.stderr[-3000:]}")

        assert result.returncode == 0, (
            f"train_mipcgrl.py exited with code {result.returncode}\n"
            f"stderr:\n{result.stderr[-3000:]}"
        )

        combined_output = result.stdout + result.stderr
        assert "encoder checkpoint" in combined_output.lower(), (
            "인코더 체크포인트 로딩 로그를 찾을 수 없습니다.\n"
            f"stdout (last 2000):\n{result.stdout[-2000:]}\n"
            f"stderr (last 2000):\n{result.stderr[-2000:]}"
        )

    def test_rl_exp_dir_uses_mipcgrl_prefix(self, mipcgrl_encoder_ckpt, tmp_base):
        """RL 측 exp_dir 도 ``mipcgrl_`` prefix 를 사용해 IPCGRL (``ipcgrl_...``)
        체크포인트와 디스크/wandb 에서 분리되어야 한다."""
        rl_saves_dir = os.path.join(tmp_base, "mipcgrl_rl")
        assert os.path.isdir(rl_saves_dir), f"RL saves_dir 없음: {rl_saves_dir}"

        rl_entries = [
            e for e in os.listdir(rl_saves_dir)
            if os.path.isdir(os.path.join(rl_saves_dir, e))
        ]
        assert rl_entries, f"RL exp_dir 항목 없음: {os.listdir(rl_saves_dir)}"
        assert any(e.startswith("mipcgrl_") for e in rl_entries), (
            f"mipcgrl_ prefix 인 exp_dir 가 없습니다: {rl_entries}"
        )
        assert not any(e.startswith("ipcgrl_") for e in rl_entries), (
            f"ipcgrl_ prefix exp_dir 가 생성됐습니다 (분리 실패): {rl_entries}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
