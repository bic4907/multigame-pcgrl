"""tests/test_vipcgrl_checkpoint.py

train_clip → train_vipcgrl 체크포인트 파이프라인 통합 테스트.

1. train_clip.py 를 max_samples 로 dry-run 하여 CLIP 인코더 체크포인트를 /tmp 에 저장
2. 저장된 체크포인트를 /tmp/pretrained_encoders 구조로 구성
3. train_vipcgrl.py 에서 encoder.ckpt_path 로 해당 체크포인트를 로드하여 학습

모든 파일은 /tmp 에 생성되며, 작업 디렉토리에는 아무것도 남지 않는다.

실행
----
    python -m pytest tests/test_vipcgrl_checkpoint.py -v -s
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
    base = tempfile.mkdtemp(prefix="vipcgrl_ckpt_test_")
    logger.info(f"Temporary base directory: {base}")
    yield base
    shutil.rmtree(base, ignore_errors=True)


@pytest.fixture(scope="module")
def clip_ckpt_dir(tmp_base):
    """train_clip.py dry-run → CLIP 인코더 체크포인트 경로를 반환한다.

    /tmp/.../clip_exp/ckpts/<step>/ 구조로 저장된다.
    """
    clip_exp_dir = os.path.join(tmp_base, "clip_exp")
    os.makedirs(clip_exp_dir, exist_ok=True)

    # Hydra 는 outputs/ 를 cwd 에 만들므로 /tmp 로 격리
    hydra_run_dir = os.path.join(tmp_base, "hydra_clip")

    result = subprocess.run(
        [
            sys.executable, "-m", "train_clip",
            "overwrite=true",
            "n_epochs=1",
            "ckpt_freq=1",
            f"max_samples=32",
            "batch_size=8",
            "seed=0",
            "encoder.model=cnnclip",
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

    logger.info(f"[train_clip] stdout (last 3000):\n{result.stdout[-3000:]}")
    if result.returncode != 0:
        logger.error(f"[train_clip] stderr:\n{result.stderr[-3000:]}")
    assert result.returncode == 0, (
        f"train_clip.py exited with code {result.returncode}\n"
        f"stderr:\n{result.stderr[-3000:]}"
    )

    # train_clip 이 저장한 체크포인트 탐색 — saves/ 또는 hydra cwd 아래
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
        "train_clip.py 가 체크포인트를 저장하지 않았습니다.\n"
        f"searched: {_ROOT}, {hydra_run_dir}"
    )

    # /tmp/pretrained_encoders/test_clip/ckpts 구조로 복사
    pretrained_ckpt_dir = os.path.join(tmp_base, "pretrained_encoders", "test_clip", "ckpts")
    if os.path.exists(pretrained_ckpt_dir):
        shutil.rmtree(pretrained_ckpt_dir)
    shutil.copytree(found_ckpts, pretrained_ckpt_dir)

    logger.info(f"[clip_ckpt_dir] source: {found_ckpts}")
    logger.info(f"[clip_ckpt_dir] copied to: {pretrained_ckpt_dir}")
    logger.info(f"[clip_ckpt_dir] contents: {os.listdir(pretrained_ckpt_dir)}")

    return pretrained_ckpt_dir


# ═══════════════════════════════════════════════════════════════════════════════
#  1. CLIP 체크포인트 생성 테스트
# ═══════════════════════════════════════════════════════════════════════════════

class TestCLIPCheckpoint:
    """train_clip dry-run 후 체크포인트가 존재하는지 검증."""

    def test_ckpt_dir_exists(self, clip_ckpt_dir):
        assert os.path.isdir(clip_ckpt_dir)

    def test_ckpt_has_steps(self, clip_ckpt_dir):
        """체크포인트 디렉토리에 step 숫자 항목이 존재해야 한다."""
        entries = os.listdir(clip_ckpt_dir)
        step_entries = [e for e in entries if e.isdigit()]
        logger.info(f"checkpoint entries: {entries}, step_entries: {step_entries}")
        assert len(step_entries) > 0, f"No step entries in {clip_ckpt_dir}: {entries}"

# ═══════════════════════════════════════════════════════════════════════════════
#  2. VIPCGRL 이 CLIP 체크포인트를 로드하여 학습하는 E2E 테스트
# ═══════════════════════════════════════════════════════════════════════════════

class TestVIPCGRLWithCLIPCheckpoint:
    """train_vipcgrl.py 가 CLIP 체크포인트를 로드하여 정상 학습하는지 검증."""

    def test_vipcgrl_loads_clip_ckpt_and_logs_encoder(self, clip_ckpt_dir, tmp_base):
        """train_vipcgrl.py 에서 encoder.ckpt_path 로 CLIP 체크포인트를 지정하고
        정상 종료(exit 0) 및 인코더 로딩 로그를 동시에 확인한다."""

        hydra_run_dir = os.path.join(tmp_base, "hydra_vipcgrl")

        # decoder_ckpt_dir = {tmp_base}/pretrained_decoders/test_decoder/ckpts
        ckpt_name = os.path.basename(os.path.dirname(clip_ckpt_dir))   # "test_decoder"
        ckpt_dir  = os.path.dirname(os.path.dirname(clip_ckpt_dir))    # "{tmp_base}/pretrained_decoders"

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
                "exp_name=test_vipcgrl_ckpt",
                f"encoder.ckpt_dir={ckpt_dir}",
                f"encoder.ckpt_name={ckpt_name}",
                f"hydra.run.dir={hydra_run_dir}",
            ],
            cwd=_ROOT,
            capture_output=True,
            text=True,
            timeout=600,
            env={**os.environ, "WANDB_MODE": "disabled"},
        )

        logger.info(f"[train_vipcgrl] stdout (last 3000):\n{result.stdout[-3000:]}")
        if result.returncode != 0:
            logger.error(f"[train_vipcgrl] stderr:\n{result.stderr[-3000:]}")

        assert result.returncode == 0, (
            f"train_vipcgrl.py exited with code {result.returncode}\n"
            f"stderr:\n{result.stderr[-3000:]}"
        )


        combined_output = result.stdout + result.stderr
        assert "Encoder checkpoint" in combined_output or "encoder checkpoint" in combined_output.lower(), (
            "인코더 체크포인트 로딩 로그를 찾을 수 없습니다.\n"
            f"stdout (last 2000):\n{result.stdout[-2000:]}\n"
            f"stderr (last 2000):\n{result.stderr[-2000:]}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

