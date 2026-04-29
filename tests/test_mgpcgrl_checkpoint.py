"""tests/test_mgpcgrl_checkpoint.py

train_clip_decoder -> train_mg_pcgrl 체크포인트 파이프라인 통합 테스트.

1) train_clip_decoder.py 를 dry-run 하여 디코더 체크포인트를 /tmp 에 저장
2) 저장된 체크포인트를 /tmp/pretrained_decoders 구조로 구성
3) train_mg_pcgrl.py 에서 encoder.ckpt_dir + encoder.ckpt_name 으로 로드하여 학습

실행
----
    python -m pytest tests/test_mgpcgrl_checkpoint.py -v -s
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


@pytest.fixture(scope="module")
def tmp_base():
    base = tempfile.mkdtemp(prefix="mgpcgrl_ckpt_test_")
    logger.info(f"Temporary base directory: {base}")
    yield base
    shutil.rmtree(base, ignore_errors=True)


@pytest.fixture(scope="module")
def decoder_ckpt_dir(tmp_base):
    """train_clip_decoder.py dry-run 후 디코더 ckpt 디렉토리 경로를 반환."""
    hydra_run_dir = os.path.join(tmp_base, "hydra_clip_decoder")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "train_clip_decoder",
            "overwrite=true",
            "n_epochs=1",
            "ckpt_freq=1",
            "max_samples=32",
            "batch_size=8",
            "seed=0",
            "encoder.model=cnnclip",
            "encoder.state=true",
            "decoder.num_reward_classes=5",
            "unseen_games=dg",
            f"hydra.run.dir={hydra_run_dir}",
        ],
        cwd=_ROOT,
        capture_output=True,
        text=True,
        timeout=900,
        env={**os.environ, "WANDB_MODE": "disabled"},
    )

    logger.info(f"[train_clip_decoder] stdout (last 3000):\n{result.stdout[-3000:]}")
    if result.returncode != 0:
        logger.error(f"[train_clip_decoder] stderr:\n{result.stderr[-3000:]}")
    assert result.returncode == 0, (
        f"train_clip_decoder.py exited with {result.returncode}\n"
        f"stderr:\n{result.stderr[-3000:]}"
    )

    found_ckpts = None
    for search_root in [_ROOT, hydra_run_dir]:
        for dirpath, _, _ in os.walk(search_root):
            if os.path.basename(dirpath) == "ckpts":
                entries = os.listdir(dirpath)
                if any(e.isdigit() for e in entries):
                    found_ckpts = dirpath
                    break
        if found_ckpts:
            break

    assert found_ckpts is not None, (
        "train_clip_decoder.py 가 체크포인트를 저장하지 않았습니다.\n"
        f"searched: {_ROOT}, {hydra_run_dir}"
    )

    dst = os.path.join(tmp_base, "pretrained_decoders", "test_decoder", "ckpts")
    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.copytree(found_ckpts, dst)

    logger.info(f"[decoder_ckpt_dir] source: {found_ckpts}")
    logger.info(f"[decoder_ckpt_dir] copied to: {dst}")
    logger.info(f"[decoder_ckpt_dir] contents: {os.listdir(dst)}")

    return dst


class TestDecoderCheckpoint:
    def test_ckpt_dir_exists(self, decoder_ckpt_dir):
        assert os.path.isdir(decoder_ckpt_dir)

    def test_ckpt_has_steps(self, decoder_ckpt_dir):
        entries = os.listdir(decoder_ckpt_dir)
        steps = [e for e in entries if e.isdigit()]
        logger.info(f"decoder checkpoint entries: {entries}, steps: {steps}")
        assert len(steps) > 0


class TestMGPCGRLWithDecoderCheckpoint:
    def test_mgpcgrl_loads_decoder_ckpt_and_logs(self, decoder_ckpt_dir, tmp_base):
        """train_mg_pcgrl.py가 encoder.ckpt_dir + encoder.ckpt_name을 로드해 정상 종료하고
        디코더 로딩 로그가 포함되어야 한다."""
        hydra_run_dir = os.path.join(tmp_base, "hydra_mgpcgrl")

        # decoder_ckpt_dir = {tmp_base}/pretrained_decoders/test_decoder/ckpts
        ckpt_name = os.path.basename(os.path.dirname(decoder_ckpt_dir))   # "test_decoder"
        ckpt_dir  = os.path.dirname(os.path.dirname(decoder_ckpt_dir))    # "{tmp_base}/pretrained_decoders"

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "train_mg_pcgrl",
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
                "exp_name=test_mgpcgrl_ckpt",
                f"encoder.ckpt_dir={ckpt_dir}",
                f"encoder.ckpt_name={ckpt_name}",
                f"hydra.run.dir={hydra_run_dir}",
            ],
            cwd=_ROOT,
            capture_output=True,
            text=True,
            timeout=900,
            env={**os.environ, "WANDB_MODE": "disabled"},
        )

        logger.info(f"[train_mg_pcgrl] stdout (last 3000):\n{result.stdout[-3000:]}")
        if result.returncode != 0:
            logger.error(f"[train_mg_pcgrl] stderr:\n{result.stderr[-3000:]}")

        assert result.returncode == 0, (
            f"train_mg_pcgrl.py exited with code {result.returncode}\n"
            f"stderr:\n{result.stderr[-3000:]}"
        )


        combined_output = result.stdout + result.stderr
        assert (
            "top-level keys: ['decoder', 'encoders_state', 'encoders_text', 'text_state_temperature']" in combined_output
            or "top-level keys: ['decoder', 'encoders_state', 'encoders_text', 'text_state_temperature']" in combined_output.lower()
        ), (
            "디코더 체크포인트 로딩 로그를 찾을 수 없습니다.\n"
            f"stdout (last 2000):\n{result.stdout[-2000:]}\n"
            f"stderr (last 2000):\n{result.stderr[-2000:]}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
