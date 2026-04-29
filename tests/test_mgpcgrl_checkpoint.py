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

    # NOTE: 사용자가 검증한 working 설정 (`train_clip_decoder game=dg ...`) 을 그대로 따른다.
    #   - game=dg  : seen 게임을 dungeon 으로 고정 (실제 학습 샘플 발생)
    #   - unseen_games 는 schema 기본값(zd) 그대로 사용
    # (기존 unseen_games=dg 는 seen_games=[] 가 되어 train pool 이 비고,
    #  결과 디코더 ckpt 의 인코더 채널 수가 train_mg_pcgrl 의 game=dg 초기화와
    #  일치하지 않아 train_mg_pcgrl JIT 단계에서 실패한다.)
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
            "game=dg",
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

    # train_clip_decoder 가 새로 저장한 ckpt 만 찾는다.
    # 주의: _ROOT 전체를 탐색하면 `pretrained_encoders/vipcgrl/default2/ckpts`
    # 같은 기존 ckpt 가 먼저 매칭되어 train_mg_pcgrl 에서 채널/구조 불일치로 실패한다.
    # 따라서 (1) train_clip_decoder 의 출력 루트인 `saves/`,
    # (2) hydra.run.dir 두 곳만 탐색한다.
    found_ckpts = None
    candidates = []
    for search_root in [os.path.join(_ROOT, "saves"), hydra_run_dir]:
        if not os.path.isdir(search_root):
            continue
        for dirpath, _, _ in os.walk(search_root):
            if os.path.basename(dirpath) != "ckpts":
                continue
            entries = os.listdir(dirpath)
            if not any(e.isdigit() for e in entries):
                continue
            # train_clip_decoder 가 만든 디렉토리만 (clipdec prefix) 받아들인다.
            if "clipdec" not in dirpath:
                continue
            candidates.append((os.path.getmtime(dirpath), dirpath))

    if candidates:
        # 가장 최근에 생성된 ckpt 디렉토리를 선택
        candidates.sort(reverse=True)
        found_ckpts = candidates[0][1]

    assert found_ckpts is not None, (
        "train_clip_decoder.py 가 체크포인트를 저장하지 않았습니다.\n"
        f"searched: {os.path.join(_ROOT, 'saves')}, {hydra_run_dir}"
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
                "train_mgpcgrl",
                # game=dg : 디코더 ckpt 가 game=dg 환경에서 학습되었으므로
                # train_mg_pcgrl 도 동일한 게임으로 인코더를 초기화해야
                # Conv_0 입력 채널 수(=base+onehot=10) 가 일치한다.
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
