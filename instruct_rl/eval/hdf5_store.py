"""
instruct_rl/eval/hdf5_store.py
================================
HDF5 기반 평가 결과 저장소.

구조:
  {eval_dir}/eval.h5
    /{folder_name}/seed_{seed_i}/state    → (H, W[, C]) 최종 맵 (uint8)
    /{folder_name}/seed_{seed_i}/frames   → (n_frames, H, W, 3) RGB 프레임 (uint8)

사용 예:
    with open_eval_store(config.eval_dir, mode="a") as h5:
        write_sample(h5, folder_name, seed_i, frames_rgb, state)

    with open_eval_store(config.eval_dir, mode="r") as h5:
        state = read_state(h5, folder_name, seed_i)
"""
from __future__ import annotations

import contextlib
import os
from typing import Optional

import numpy as np

STORE_FILENAME = "eval.h5"


def store_path(eval_dir: str) -> str:
    return os.path.join(eval_dir, STORE_FILENAME)


@contextlib.contextmanager
def open_eval_store(eval_dir: str, mode: str = "a"):
    """eval.h5 를 열고 h5py.File 을 yield 한 뒤 닫는 컨텍스트 매니저."""
    import h5py
    path = store_path(eval_dir)
    with h5py.File(path, mode) as h5:
        yield h5


def write_sample(
    h5,
    folder_name: str,
    seed_i: int,
    frames_rgb: np.ndarray,   # (n_frames, H, W, 3) uint8
    state: np.ndarray,         # (H, W[, C]) uint8
) -> None:
    """하나의 (sample, seed) 결과를 HDF5에 기록한다."""
    key = f"{folder_name}/seed_{seed_i}"
    grp = h5.require_group(key)
    # 기존 데이터 덮어쓰기
    for name, data in [("frames", np.asarray(frames_rgb, dtype=np.uint8)),
                       ("state",  np.asarray(state,      dtype=np.uint8))]:
        if name in grp:
            del grp[name]
        grp.create_dataset(name, data=data, compression="lzf")


def read_state(
    h5,
    folder_name: str,
    seed_i: int,
    frame_i: int = 0,
) -> np.ndarray:
    """저장된 state 배열을 반환한다."""
    return h5[f"{folder_name}/seed_{seed_i}/state"][()]


def read_frames(
    h5,
    folder_name: str,
    seed_i: int,
) -> np.ndarray:
    """저장된 frames 배열 (n_frames, H, W, 3) 을 반환한다."""
    return h5[f"{folder_name}/seed_{seed_i}/frames"][()]


def list_folder_names(h5) -> list[str]:
    """저장된 모든 folder_name 목록을 반환한다."""
    return list(h5.keys())

