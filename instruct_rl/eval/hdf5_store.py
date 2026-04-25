"""
instruct_rl/eval/hdf5_store.py
================================
HDF5 기반 평가 결과 저장소.

구조:
  {eval_dir}/eval.h5
    /{folder_name}/seed_{seed_i}/state  → (H, W[, C]) 최종 env_map (uint8, gzip-4)

NOTE:
  - rendered_image는 HDF5에 저장하지 않는다.
    ViTScore 등에서 필요할 때는 state(env_map)를 읽어
    render_multigame_map() 으로 동적 렌더링한다.
  - 비동기 쓰기: AsyncH5Writer 를 사용하면 별도 쓰레드에서 HDF5에 기록하므로
    GPU(JAX) 연산과 디스크 I/O 가 겹쳐서 처리 속도가 향상된다.

사용 예 (동기):
    with open_eval_store(config.eval_dir, mode="a") as h5:
        write_sample(h5, folder_name, seed_i, state)

사용 예 (비동기):
    with AsyncH5Writer(config.eval_dir) as writer:
        writer.write(folder_name, seed_i, state)
    # with 블록을 나오면 queue flush + thread join 자동 수행

    with open_eval_store(config.eval_dir, mode="r") as h5:
        state = read_state(h5, folder_name, seed_i)
"""
from __future__ import annotations

import contextlib
import logging
import os
import queue
import threading
from typing import Optional  # noqa: F401 — kept for backward compat

import numpy as np

logger = logging.getLogger(__name__)

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
    state: np.ndarray,   # (H, W[, C]) uint8
) -> None:
    """하나의 (sample, seed) 결과를 HDF5에 기록한다. state(env_map)만 저장."""
    key = f"{folder_name}/seed_{seed_i}"
    grp = h5.require_group(key)
    for name, data in [("state", np.asarray(state, dtype=np.uint8))]:
        if name in grp:
            del grp[name]
        grp.create_dataset(name, data=data, compression="gzip", compression_opts=4)


# ── 비동기 HDF5 Writer ────────────────────────────────────────────────────────

_SENTINEL = None   # queue 종료 신호


class AsyncH5Writer:
    """Writes HDF5 samples on a background thread to overlap GPU work with disk I/O.

    The main thread enqueues states immediately after each rollout batch and
    continues to the next batch without waiting for disk writes.  The HDF5 file
    handle is owned exclusively by the writer thread.

    Parameters
    ----------
    eval_dir : Directory where ``eval.h5`` will be written.
    maxsize  : Maximum queue depth (0 = unlimited). Large values use more RAM.

    Usage::

        with AsyncH5Writer(config.eval_dir) as writer:
            writer.write(folder_name, seed_i, state)
        # Exiting the context flushes the queue and joins the thread.
    """

    def __init__(self, eval_dir: str, maxsize: int = 16):
        self._eval_dir = eval_dir
        self._q: queue.Queue = queue.Queue(maxsize=maxsize)
        self._exc: BaseException | None = None
        self._thread = threading.Thread(
            target=self._worker, name="AsyncH5Writer", daemon=True
        )
        self._thread.start()

    def _worker(self) -> None:
        """Background loop: opens the HDF5 file and drains the queue."""
        try:
            with open_eval_store(self._eval_dir, mode="a") as h5:
                while True:
                    item = self._q.get()
                    if item is _SENTINEL:
                        self._q.task_done()
                        break
                    folder_name, seed_i, state = item
                    write_sample(h5, folder_name, seed_i, state)
                    self._q.task_done()
        except Exception as exc:
            logger.error("[AsyncH5Writer] Background thread error: %s", exc, exc_info=True)
            self._exc = exc

    def write(self, folder_name: str, seed_i: int, state: np.ndarray) -> None:
        """Enqueue a state for writing. Blocks if the queue is full."""
        if self._exc is not None:
            raise RuntimeError("AsyncH5Writer thread terminated with an error") from self._exc
        # Copy to numpy immediately so the JAX device buffer can be freed.
        self._q.put((folder_name, seed_i, np.array(state, dtype=np.uint8)))

    def flush(self) -> None:
        """Block until all queued writes have been committed. The thread stays alive.

        Call this after the rollout loop and before any code that reads from the
        HDF5 file (ViTScore, TPKL, Diversity, …) to guarantee all data is on disk.
        """
        if self._exc is not None:
            raise RuntimeError("AsyncH5Writer thread terminated with an error") from self._exc
        self._q.join()
        logger.info("[AsyncH5Writer] Flush complete — all pending HDF5 writes are on disk.")

    def close(self) -> None:
        """Drain the queue, stop the background thread, and close the HDF5 file."""
        self._q.put(_SENTINEL)
        self._thread.join()
        if self._exc is not None:
            raise RuntimeError("AsyncH5Writer thread terminated with an error") from self._exc
        logger.info("[AsyncH5Writer] Writer closed — all writes complete.")

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


# ── 읽기 유틸 ─────────────────────────────────────────────────────────────────

def write_rendered_image(h5, folder_name, seed_i, image, **_):
    """Deprecated: 렌더링 이미지는 더 이상 HDF5에 저장하지 않는다. (no-op)"""
    pass  # backward-compat stub


def read_rendered_image(h5, folder_name, seed_i):
    """Deprecated: read_state() + render_multigame_map() 를 대신 사용하라."""
    raise NotImplementedError(
        "rendered_image는 더 이상 HDF5에 저장되지 않습니다. "
        "read_state() 후 render_multigame_map()으로 동적 렌더링하세요."
    )


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
