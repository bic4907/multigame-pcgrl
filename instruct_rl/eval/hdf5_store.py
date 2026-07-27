"""
instruct_rl/eval/hdf5_store.py
================================
HDF5-based evaluation result storage.

structure:
  {eval_dir}/eval.h5
    /{folder_name}/seed_{seed_i}/state  -> final env_map (H, W[, C]) (uint8, gzip-4)

NOTE:
  - rendered_image is not stored in HDF5.
    Consumers such as ViTScore read state (env_map) when needed and render it
    dynamically with render_multigame_map().
  - Asynchronous writes: AsyncH5Writer records HDF5 data on a separate thread,
    overlapping GPU (JAX) computation with disk I/O for better throughput.

Example (synchronous):
    with open_eval_store(config.eval_dir, mode="a") as h5:
        write_sample(h5, folder_name, seed_i, state)

Example (asynchronous):
    with AsyncH5Writer(config.eval_dir) as writer:
        writer.write(folder_name, seed_i, state)
    # Leaving the with block automatically flushes the queue and joins the thread

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
    """Context manager that opens eval.h5, yields an h5py.File, and closes it."""
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
    """Write one (sample, seed) result to HDF5, storing only state (env_map)."""
    key = f"{folder_name}/seed_{seed_i}"
    grp = h5.require_group(key)
    for name, data in [("state", np.asarray(state, dtype=np.uint8))]:
        if name in grp:
            del grp[name]
        grp.create_dataset(name, data=data, compression="gzip", compression_opts=4)


# ── asynchronous HDF5 Writer ────────────────────────────────────────────────────────

_SENTINEL = None   # Queue termination signal


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


# ── read utility ─────────────────────────────────────────────────────────────────

def write_rendered_image(h5, folder_name, seed_i, image, **_):
    """Deprecated: rendered images are no longer stored in HDF5 (no-op)."""
    pass  # backward-compat stub


def read_rendered_image(h5, folder_name, seed_i):
    """Deprecated: use read_state() with render_multigame_map() instead."""
    raise NotImplementedError(
        "rendered_image is no longer stored in HDF5. "
        "Use read_state() and render dynamically with render_multigame_map()."
    )


def read_state(
    h5,
    folder_name: str,
    seed_i: int,
    frame_i: int = 0,
) -> np.ndarray:
    """Return the stored state array."""
    return h5[f"{folder_name}/seed_{seed_i}/state"][()]


def read_frames(
    h5,
    folder_name: str,
    seed_i: int,
) -> np.ndarray:
    """Return stored frames with shape (n_frames, H, W, 3)."""
    return h5[f"{folder_name}/seed_{seed_i}/frames"][()]


def list_folder_names(h5) -> list[str]:
    """Return all stored folder names."""
    return list(h5.keys())
