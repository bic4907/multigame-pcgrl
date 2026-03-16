"""Cache helpers for MultiGameDataset.

Cache key = hash(code files + init args + schema version).
Cache payload = arrays.npz + meta.json (git-committable local files).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .base import GameSample

CACHE_SCHEMA_VERSION = 1


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _stable_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=True, separators=(",", ":"))


def hash_code_files(code_root: Path) -> str:
    """Hash relevant source files under dataset/multigame."""
    # Exclude cache and tests from the code signature.
    py_files = sorted(
        p for p in code_root.rglob("*.py")
        if "tests" not in p.parts and "__pycache__" not in p.parts and "cache" not in p.parts
    )
    h = hashlib.sha256()
    for p in py_files:
        h.update(str(p.relative_to(code_root)).encode("utf-8"))
        h.update(p.read_bytes())
    return h.hexdigest()


def build_cache_key(args_dict: Dict[str, Any], *, code_root: Path) -> str:
    payload = {
        "schema": CACHE_SCHEMA_VERSION,
        "args": args_dict,
        "code_hash": hash_code_files(code_root),
    }
    return _sha256_bytes(_stable_json(payload).encode("utf-8"))


def _cache_paths(cache_dir: Path, key: str) -> tuple[Path, Path]:
    base = cache_dir / key
    npz_path = base.with_suffix(".npz")
    meta_path = base.with_suffix(".json")
    return npz_path, meta_path


def save_samples_to_cache(cache_dir: Path, key: str, samples: List[GameSample]) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    npz_path, meta_path = _cache_paths(cache_dir, key)

    if samples:
        arrays = np.stack([s.array for s in samples], axis=0)
    else:
        arrays = np.zeros((0, 16, 16), dtype=np.int32)

    np.savez_compressed(npz_path, arrays=arrays)

    meta: List[Dict[str, Any]] = []
    for s in samples:
        meta.append(
            {
                "game": s.game,
                "source_id": s.source_id,
                "instruction": s.instruction,
                "order": s.order,
                "meta": s.meta,
            }
        )
    meta_path.write_text(_stable_json(meta), encoding="utf-8")


def load_samples_from_cache(cache_dir: Path, key: str) -> Optional[List[GameSample]]:
    npz_path, meta_path = _cache_paths(cache_dir, key)
    if not npz_path.exists() or not meta_path.exists():
        return None

    arrays = np.load(npz_path)["arrays"]
    meta: List[Dict[str, Any]] = json.loads(meta_path.read_text(encoding="utf-8"))
    if len(meta) != len(arrays):
        return None

    samples: List[GameSample] = []
    for i, m in enumerate(meta):
        samples.append(
            GameSample(
                game=m["game"],
                source_id=m["source_id"],
                array=arrays[i].astype(np.int32),
                char_grid=None,
                legend=None,
                instruction=m.get("instruction"),
                order=m.get("order"),
                meta=m.get("meta", {}),
            )
        )
    return samples

