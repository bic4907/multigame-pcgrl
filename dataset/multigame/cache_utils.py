"""Cache helpers for MultiGameDataset.

Cache key = hash(code files + init args + schema version).
Cache payload = arrays.npz + meta.json (git-committable local files).

캐시 키 계산 시 절대 경로는 프로젝트 루트(dataset/ 두 단계 위)를 기준으로
상대 경로로 정규화하므로, 다른 PC에서도 동일한 키가 생성된다.
"""
from __future__ import annotations

import hashlib
import json
import logging
import socket
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .base import GameSample

logger = logging.getLogger(__name__)


def _cache_log(msg: str) -> None:
    """logger.info + print fallback.

    logging 핸들러가 설정돼 있으면 logger.info로, 없으면 print로 출력해
    logging.basicConfig 없이도 캐시 메타가 콘솔에 보이도록 한다.
    """
    logger.info(msg)
    # 루트 로거에 핸들러가 없고 패키지 로거 자체에도 NullHandler만 있으면
    # logging 경로로는 아무것도 출력되지 않는다 → print로 보완.
    root_has_handlers = bool(logging.root.handlers)
    pkg_has_real_handler = any(
        not isinstance(h, logging.NullHandler)
        for h in logging.getLogger("dataset.multigame").handlers
    )
    if not root_has_handlers and not pkg_has_real_handler:
        print(msg)

CACHE_SCHEMA_VERSION = 1

# dataset/multigame/ → dataset/ → project_root
_PROJECT_ROOT: Path = Path(__file__).parent.parent.parent


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _stable_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=True, separators=(",", ":"))


def _normalize_path(raw: str) -> str:
    """절대 경로를 프로젝트 루트 기준 상대 경로 문자열로 변환.

    프로젝트 루트 밖이면 경로의 마지막 두 컴포넌트만 사용해 식별한다.
    """
    p = Path(raw).resolve()
    try:
        return str(p.relative_to(_PROJECT_ROOT.resolve()))
    except ValueError:
        return str(Path(*p.parts[-2:]))


def _normalize_args(args_dict: Dict[str, Any]) -> Dict[str, Any]:
    """args_dict 에서 경로 값을 상대 경로로 정규화한다."""
    path_keys = {"vglc_root", "dungeon_root"}
    normalized: Dict[str, Any] = {}
    for k, v in args_dict.items():
        if k in path_keys and isinstance(v, str):
            normalized[k] = _normalize_path(v)
        else:
            normalized[k] = v
    return normalized


def hash_code_files(code_root: Path) -> str:
    """Hash relevant source files under dataset/multigame."""
    py_files = sorted(
        p for p in code_root.rglob("*.py")
        if "tests" not in p.parts and "__pycache__" not in p.parts and "cache" not in p.parts and "viewer" not in p.parts
    )
    h = hashlib.sha256()
    for p in py_files:
        h.update(str(p.relative_to(code_root)).encode("utf-8"))
        h.update(p.read_bytes())
    return h.hexdigest()


def build_cache_key(args_dict: Dict[str, Any], *, code_root: Path) -> str:
    payload = {
        "schema": CACHE_SCHEMA_VERSION,
        "args": _normalize_args(args_dict),   # ← 상대 경로로 정규화
        "code_hash": hash_code_files(code_root),
    }
    return _sha256_bytes(_stable_json(payload).encode("utf-8"))


def _cache_paths(cache_dir: Path, key: str) -> tuple[Path, Path, Path]:
    base = cache_dir / key
    npz_path  = base.with_suffix(".npz")
    meta_path = base.with_suffix(".json")
    info_path = base.with_suffix(".info.json")   # 캐시 생성 메타데이터
    return npz_path, meta_path, info_path


def _collect_info(samples: List[GameSample]) -> Dict[str, Any]:
    """캐시 생성 당시 환경 정보를 수집한다."""
    game_counts: Dict[str, int] = {}
    for s in samples:
        game_counts[s.game] = game_counts.get(s.game, 0) + 1

    try:
        hostname = socket.gethostname()
    except Exception:
        hostname = "unknown"

    return {
        "created_at": datetime.now(tz=timezone.utc).isoformat(),
        "hostname": hostname,
        "total_samples": len(samples),
        "game_counts": game_counts,
    }


def _purge_old_caches(cache_dir: Path, keep_key: str) -> None:
    """cache_dir 안의 캐시 파일 중 keep_key에 해당하지 않는 것을 모두 삭제한다."""
    removed: List[Path] = []
    for f in cache_dir.iterdir():
        if not f.is_file():
            continue
        # 캐시 파일 패턴: <key>.npz / <key>.json / <key>.info.json
        stem = f.name
        for ext in (".npz", ".json", ".info.json"):
            if stem.endswith(ext):
                candidate_key = stem[: -len(ext)]
                if candidate_key != keep_key:
                    f.unlink(missing_ok=True)
                    removed.append(f)
                break
    if removed:
        _cache_log(
            f"[MultiGameDataset] Removed {len(removed)} stale cache file(s) "
            f"from {cache_dir}"
        )


def save_samples_to_cache(cache_dir: Path, key: str, samples: List[GameSample]) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    npz_path, meta_path, info_path = _cache_paths(cache_dir, key)

    # 기존 캐시 파일 전부 삭제 (stale 파일 방지)
    _purge_old_caches(cache_dir, keep_key=key)

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

    # 캐시 생성 환경 정보 저장
    info = _collect_info(samples)
    info_path.write_text(_stable_json(info), encoding="utf-8")
    _cache_log(
        f"[MultiGameDataset] Cache saved → {npz_path.name}  "
        f"(total={info['total_samples']}, games={info['game_counts']}, "
        f"created_at={info['created_at']}, host={info['hostname']})"
    )


def load_samples_from_cache(cache_dir: Path, key: str) -> Optional[List[GameSample]]:
    npz_path, meta_path, info_path = _cache_paths(cache_dir, key)
    if not npz_path.exists() or not meta_path.exists():
        return None

    arrays = np.load(npz_path)["arrays"]
    meta: List[Dict[str, Any]] = json.loads(meta_path.read_text(encoding="utf-8"))
    if len(meta) != len(arrays):
        return None

    # 캐시 생성 메타데이터 로그 출력
    if info_path.exists():
        try:
            info: Dict[str, Any] = json.loads(info_path.read_text(encoding="utf-8"))
            _cache_log(
                f"[MultiGameDataset] Loaded from cache  "
                f"total={info.get('total_samples', len(meta))} | "
                f"games={info.get('game_counts', {})} | "
                f"created_at={info.get('created_at', '?')} | "
                f"host={info.get('hostname', '?')}"
            )
        except Exception:
            pass
    else:
        _cache_log(
            f"[MultiGameDataset] Loaded from cache  total={len(meta)} (no info file)"
        )

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

