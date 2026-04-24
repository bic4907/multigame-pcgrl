"""Cache helpers for MultiGameDataset.

v1 (legacy): 단일 캐시 — hash(모든 init args + 전체 코드) → 하나의 npz/json
v2 (현재):   게임별 캐시 — artifacts/{game}/{key}.npz|json|info.json
             각 게임의 캐시 키는 해당 게임의 root, handler_config, 핸들러 코드만으로 결정.
             원본 데이터셋 없이 artifact만으로도 로드 가능.
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

CACHE_SCHEMA_VERSION = 2
ANN_SCHEMA_VERSION = 1

# dataset/multigame/ → dataset/ → project_root
_HERE = Path(__file__).parent
_PROJECT_ROOT: Path = _HERE.parent.parent

# ── 게임별 핸들러 파일 매핑 ──────────────────────────────────────────────────
GAME_HANDLER_FILES: Dict[str, List[str]] = {
    "dungeon": ["handlers/dungeon_handler.py"],
    "sokoban": ["handlers/boxoban_handler.py"],
    "zelda":   ["handlers/zelda_handler.py", "handlers/vglc_handler.py"],
    "pokemon": ["handlers/pokemon_handler.py"],
    "doom":    ["handlers/doom_handler.py"],
    "doom2":   ["handlers/doom_handler.py"],
}


def _cache_log(msg: str, level: str = "info") -> None:
    """logger.info/debug + print fallback."""
    getattr(logger, level)(msg)
    if level == "debug":
        return
    root_has_handlers = bool(logging.root.handlers)
    pkg_has_real_handler = any(
        not isinstance(h, logging.NullHandler)
        for h in logging.getLogger("dataset.multigame").handlers
    )
    if not root_has_handlers and not pkg_has_real_handler:
        print(msg)


# ═══════════════════════════════════════════════════════════════════════════════
#  공통 유틸
# ═══════════════════════════════════════════════════════════════════════════════

def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _stable_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=True, separators=(",", ":"))


def _normalize_path(raw: str) -> str:
    p = Path(raw).resolve()
    try:
        return str(p.relative_to(_PROJECT_ROOT.resolve()))
    except ValueError:
        return str(Path(*p.parts[-2:]))


# ═══════════════════════════════════════════════════════════════════════════════
#  Per-game 캐시 (v2)
# ═══════════════════════════════════════════════════════════════════════════════

def hash_handler_files(game: str) -> str:
    """게임별 핸들러 파일만 해싱한다."""
    handler_paths = GAME_HANDLER_FILES.get(game, [])
    common_files = ["base.py", "tile_utils.py", "handlers/handler_config.py"]
    all_files = sorted(set(handler_paths + common_files))

    h = hashlib.sha256()
    for rel in all_files:
        p = _HERE / rel
        if p.is_file():
            h.update(rel.encode("utf-8"))
            h.update(p.read_bytes())
        elif p.is_dir():
            for f in sorted(p.rglob("*.py")):
                h.update(str(f.relative_to(_HERE)).encode("utf-8"))
                h.update(f.read_bytes())
    return h.hexdigest()


def build_per_game_cache_key(
    game: str,
    game_root: str,
    handler_config_dict: Dict[str, Any],
) -> str:
    """게임별 캐시 키를 생성한다."""
    payload = {
        "schema": CACHE_SCHEMA_VERSION,
        "game": game,
        "game_root": _normalize_path(game_root),
        "handler_config": handler_config_dict,
        "code_hash": hash_handler_files(game),
    }
    return _sha256_bytes(_stable_json(payload).encode("utf-8"))


def build_combined_doom_cache_key(
    doom_root: str,
    doom2_root: str,
    include_doom: bool,
    include_doom2: bool,
    handler_config_dict: Dict[str, Any],
) -> str:
    """doom + doom2 통합 캐시 키를 생성한다."""
    payload = {
        "schema": CACHE_SCHEMA_VERSION,
        "game": "doom",
        "doom_root": _normalize_path(doom_root) if include_doom else None,
        "doom2_root": _normalize_path(doom2_root) if include_doom2 else None,
        "handler_config": handler_config_dict,
        "code_hash": hash_handler_files("doom"),
    }
    return _sha256_bytes(_stable_json(payload).encode("utf-8"))


def _game_cache_dir(cache_dir: Path, game: str) -> Path:
    return cache_dir / game


def _game_cache_paths(cache_dir: Path, game: str, key: str):
    d = _game_cache_dir(cache_dir, game)
    base = d / key
    return base.with_suffix(".npz"), base.with_suffix(".json"), base.with_suffix(".info.json")


def _game_ann_path(cache_dir: Path, game: str, key: str) -> Path:
    """게임별 annotation 캐시 파일 경로."""
    return _game_cache_dir(cache_dir, game) / f"{key}.ann.json"


def _purge_old_game_caches(game_dir: Path, keep_key: str) -> None:
    """게임별 캐시 디렉토리에서 keep_key 외 파일 삭제."""
    if not game_dir.exists():
        return
    removed: List[Path] = []
    for f in game_dir.iterdir():
        if not f.is_file():
            continue
        stem = f.name
        # 긴 확장자를 먼저 체크해야 한다: .ann.json/.info.json이 .json보다 먼저
        for ext in (".ann.json", ".info.json", ".npz", ".json"):
            if stem.endswith(ext):
                candidate_key = stem[: -len(ext)]
                if candidate_key != keep_key:
                    f.unlink(missing_ok=True)
                    removed.append(f)
                break
    if removed:
        _cache_log(
            f"[MultiGameDataset] Removed {len(removed)} stale cache file(s) "
            f"from {game_dir}"
        )


def _collect_info(samples: List[GameSample], game: str = "") -> Dict[str, Any]:
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
        "game": game,
        "game_counts": game_counts,
    }


def save_game_samples_to_cache(
    cache_dir: Path, game: str, key: str, samples: List[GameSample]
) -> None:
    """게임별 캐시 저장."""
    game_dir = _game_cache_dir(cache_dir, game)
    game_dir.mkdir(parents=True, exist_ok=True)

    _purge_old_game_caches(game_dir, keep_key=key)

    npz_path, meta_path, info_path = _game_cache_paths(cache_dir, game, key)

    if samples:
        arrays = np.stack([s.array for s in samples], axis=0)
    else:
        arrays = np.zeros((0, 16, 16), dtype=np.int32)

    np.savez_compressed(npz_path, arrays=arrays)

    # 맵 정보 + ann_keys만 저장 (instruction/meta는 ann.json에서 관리)
    meta: List[Dict[str, Any]] = []
    for s in samples:
        entry: Dict[str, Any] = {
            "game":      s.game,
            "source_id": s.source_id,
            "order":     s.order,
        }
        ann_keys = s.meta.get("ann_keys") if s.meta else None
        if ann_keys:
            entry["ann_keys"] = ann_keys
        meta.append(entry)
    meta_path.write_text(_stable_json(meta), encoding="utf-8")

    info = _collect_info(samples, game=game)
    info_path.write_text(_stable_json(info), encoding="utf-8")
    _cache_log(
        f"[MultiGameDataset] Cache saved → {game}/{npz_path.name}  "
        f"(total={info['total_samples']}, game={game})"
    )


def load_game_samples_from_cache(
    cache_dir: Path, game: str, key: str
) -> Optional[List[GameSample]]:
    """게임별 캐시 로드. 없으면 None 반환."""
    npz_path, meta_path, info_path = _game_cache_paths(cache_dir, game, key)
    if not npz_path.exists() or not meta_path.exists():
        return None

    arrays = np.load(npz_path)["arrays"]
    meta: List[Dict[str, Any]] = json.loads(meta_path.read_text(encoding="utf-8"))
    if len(meta) != len(arrays):
        return None

    if info_path.exists():
        try:
            info: Dict[str, Any] = json.loads(info_path.read_text(encoding="utf-8"))
            _cache_log(
                f"[MultiGameDataset] Loaded {game} from cache  "
                f"total={info.get('total_samples', len(meta))} | "
                f"created_at={info.get('created_at', '?')} | "
                f"host={info.get('hostname', '?')}",
                level="debug",
            )
        except Exception:
            pass
    else:
        _cache_log(
            f"[MultiGameDataset] Loaded {game} from cache  total={len(meta)}",
            level="debug",
        )

    samples: List[GameSample] = []
    for i, m in enumerate(meta):
        # 신규 포맷: game/source_id/order/ann_keys만 저장
        # 구 포맷(instruction/meta 포함)도 읽을 수 있도록 호환 처리
        sample_meta: Dict[str, Any] = {}
        if "ann_keys" in m:
            sample_meta["ann_keys"] = m["ann_keys"]
        elif "meta" in m:
            old_meta = m["meta"]
            if isinstance(old_meta, dict):
                sample_meta = {k: v for k, v in old_meta.items()
                               if k in ("level_id", "sample_id", "instruction_slug")}
        samples.append(
            GameSample(
                game=m["game"],
                source_id=m["source_id"],
                array=arrays[i].astype(np.int32),
                char_grid=None,
                legend=None,
                instruction=None,      # instruction은 ann.json에서 로드
                order=m.get("order"),
                meta=sample_meta,
            )
        )
    return samples


def list_cached_games(cache_dir: Path) -> List[str]:
    """캐시 디렉토리에서 사용 가능한 게임 목록을 반환한다."""
    if not cache_dir.exists():
        return []
    games = []
    for d in sorted(cache_dir.iterdir()):
        if d.is_dir() and any(d.glob("*.npz")):
            games.append(d.name)
    return games


def load_any_game_cache(cache_dir: Path, game: str) -> Optional[List[GameSample]]:
    """게임 디렉토리에 있는 아무 캐시나 로드한다 (키를 모를 때).

    artifact-only 모드: 원본 데이터도 없고 현재 키로 매칭되는 캐시도 없지만,
    해당 게임 디렉토리에 npz 파일이 존재하면 그것을 로드한다.
    """
    game_dir = _game_cache_dir(cache_dir, game)
    if not game_dir.exists():
        return None
    npz_files = sorted(game_dir.glob("*.npz"))
    if not npz_files:
        return None
    # 가장 최근 npz 사용
    npz_path = npz_files[-1]
    key = npz_path.stem
    return load_game_samples_from_cache(cache_dir, game, key)


# ═══════════════════════════════════════════════════════════════════════════════
#  Annotation cache (ann.json)
# ═══════════════════════════════════════════════════════════════════════════════

def save_game_annotations_to_cache(
    cache_dir: Path,
    game: str,
    key: str,
    annotations: List[Dict[str, Any]],
    has_instructions: bool = False,
    n_samples: int = 0,
    batch_id: Optional[str] = None,
) -> None:
    """게임별 annotation을 {key}.ann.json에 저장.

    annotations: _make_rows()가 반환하는 dict 리스트.
                 각 dict: key, source_id, reward_enum, feature_name,
                           sub_condition, condition_0..4,
                           instruction_raw, instruction_uni
    batch_id: OpenAI 배치 제출 시 기록. None이면 필드 생략.
    """
    game_dir = _game_cache_dir(cache_dir, game)
    game_dir.mkdir(parents=True, exist_ok=True)
    ann_path = _game_ann_path(cache_dir, game, key)
    payload: Dict[str, Any] = {
        "schema": ANN_SCHEMA_VERSION,
        "game": game,
        "n_samples": n_samples,
        "has_instructions": has_instructions,
        "annotations": annotations,
    }
    if batch_id is not None:
        payload["batch_id"] = batch_id
    ann_path.write_text(_stable_json(payload), encoding="utf-8")
    _cache_log(
        f"[MultiGameDataset] Annotations saved → {game}/{ann_path.name}  "
        f"({len(annotations)} rows, has_instructions={has_instructions})"
    )


def load_game_annotations_from_cache(
    cache_dir: Path,
    game: str,
    key: str,
) -> Optional[Dict[str, Any]]:
    """게임별 annotation 로드. 없거나 파싱 실패 시 None 반환.

    반환값 구조:
      {"schema": 1, "game": ..., "n_samples": ...,
       "has_instructions": bool, "annotations": List[dict]}
    """
    ann_path = _game_ann_path(cache_dir, game, key)
    if not ann_path.exists():
        return None
    try:
        data = json.loads(ann_path.read_text(encoding="utf-8"))
        _cache_log(
            f"[MultiGameDataset] Annotations loaded ← {game}/{ann_path.name}  "
            f"({len(data.get('annotations', []))} rows, "
            f"has_instructions={data.get('has_instructions', False)})",
            level="debug",
        )
        return data
    except Exception as e:
        _cache_log(f"[MultiGameDataset] Failed to load ann.json for {game}: {e}")
        return None


def update_json_with_ann_keys(
    cache_dir: Path, game: str, key: str, ann_data: Dict[str, Any]
) -> None:
    """ann.json의 키 정보를 {key}.json 메타데이터에 ann_keys로 기록한다.

    각 샘플의 ann_keys = [key_r0, key_r1, ..., key_r{n_rewards-1}]
    ann.json 행 순서: reward_enum 0 전체 → 1 전체 → … → 4 전체
    sample i, reward_enum r → ann row r * n_samples + i
    """
    _, meta_path, _ = _game_cache_paths(cache_dir, game, key)
    if not meta_path.exists():
        return

    annotations = ann_data.get("annotations", [])
    n_samples = ann_data.get("n_samples", 0)
    if not annotations or n_samples == 0:
        return

    sorted_anns = sorted(annotations, key=lambda r: r["key"])
    n_rewards = len(sorted_anns) // n_samples
    if n_rewards == 0:
        return

    # sample i → [key_r0, key_r1, ..., key_r{n-1}]
    sample_ann_keys: Dict[int, List[str]] = {}
    for r in range(n_rewards):
        for i in range(n_samples):
            row_idx = r * n_samples + i
            if row_idx < len(sorted_anns):
                sample_ann_keys.setdefault(i, []).append(sorted_anns[row_idx]["key"])

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    updated = 0
    for i, entry in enumerate(meta):
        if i in sample_ann_keys:
            entry["ann_keys"] = sample_ann_keys[i]
            # 구 포맷 필드 제거
            entry.pop("instruction", None)
            entry.pop("meta", None)
            updated += 1
    meta_path.write_text(_stable_json(meta), encoding="utf-8")
    _cache_log(
        f"[MultiGameDataset] ann_keys updated → {game}/{meta_path.name}  "
        f"({updated} samples, {n_rewards} keys each)"
    )




def update_ann_batch_id(cache_dir: Path, game: str, key: str, batch_id: str) -> None:
    """ann.json에 batch_id 필드를 기록한다 (instruction 배치 제출 추적용)."""
    ann_path = _game_ann_path(cache_dir, game, key)
    if not ann_path.exists():
        return
    try:
        data = json.loads(ann_path.read_text(encoding="utf-8"))
        data["batch_id"] = batch_id
        ann_path.write_text(_stable_json(data), encoding="utf-8")
        _cache_log(
            f"[MultiGameDataset] ann.json batch_id recorded → {game}/{ann_path.name}  "
            f"(batch_id={batch_id})"
        )
    except Exception as e:
        _cache_log(f"[MultiGameDataset] Failed to record ann.json batch_id ({game}): {e}")


def find_game_cache_key(cache_dir: Path, game: str) -> Optional[str]:
    """게임 캐시 디렉토리에서 npz 파일 이름으로 캐시 키를 찾는다."""
    game_dir = _game_cache_dir(cache_dir, game)
    if not game_dir.exists():
        return None
    npz_files = sorted(game_dir.glob("*.npz"))
    if not npz_files:
        return None
    return npz_files[-1].stem


# ═══════════════════════════════════════════════════════════════════════════════
#  Legacy (v1) — 하위 호환용
# ═══════════════════════════════════════════════════════════════════════════════

def _normalize_args(args_dict: Dict[str, Any]) -> Dict[str, Any]:
    path_keys = {"vglc_root", "dungeon_root"}
    normalized: Dict[str, Any] = {}
    for k, v in args_dict.items():
        if k in path_keys and isinstance(v, str):
            normalized[k] = _normalize_path(v)
        else:
            normalized[k] = v
    return normalized


def hash_code_files(code_root: Path) -> str:
    """[Legacy] 전체 코드 해싱."""
    py_files = sorted(
        p for p in code_root.rglob("*.py")
        if "tests" not in p.parts and "__pycache__" not in p.parts
        and "cache" not in p.parts and "viewer" not in p.parts
    )
    h = hashlib.sha256()
    for p in py_files:
        h.update(str(p.relative_to(code_root)).encode("utf-8"))
        h.update(p.read_bytes())
    return h.hexdigest()


def build_cache_key(args_dict: Dict[str, Any], *, code_root: Path) -> str:
    """[Legacy] 단일 캐시 키 — 하위 호환용."""
    payload = {
        "schema": 1,
        "args": _normalize_args(args_dict),
        "code_hash": hash_code_files(code_root),
    }
    return _sha256_bytes(_stable_json(payload).encode("utf-8"))


def _cache_paths(cache_dir: Path, key: str):
    base = cache_dir / key
    return base.with_suffix(".npz"), base.with_suffix(".json"), base.with_suffix(".info.json")


def _purge_old_caches(cache_dir: Path, keep_key: str) -> None:
    """[Legacy] cache_dir 안의 캐시 파일 중 keep_key에 해당하지 않는 것을 모두 삭제."""
    removed: List[Path] = []
    for f in cache_dir.iterdir():
        if not f.is_file():
            continue
        stem = f.name
        for ext in (".info.json", ".npz", ".json"):
            if stem.endswith(ext):
                candidate_key = stem[: -len(ext)]
                if candidate_key != keep_key:
                    f.unlink(missing_ok=True)
                    removed.append(f)
                break
    if removed:
        _cache_log(
            f"[MultiGameDataset] Removed {len(removed)} stale legacy cache file(s)"
        )


def save_samples_to_cache(cache_dir: Path, key: str, samples: List[GameSample]) -> None:
    """[Legacy] 단일 캐시 저장."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    npz_path, meta_path, info_path = _cache_paths(cache_dir, key)
    _purge_old_caches(cache_dir, keep_key=key)

    if samples:
        arrays = np.stack([s.array for s in samples], axis=0)
    else:
        arrays = np.zeros((0, 16, 16), dtype=np.int32)

    np.savez_compressed(npz_path, arrays=arrays)

    meta: List[Dict[str, Any]] = []
    for s in samples:
        meta.append({
            "game": s.game,
            "source_id": s.source_id,
            "instruction": s.instruction,
            "order": s.order,
            "meta": s.meta,
        })
    meta_path.write_text(_stable_json(meta), encoding="utf-8")
    info_data = _collect_info(samples)
    info_path.write_text(_stable_json(info_data), encoding="utf-8")
    _cache_log(f"[Legacy] Cache saved → {npz_path.name} (total={len(samples)})")


def load_samples_from_cache(cache_dir: Path, key: str) -> Optional[List[GameSample]]:
    """[Legacy] 단일 캐시 로드."""
    npz_path, meta_path, info_path = _cache_paths(cache_dir, key)
    if not npz_path.exists() or not meta_path.exists():
        return None

    arrays = np.load(npz_path)["arrays"]
    meta: List[Dict[str, Any]] = json.loads(meta_path.read_text(encoding="utf-8"))
    if len(meta) != len(arrays):
        return None

    if info_path.exists():
        try:
            info: Dict[str, Any] = json.loads(info_path.read_text(encoding="utf-8"))
            _cache_log(
                f"[Legacy] Loaded from cache  total={info.get('total_samples', len(meta))}"
            )
        except Exception:
            pass

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

