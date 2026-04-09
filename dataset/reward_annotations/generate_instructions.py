#!/usr/bin/env python3
"""
dataset/reward_annotations/generate_instructions.py
=====================================================
OpenAI Batch API (gpt-4o-2024-08-06)를 사용하여
reward annotation CSV의 instruction_raw / instruction_uni 필드를 채운다.

설정 상수: instruction_config.py
시스템 프롬프트: system_prompt.txt
배치 추적: batches/batch_log.csv  (단일 CSV)

Usage:
  # JSONL 생성 + 배치 제출
  python dataset/reward_annotations/generate_instructions.py --submit

  # 특정 게임/enum만 처리
  python dataset/reward_annotations/generate_instructions.py --submit \\
      --games doom zelda --enums 0 1

  # 결과 조회 + CSV 업데이트
  python dataset/reward_annotations/generate_instructions.py --retrieve BATCH_ID

  # 배치 상태 확인
  python dataset/reward_annotations/generate_instructions.py --status BATCH_ID

  # 제출 후 완료까지 대기 → 자동 CSV 업데이트
  python dataset/reward_annotations/generate_instructions.py --run
"""
from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import logging
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ── 프로젝트 루트 ─────────────────────────────────────────────────────────────────
_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE.parent.parent))

import numpy as np
from PIL import Image, ImageDraw
from dotenv import load_dotenv

load_dotenv(_HERE.parent.parent / ".env")

from dataset.multigame.tile_utils import to_unified, CATEGORY_COLORS, UNIFIED_CATEGORIES
from instruction_config import (
    CUSTOM_THRESHOLDS,
    RAW_TILE_COLORS, RAW_TILE_NAMES, RAW_TILE_DESCS,
    FEATURE_TILE_DESCS, FEATURE_COUNT_TILE_IDS,
    GAME_DESCRIPTIONS, FEATURE_DESCRIPTIONS,
    UNIFIED_COLOR_DESCS, FEATURE_ZONE_LABELS, VOCAB_SETS,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)   # openai SDK HTTP 요청 로그 숨김
logger = logging.getLogger(__name__)

# ── 경로 ─────────────────────────────────────────────────────────────────────────
_CACHE_DIR   = _HERE.parent / "multigame" / "cache" / "artifacts"
_ANNOT_DIR   = _HERE
_BATCH_DIR   = _HERE / "batches"
_BATCH_LOG   = _BATCH_DIR / "batch_log.csv"          # 단일 배치 추적 CSV
_SYSTEM_PROMPT_FILE = _HERE / "system_prompt.txt"

# ── 모델 설정 ─────────────────────────────────────────────────────────────────────
MODEL       = "gpt-5.4-mini"
MAX_TOKENS  = 300
TEMPERATURE = 2.0

# ── 배치 로그 CSV 헤더 ────────────────────────────────────────────────────────────
_LOG_HEADER = ["batch_id", "jsonl_file", "games", "enums",
               "n_requests", "status", "submitted_at", "completed_at"]

# ── reward_enum → condition 컬럼 ─────────────────────────────────────────────────
_ENUM_TO_COND_COL = {
    0: "condition_0", 1: "condition_1", 2: "condition_2",
    3: "condition_3", 4: "condition_4",
}

_MAX_SAMPLES: Dict[str, int] = {
    "doom": 1000, "zelda": 1000, "sokoban": 1000, "pokemon": 1000, "dungeon": 4000,
}


# ── 시스템 프롬프트 로드 ──────────────────────────────────────────────────────────

def load_system_prompt() -> str:
    """system_prompt.txt 에서 시스템 프롬프트를 읽는다."""
    return _SYSTEM_PROMPT_FILE.read_text(encoding="utf-8").strip()


# ── Zone 계산 ─────────────────────────────────────────────────────────────────────

def get_zone_label(value: float, feature: str, thresholds: Optional[List[float]]) -> str:
    if thresholds is None:
        return "N/A (no threshold defined)"
    sorted_t = sorted(thresholds)
    for i, t in enumerate(sorted_t):
        if value <= t:
            idx = i
            break
    else:
        idx = len(sorted_t)
    labels = FEATURE_ZONE_LABELS.get(feature, ["very few", "somewhat few", "somewhat many", "very many"])
    return labels[min(idx, len(labels) - 1)]


# ── 렌더링 ────────────────────────────────────────────────────────────────────────

def _render_png(
    array: np.ndarray,
    color_map: Dict[int, Tuple[int, int, int]],
    tile_size: int = 16,
) -> bytes:
    h, w = array.shape
    img  = Image.new("RGB", (w * tile_size, h * tile_size), (200, 200, 200))
    draw = ImageDraw.Draw(img)
    for r in range(h):
        for c in range(w):
            color = color_map.get(int(array[r, c]), (128, 0, 128))
            x0, y0 = c * tile_size, r * tile_size
            draw.rectangle([x0, y0, x0 + tile_size - 1, y0 + tile_size - 1], fill=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def render_raw_png(array: np.ndarray, game: str, tile_size: int = 16) -> bytes:
    return _render_png(array, RAW_TILE_COLORS.get(game, {}), tile_size)


def render_unified_png(array: np.ndarray, game: str, tile_size: int = 16) -> bytes:
    unified   = to_unified(array, game, warn_unmapped=False)
    color_map = {int(k): tuple(v) for k, v in CATEGORY_COLORS.items()}
    return _render_png(unified, color_map, tile_size)


# ── 유저 프롬프트 빌더 ────────────────────────────────────────────────────────────

_COUNT_FEATURES = {"interactable_count", "hazard_count", "collectable_count"}


def build_user_prompt(
    game: str,
    feature_name: str,
    condition_value: float,
    sub_condition: str,
    thresholds: Optional[List[float]],
    zone_label: str,
) -> str:
    lines: List[str] = []

    lines.append(f"## Game\n{GAME_DESCRIPTIONS.get(game, game)}\n")

    # zone_label → 0-based index for vocab lookup, 1-based for display
    zone_idx_0: Optional[int] = None
    zone_display: Optional[str] = None
    if thresholds is not None:
        feat_zones = FEATURE_ZONE_LABELS.get(feature_name, [])
        try:
            zone_idx_0 = feat_zones.index(zone_label)        # 0-based
            zone_display = f"intensity level {zone_idx_0 + 1}/4"
        except ValueError:
            pass

    lines.append("## Condition")
    lines.append(f"- Feature: {feature_name}")
    lines.append(f"- Description: {FEATURE_DESCRIPTIONS.get(feature_name, feature_name)}")
    if zone_display is not None:
        lines.append(f"- Intensity level: {zone_display} (scale: 1=lowest → 4=highest)")
    elif thresholds is not None:
        lines.append("- Intensity level: N/A (threshold not defined for this combination)")
    else:
        lines.append("- Intensity level: N/A (threshold not defined for this combination)")
    lines.append("")

    if thresholds is not None:
        lines.append("## Intensity Reference")
        lines.append(f"The measured intensity for this map is {zone_display} on a 4-point scale (1=lowest, 4=highest).")
        lines.append("")
    else:
        lines.append("## Intensity Reference\nNo threshold defined — describe based on what you observe in the map.\n")

    lines.append("## Image 1 — Raw Map (game-specific tile colors)")
    tile_names  = RAW_TILE_NAMES.get(game, {})
    tile_descs  = RAW_TILE_DESCS.get(game, {})
    tile_colors = RAW_TILE_COLORS.get(game, {})
    lines.append("Tile legend:")
    for tid in sorted(tile_names.keys()):
        name = tile_names[tid]
        desc = tile_descs.get(tid, "")
        r, g, b = tile_colors.get(tid, (128, 0, 128))
        lines.append(f"  ID={tid}  {name:10s}  color=RGB({r},{g},{b})  — {desc}")
    raw_desc = FEATURE_TILE_DESCS.get(game, {}).get(feature_name, ("", ""))[0]
    lines.append(f"Count basis: {raw_desc if raw_desc else sub_condition}")

    lines.append("")

    lines.append("## Image 2 — Unified Map (unified category colors)")
    cat_info = {
        0: ("empty",       "background / void / walkable space"),
        1: ("wall",        "solid impassable obstacle"),
        2: ("interactive", "interactable tiles (doors, objects, spawns)"),
        3: ("hazard",      "enemy / damaging entity"),
        4: ("collectable", "collectible item / pickup"),
    }
    lines.append("Tile legend:")
    for cid, (cname, cdesc) in cat_info.items():
        color_str = UNIFIED_COLOR_DESCS.get(cid, "")
        lines.append(f"  ID={cid}  {cname:14s}  {color_str}  — {cdesc}")
    uni_desc = FEATURE_TILE_DESCS.get(game, {}).get(feature_name, ("", ""))[1]
    lines.append(f"Count basis: {uni_desc}")
    lines.append("")

    # 어휘 세트: feature × level
    vocab_hint = ""
    if zone_idx_0 is not None:
        vocab_list = VOCAB_SETS.get(feature_name, [])
        if zone_idx_0 < len(vocab_list):
            word = random.choice(vocab_list[zone_idx_0])
            vocab_hint = (
                f"Suggested vocabulary (feel free to use variations or different expressions): "
                f"{repr(word)}"
            )

    lines.append("## Task")
    if feature_name in _COUNT_FEATURES:
        if zone_display is not None:
            lines.append(
                f"Write one short sentence describing this map's {feature_name} ({zone_display})."
            )
        else:
            lines.append(
                f"Write one short sentence describing this map's {feature_name} based on what you see."
            )
        if vocab_hint:
            lines.append(vocab_hint)
        lines.append(
            "- instruction_raw: use game-specific tile names to describe the intensity."
        )
        lines.append(
            "- instruction_uni: use unified category names (empty/wall/interactive/hazard/collectable) only; "
            "do NOT reference specific tile names — describe only the overall intensity level."
        )
        lines.append("Neither sentence should contain any numbers or measured values.")
    else:
        if zone_display is not None:
            lines.append(
                f"Write one short sentence describing this map's {feature_name} ({zone_display}). "
                "No numbers. instruction_raw uses raw tile names; instruction_uni uses unified category names."
            )
        else:
            lines.append(
                f"Write one short sentence describing this map's {feature_name} based on what you see. "
                "No numbers. instruction_raw uses raw tile names; instruction_uni uses unified category names."
            )
        if vocab_hint:
            lines.append(vocab_hint)

    return "\n".join(lines)


# ── 배치 요청 빌더 ────────────────────────────────────────────────────────────────

def build_batch_request(
    custom_id: str,
    game: str,
    feature_name: str,
    condition_value: float,
    sub_condition: str,
    array: np.ndarray,
    system_prompt: str,
) -> dict:
    thresholds = CUSTOM_THRESHOLDS.get(f"{game}_{feature_name}")
    zone_label = get_zone_label(condition_value, feature_name, thresholds)

    raw_b64 = base64.b64encode(render_raw_png(array, game)).decode()
    uni_b64 = base64.b64encode(render_unified_png(array, game)).decode()
    user_text = build_user_prompt(
        game, feature_name, condition_value, sub_condition, thresholds, zone_label,
    )

    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/responses",
        "body": {
            "model": MODEL,
            "temperature": TEMPERATURE,
            "instructions": system_prompt,
            "input": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": user_text
                        },
                        {
                            "type": "input_image",
                            "image_url": f"data:image/png;base64,{raw_b64}",
                            "detail": "low"
                        },
                        {
                            "type": "input_image",
                            "image_url": f"data:image/png;base64,{uni_b64}",
                            "detail": "low"
                        }
                    ]
                }
            ],
            "max_output_tokens": MAX_TOKENS,
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "level_instructions",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "instruction_raw": {"type": "string"},
                            "instruction_uni": {"type": "string"}
                        },
                        "required": ["instruction_raw", "instruction_uni"],
                        "additionalProperties": False
                    }
                }
            }
        }
    }

# ── 캐시 로드 ─────────────────────────────────────────────────────────────────────

def _load_cache(cache_dir: Path) -> Optional[List[dict]]:
    json_files = sorted(f for f in cache_dir.glob("*.json") if not f.name.endswith(".info.json"))
    if not json_files:
        return None
    meta_path = json_files[-1]
    npz_path  = meta_path.with_suffix(".npz")
    if not npz_path.exists():
        return None
    arrays = np.load(npz_path)["arrays"]
    meta   = json.loads(meta_path.read_text(encoding="utf-8"))
    assert len(meta) == len(arrays)
    return [{**m, "array": arrays[i].astype(np.int32)} for i, m in enumerate(meta)]


def _shorten_source_id(source_id: str, game: str) -> str:
    if game == "doom":
        path_part, slice_idx = (source_id.rsplit("|", 1) if "|" in source_id else (source_id, "0"))
        p = Path(path_part)
        version = "Doom2" if any("Doom2" in pt for pt in p.parts) else "Doom1"
        return f"{version}_{p.stem}_{int(slice_idx):03d}"
    if game == "sokoban":
        path_part, lvl_idx = (source_id.rsplit("#", 1) if "#" in source_id else (source_id, "0"))
        p = Path(path_part)
        difficulty = "hard" if any("hard" in pt for pt in p.parts) else "medium"
        return f"{difficulty}_{p.stem}_{int(lvl_idx):03d}"
    return source_id


def load_cache_by_game(cache_dir: Path) -> Dict[str, Dict[str, np.ndarray]]:
    """캐시를 {game: {sample_id: array}} 로 반환한다."""
    by_game: Dict[str, List[dict]] = {}
    if cache_dir.is_dir():
        for sub in sorted(cache_dir.iterdir()):
            if not sub.is_dir():
                continue
            samples = _load_cache(sub)
            if samples:
                for s in samples:
                    g = s.get("game", "")
                    if g:
                        by_game.setdefault(g, []).append(s)
    if not by_game:
        all_s = _load_cache(cache_dir)
        if all_s:
            for s in all_s:
                by_game.setdefault(s.get("game", ""), []).append(s)
    for g in list(by_game.keys()):
        limit = _MAX_SAMPLES.get(g)
        if limit and len(by_game[g]) > limit:
            by_game[g] = by_game[g][:limit]
    result: Dict[str, Dict[str, np.ndarray]] = {}
    for g, samples in by_game.items():
        sid_map: Dict[str, np.ndarray] = {}
        for i, s in enumerate(samples):
            sid = _shorten_source_id(s.get("source_id", str(i)), g)
            sid_map[sid] = s["array"]
        result[g] = sid_map
    return result


# ── JSONL 생성 (단일 파일) ────────────────────────────────────────────────────────

def _is_none_threshold(game: str, feature_name: str) -> bool:
    """CUSTOM_THRESHOLDS 가 None 인 (game, feature) 조합 여부."""
    return CUSTOM_THRESHOLDS.get(f"{game}_{feature_name}") is None


def fill_none_instructions(
    games: List[str],
    enums: List[int],
    annot_dir: Path,
    force: bool = False,
) -> int:
    """
    CUSTOM_THRESHOLDS 가 None 인 행은 GPT 없이 instruction_raw / instruction_uni 를
    "None" 으로 직접 채운다. 업데이트된 행 수를 반환한다.
    """
    none_results: Dict[str, dict] = {}

    for game in games:
        csv_path = annot_dir / f"{game}_reward_annotations.csv"
        if not csv_path.exists():
            continue
        with csv_path.open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                reward_enum = int(row["reward_enum"])
                if reward_enum not in enums:
                    continue
                if not force and row.get("instruction_raw", "").strip():
                    continue
                if _is_none_threshold(game, row["feature_name"]):
                    none_results[row["key"]] = {
                        "instruction_raw": "None",
                        "instruction_uni": "None",
                    }

    if none_results:
        n = update_csvs(none_results, annot_dir, games)
        logger.info(f"threshold=None 행 {n}개 → 'None' 으로 직접 채움")
        return n
    return 0


def build_jsonl(
    games: List[str],
    enums: List[int],
    annot_dir: Path,
    cache_by_game: Dict[str, Dict[str, np.ndarray]],
    system_prompt: str,
    force: bool = False,
    limit: Optional[int] = None,
) -> Optional[Path]:
    """
    처리 대상 행에 대한 JSONL 파일을 단일 파일로 생성하여 경로를 반환한다.
    threshold=None 인 행은 제외 (fill_none_instructions 에서 별도 처리).
    생성할 행이 없으면 None을 반환한다.
    """
    _BATCH_DIR.mkdir(parents=True, exist_ok=True)

    lines:  List[str] = []
    n_skip = 0
    n_none = 0

    for game in games:
        csv_path = annot_dir / f"{game}_reward_annotations.csv"
        if not csv_path.exists():
            logger.warning(f"{csv_path.name} 없음, 건너뜀")
            continue
        sid_map = cache_by_game.get(game, {})
        if not sid_map:
            logger.warning(f"{game}: 캐시 없음, 건너뜀")
            continue

        with csv_path.open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                reward_enum = int(row["reward_enum"])
                if reward_enum not in enums:
                    continue
                if not force and row.get("instruction_raw", "").strip() and row.get("instruction_uni", "").strip():
                    n_skip += 1
                    continue

                # threshold=None → 별도 처리 (GPT 호출 안 함)
                if _is_none_threshold(game, row["feature_name"]):
                    n_none += 1
                    continue

                cond_col = _ENUM_TO_COND_COL.get(reward_enum)
                raw_val  = row.get(cond_col, "")
                if not raw_val:
                    continue
                try:
                    cond_val = float(raw_val)
                except ValueError:
                    continue

                array = sid_map.get(row["sample_id"])
                if array is None:
                    continue

                req  = build_batch_request(
                    row["key"], game, row["feature_name"],
                    cond_val, row.get("sub_condition", ""),
                    array, system_prompt,
                )
                lines.append(json.dumps(req, ensure_ascii=False))

                if limit and len(lines) >= limit:
                    break
        if limit and len(lines) >= limit:
            break

    if n_skip:
        logger.info(f"이미 채워진 행 {n_skip}개 건너뜀 (--force 로 재생성 가능)")
    if n_none:
        logger.info(f"threshold=None 행 {n_none}개 제외 (--submit 전 자동 처리됨)")

    if not lines:
        logger.info("생성할 요청이 없습니다.")
        return None

    ts       = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = _BATCH_DIR / f"batch_{ts}.jsonl"
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info(f"JSONL 생성: {out_path.name}  ({len(lines)} requests)")
    return out_path


# ── 배치 로그 CSV 관리 ────────────────────────────────────────────────────────────

def _read_batch_log() -> List[dict]:
    if not _BATCH_LOG.exists():
        return []
    with _BATCH_LOG.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_batch_log(rows: List[dict]) -> None:
    _BATCH_DIR.mkdir(parents=True, exist_ok=True)
    with _BATCH_LOG.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_LOG_HEADER)
        writer.writeheader()
        writer.writerows(rows)


def _append_batch_log(row: dict) -> None:
    rows = _read_batch_log()
    rows.append(row)
    _write_batch_log(rows)


def _update_batch_log(batch_id: str, **kwargs) -> None:
    rows = _read_batch_log()
    for r in rows:
        if r["batch_id"] == batch_id:
            r.update(kwargs)
    _write_batch_log(rows)


# ── OpenAI Batch API ──────────────────────────────────────────────────────────────

def submit_batch(jsonl_path: Path, games: List[str], enums: List[int], n: int) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    logger.info(f"파일 업로드: {jsonl_path.name}")
    with jsonl_path.open("rb") as f:
        file_obj = client.files.create(file=f, purpose="batch")
    logger.info(f"  file_id: {file_obj.id}")

    batch = client.batches.create(
        input_file_id=file_obj.id,
        endpoint="/v1/responses",
        completion_window="24h",
        metadata={"source_file": jsonl_path.name},
    )
    logger.info(f"  batch_id: {batch.id}  status: {batch.status}")

    _append_batch_log({
        "batch_id":     batch.id,
        "jsonl_file":   jsonl_path.name,
        "games":        ",".join(games),
        "enums":        ",".join(map(str, enums)),
        "n_requests":   n,
        "status":       batch.status,
        "submitted_at": datetime.now(timezone.utc).isoformat(),
        "completed_at": "",
    })
    return batch.id


def check_batch_status(batch_id: str) -> dict:
    from openai import OpenAI
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    b = client.batches.retrieve(batch_id)
    counts = b.request_counts
    return {
        "id":     b.id,
        "status": b.status,
        "output_file_id": getattr(b, "output_file_id", None),
        "error_file_id":  getattr(b, "error_file_id", None),
        "request_counts": {
            "total":     getattr(counts, "total",     0) if counts else 0,
            "completed": getattr(counts, "completed", 0) if counts else 0,
            "failed":    getattr(counts, "failed",    0) if counts else 0,
        },
    }


def _extract_text_from_response_body(body: dict) -> Optional[str]:
    msg = body.get("output_text")
    if isinstance(msg, str) and msg.strip():
        return msg.strip()

    texts: List[str] = []
    for item in body.get("output", []):
        if item.get("type") != "message":
            continue
        for content in item.get("content", []):
            if content.get("type") == "output_text":
                text = content.get("text")
                if isinstance(text, str) and text.strip():
                    texts.append(text.strip())

    if texts:
        return "\n".join(texts).strip()

    return None


def retrieve_batch_results(batch_id: str) -> Dict[str, dict]:
    """완료된 배치 결과를 {custom_id: {instruction_raw, instruction_uni}} 로 반환."""
    from openai import OpenAI
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    b = client.batches.retrieve(batch_id)
    if b.status != "completed":
        raise RuntimeError(f"배치가 아직 완료되지 않았습니다. 상태: {b.status}")
    if not b.output_file_id:
        raise RuntimeError("output_file_id 없음")

    content = client.files.content(b.output_file_id).content
    results: Dict[str, dict] = {}

    for line in content.decode("utf-8").splitlines():
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
            cid = obj["custom_id"]

            response = obj.get("response") or {}
            status_code = response.get("status_code")
            body = response.get("body") or {}

            if status_code != 200:
                err = body.get("error", {})
                logger.warning(
                    f"요청 실패 ({cid}): status={status_code}, "
                    f"code={err.get('code')}, message={err.get('message')}"
                )
                continue

            msg = _extract_text_from_response_body(body)
            if not msg:
                logger.warning(f"응답 텍스트 없음 ({cid})")
                continue

            parsed = json.loads(msg)
            results[cid] = {
                "instruction_raw": parsed.get("instruction_raw", ""),
                "instruction_uni": parsed.get("instruction_uni", ""),
            }

        except Exception as e:
            logger.warning(f"파싱 실패 ({line[:60]}…): {e}")

    logger.info(f"결과 파싱: {len(results)}개 성공")
    _update_batch_log(
        batch_id,
        status="completed",
        completed_at=datetime.now(timezone.utc).isoformat(),
    )
    return results

# ── CSV 업데이트 ──────────────────────────────────────────────────────────────────

def update_csvs(results: Dict[str, dict], annot_dir: Path, games: List[str]) -> int:
    total = 0
    for game in games:
        csv_path = annot_dir / f"{game}_reward_annotations.csv"
        if not csv_path.exists():
            continue
        rows: List[dict] = []
        updated = 0
        with csv_path.open(encoding="utf-8") as f:
            reader     = csv.DictReader(f)
            fieldnames = list(reader.fieldnames or [])
            # 이전 형식(instruction) → 새 형식 자동 변환
            if "instruction" in fieldnames and "instruction_raw" not in fieldnames:
                idx = fieldnames.index("instruction")
                fieldnames = fieldnames[:idx] + ["instruction_raw", "instruction_uni"] + fieldnames[idx+1:]
            for row in reader:
                if row["key"] in results:
                    row["instruction_raw"] = results[row["key"]]["instruction_raw"]
                    row["instruction_uni"] = results[row["key"]]["instruction_uni"]
                    updated += 1
                rows.append(row)

        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)

        logger.info(f"  {csv_path.name}: {updated}개 업데이트")
        total += updated
    return total


# ── CLI ───────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="OpenAI Batch API로 instruction_raw / instruction_uni 생성"
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--submit",   action="store_true",
                      help="JSONL 생성 + Batch API 제출")
    mode.add_argument("--retrieve", metavar="BATCH_ID",
                      help="결과 조회 + CSV 업데이트")
    mode.add_argument("--status",   metavar="BATCH_ID",
                      help="배치 상태 확인")
    mode.add_argument("--run",      action="store_true",
                      help="제출 → 대기 → CSV 업데이트 (소규모 테스트)")
    mode.add_argument("--log",      action="store_true",
                      help="제출된 배치 로그 출력 (batch_log.csv)")

    parser.add_argument("--games", nargs="+",
                        default=["doom", "zelda", "sokoban", "pokemon", "dungeon"],
                        choices=["doom", "zelda", "sokoban", "pokemon", "dungeon"])
    parser.add_argument("--enums", nargs="+", type=int, default=[0, 1, 2, 3, 4],
                        choices=[0, 1, 2, 3, 4],
                        help="0=region 1=path_length 2=interactable 3=hazard 4=collectable")
    parser.add_argument("--cache-dir",    type=Path, default=_CACHE_DIR)
    parser.add_argument("--annot-dir",    type=Path, default=_ANNOT_DIR)
    parser.add_argument("--limit",        type=int,  default=None,
                        help="처리할 최대 행 수 (테스트용)")
    parser.add_argument("--force",        action="store_true",
                        help="이미 채워진 instruction도 재생성")
    parser.add_argument("--poll-interval",type=int,  default=60)
    args = parser.parse_args()

    # ── log ──
    if args.log:
        rows = _read_batch_log()
        if not rows:
            print("(배치 로그 없음)")
        else:
            for r in rows:
                print(r)
        return

    # ── status ──
    if args.status:
        info = check_batch_status(args.status)
        print(json.dumps(info, indent=2, ensure_ascii=False))
        return

    # ── retrieve ──
    if args.retrieve:
        results = retrieve_batch_results(args.retrieve)
        n = update_csvs(results, args.annot_dir, args.games)
        logger.info(f"총 {n}개 행 업데이트 완료")
        return

    # ── submit / run ──
    system_prompt = load_system_prompt()

    logger.info(f"캐시 로드: {args.cache_dir}")
    cache_by_game = load_cache_by_game(args.cache_dir)
    if not cache_by_game:
        logger.error("캐시 없음")
        return
    logger.info("캐시: " + ", ".join(f"{g}={len(v)}" for g, v in sorted(cache_by_game.items())))

    # threshold=None 행은 GPT 없이 "None" 으로 직접 채움
    fill_none_instructions(
        games=args.games, enums=args.enums,
        annot_dir=args.annot_dir, force=args.force,
    )

    # game × enum 조합별로 배치를 분리하여 제출
    _ENUM_NAMES = {
        0: "region", 1: "path_length", 2: "interactable_count",
        3: "hazard_count", 4: "collectable_count",
    }

    submitted_batches: List[Tuple[str, str]] = []  # (batch_id, game)

    for game in args.games:
        logger.info(f"\n── {game} ──")

        try:
            jsonl_path = build_jsonl(
                games=[game], enums=args.enums,
                annot_dir=args.annot_dir, cache_by_game=cache_by_game,
                system_prompt=system_prompt, force=args.force,
                limit=args.limit,
            )
            if jsonl_path is None:
                logger.info(f"  {game}: 생성할 요청 없음, 건너뜀")
                continue

            n_requests = sum(1 for _ in jsonl_path.open(encoding="utf-8"))
            batch_id = submit_batch(jsonl_path, [game], args.enums, n_requests)
            submitted_batches.append((batch_id, game))
        except Exception as e:
            logger.error(f"  {game}: 제출 실패 → {e}, 건너뜀")

    if not submitted_batches:
        logger.info("제출된 배치 없음")
        return

    if args.run:
        logger.info(f"\n완료 대기 중 (interval={args.poll_interval}s) …")
        pending = list(submitted_batches)
        total_updated = 0
        while pending:
            time.sleep(args.poll_interval)
            still_pending = []
            for batch_id, game in pending:
                info   = check_batch_status(batch_id)
                status = info["status"]
                counts = info["request_counts"]
                logger.info(f"  [{game}] {batch_id}: {status}  "
                            f"{counts['completed']}/{counts['total']} completed")
                if status == "completed":
                    try:
                        results = retrieve_batch_results(batch_id)
                        n = update_csvs(results, args.annot_dir, [game])
                        total_updated += n
                    except Exception as e:
                        logger.error(f"  [{game}] 결과 조회/업데이트 실패 → {e}, 건너뜀")
                elif status in ("failed", "expired", "cancelled"):
                    logger.error(f"  [{game}] 배치 실패/만료/취소: {status}")
                else:
                    still_pending.append((batch_id, game))
            pending = still_pending
        logger.info(f"총 {total_updated}개 행 업데이트 완료")
    else:
        logger.info("\n배치 제출 완료. 결과 조회:")
        for batch_id, game in submitted_batches:
            logger.info(
                f"  [{game}] "
                f"python dataset/reward_annotations/generate_instructions.py "
                f"--retrieve {batch_id} --games {game}"
            )


if __name__ == "__main__":
    main()
