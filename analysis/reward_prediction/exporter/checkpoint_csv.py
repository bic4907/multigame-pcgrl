"""
exporter/checkpoint_csv.py
==========================
MGPCGRL 인코더 체크포인트(들)를 순회하며 보상 예측 결과를 CSV로 저장한다.

- 체크포인트별 CSV : <output_dir>/<ckpt_name>.csv
- 통합 CSV        : <output_dir>/all_checkpoints.csv
- 요약 CSV        : <output_dir>/summary.csv
"""
from __future__ import annotations

import csv
import json
import math
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np

# 프로젝트 루트가 sys.path에 없으면 추가
_ROOT = Path(__file__).resolve().parents[4]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from conf.config import MGPCGRLConfig
from conf.game_utils import CANONICAL_GAMES, compute_seen_unseen_split
from instruct_rl.utils.dataset_loader import load_dataset_instruct
from instruct_rl.utils.log_utils import get_logger, suppress_jax_debug_logs

suppress_jax_debug_logs()
logger = get_logger(__file__)


# ─────────────────────────────────────────────────────────
# 설정 컨테이너
# ─────────────────────────────────────────────────────────

@dataclass
class ExportConfig:
    """pipeline.py 에서 export 단계를 구성하는 파라미터."""

    ckpt_dir: str = "/mnt/nas/mgpcgrl_encoder_unseen"
    output_dir: Path = Path("results/mgpcgrl_reward_pred_csv")
    dataset_game: str = "all"
    dataset_reward_enum: Any = "all"   # "all" | int | str
    reward_decoder_mode: str = "all"   # "all" | "unseen" | "noop"
    max_samples_per_game: int = 0
    max_checkpoints: int = 0
    fail_on_missing: bool = False

    # 실행 중 채워지는 결과 경로
    all_csv_path: Path = field(default=None, init=False)
    summary_csv_path: Path = field(default=None, init=False)


# ─────────────────────────────────────────────────────────
# 내부 유틸리티
# ─────────────────────────────────────────────────────────

def _to_number_or_str(value: str) -> Any:
    if value.lower() == "all":
        return "all"
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _scan_checkpoint_dir(ckpt_dir: str) -> List[str]:
    """ckpt_dir 내의 유효한 체크포인트 이름 목록 반환.

    유효 체크포인트 = <ckpt_dir>/<name>/ckpts/ 디렉토리가 존재하는 것.
    """
    root = Path(ckpt_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"ckpt_dir 를 찾을 수 없습니다: {root}")
    names = sorted(p.name for p in root.iterdir() if p.is_dir() and (p / "ckpts").is_dir())
    if not names:
        raise ValueError(f"유효한 체크포인트(ckpts/ 포함 서브디렉토리)가 없습니다: {root}")
    return names


def _canonical_game_name(game: str) -> str:
    return "doom" if game == "doom2" else game


def _build_seen_unseen_flag_columns(
    seen_games: Sequence[str],
    unseen_games: Sequence[str],
) -> Dict[str, int]:
    seen_set = set(seen_games)
    unseen_set = set(unseen_games)
    out: Dict[str, int] = {}
    for g in CANONICAL_GAMES:
        out[f"seen_{g}"] = int(g in seen_set)
        out[f"unseen_{g}"] = int(g in unseen_set)
    return out


def _apply_dataset_setting_from_encoder(config: MGPCGRLConfig) -> Tuple[List[str], List[str]]:
    """train_mgpcgrl.py 와 동일하게 dataset_setting.json 을 반영한다."""
    dataset_setting_path = (
        Path(config.encoder.ckpt_dir) / str(config.encoder.ckpt_name) / "dataset_setting.json"
    )
    if not dataset_setting_path.is_file():
        logger.warning("dataset_setting.json not found: %s", dataset_setting_path)
        return [], []

    with dataset_setting_path.open("r", encoding="utf-8") as f:
        dataset_setting = json.load(f)

    seen_ratio = float(dataset_setting.get("seen_ratio", config.dataset_seen_ratio))
    config.dataset_seen_ratio = seen_ratio
    seen_games = list(dataset_setting.get("seen_games", []) or [])
    config.reward_seen_games = seen_games
    seen_games_canonical, unseen_games_canonical = compute_seen_unseen_split(seen_games)
    logger.info(
        "dataset_setting loaded for %s: seen_ratio=%.4f, seen=%s, unseen=%s",
        config.encoder.ckpt_name,
        config.dataset_seen_ratio,
        seen_games_canonical,
        unseen_games_canonical,
    )
    return seen_games_canonical, unseen_games_canonical


def _build_config_for_ckpt(
    ckpt_dir: str,
    ckpt_name: str,
    cfg_export: ExportConfig,
) -> MGPCGRLConfig:
    cfg = MGPCGRLConfig()
    cfg.dataset_game = cfg_export.dataset_game
    cfg.dataset_reward_enum = cfg_export.dataset_reward_enum
    cfg.reward_decoder_mode = cfg_export.reward_decoder_mode
    cfg.max_samples_per_game = cfg_export.max_samples_per_game
    cfg.encoder.ckpt_dir = ckpt_dir
    cfg.encoder.ckpt_name = ckpt_name
    cfg.encoder.ckpt_path = str(Path(ckpt_dir) / ckpt_name / "ckpts")
    return cfg


def _get_actual_condition_vector(sample, num_classes: int) -> List[float]:
    conds = sample.meta.get("conditions", {}) or {}
    return [float(conds.get(i, conds.get(str(i), -1.0)) or -1.0) for i in range(num_classes)]


def _safe_float(v) -> float:
    if v is None:
        return math.nan
    try:
        return float(v)
    except Exception:
        return math.nan


# ─────────────────────────────────────────────────────────
# 행(row) 빌더
# ─────────────────────────────────────────────────────────

def build_rows_for_checkpoint(
    ckpt_name: str,
    instruct,
    samples: Sequence[Any],
    num_classes: int,
    seen_games: Sequence[str],
    unseen_games: Sequence[str],
) -> List[Dict[str, Any]]:
    pred_reward = np.asarray(instruct.reward_i, dtype=np.int32).reshape(-1)
    pred_condition = np.asarray(instruct.condition, dtype=np.float32)
    seen_games_str = ",".join(seen_games)
    unseen_games_str = ",".join(unseen_games)
    split_flags = _build_seen_unseen_flag_columns(seen_games, unseen_games)
    seen_set = set(seen_games)
    unseen_set = set(unseen_games)

    rows: List[Dict[str, Any]] = []
    for idx, sample in enumerate(samples):
        actual_reward = int(sample.meta["reward_enum"])
        actual_cond_vec = _get_actual_condition_vector(sample, num_classes)
        pred_reward_i = int(pred_reward[idx])
        pred_cond_vec = pred_condition[idx].tolist()
        sample_game_raw = str(getattr(sample, "game", ""))
        sample_game = _canonical_game_name(sample_game_raw)
        if sample_game in seen_set:
            game_split = "seen"
        elif sample_game in unseen_set:
            game_split = "unseen"
        else:
            game_split = "unknown"

        actual_cond_active = (
            _safe_float(actual_cond_vec[actual_reward])
            if 0 <= actual_reward < num_classes
            else math.nan
        )
        pred_cond_active = (
            _safe_float(pred_cond_vec[pred_reward_i])
            if 0 <= pred_reward_i < num_classes
            else math.nan
        )
        pred_cond_at_actual = (
            _safe_float(pred_cond_vec[actual_reward])
            if 0 <= actual_reward < num_classes
            else math.nan
        )

        row: Dict[str, Any] = {
            "encoder_name": ckpt_name,
            "checkpoint_name": ckpt_name,
            "seen_games": seen_games_str,
            "unseen_games": unseen_games_str,
            "sample_index": idx,
            "game": sample_game_raw,
            "game_canonical": sample_game,
            "game_seen_unseen": game_split,
            "source_id": str(getattr(sample, "source_id", "")),
            "instruction": str(getattr(sample, "instruction", "") or ""),
            "actual_reward_enum": actual_reward,
            "pred_reward_enum": pred_reward_i,
            "reward_enum_match": int(actual_reward == pred_reward_i),
            "actual_condition_active": actual_cond_active,
            "pred_condition_active": pred_cond_active,
            "pred_condition_at_actual_enum": pred_cond_at_actual,
        }
        row["diff_condition_at_actual_enum"] = (
            pred_cond_at_actual - actual_cond_active
            if not (math.isnan(actual_cond_active) or math.isnan(pred_cond_at_actual))
            else math.nan
        )
        for j in range(num_classes):
            a = _safe_float(actual_cond_vec[j])
            p = _safe_float(pred_cond_vec[j])
            row[f"actual_condition_{j}"] = a
            row[f"pred_condition_{j}"] = p
            row[f"diff_condition_{j}"] = (
                p - a if not (math.isnan(a) or math.isnan(p)) else math.nan
            )
        row.update(split_flags)
        rows.append(row)
    return rows


# ─────────────────────────────────────────────────────────
# 요약
# ─────────────────────────────────────────────────────────

def summarize_checkpoint(
    ckpt_name: str,
    rows: Sequence[Dict[str, Any]],
    seen_games: Sequence[str],
    unseen_games: Sequence[str],
) -> Dict[str, Any]:
    n = len(rows)
    seen_games_str = ",".join(seen_games)
    unseen_games_str = ",".join(unseen_games)
    split_flags = _build_seen_unseen_flag_columns(seen_games, unseen_games)

    if n == 0:
        out = {
            "encoder_name": ckpt_name,
            "checkpoint_name": ckpt_name,
            "seen_games": seen_games_str,
            "unseen_games": unseen_games_str,
            "num_samples": 0,
            "reward_enum_accuracy": math.nan,
            "condition_mae_when_enum_match": math.nan,
        }
        out.update(split_flags)
        return out

    matches = np.array([int(r["reward_enum_match"]) for r in rows], dtype=np.float32)
    acc = float(matches.mean())
    cond_errs = [
        abs(_safe_float(r["pred_condition_active"]) - _safe_float(r["actual_condition_active"]))
        for r in rows
        if int(r["reward_enum_match"]) == 1
        and not math.isnan(_safe_float(r["actual_condition_active"]))
        and not math.isnan(_safe_float(r["pred_condition_active"]))
    ]
    cond_mae = float(np.mean(cond_errs)) if cond_errs else math.nan

    out = {
        "encoder_name": ckpt_name,
        "checkpoint_name": ckpt_name,
        "seen_games": seen_games_str,
        "unseen_games": unseen_games_str,
        "num_samples": n,
        "reward_enum_accuracy": acc,
        "condition_mae_when_enum_match": cond_mae,
    }
    out.update(split_flags)
    return out


# ─────────────────────────────────────────────────────────
# CSV I/O
# ─────────────────────────────────────────────────────────

def write_csv(rows: Sequence[Dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        logger.warning("No rows to write: %s", out_path)
        return
    fieldnames = list(rows[0].keys())
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Saved CSV: %s (%d rows)", out_path, len(rows))


# ─────────────────────────────────────────────────────────
# 메인 export 실행 함수
# ─────────────────────────────────────────────────────────

def run_export(cfg: ExportConfig) -> ExportConfig:
    """
    ExportConfig 를 받아 전체 export 파이프라인을 실행하고
    all_csv_path / summary_csv_path 가 채워진 cfg를 반환한다.
    """
    _log_nas_mount_status()

    ckpt_dir = str(cfg.ckpt_dir)
    ckpt_names = _scan_checkpoint_dir(ckpt_dir)
    ckpt_names = _iter_checkpoints(ckpt_names, cfg.max_checkpoints)

    logger.info("Checkpoint dir   : %s", ckpt_dir)
    logger.info("Checkpoint count : %d", len(ckpt_names))
    logger.info("dataset_game     : %s", cfg.dataset_game)
    logger.info("dataset_reward   : %s", cfg.dataset_reward_enum)
    logger.info("decoder_mode     : %s", cfg.reward_decoder_mode)

    all_rows: List[Dict[str, Any]] = []
    summaries: List[Dict[str, Any]] = []

    for i, ckpt_name in enumerate(ckpt_names, start=1):
        ckpt_path = Path(ckpt_dir) / ckpt_name / "ckpts"
        if not ckpt_path.is_dir():
            msg = f"[{i}/{len(ckpt_names)}] missing checkpoint: {ckpt_path}"
            if cfg.fail_on_missing:
                raise FileNotFoundError(msg)
            logger.warning(msg + " (skip)")
            continue

        logger.info("[%d/%d] Processing checkpoint: %s", i, len(ckpt_names), ckpt_name)
        mgpcgrl_cfg = _build_config_for_ckpt(ckpt_dir, ckpt_name, cfg)
        seen_games, unseen_games = _apply_dataset_setting_from_encoder(mgpcgrl_cfg)

        instruct, _, samples = load_dataset_instruct(mgpcgrl_cfg)
        num_classes = int(mgpcgrl_cfg.decoder.num_reward_classes)
        rows = build_rows_for_checkpoint(
            ckpt_name=ckpt_name,
            instruct=instruct,
            samples=samples,
            num_classes=num_classes,
            seen_games=seen_games,
            unseen_games=unseen_games,
        )
        summaries.append(summarize_checkpoint(ckpt_name, rows, seen_games, unseen_games))

        ckpt_csv = cfg.output_dir / f"{ckpt_name}.csv"
        write_csv(rows, ckpt_csv)
        all_rows.extend(rows)

    all_csv = cfg.output_dir / "all_checkpoints.csv"
    summary_csv = cfg.output_dir / "summary.csv"

    if all_rows:
        write_csv(all_rows, all_csv)
    if summaries:
        write_csv(summaries, summary_csv)

    cfg.all_csv_path = all_csv
    cfg.summary_csv_path = summary_csv

    logger.info("Export done. output_dir=%s", cfg.output_dir.resolve())
    return cfg


# ─────────────────────────────────────────────────────────
# 내부 헬퍼
# ─────────────────────────────────────────────────────────

def _log_nas_mount_status(nas_path: Path = Path("/mnt/nas")) -> None:
    exists = nas_path.exists()
    is_dir = nas_path.is_dir()
    logger.info("NAS check: path=%s exists=%s is_dir=%s", nas_path, exists, is_dir)
    if not exists or not is_dir:
        return
    try:
        entries = sorted(os.listdir(nas_path))
        logger.info("NAS entries: count=%d sample=%s", len(entries), entries[:10])
    except Exception as e:
        logger.warning("NAS list failed: path=%s err=%s", nas_path, e)


def _iter_checkpoints(ckpt_names: Iterable[str], max_checkpoints: int) -> List[str]:
    names = list(ckpt_names)
    return names[:max_checkpoints] if max_checkpoints > 0 else names

