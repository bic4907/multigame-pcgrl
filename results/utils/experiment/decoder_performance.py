"""
Download decoder_prediction_csv artifacts and plot decoder training performance.

Inputs:
  - W&B project runs from results/config.json experiment target_projects
  - logged artifact named decoder_prediction_csv
  - W&B history metric train(direction)/continuous_delta_loss

Outputs:
  - decoder_prediction_rows.csv
  - decoder_epoch_metrics.csv
  - continuous_delta_loss_history.csv
  - continuous_delta_loss.png/.pdf
  - prediction_loss.png/.pdf
  - prediction_loss_continuous_delta_loss.png/.pdf
  - prediction_loss_by_unseen_game.png/.pdf
  - by_game/<game>/prediction_loss.png/.pdf
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

_HERE = Path(__file__).resolve().parent
_RESULTS_DIR = _HERE.parent.parent
_ROOT = _RESULTS_DIR.parent
if str(_RESULTS_DIR) not in sys.path:
    sys.path.insert(0, str(_RESULTS_DIR))
if str(_ROOT) not in sys.path:
    sys.path.append(str(_ROOT))

from sweep.wandb_utils.config import DEFAULT_ENTITY, DEFAULT_NUM_WORKERS
from sweep.wandb_utils.downloader import get_api
from utils.core.run_output import load_cfg, setup_logger

_CFG = load_cfg()
_DEFAULT_LOSS_KEY = "train(direction)/continuous_delta_loss"


def _experiment_names() -> list[str]:
    return list(_CFG.get("experiments", {}).keys())


def _safe_name(value: str) -> str:
    return "".join(c if c.isalnum() or c in "._=-" else "_" for c in value)


def _as_float(value: object) -> float | None:
    try:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_bool(value: object) -> bool | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y", "on"}:
        return True
    if text in {"false", "0", "no", "n", "off"}:
        return False
    return None


def _run_config_value(config: dict, *keys: str) -> object | None:
    for key in keys:
        if key in config:
            return config.get(key)
        current: object = config
        found = True
        for part in key.split("."):
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                found = False
                break
        if found:
            return current
    return None


def _safe_pearson(a: pd.Series, b: pd.Series) -> float:
    aa = pd.to_numeric(a, errors="coerce")
    bb = pd.to_numeric(b, errors="coerce")
    mask = aa.notna() & bb.notna()
    if mask.sum() < 2:
        return float("nan")
    aa = aa[mask]
    bb = bb[mask]
    if float(aa.std(ddof=0)) <= 1e-12 or float(bb.std(ddof=0)) <= 1e-12:
        return float("nan")
    return float(aa.corr(bb))


def _canonical_game_set(value: object) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "unknown"
    if isinstance(value, (list, tuple, set)):
        vals = [str(v).strip() for v in value if str(v).strip()]
        return "+".join(sorted(vals)) if vals else "unknown"
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return "unknown"
    try:
        parsed = json.loads(text)
    except Exception:
        parsed = None
    if isinstance(parsed, (list, tuple, set)):
        vals = [str(v).strip() for v in parsed if str(v).strip()]
        return "+".join(sorted(vals)) if vals else "unknown"
    return text


def _resolve_projects(experiment: str | None, projects: list[str] | None) -> list[str]:
    if projects:
        return projects
    if experiment:
        exp_cfg = _CFG.get("experiments", {}).get(experiment, {})
        configured = list(exp_cfg.get("target_projects", []))
        if not configured:
            raise SystemExit(f"Experiment '{experiment}' has no target_projects in results/config.json.")
        return configured
    return ["aaai27_encoder_mgpcgrl__decoderperformance"]


def _experiment_cfg(experiment: str | None) -> dict:
    if not experiment:
        return {}
    return _CFG.get("experiments", {}).get(experiment, {})


def _make_decoder_run_dir(cfg: dict) -> Path:
    """Use a single decoder_performance folder inside pipeline outputs."""
    pipeline_run_dir = os.environ.get("PIPELINE_RUN_DIR")
    pipeline_experiment = os.environ.get("PIPELINE_EXPERIMENT", "")
    if pipeline_run_dir:
        run_dir = Path(pipeline_run_dir) / pipeline_experiment
        run_dir.mkdir(parents=True, exist_ok=True)
        with (run_dir / "used_config.json").open("w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)
        return run_dir

    from utils.core.run_output import make_run_dir

    return make_run_dir("decoder_performance", cfg=cfg)


def _default_cache_dir() -> Path:
    return _RESULTS_DIR / "wandb_projects" / "decoder_performance_cache"


_PRETENDARD_CANDIDATES: tuple[Path, ...] = (
    Path(os.environ.get("PRETENDARD_MEDIUM_PATH", "")),
    Path(os.environ.get("PRETENDARD_REGULAR_PATH", "")),
    Path("/Users/inchang/Desktop/MuCap-fin2/mucap/fonts/Pretendard-Medium.otf"),
    Path.home() / "Library/Fonts/Pretendard-Medium.otf",
    Path.home() / "Library/Fonts/Pretendard-Regular.otf",
    Path("/Library/Fonts/Pretendard-Medium.otf"),
    Path("/Library/Fonts/Pretendard-Regular.otf"),
)


def _apply_plot_style(mpl) -> None:
    font_family = "Pretendard"
    from matplotlib import font_manager

    for font_path in _PRETENDARD_CANDIDATES:
        if not str(font_path) or not font_path.is_file():
            continue
        font_manager.fontManager.addfont(str(font_path))
        font_family = font_manager.FontProperties(fname=str(font_path)).get_name()
        break
    mpl.rcParams.update(
        {
            "font.family": font_family,
            "font.sans-serif": [font_family, "Pretendard", "Arial", "Helvetica", "DejaVu Sans"],
            "font.weight": "regular",
            "axes.labelweight": "regular",
            "axes.titleweight": "regular",
            "mathtext.fontset": "custom",
            "mathtext.rm": "Times New Roman",
            "mathtext.it": "Times New Roman:italic",
            "mathtext.bf": "Times New Roman:bold",
        }
    )


def _run_cache_dir(cache_dir: Path, project: str, run) -> Path:
    return cache_dir / project / _safe_name(run.id)


def _is_finished_run(run) -> bool:
    return str(getattr(run, "state", "")).lower() == "finished"


def _select_decoder_artifact(run) -> object | None:
    artifacts = list(run.logged_artifacts())
    candidates = []
    for artifact in artifacts:
        base_name = artifact.name.split(":", 1)[0]
        if base_name == "decoder_prediction_csv" and artifact.type == "dataset":
            candidates.append(artifact)
    return candidates[-1] if candidates else None


def _download_prediction_artifact(
    run,
    project: str,
    cache_dir: Path,
    force: bool,
) -> list[Path]:
    run_dir = _run_cache_dir(cache_dir, project, run)
    artifact_dir = run_dir / "decoder_prediction_csv"
    if _is_finished_run(run) and artifact_dir.exists() and not force:
        cached_csvs = sorted(artifact_dir.rglob("*.csv"))
        if cached_csvs:
            return cached_csvs

    artifact = _select_decoder_artifact(run)
    if artifact is None:
        return []

    if force and artifact_dir.exists():
        shutil.rmtree(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    if not any(artifact_dir.glob("*.csv")):
        artifact.download(root=str(artifact_dir))
    return sorted(artifact_dir.rglob("*.csv"))


def _read_prediction_csvs(
    csv_paths: list[Path],
    *,
    project: str,
    run,
    delta_weight: float | None,
    decoder_nograd: bool | None,
) -> pd.DataFrame:
    frames = []
    for path in csv_paths:
        try:
            df = pd.read_csv(path, low_memory=False)
        except Exception:
            continue
        if df.empty:
            continue
        df["project"] = project
        df["run_id"] = run.id
        df["run_name"] = run.name
        df["source_csv"] = path.name
        if "delta_weight" not in df.columns or df["delta_weight"].isna().all():
            df["delta_weight"] = delta_weight
        df["decoder_nograd"] = decoder_nograd
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _prepare_prediction_frame(path: Path, usecols: set[str]) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, usecols=lambda c: c in usecols, low_memory=False)
    except Exception:
        return pd.DataFrame()
    if df.empty:
        return df

    if "epoch_num" in df.columns:
        df["epoch"] = pd.to_numeric(df["epoch_num"], errors="coerce")
    elif "epoch" in df.columns:
        df["epoch"] = pd.to_numeric(df["epoch"], errors="coerce")
    else:
        return pd.DataFrame()
    df = df[df["epoch"] > 0]
    if df.empty:
        return df

    if "reward_enum_target" in df.columns:
        df["reward_enum"] = pd.to_numeric(df["reward_enum_target"], errors="coerce")
    else:
        df["reward_enum"] = -1
    if "eval_unseen_games" in df.columns:
        df["unseen_game"] = df["eval_unseen_games"].map(_canonical_game_set)
    elif "train_unseen_games" in df.columns:
        df["unseen_game"] = df["train_unseen_games"].map(_canonical_game_set)
    else:
        df["unseen_game"] = "unknown"

    for col in [
        "condition_target_raw",
        "condition_pred_raw",
        "condition_target_norm",
        "condition_pred_norm",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _prediction_norm_scales_from_csvs(
    csv_paths_by_run: list[list[Path]],
    norm_group_cols: list[str],
) -> pd.DataFrame:
    if not csv_paths_by_run or not norm_group_cols:
        return pd.DataFrame()
    usecols = {
        "epoch",
        "epoch_num",
        "reward_enum_target",
        "eval_unseen_games",
        "train_unseen_games",
        "condition_target_raw",
    }
    frames = []
    for csv_paths in tqdm(csv_paths_by_run, desc="norm scale", unit="run"):
        for path in csv_paths:
            df = _prepare_prediction_frame(path, usecols)
            if df.empty or "condition_target_raw" not in df.columns:
                continue
            valid_group_cols = [c for c in norm_group_cols if c in df.columns]
            if not valid_group_cols:
                continue
            scale = (
                df.dropna(subset=valid_group_cols + ["condition_target_raw"])
                .groupby(valid_group_cols, dropna=False)["condition_target_raw"]
                .agg(target_min="min", target_max="max")
                .reset_index()
            )
            if not scale.empty:
                frames.append(scale)
    if not frames:
        return pd.DataFrame()
    merged = pd.concat(frames, ignore_index=True)
    valid_group_cols = [c for c in norm_group_cols if c in merged.columns]
    if not valid_group_cols:
        return pd.DataFrame()
    return (
        merged.groupby(valid_group_cols, dropna=False)
        .agg(target_min=("target_min", "min"), target_max=("target_max", "max"))
        .reset_index()
    )


def _prediction_epoch_metrics_from_csvs(
    csv_paths: list[Path],
    *,
    project: str,
    run,
    delta_weight: float | None,
    decoder_nograd: bool | None,
    norm_group_cols: list[str] | None = None,
    norm_scales: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, int]:
    frames = []
    n_rows = 0
    usecols = {
        "epoch",
        "epoch_num",
        "reward_enum_target",
        "eval_unseen_games",
        "train_unseen_games",
        "condition_target_norm",
        "condition_pred_norm",
        "condition_target_raw",
        "condition_pred_raw",
    }
    for path in csv_paths:
        df = _prepare_prediction_frame(path, usecols)
        if df.empty:
            continue
        n_rows += len(df)

        required_raw = {"epoch", "reward_enum", "condition_target_raw", "condition_pred_raw"}
        if not required_raw.issubset(df.columns):
            continue
        df = df.dropna(subset=list(required_raw))
        if df.empty:
            continue

        df["regression_abs_error_raw"] = (
            df["condition_target_raw"] - df["condition_pred_raw"]
        ).abs()
        df["regression_sq_error_raw"] = (
            df["condition_target_raw"] - df["condition_pred_raw"]
        ).pow(2)

        norm_group_cols = norm_group_cols or []
        valid_norm_group_cols = [c for c in norm_group_cols if c in df.columns]
        if (
            valid_norm_group_cols
            and norm_scales is not None
            and not norm_scales.empty
            and set(valid_norm_group_cols + ["target_min", "target_max"]).issubset(norm_scales.columns)
        ):
            df = df.merge(
                norm_scales[valid_norm_group_cols + ["target_min", "target_max"]],
                on=valid_norm_group_cols,
                how="left",
            )
            missing_scale = df["target_min"].isna() | df["target_max"].isna()
            if missing_scale.any():
                fallback_min = df.groupby(valid_norm_group_cols)["condition_target_raw"].transform("min")
                fallback_max = df.groupby(valid_norm_group_cols)["condition_target_raw"].transform("max")
                df.loc[missing_scale, "target_min"] = fallback_min[missing_scale]
                df.loc[missing_scale, "target_max"] = fallback_max[missing_scale]
            ranges = (df["target_max"] - df["target_min"]).clip(lower=1e-12)
            df["condition_target_norm_local"] = (
                df["condition_target_raw"] - df["target_min"]
            ) / ranges
            df["condition_pred_norm_local"] = (
                df["condition_pred_raw"] - df["target_min"]
            ) / ranges
            df["regression_abs_error_norm"] = (
                df["condition_target_norm_local"] - df["condition_pred_norm_local"]
            ).abs()
            df["regression_sq_error_norm"] = (
                df["condition_target_norm_local"] - df["condition_pred_norm_local"]
            ).pow(2)
        elif valid_norm_group_cols:
            target_min = df.groupby(valid_norm_group_cols)["condition_target_raw"].transform("min")
            target_max = df.groupby(valid_norm_group_cols)["condition_target_raw"].transform("max")
            ranges = (target_max - target_min).clip(lower=1e-12)
            df["condition_target_norm_local"] = (
                df["condition_target_raw"] - target_min
            ) / ranges
            df["condition_pred_norm_local"] = (
                df["condition_pred_raw"] - target_min
            ) / ranges
            df["regression_abs_error_norm"] = (
                df["condition_target_norm_local"] - df["condition_pred_norm_local"]
            ).abs()
            df["regression_sq_error_norm"] = (
                df["condition_target_norm_local"] - df["condition_pred_norm_local"]
            ).pow(2)
        elif {"condition_target_norm", "condition_pred_norm"}.issubset(df.columns):
            df["regression_abs_error_norm"] = (
                df["condition_target_norm"] - df["condition_pred_norm"]
            ).abs()
            df["regression_sq_error_norm"] = (
                df["condition_target_norm"] - df["condition_pred_norm"]
            ).pow(2)
        else:
            ranges = (
                df.groupby("reward_enum")["condition_target_raw"]
                .transform(lambda s: max(float(s.max() - s.min()), 1e-12))
            )
            df["regression_abs_error_norm"] = df["regression_abs_error_raw"] / ranges
            df["regression_sq_error_norm"] = df["regression_abs_error_norm"].pow(2)

        grouped_rows = []
        for (unseen_game, reward_enum, epoch), group in df.groupby(
            ["unseen_game", "reward_enum", "epoch"],
            dropna=False,
            sort=True,
        ):
            pearson_raw = _safe_pearson(
                group["condition_target_raw"],
                group["condition_pred_raw"],
            )
            if {"condition_target_norm_local", "condition_pred_norm_local"}.issubset(group.columns):
                pearson_norm = _safe_pearson(
                    group["condition_target_norm_local"],
                    group["condition_pred_norm_local"],
                )
            elif {"condition_target_norm", "condition_pred_norm"}.issubset(group.columns):
                pearson_norm = _safe_pearson(
                    group["condition_target_norm"],
                    group["condition_pred_norm"],
                )
            else:
                pearson_norm = pearson_raw
            grouped_rows.append({
                "unseen_game": unseen_game,
                "reward_enum": reward_enum,
                "epoch": epoch,
                "regression_mae_raw": group["regression_abs_error_raw"].mean(),
                "regression_mae_norm": group["regression_abs_error_norm"].mean(),
                "regression_rmse_raw": group["regression_sq_error_raw"].mean() ** 0.5,
                "regression_rmse_norm": group["regression_sq_error_norm"].mean() ** 0.5,
                "regression_pearson_raw": pearson_raw,
                "regression_pearson_norm": pearson_norm,
                "n_predictions": len(group),
            })
        grouped = pd.DataFrame(grouped_rows)
        grouped["project"] = project
        grouped["run_id"] = run.id
        grouped["run_name"] = run.name
        grouped["delta_weight"] = delta_weight
        grouped["decoder_nograd"] = decoder_nograd
        frames.append(grouped)

    if not frames:
        return pd.DataFrame(), n_rows
    result = pd.concat(frames, ignore_index=True)
    cols = [
        "project",
        "run_id",
        "run_name",
        "delta_weight",
        "decoder_nograd",
        "unseen_game",
        "reward_enum",
        "epoch",
        "regression_mae_raw",
        "regression_mae_norm",
        "regression_rmse_raw",
        "regression_rmse_norm",
        "regression_pearson_raw",
        "regression_pearson_norm",
        "n_predictions",
    ]
    return result[cols], n_rows


def _history_rows(
    project: str,
    run,
    delta_weight: float | None,
    decoder_nograd: bool | None,
    loss_key: str,
    epoch_offset: float,
    cache_dir: Path,
    force: bool,
) -> pd.DataFrame:
    history_cache = (
        _run_cache_dir(cache_dir, project, run)
        / f"{_safe_name(loss_key)}_epochoffset-{epoch_offset:g}_history.csv"
    )
    if _is_finished_run(run) and history_cache.is_file() and not force:
        try:
            cached = pd.read_csv(history_cache)
        except Exception:
            cached = pd.DataFrame()
        if not cached.empty:
            cached["decoder_nograd"] = decoder_nograd
            return cached

    try:
        rows = list(run.scan_history(keys=[loss_key, "total/epoch"]))
    except Exception:
        return pd.DataFrame()
    if not rows:
        return pd.DataFrame()
    hist = pd.DataFrame(rows)
    if hist.empty or loss_key not in hist.columns:
        return pd.DataFrame()

    keep = [c for c in ["_step", "total/epoch", loss_key] if c in hist.columns]
    hist = hist[keep].copy()
    hist = hist.dropna(subset=[loss_key])
    if hist.empty:
        return hist

    if "total/epoch" in hist.columns:
        hist["epoch"] = pd.to_numeric(hist["total/epoch"], errors="coerce") + epoch_offset
    else:
        hist["epoch"] = pd.to_numeric(hist.get("_step"), errors="coerce")
    hist["project"] = project
    hist["run_id"] = run.id
    hist["run_name"] = run.name
    hist["delta_weight"] = delta_weight
    hist["decoder_nograd"] = decoder_nograd
    hist = hist.rename(columns={loss_key: "history_loss"})
    hist = hist[
        ["project", "run_id", "run_name", "delta_weight", "decoder_nograd", "epoch", "history_loss"]
    ].dropna(subset=["epoch"])
    if _is_finished_run(run) and not hist.empty:
        history_cache.parent.mkdir(parents=True, exist_ok=True)
        hist.to_csv(history_cache, index=False)
    return hist


def _prediction_epoch_metrics(pred_rows: pd.DataFrame) -> pd.DataFrame:
    if pred_rows.empty:
        return pd.DataFrame()

    df = pred_rows.copy()
    if "epoch_num" in df.columns:
        df["epoch"] = pd.to_numeric(df["epoch_num"], errors="coerce")
    elif "epoch" in df.columns:
        df["epoch"] = pd.to_numeric(df["epoch"], errors="coerce") + 1
    else:
        return pd.DataFrame()

    required = {
        "project",
        "run_id",
        "run_name",
        "delta_weight",
        "epoch",
        "condition_target_raw",
        "condition_pred_raw",
    }
    missing = required.difference(df.columns)
    if missing:
        return pd.DataFrame()

    for col in [
        "delta_weight",
        "condition_target_raw",
        "condition_pred_raw",
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(
        subset=[
            "delta_weight",
            "epoch",
            "condition_target_raw",
            "condition_pred_raw",
        ]
    )
    if df.empty:
        return pd.DataFrame()

    df["regression_abs_error_raw"] = (
        df["condition_target_raw"] - df["condition_pred_raw"]
    ).abs()
    df["regression_sq_error_raw"] = (
        df["condition_target_raw"] - df["condition_pred_raw"]
    ).pow(2)
    if {"condition_target_norm", "condition_pred_norm"}.issubset(df.columns):
        df["condition_target_norm"] = pd.to_numeric(df["condition_target_norm"], errors="coerce")
        df["condition_pred_norm"] = pd.to_numeric(df["condition_pred_norm"], errors="coerce")
        df["regression_abs_error_norm"] = (
            df["condition_target_norm"] - df["condition_pred_norm"]
        ).abs()
        df["regression_sq_error_norm"] = (
            df["condition_target_norm"] - df["condition_pred_norm"]
        ).pow(2)
    else:
        df["regression_abs_error_norm"] = float("nan")
        df["regression_sq_error_norm"] = float("nan")

    group_cols = ["project", "run_id", "run_name", "delta_weight", "epoch"]
    if "decoder_nograd" in df.columns:
        group_cols.insert(4, "decoder_nograd")
    grouped = df.groupby(group_cols, dropna=False).agg(
        regression_mae_raw=("regression_abs_error_raw", "mean"),
        regression_mae_norm=("regression_abs_error_norm", "mean"),
        regression_mse_raw=("regression_sq_error_raw", "mean"),
        regression_mse_norm=("regression_sq_error_norm", "mean"),
        n_predictions=("regression_abs_error_raw", "size"),
    )
    grouped = grouped.reset_index()
    grouped["regression_rmse_raw"] = grouped["regression_mse_raw"].pow(0.5)
    grouped["regression_rmse_norm"] = grouped["regression_mse_norm"].pow(0.5)
    return grouped.drop(columns=["regression_mse_raw", "regression_mse_norm"])


def _epoch_bin(values: pd.Series, bin_size: int) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    if bin_size <= 1:
        return numeric.round().astype("Int64")
    return (((numeric - 1) // bin_size) * bin_size + 1).round().astype("Int64")


def _auto_bin_size(epoch_metrics: pd.DataFrame, loss_history: pd.DataFrame, max_points: int) -> int:
    epochs = []
    for df in (epoch_metrics, loss_history):
        if not df.empty and "epoch" in df.columns:
            epochs.extend(pd.to_numeric(df["epoch"], errors="coerce").dropna().tolist())
    if not epochs or max_points <= 0:
        return 1
    n_unique = len(set(int(round(e)) for e in epochs))
    return max(1, int(math.ceil(n_unique / max_points)))


def _aggregate_for_plot(df: pd.DataFrame, value_col: str, epoch_bin_size: int) -> pd.DataFrame:
    if df.empty or value_col not in df.columns:
        return pd.DataFrame()
    work = _with_method_columns(df)
    work["epoch_bin"] = _epoch_bin(work["epoch"], epoch_bin_size)
    work[value_col] = pd.to_numeric(work[value_col], errors="coerce")
    work = work.dropna(subset=["method_key", "epoch_bin", value_col])
    if work.empty:
        return pd.DataFrame()
    method_cols = ["method_key", "method_label", "method_order"]
    if "run_id" in work.columns:
        work = (
            work.groupby(method_cols + ["epoch_bin", "run_id"], dropna=False)[value_col]
            .mean()
            .reset_index()
        )
    grouped = (
        work.groupby(method_cols + ["epoch_bin"], dropna=False)[value_col]
        .agg(["mean", "std", "count"])
        .reset_index()
        .sort_values(["method_order", "epoch_bin"])
    )
    grouped["sem"] = grouped["std"].fillna(0.0) / grouped["count"].pow(0.5)
    grouped = grouped.rename(columns={"epoch_bin": "epoch", "mean": value_col})
    return grouped


def _aggregate_by_game_for_plot(
    df: pd.DataFrame,
    value_col: str,
    epoch_bin_size: int,
) -> pd.DataFrame:
    if df.empty or value_col not in df.columns or "unseen_game" not in df.columns:
        return pd.DataFrame()
    work = _with_method_columns(df)
    work["epoch_bin"] = _epoch_bin(work["epoch"], epoch_bin_size)
    work[value_col] = pd.to_numeric(work[value_col], errors="coerce")
    work = work.dropna(subset=["method_key", "epoch_bin", "unseen_game", value_col])
    work = work[work["unseen_game"].astype(str) != "unknown"]
    if work.empty:
        return pd.DataFrame()
    method_cols = ["method_key", "method_label", "method_order"]
    if "run_id" in work.columns:
        work = (
            work.groupby(
                ["unseen_game"] + method_cols + ["epoch_bin", "run_id"],
                dropna=False,
            )[value_col]
            .mean()
            .reset_index()
        )
    grouped = (
        work.groupby(["unseen_game"] + method_cols + ["epoch_bin"], dropna=False)[value_col]
        .agg(["mean", "std", "count"])
        .reset_index()
        .sort_values(["unseen_game", "method_order", "epoch_bin"])
        .rename(columns={"epoch_bin": "epoch"})
    )
    grouped["sem"] = grouped["std"].fillna(0.0) / grouped["count"].pow(0.5)
    grouped = grouped.rename(columns={"mean": value_col})
    return grouped


def _filter_excluded_delta_weights(
    df: pd.DataFrame,
    excluded_delta_weights: list[float],
) -> pd.DataFrame:
    if df.empty or "delta_weight" not in df.columns or not excluded_delta_weights:
        return df
    work = df.copy()
    weights = pd.to_numeric(work["delta_weight"], errors="coerce")
    keep = pd.Series(True, index=work.index)
    for excluded in excluded_delta_weights:
        keep &= ~weights.map(lambda value: pd.notna(value) and math.isclose(float(value), excluded, abs_tol=1e-9))
    return work[keep].reset_index(drop=True)


def _method_key(delta_weight: object, decoder_nograd: object = None) -> str:
    weight = _as_float(delta_weight)
    nograd = _as_bool(decoder_nograd)
    if nograd is True:
        return "detach"
    if weight is not None and math.isclose(weight, 0.0, abs_tol=1e-9):
        return "mgpcgrl"
    if weight is not None and math.isclose(weight, 0.03, abs_tol=1e-9):
        return "mgpcgrl_da"
    return f"delta_{weight:g}" if weight is not None else "unknown"


def _method_label(method_key: str) -> str:
    labels = {
        "mgpcgrl": r"ReWARD ($-\mathcal{L}_{\mathrm{dir}}$)",
        "mgpcgrl_da": "ReWARD",
        "detach": "Detach",
        "unknown": "Unknown",
    }
    if method_key.startswith("delta_"):
        return rf"$\lambda_{{\mathrm{{dir}}}}={method_key.removeprefix('delta_')}$"
    return labels.get(method_key, method_key)


def _method_sort_order(method_key: str) -> int:
    order = {
        "mgpcgrl": 0,
        "mgpcgrl_da": 1,
        "detach": 2,
        "unknown": 99,
    }
    return order.get(method_key, 50)


def _with_method_columns(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    if "delta_weight" not in work.columns:
        work["delta_weight"] = float("nan")
    if "decoder_nograd" not in work.columns:
        work["decoder_nograd"] = None
    work["delta_weight"] = pd.to_numeric(work["delta_weight"], errors="coerce")
    work["method_key"] = [
        _method_key(delta_weight, decoder_nograd)
        for delta_weight, decoder_nograd in zip(
            work["delta_weight"],
            work["decoder_nograd"],
            strict=False,
        )
    ]
    work["method_label"] = work["method_key"].map(_method_label)
    work["method_order"] = work["method_key"].map(_method_sort_order)
    return work


def _display_game_name(game: str) -> str:
    return str(game).replace("_", " ").title()


def _method_color(method_key: str) -> str:
    colors = {
        "mgpcgrl": "#0F9D58",
        "mgpcgrl_da": "#4285F4",
        "detach": "#DB4437",
    }
    return colors.get(method_key, "#777777")


def _method_marker(method_key: str) -> str:
    markers = {
        "mgpcgrl": "o",
        "mgpcgrl_da": "s",
        "detach": "^",
    }
    return markers.get(method_key, "D")


def _method_line_style(method_key: str) -> str:
    return "-"


def _method_marker_size(method_key: str) -> float:
    return 4.5


def _plot_by_unseen_game(
    epoch_metrics: pd.DataFrame,
    output_path: Path,
    epoch_bin_size: int,
    max_points_per_line: int,
    prediction_metric: str,
    prediction_metric_label: str,
    ylim_exclude_epoch_max: float | None,
) -> None:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FuncFormatter, MultipleLocator

    if epoch_bin_size <= 0:
        epoch_bin_size = _auto_bin_size(epoch_metrics, pd.DataFrame(), max_points_per_line)

    plt.style.use("seaborn-v0_8-whitegrid")
    _apply_plot_style(mpl)

    plot_df = _aggregate_by_game_for_plot(epoch_metrics, prediction_metric, epoch_bin_size)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(3.0, 2.5))

    def _epoch_formatter(value: float, _pos: int) -> str:
        if abs(value) >= 1000:
            scaled = value / 1000.0
            if abs(scaled - round(scaled)) < 1e-6:
                return f"{int(round(scaled))}e3"
            return f"{scaled:g}e3"
        if abs(value - round(value)) < 1e-6:
            return str(int(round(value)))
        return f"{value:g}"

    def _marker_indices(n_points: int) -> list[int]:
        if n_points <= 0:
            return []
        step = max(1, n_points // 10)
        indices = list(range(0, n_points, step))
        if indices[-1] != n_points - 1:
            indices.append(n_points - 1)
        return indices

    if plot_df.empty:
        ax.text(0.5, 0.5, f"No data: {prediction_metric}", ha="center", va="center")
        ax.set_axis_off()
    else:
        game_colors = _CFG.get("games", {}).get("colors", {})
        fallback_colors = mpl.colormaps.get_cmap("tab10")
        games = sorted(str(g) for g in plot_df["unseen_game"].dropna().unique())
        methods = (
            plot_df[["method_key", "method_label", "method_order"]]
            .drop_duplicates()
            .sort_values("method_order")
            .to_dict("records")
        )
        game_color_map = {
            game: game_colors.get(game, fallback_colors(idx % 10))
            for idx, game in enumerate(games)
        }

        for game in games:
            for method in methods:
                method_key = str(method["method_key"])
                line_df = plot_df[
                    (plot_df["unseen_game"].astype(str) == game)
                    & (plot_df["method_key"].astype(str) == method_key)
                ].sort_values("epoch")
                if line_df.empty:
                    continue
                x = line_df["epoch"].astype(float).to_numpy()
                y = line_df[prediction_metric].astype(float).to_numpy()
                sem = line_df["sem"].fillna(0.0).astype(float).to_numpy()
                marker_idx = _marker_indices(len(x))
                ax.plot(
                    x,
                    y,
                    color=game_color_map[game],
                    linewidth=1.8,
                    linestyle=_method_line_style(method_key),
                    marker=_method_marker(method_key),
                    markevery=marker_idx,
                    markersize=_method_marker_size(method_key),
                    alpha=0.95,
                )
                ax.fill_between(
                    x,
                    y - sem,
                    y + sem,
                    color=game_color_map[game],
                    alpha=0.08,
                    linewidth=0,
                )

        ax.set_ylabel(prediction_metric_label)
        ax.xaxis.set_major_locator(MultipleLocator(1000))
        ax.xaxis.set_major_formatter(FuncFormatter(_epoch_formatter))

        game_handles = [
            Line2D(
                [0],
                [0],
                color=game_color_map[game],
                linewidth=2.0,
                label=_display_game_name(game),
            )
            for game in games
        ]
        method_handles = [
            Line2D(
                [0],
                [0],
                color="#333333",
                linewidth=2.0,
                linestyle=_method_line_style(str(method["method_key"])),
                marker=_method_marker(str(method["method_key"])),
                markersize=_method_marker_size(str(method["method_key"])),
                label=str(method["method_label"]),
            )
            for method in methods
        ]
        first_legend = ax.legend(
            handles=game_handles,
            title="Unseen game",
            loc="upper right",
            bbox_to_anchor=(1.0, 1.24),
            frameon=False,
            fontsize=8,
            title_fontsize=9,
        )
        ax.add_artist(first_legend)
        ax.legend(
            handles=method_handles,
            title="Method",
            loc="upper left",
            bbox_to_anchor=(0.0, 1.24),
            frameon=False,
            fontsize=8,
            title_fontsize=9,
        )

        if ylim_exclude_epoch_max is not None:
            ylim_df = plot_df[
                pd.to_numeric(plot_df["epoch"], errors="coerce") > ylim_exclude_epoch_max
            ].copy()
            if not ylim_df.empty:
                y = pd.to_numeric(ylim_df[prediction_metric], errors="coerce")
                sem = pd.to_numeric(ylim_df.get("sem", 0.0), errors="coerce").fillna(0.0)
                lo = (y - sem).min()
                hi = (y + sem).max()
                if pd.notna(lo) and pd.notna(hi) and hi > lo:
                    pad = (hi - lo) * 0.08
                    ax.set_ylim(max(0.0, lo - pad), hi + pad)

    ax.set_xlabel("Epoch")
    fig.tight_layout(pad=0.15)
    fig.savefig(output_path.with_suffix(".png"), dpi=240, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _last_epoch_bar_data(epoch_metrics: pd.DataFrame, value_col: str) -> pd.DataFrame:
    if (
        epoch_metrics.empty
        or value_col not in epoch_metrics.columns
        or "unseen_game" not in epoch_metrics.columns
    ):
        return pd.DataFrame()
    work = _with_method_columns(epoch_metrics)
    work["epoch"] = pd.to_numeric(work["epoch"], errors="coerce")
    work[value_col] = pd.to_numeric(work[value_col], errors="coerce")
    work = work.dropna(subset=["method_key", "epoch", "unseen_game", value_col])
    work = work[work["unseen_game"].astype(str) != "unknown"]
    if work.empty:
        return pd.DataFrame()

    method_cols = ["method_key", "method_label", "method_order"]
    seed_cols = ["unseen_game"] + method_cols
    if "run_id" in work.columns:
        seed_cols.append("run_id")
    max_epoch = work.groupby(seed_cols, dropna=False)["epoch"].transform("max")
    work = work[work["epoch"] == max_epoch]
    if work.empty:
        return pd.DataFrame()

    if "run_id" in work.columns:
        seed_values = (
            work.groupby(["unseen_game"] + method_cols + ["run_id"], dropna=False)[value_col]
            .mean()
            .reset_index()
        )
    else:
        seed_values = work[["unseen_game"] + method_cols + [value_col]].copy()

    grouped = (
        seed_values.groupby(["unseen_game"] + method_cols, dropna=False)[value_col]
        .agg(["mean", "std", "count"])
        .reset_index()
        .sort_values(["unseen_game", "method_order"])
    )
    grouped["sem"] = grouped["std"].fillna(0.0) / grouped["count"].pow(0.5)
    return grouped.rename(columns={"mean": value_col})


def _plot_last_epoch_bar(
    epoch_metrics: pd.DataFrame,
    output_path: Path,
    prediction_metric: str,
    prediction_metric_label: str,
    ylim_exclude_epoch_max: float | None,
) -> None:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import numpy as np

    plt.style.use("seaborn-v0_8-whitegrid")
    _apply_plot_style(mpl)

    bar_df = _last_epoch_bar_data(epoch_metrics, prediction_metric)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(3.0, 2.25))
    if bar_df.empty:
        ax.text(0.5, 0.5, f"No data: {prediction_metric}", ha="center", va="center")
        ax.set_axis_off()
    else:
        configured_games = list(_CFG.get("games", {}).get("colors", {}).keys())
        present_games = [str(g) for g in bar_df["unseen_game"].dropna().unique()]
        games = [g for g in configured_games if g in present_games]
        games.extend(sorted(g for g in present_games if g not in games))
        methods = (
            bar_df[["method_key", "method_label", "method_order"]]
            .drop_duplicates()
            .sort_values("method_order")
            .to_dict("records")
        )
        x = np.arange(len(games), dtype=float)
        total_width = 0.74
        bar_width = total_width / max(1, len(methods))
        offsets = (np.arange(len(methods)) - (len(methods) - 1) / 2.0) * bar_width

        for idx, method in enumerate(methods):
            method_key = str(method["method_key"])
            values = []
            errors = []
            for game in games:
                row = bar_df[
                    (bar_df["unseen_game"].astype(str) == game)
                    & (bar_df["method_key"].astype(str) == method_key)
                ]
                if row.empty:
                    values.append(float("nan"))
                    errors.append(0.0)
                else:
                    values.append(float(row.iloc[0][prediction_metric]))
                    errors.append(float(row.iloc[0]["sem"]))
            color = _method_color(method_key)
            centers = x + offsets[idx]
            bars = ax.bar(
                centers,
                values,
                width=bar_width * 0.88,
                color=color,
                alpha=0.82,
                label=str(method["method_label"]),
                yerr=errors,
                error_kw={"elinewidth": 1.0, "capsize": 2.5, "capthick": 1.0},
                edgecolor="white",
                hatch="////" if method_key == "mgpcgrl_da" else ("...." if method_key == "detach" else None),
                linewidth=0.6,
            )
            for bar, value, error in zip(bars, values, errors, strict=False):
                if not np.isfinite(value):
                    continue
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    value + error + 0.003,
                    f"{value:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=5,
                )

        ax.set_xticks(x)
        ax.set_xticklabels(
            [_display_game_name(game) for game in games],
            rotation=20,
            ha="right",
            fontsize=8,
        )
        ax.tick_params(axis="y", labelsize=8)
        ax.set_xlabel("Unseen Game", fontsize=9)
        ax.set_ylabel(prediction_metric_label, fontsize=9)
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, 1.2),
            ncol=max(1, len(methods)),
            frameon=False,
            fontsize=8,
            handlelength=1.4,
            columnspacing=1.2,
        )

        if ylim_exclude_epoch_max is not None:
            y = pd.to_numeric(bar_df[prediction_metric], errors="coerce")
            sem = pd.to_numeric(bar_df.get("sem", 0.0), errors="coerce").fillna(0.0)
            lo = (y - sem).min()
            hi = (y + sem).max()
            if pd.notna(lo) and pd.notna(hi) and hi > lo:
                pad = (hi - lo) * 0.28
                ax.set_ylim(max(0.0, lo - pad), hi + pad)

    fig.tight_layout(pad=0.15)
    fig.savefig(output_path.with_suffix(".png"), dpi=240, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _plot_performance(
    epoch_metrics: pd.DataFrame,
    loss_history: pd.DataFrame,
    output_base: Path,
    epoch_bin_size: int,
    max_points_per_line: int,
    history_loss_label: str,
    history_loss_output_name: str,
    prediction_metric: str,
    prediction_metric_label: str,
    prediction_metric_output_name: str,
    ylim_exclude_epoch_max: float | None,
) -> None:
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    from matplotlib.ticker import FuncFormatter, MultipleLocator

    if epoch_bin_size <= 0:
        epoch_bin_size = _auto_bin_size(epoch_metrics, loss_history, max_points_per_line)

    plt.style.use("seaborn-v0_8-whitegrid")
    _apply_plot_style(mpl)

    specs = []
    if not loss_history.empty:
        specs.append(
            (
                loss_history,
                "history_loss",
                history_loss_label,
                output_base.parent / history_loss_output_name,
            )
        )
    if not epoch_metrics.empty:
        specs.append(
            (
                epoch_metrics,
                prediction_metric,
                prediction_metric_label,
                output_base.parent / prediction_metric_output_name,
            )
        )

    methods_by_key: dict[str, dict] = {}
    for source, value_col, _, _ in specs:
        plot_df = _aggregate_for_plot(source, value_col, epoch_bin_size)
        if not plot_df.empty:
            for method in (
                plot_df[["method_key", "method_label", "method_order"]]
                .drop_duplicates()
                .to_dict("records")
            ):
                methods_by_key[str(method["method_key"])] = method
    methods = sorted(methods_by_key.values(), key=lambda item: int(item["method_order"]))

    def _epoch_formatter(value: float, _pos: int) -> str:
        if abs(value) >= 1000:
            scaled = value / 1000.0
            if abs(scaled - round(scaled)) < 1e-6:
                return f"{int(round(scaled))}e3"
            return f"{scaled:g}e3"
        if abs(value - round(value)) < 1e-6:
            return str(int(round(value)))
        return f"{value:g}"

    def _marker_indices(n_points: int) -> list[int]:
        if n_points <= 0:
            return []
        step = max(1, n_points // 12)
        indices = list(range(0, n_points, step))
        if indices[-1] != n_points - 1:
            indices.append(n_points - 1)
        return indices

    for source, value_col, ylabel, out_path in specs:
        plot_df = _aggregate_for_plot(source, value_col, epoch_bin_size)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(3.0, 2.25))
        if plot_df.empty:
            ax.text(0.5, 0.5, f"No data: {value_col}", ha="center", va="center")
            ax.set_axis_off()
        else:
            for method in methods:
                method_key = str(method["method_key"])
                wdf = plot_df[plot_df["method_key"].astype(str) == method_key].sort_values("epoch")
                if wdf.empty:
                    continue
                color = _method_color(method_key)
                x = wdf["epoch"].astype(float).to_numpy()
                y = wdf[value_col].astype(float).to_numpy()
                sem = wdf["sem"].fillna(0.0).astype(float).to_numpy()
                marker_idx = _marker_indices(len(x))
                x_marked = x[marker_idx]
                y_marked = y[marker_idx]
                sem_marked = sem[marker_idx]
                ax.plot(
                    x_marked,
                    y_marked,
                    color=color,
                    linewidth=1.9,
                    linestyle=_method_line_style(method_key),
                    marker=_method_marker(method_key),
                    markersize=_method_marker_size(method_key),
                    label=str(method["method_label"]),
                )
                ax.fill_between(
                    x_marked,
                    y_marked - sem_marked,
                    y_marked + sem_marked,
                    color=color,
                    alpha=0.16,
                    linewidth=0,
                )
            ax.set_ylabel(ylabel)
            ax.xaxis.set_major_locator(MultipleLocator(1000))
            ax.xaxis.set_major_formatter(FuncFormatter(_epoch_formatter))
            ax.legend(
                loc="upper center",
                bbox_to_anchor=(0.5, 1.22),
                ncol=max(1, len(methods)),
                frameon=False,
                handlelength=1.6,
                columnspacing=1.2,
            )
            if ylim_exclude_epoch_max is not None:
                ylim_df = plot_df[
                    pd.to_numeric(plot_df["epoch"], errors="coerce") > ylim_exclude_epoch_max
                ].copy()
                if not ylim_df.empty:
                    y = pd.to_numeric(ylim_df[value_col], errors="coerce")
                    sem = pd.to_numeric(ylim_df.get("sem", 0.0), errors="coerce").fillna(0.0)
                    lo = (y - sem).min()
                    hi = (y + sem).max()
                    if pd.notna(lo) and pd.notna(hi) and hi > lo:
                        pad = (hi - lo) * 0.08
                        ax.set_ylim(max(0.0, lo - pad), hi + pad)

        ax.set_xlabel("Epoch")
        fig.tight_layout(pad=0.15)
        fig.savefig(out_path.with_suffix(".png"), dpi=240)
        fig.savefig(out_path.with_suffix(".pdf"))
        plt.close(fig)


def _plot_prediction_and_history_subplots(
    *,
    epoch_metrics: pd.DataFrame,
    loss_history: pd.DataFrame,
    output_path: Path,
    epoch_bin_size: int,
    max_points_per_line: int,
    prediction_metric: str,
    prediction_metric_label: str,
    history_loss_label: str,
    ylim_exclude_epoch_max: float | None,
) -> None:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FuncFormatter, MultipleLocator

    if epoch_metrics.empty or loss_history.empty:
        return
    if epoch_bin_size <= 0:
        epoch_bin_size = _auto_bin_size(epoch_metrics, loss_history, max_points_per_line)

    plt.style.use("seaborn-v0_8-whitegrid")
    _apply_plot_style(mpl)

    specs = [
        (epoch_metrics, prediction_metric, prediction_metric_label),
        (loss_history, "history_loss", history_loss_label),
    ]
    plot_specs = [
        (_aggregate_for_plot(source, value_col, epoch_bin_size), value_col, ylabel)
        for source, value_col, ylabel in specs
    ]

    methods_by_key: dict[str, dict] = {}
    for plot_df, _, _ in plot_specs:
        if plot_df.empty:
            continue
        for method in (
            plot_df[["method_key", "method_label", "method_order"]]
            .drop_duplicates()
            .to_dict("records")
        ):
            methods_by_key[str(method["method_key"])] = method
    methods = sorted(methods_by_key.values(), key=lambda item: int(item["method_order"]))

    def _epoch_formatter(value: float, _pos: int) -> str:
        if abs(value) >= 1000:
            scaled = value / 1000.0
            if abs(scaled - round(scaled)) < 1e-6:
                return f"{int(round(scaled))}e3"
            return f"{scaled:g}e3"
        if abs(value - round(value)) < 1e-6:
            return str(int(round(value)))
        return f"{value:g}"

    def _marker_indices(n_points: int) -> list[int]:
        if n_points <= 0:
            return []
        step = max(1, n_points // 12)
        indices = list(range(0, n_points, step))
        if indices[-1] != n_points - 1:
            indices.append(n_points - 1)
        return indices

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(5.2, 2.05), sharex=False)
    for ax, (plot_df, value_col, ylabel) in zip(axes, plot_specs, strict=False):
        if plot_df.empty:
            ax.text(0.5, 0.5, f"No data: {value_col}", ha="center", va="center")
            ax.set_axis_off()
            continue
        for method in methods:
            method_key = str(method["method_key"])
            wdf = plot_df[plot_df["method_key"].astype(str) == method_key].sort_values("epoch")
            if wdf.empty:
                continue
            x = wdf["epoch"].astype(float).to_numpy()
            y = wdf[value_col].astype(float).to_numpy()
            sem = wdf["sem"].fillna(0.0).astype(float).to_numpy()
            ax.plot(
                x,
                y,
                color=_method_color(method_key),
                linewidth=1.9,
                linestyle=_method_line_style(method_key),
                marker=_method_marker(method_key),
                markevery=_marker_indices(len(x)),
                markersize=_method_marker_size(method_key),
            )
            ax.fill_between(
                x,
                y - sem,
                y + sem,
                color=_method_color(method_key),
                alpha=0.16,
                linewidth=0,
            )
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.xaxis.set_major_locator(MultipleLocator(1000))
        ax.xaxis.set_major_formatter(FuncFormatter(_epoch_formatter))
        if ylim_exclude_epoch_max is not None:
            ylim_df = plot_df[
                pd.to_numeric(plot_df["epoch"], errors="coerce") > ylim_exclude_epoch_max
            ].copy()
            if not ylim_df.empty:
                y = pd.to_numeric(ylim_df[value_col], errors="coerce")
                sem = pd.to_numeric(ylim_df.get("sem", 0.0), errors="coerce").fillna(0.0)
                lo = (y - sem).min()
                hi = (y + sem).max()
                if pd.notna(lo) and pd.notna(hi) and hi > lo:
                    pad = (hi - lo) * 0.08
                    ax.set_ylim(max(0.0, lo - pad), hi + pad)

    handles = [
        Line2D(
            [0],
            [0],
            color=_method_color(str(method["method_key"])),
            linewidth=1.9,
            linestyle=_method_line_style(str(method["method_key"])),
            marker=_method_marker(str(method["method_key"])),
            markersize=_method_marker_size(str(method["method_key"])),
            label=str(method["method_label"]),
        )
        for method in methods
    ]
    if handles:
        fig.legend(
            handles=handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.04),
            ncol=len(handles),
            frameon=False,
            handlelength=1.7,
            columnspacing=1.4,
        )
    fig.tight_layout(pad=0.2, rect=(0, 0, 1, 0.9))
    fig.subplots_adjust(wspace=0.42)
    fig.savefig(output_path.with_suffix(".png"), dpi=240, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _plot_decoder_outputs(
    *,
    epoch_metrics: pd.DataFrame,
    loss_history: pd.DataFrame,
    run_dir: Path,
    loss_plot_specs: list[tuple[str, str, str]],
    args: argparse.Namespace,
    history_loss_label: str,
    history_loss_output_name: str,
    prediction_metric: str,
    prediction_metric_label: str,
    prediction_metric_output_name: str,
    ylim_exclude_epoch_max: float | None,
) -> None:
    if not epoch_metrics.empty and not loss_history.empty:
        _plot_prediction_and_history_subplots(
            epoch_metrics=epoch_metrics,
            loss_history=loss_history,
            output_path=run_dir / f"{prediction_metric_output_name}_{history_loss_output_name}",
            epoch_bin_size=args.epoch_bin_size,
            max_points_per_line=args.max_points_per_line,
            prediction_metric=prediction_metric,
            prediction_metric_label=prediction_metric_label,
            history_loss_label=history_loss_label,
            ylim_exclude_epoch_max=ylim_exclude_epoch_max,
        )
    if not loss_history.empty:
        _plot_performance(
            epoch_metrics=pd.DataFrame(),
            loss_history=loss_history,
            output_base=run_dir / "decoder_performance_by_delta_weight",
            epoch_bin_size=args.epoch_bin_size,
            max_points_per_line=args.max_points_per_line,
            history_loss_label=history_loss_label,
            history_loss_output_name=history_loss_output_name,
            prediction_metric=prediction_metric,
            prediction_metric_label=prediction_metric_label,
            prediction_metric_output_name=prediction_metric_output_name,
            ylim_exclude_epoch_max=ylim_exclude_epoch_max,
        )
    for metric_name, metric_label, output_name in loss_plot_specs:
        _plot_performance(
            epoch_metrics=epoch_metrics,
            loss_history=pd.DataFrame(),
            output_base=run_dir / "decoder_performance_by_delta_weight",
            epoch_bin_size=args.epoch_bin_size,
            max_points_per_line=args.max_points_per_line,
            history_loss_label=history_loss_label,
            history_loss_output_name=history_loss_output_name,
            prediction_metric=metric_name,
            prediction_metric_label=metric_label,
            prediction_metric_output_name=output_name,
            ylim_exclude_epoch_max=ylim_exclude_epoch_max,
        )
        if epoch_metrics.empty or "unseen_game" not in epoch_metrics.columns:
            continue

        _plot_by_unseen_game(
            epoch_metrics=epoch_metrics,
            output_path=run_dir / f"{output_name}_by_unseen_game",
            epoch_bin_size=args.epoch_bin_size,
            max_points_per_line=args.max_points_per_line,
            prediction_metric=metric_name,
            prediction_metric_label=metric_label,
            ylim_exclude_epoch_max=ylim_exclude_epoch_max,
        )
        _plot_last_epoch_bar(
            epoch_metrics=epoch_metrics,
            output_path=run_dir / f"{output_name}_last_epoch_bar",
            prediction_metric=metric_name,
            prediction_metric_label=metric_label,
            ylim_exclude_epoch_max=ylim_exclude_epoch_max,
        )

        unseen_games = sorted(
            g for g in epoch_metrics["unseen_game"].dropna().unique()
            if str(g) and str(g) != "unknown"
        )
        for unseen_game in unseen_games:
            game_metrics = epoch_metrics[epoch_metrics["unseen_game"] == unseen_game]
            if game_metrics.empty:
                continue
            _plot_performance(
                epoch_metrics=game_metrics,
                loss_history=pd.DataFrame(),
                output_base=run_dir / "by_game" / _safe_name(str(unseen_game)) / "decoder_performance_by_delta_weight",
                epoch_bin_size=args.epoch_bin_size,
                max_points_per_line=args.max_points_per_line,
                history_loss_label=history_loss_label,
                history_loss_output_name=history_loss_output_name,
                prediction_metric=metric_name,
                prediction_metric_label=metric_label,
                prediction_metric_output_name=output_name,
                ylim_exclude_epoch_max=ylim_exclude_epoch_max,
            )


def parse_args() -> argparse.Namespace:
    exp_names = _experiment_names()
    parser = argparse.ArgumentParser(
        description="Download decoder_prediction_csv artifacts and plot decoder performance.",
    )
    parser.add_argument("--experiment", choices=exp_names if exp_names else None, default="decoder_performance")
    parser.add_argument("--projects", nargs="+", default=None)
    parser.add_argument("--entity", default=DEFAULT_ENTITY)
    parser.add_argument("--finished-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--workers", type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument(
        "--no-prediction-artifacts",
        action="store_true",
        help="Skip decoder_prediction_csv artifact download/read and process W&B history only.",
    )
    parser.add_argument(
        "--no-save-prediction-rows",
        action="store_true",
        help="Do not write concatenated raw decoder_prediction_rows.csv.",
    )
    parser.add_argument(
        "--history-loss-key",
        default=None,
        help=(
            "W&B history metric to plot by epoch. "
            f"Default: experiment config history_loss_key or {_DEFAULT_LOSS_KEY!r}."
        ),
    )
    parser.add_argument(
        "--history-loss-label",
        default=None,
        help="Y-axis label for the history loss plot.",
    )
    parser.add_argument(
        "--history-loss-output-name",
        default=None,
        help="Output basename for the history loss plot, without extension.",
    )
    parser.add_argument(
        "--prediction-metric",
        default=None,
        choices=[
            "regression_mae_raw",
            "regression_mae_norm",
            "regression_rmse_raw",
            "regression_rmse_norm",
            "regression_pearson_raw",
            "regression_pearson_norm",
        ],
        help="Prediction metric column to plot from decoder_prediction_csv artifacts.",
    )
    parser.add_argument(
        "--prediction-metric-label",
        default=None,
        help="Y-axis label for artifact-based prediction metric plot.",
    )
    parser.add_argument(
        "--prediction-metric-output-name",
        default=None,
        help="Output basename for artifact-based prediction metric plot, without extension.",
    )
    parser.add_argument(
        "--ylim-exclude-epoch-max",
        type=float,
        default=None,
        help="Exclude epochs <= this value when computing plot y-limits.",
    )
    parser.add_argument(
        "--cache-dir",
        default=str(_default_cache_dir()),
        help="Persistent cache directory for finished W&B runs.",
    )
    parser.add_argument(
        "--epoch-bin-size",
        type=int,
        default=0,
        help="Epoch bin size for plotting. 0 chooses an automatic bin size.",
    )
    parser.add_argument(
        "--max-points-per-line",
        type=int,
        default=80,
        help="Used only when --epoch-bin-size=0 to avoid overcrowded x-axis lines.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = _make_decoder_run_dir(_CFG)
    log = setup_logger(run_dir, name="decoder_performance")

    projects = _resolve_projects(args.experiment, args.projects)
    exp_cfg = _experiment_cfg(args.experiment)
    history_loss_key = (
        args.history_loss_key
        or exp_cfg.get("history_loss_key")
        or _DEFAULT_LOSS_KEY
    )
    history_loss_label = (
        args.history_loss_label
        or exp_cfg.get("history_loss_label")
        or ("Embedding Alignment Loss" if history_loss_key == _DEFAULT_LOSS_KEY else history_loss_key)
    )
    history_loss_output_name = (
        args.history_loss_output_name
        or exp_cfg.get("history_loss_output_name")
        or ("delta_loss" if history_loss_key == _DEFAULT_LOSS_KEY else _safe_name(history_loss_key))
    )
    prediction_metric = (
        args.prediction_metric
        or exp_cfg.get("prediction_metric")
        or "regression_mae_raw"
    )
    prediction_metric_label = (
        args.prediction_metric_label
        or exp_cfg.get("prediction_metric_label")
        or "Reward Prediction Loss"
    )
    prediction_metric_output_name = (
        args.prediction_metric_output_name
        or exp_cfg.get("prediction_metric_output_name")
        or "regression"
    )
    ylim_exclude_epoch_max = (
        args.ylim_exclude_epoch_max
        if args.ylim_exclude_epoch_max is not None
        else exp_cfg.get("ylim_exclude_epoch_max")
    )
    history_epoch_offset = float(exp_cfg.get("history_epoch_offset", 1.0))
    download_prediction_artifacts = bool(exp_cfg.get("download_prediction_artifacts", True))
    if args.no_prediction_artifacts:
        download_prediction_artifacts = False
    download_history_loss = bool(exp_cfg.get("download_history_loss", True))
    save_prediction_rows = bool(exp_cfg.get("save_prediction_rows", True))
    if args.no_save_prediction_rows:
        save_prediction_rows = False
    prediction_norm_group_cols = list(exp_cfg.get("prediction_norm_group_cols", []))
    excluded_delta_weights = [
        float(w)
        for w in exp_cfg.get("exclude_delta_weights", [])
        if _as_float(w) is not None
    ]
    default_decoder_nograd = _as_bool(exp_cfg.get("default_decoder_nograd"))
    cache_dir = Path(args.cache_dir)
    if not cache_dir.is_absolute():
        cache_dir = (_ROOT / cache_dir).resolve()
    api = get_api()
    filters = {"state": "finished"} if args.finished_only else {}

    log.info("experiment : %s", args.experiment)
    log.info("projects   : %s", projects)
    log.info("entity     : %s", args.entity)
    log.info("loss key   : %s", history_loss_key)
    log.info("artifacts  : %s", "enabled" if download_prediction_artifacts else "disabled")
    log.info("history    : %s", "enabled" if download_history_loss else "disabled")
    log.info("pred metric: %s", prediction_metric)
    log.info("norm groups: %s", prediction_norm_group_cols or "artifact/default")
    log.info("excl delta : %s", excluded_delta_weights or "none")
    log.info("ylim excl. : %s", ylim_exclude_epoch_max)
    log.info("run_dir    : %s", run_dir)
    log.info("cache_dir  : %s", cache_dir)

    prediction_frames: list[pd.DataFrame] = []
    epoch_metric_frames: list[pd.DataFrame] = []
    history_frames: list[pd.DataFrame] = []
    artifact_records: list[dict] = []
    scanned_runs = 0
    artifact_runs = 0
    prediction_row_count = 0

    for project in projects:
        runs = list(api.runs(f"{args.entity}/{project}", filters=filters, per_page=200))
        for run in tqdm(runs, desc=project, unit="run"):
            scanned_runs += 1
            delta_weight = _as_float(_run_config_value(run.config, "delta_weight"))
            decoder_nograd = _as_bool(
                _run_config_value(
                    run.config,
                    "decoder_nograd",
                    "decoder.nograd",
                    "decoder.nograd_decoder",
                    "decoder.decoder_nograd",
                )
            )
            if decoder_nograd is None:
                decoder_nograd = default_decoder_nograd
            if download_prediction_artifacts:
                csv_paths = _download_prediction_artifact(run, project, cache_dir, force=args.force)
                if csv_paths:
                    artifact_runs += 1
                    artifact_records.append({
                        "project": project,
                        "run": run,
                        "delta_weight": delta_weight,
                        "decoder_nograd": decoder_nograd,
                        "csv_paths": csv_paths,
                    })

            if download_history_loss:
                hist = _history_rows(
                    project,
                    run,
                    delta_weight,
                    decoder_nograd,
                    loss_key=history_loss_key,
                    epoch_offset=history_epoch_offset,
                    cache_dir=cache_dir,
                    force=args.force,
                )
                if not hist.empty:
                    history_frames.append(hist)

    norm_scales = pd.DataFrame()
    if artifact_records and prediction_norm_group_cols:
        norm_scales = _prediction_norm_scales_from_csvs(
            [record["csv_paths"] for record in artifact_records],
            prediction_norm_group_cols,
        )
        if not norm_scales.empty:
            norm_scales.to_csv(run_dir / "decoder_prediction_norm_scales.csv", index=False)
        log.info("norm scale rows   : %d", len(norm_scales))

    for record in tqdm(artifact_records, desc="decoder metrics", unit="run"):
        run = record["run"]
        csv_paths = record["csv_paths"]
        metrics, n_rows = _prediction_epoch_metrics_from_csvs(
            csv_paths,
            project=record["project"],
            run=run,
            delta_weight=record["delta_weight"],
            decoder_nograd=record["decoder_nograd"],
            norm_group_cols=prediction_norm_group_cols,
            norm_scales=norm_scales,
        )
        prediction_row_count += n_rows
        if not metrics.empty:
            epoch_metric_frames.append(metrics)
        if save_prediction_rows:
            rows = _read_prediction_csvs(
                csv_paths,
                project=record["project"],
                run=run,
                delta_weight=record["delta_weight"],
                decoder_nograd=record["decoder_nograd"],
            )
            if not rows.empty:
                prediction_frames.append(rows)

    prediction_rows = (
        pd.concat(prediction_frames, ignore_index=True)
        if prediction_frames
        else pd.DataFrame()
    )
    loss_history = (
        pd.concat(history_frames, ignore_index=True)
        if history_frames
        else pd.DataFrame()
    )
    if epoch_metric_frames:
        epoch_metrics = pd.concat(epoch_metric_frames, ignore_index=True)
    else:
        epoch_metrics = _prediction_epoch_metrics(prediction_rows)
    if not prediction_rows.empty and not loss_history.empty and "delta_weight" in prediction_rows.columns:
        run_delta = (
            prediction_rows[["run_id", "delta_weight"]]
            .dropna()
            .drop_duplicates("run_id")
            .set_index("run_id")["delta_weight"]
        )
        missing_delta = loss_history["delta_weight"].isna()
        loss_history.loc[missing_delta, "delta_weight"] = (
            loss_history.loc[missing_delta, "run_id"].map(run_delta)
        )
    if not prediction_rows.empty and not loss_history.empty and "decoder_nograd" in prediction_rows.columns:
        if "decoder_nograd" not in loss_history.columns:
            loss_history["decoder_nograd"] = None
        run_decoder_nograd = (
            prediction_rows[["run_id", "decoder_nograd"]]
            .dropna()
            .drop_duplicates("run_id")
            .set_index("run_id")["decoder_nograd"]
        )
        missing_decoder_nograd = loss_history["decoder_nograd"].isna()
        loss_history.loc[missing_decoder_nograd, "decoder_nograd"] = (
            loss_history.loc[missing_decoder_nograd, "run_id"].map(run_decoder_nograd)
        )

    prediction_rows = _filter_excluded_delta_weights(prediction_rows, excluded_delta_weights)
    loss_history = _filter_excluded_delta_weights(loss_history, excluded_delta_weights)
    epoch_metrics = _filter_excluded_delta_weights(epoch_metrics, excluded_delta_weights)

    if save_prediction_rows:
        prediction_rows.to_csv(run_dir / "decoder_prediction_rows.csv", index=False)
    loss_history.to_csv(run_dir / f"{history_loss_output_name}_history.csv", index=False)
    epoch_metrics.to_csv(run_dir / "decoder_epoch_metrics.csv", index=False)

    loss_plot_specs = [
        (
            prediction_metric,
            prediction_metric_label,
            prediction_metric_output_name,
        ),
    ]
    # RMSE plots are disabled for the current figure set.
    # if "regression_rmse_norm" in epoch_metrics.columns:
    #     loss_plot_specs.append(
    #         (
    #             "regression_rmse_norm",
    #             "Normalized RMSE",
    #             "prediction_rmse",
    #         )
    #     )
    seen_plot_outputs: set[str] = set()
    loss_plot_specs = [
        spec for spec in loss_plot_specs
        if not (spec[2] in seen_plot_outputs or seen_plot_outputs.add(spec[2]))
    ]

    if not args.no_plot:
        method_labels = []
        if not epoch_metrics.empty and {"delta_weight", "decoder_nograd"}.issubset(epoch_metrics.columns):
            method_keys = {
                _method_key(row.delta_weight, row.decoder_nograd)
                for row in epoch_metrics[["delta_weight", "decoder_nograd"]]
                .drop_duplicates()
                .itertuples(index=False)
            }
            method_labels = [
                _method_label(method_key)
                for method_key in sorted(method_keys, key=_method_sort_order)
            ]
        if method_labels:
            log.info("methods    : %s", ", ".join(method_labels))
        _plot_decoder_outputs(
            epoch_metrics=epoch_metrics,
            loss_history=loss_history,
            run_dir=run_dir,
            loss_plot_specs=loss_plot_specs,
            args=args,
            history_loss_label=history_loss_label,
            history_loss_output_name=history_loss_output_name,
            prediction_metric=prediction_metric,
            prediction_metric_label=prediction_metric_label,
            prediction_metric_output_name=prediction_metric_output_name,
            ylim_exclude_epoch_max=ylim_exclude_epoch_max,
        )
        # Pearson plots are disabled for the current figure set.
        # if not epoch_metrics.empty and "regression_pearson_norm" in epoch_metrics.columns:
        #     if "unseen_game" in epoch_metrics.columns:
        #         _plot_by_unseen_game(
        #             epoch_metrics=epoch_metrics,
        #             output_path=run_dir / "prediction_pearson",
        #             epoch_bin_size=args.epoch_bin_size,
        #             max_points_per_line=args.max_points_per_line,
        #             prediction_metric="regression_pearson_norm",
        #             prediction_metric_label="Reward Prediction Pearson",
        #             ylim_exclude_epoch_max=ylim_exclude_epoch_max,
        #         )
        #         _plot_last_epoch_bar(
        #             epoch_metrics=epoch_metrics,
        #             output_path=run_dir / "prediction_pearson_last_epoch_bar",
        #             prediction_metric="regression_pearson_norm",
        #             prediction_metric_label="Reward Prediction Pearson",
        #             ylim_exclude_epoch_max=ylim_exclude_epoch_max,
        #         )
        #     else:
        #         _plot_performance(
        #             epoch_metrics=epoch_metrics,
        #             loss_history=pd.DataFrame(),
        #             output_base=run_dir / "decoder_performance_by_delta_weight",
        #             epoch_bin_size=args.epoch_bin_size,
        #             max_points_per_line=args.max_points_per_line,
        #             history_loss_label=history_loss_label,
        #             history_loss_output_name=history_loss_output_name,
        #             prediction_metric="regression_pearson_norm",
        #             prediction_metric_label="Reward Prediction Pearson",
        #             prediction_metric_output_name="prediction_pearson",
        #             ylim_exclude_epoch_max=ylim_exclude_epoch_max,
        #         )

        # Per-unseen-game plots are disabled for the current figure set.
        # if not epoch_metrics.empty and "unseen_game" in epoch_metrics.columns:
        #     unseen_games = sorted(
        #         g for g in epoch_metrics["unseen_game"].dropna().unique()
        #         if str(g) and str(g) != "unknown"
        #     )
        #     for unseen_game in unseen_games:
        #         game_metrics = epoch_metrics[epoch_metrics["unseen_game"] == unseen_game]
        #         if game_metrics.empty:
        #             continue
        #         for metric_name, metric_label, output_name in loss_plot_specs:
        #             _plot_performance(
        #                 epoch_metrics=game_metrics,
        #                 loss_history=pd.DataFrame(),
        #                 output_base=run_dir / _safe_name(str(unseen_game)) / "decoder_performance_by_delta_weight",
        #                 epoch_bin_size=args.epoch_bin_size,
        #                 max_points_per_line=args.max_points_per_line,
        #                 history_loss_label=history_loss_label,
        #                 history_loss_output_name=history_loss_output_name,
        #                 prediction_metric=metric_name,
        #                 prediction_metric_label=metric_label,
        #                 prediction_metric_output_name=output_name,
        #                 ylim_exclude_epoch_max=ylim_exclude_epoch_max,
        #             )
        #         if "regression_pearson_norm" in game_metrics.columns:
        #             _plot_performance(
        #                 epoch_metrics=game_metrics,
        #                 loss_history=pd.DataFrame(),
        #                 output_base=run_dir / _safe_name(str(unseen_game)) / "decoder_performance_by_delta_weight",
        #                 epoch_bin_size=args.epoch_bin_size,
        #                 max_points_per_line=args.max_points_per_line,
        #                 history_loss_label=history_loss_label,
        #                 history_loss_output_name=history_loss_output_name,
        #                 prediction_metric="regression_pearson_norm",
        #                 prediction_metric_label="Reward Prediction Pearson",
        #                 prediction_metric_output_name="prediction_pearson",
        #                 ylim_exclude_epoch_max=ylim_exclude_epoch_max,
        #             )

    log.info("runs scanned       : %d", scanned_runs)
    log.info("runs with artifact : %d", artifact_runs)
    log.info("prediction rows    : %d", prediction_row_count or len(prediction_rows))
    log.info("epoch metric rows  : %d", len(epoch_metrics))
    log.info("loss history rows  : %d", len(loss_history))
    log.info("outputs            : %s", run_dir)


if __name__ == "__main__":
    main()
