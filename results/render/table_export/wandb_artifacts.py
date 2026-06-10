from __future__ import annotations

import shutil
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

from .h5_utils import analyze_h5_file
from .models import RunResult
from .utils import (
    parse_wandb_run_url,
    replace_reward_enum_in_run_name,
    run_url,
    safe_slug,
)


def _select_h5_artifact(run, artifact_name: Optional[str] = None) -> tuple[Optional[object], str]:
    try:
        artifacts = list(run.logged_artifacts())
    except Exception as e:
        return None, f"failed to list artifacts: {e}"

    candidates: list[tuple[float, Any]] = []
    for art in artifacts:
        if getattr(art, "type", None) != "dataset":
            continue
        name = str(getattr(art, "name", ""))
        name_without_version = name.split(":", maxsplit=1)[0]
        if artifact_name:
            if artifact_name not in {name, name_without_version}:
                continue
        elif not name.startswith("eval_h5"):
            continue

        try:
            files = list(art.files())
        except Exception:
            continue
        if not any(Path(f.name).name == "eval.h5" for f in files):
            continue

        created_ts = getattr(art, "createdAt", None)
        candidates.append((float(created_ts.timestamp()) if created_ts else 0.0, art))

    if not candidates:
        return None, "eval_h5 artifact containing eval.h5 not found"
    candidates.sort(key=lambda x: x[0])
    return candidates[-1][1], ""


def download_eval_h5_from_run(
    run,
    output_path: Path,
    artifact_name: Optional[str] = None,
) -> tuple[Optional[Path], Optional[str], Optional[str]]:
    output_path.mkdir(parents=True, exist_ok=True)
    local_eval_h5 = output_path / "eval.h5"
    if local_eval_h5.exists():
        return local_eval_h5, None, "cached"

    artifact, err = _select_h5_artifact(run, artifact_name=artifact_name)
    if err:
        return None, err, None

    with tempfile.TemporaryDirectory() as tmp_dir:
        try:
            artifact.download(root=tmp_dir)
            downloaded = next(Path(tmp_dir).rglob("eval.h5"), None)
            if downloaded is None:
                return None, "eval.h5 was not downloaded", getattr(artifact, "name", None)
            if local_eval_h5.exists():
                local_eval_h5.unlink()
            shutil.move(str(downloaded), local_eval_h5)
            return local_eval_h5, None, getattr(artifact, "name", None)
        except Exception as e:
            return None, f"download failed: {e}", getattr(artifact, "name", None)


def _select_csv_artifact(run) -> tuple[Optional[object], str]:
    try:
        artifacts = list(run.logged_artifacts())
    except Exception as e:
        return None, f"failed to list artifacts: {e}"

    candidates: list[tuple[float, Any]] = []
    for art in artifacts:
        if getattr(art, "type", None) != "dataset":
            continue
        if not str(getattr(art, "name", "")).startswith("eval_csv"):
            continue
        created_ts = getattr(art, "createdAt", None)
        candidates.append((float(created_ts.timestamp()) if created_ts else 0.0, art))

    if not candidates:
        return None, "eval_csv artifact not found"
    candidates.sort(key=lambda x: x[0])
    return candidates[-1][1], ""


def download_eval_csv_from_run(run, output_path: Path) -> tuple[Optional[Path], Optional[str], Optional[str]]:
    output_path.mkdir(parents=True, exist_ok=True)
    csv_dir = output_path / "csv"
    if (csv_dir / "results.csv").exists() and (csv_dir / "ctrl_sim.csv").exists():
        return csv_dir, None, "cached"

    artifact, err = _select_csv_artifact(run)
    if err:
        return None, err, None

    if csv_dir.exists():
        shutil.rmtree(csv_dir)
    csv_dir.mkdir(parents=True, exist_ok=True)

    try:
        artifact.download(root=str(csv_dir))
        return csv_dir, None, getattr(artifact, "name", None)
    except Exception as e:
        return None, f"download failed: {e}", getattr(artifact, "name", None)


def _resolve_wandb_api_module():
    project_root = Path(__file__).resolve().parents[3]
    previous_sys_path = list(sys.path)
    shadow_module = sys.modules.get("wandb")

    def _is_project_root_path(path_value: str) -> bool:
        try:
            return Path(path_value or ".").resolve() == project_root
        except OSError:
            return False

    try:
        sys.path = [p for p in sys.path if not _is_project_root_path(p)]
        if shadow_module is not None and not hasattr(shadow_module, "Api"):
            sys.modules.pop("wandb", None)
        import wandb
        if not hasattr(wandb, "Api"):
            raise AttributeError("installed wandb package does not expose Api")
        return wandb.Api()
    except Exception as e:
        raise RuntimeError(
            "wandb API 초기화 실패: 실제 wandb 패키지가 필요합니다. "
            f"현재 오류: {e}"
        )
    finally:
        sys.path = previous_sys_path


def _discover_reward_enum_runs(api, entity: str, project: str, base_run, reward_enums: list[int]):
    runs = list(api.runs(f"{entity}/{project}", per_page=200))
    by_name = defaultdict(list)
    for run in runs:
        by_name[run.name].append(run)

    discovered = []
    for reward_enum in reward_enums:
        target_name = replace_reward_enum_in_run_name(base_run.name, reward_enum)
        candidates = by_name.get(target_name, [])
        if candidates:
            discovered.append((reward_enum, candidates[0]))
        elif base_run.name == target_name:
            discovered.append((reward_enum, base_run))
    return discovered


def process_config_items(
    config: list[dict[str, Any]],
    script_dir: Path,
    reward_enums: Optional[list[int]] = None,
) -> list[RunResult]:
    run_results: list[RunResult] = []
    download_root = script_dir / ".wandb_download"
    reward_enums = reward_enums or [0, 1, 2, 3, 4]
    try:
        api = _resolve_wandb_api_module()
    except RuntimeError as e:
        return [
            RunResult(
                method=item.get("method", "N/A"),
                run_url=item.get("wandb_run_url", ""),
                reward_enum=reward_enum,
                artifact_name=item.get("wandb_artifact") or item.get("artifact_name"),
                error=str(e),
            )
            for item in config
            for reward_enum in reward_enums
        ]

    for item in config:
        method = item.get("method", "N/A")
        original_run_url = item.get("wandb_run_url", "")
        artifact_name = item.get("wandb_artifact") or item.get("artifact_name")
        run_info = parse_wandb_run_url(original_run_url)
        if not original_run_url or run_info is None:
            run_results.append(
                RunResult(
                    method=method,
                    run_url=original_run_url,
                    artifact_name=artifact_name,
                    error="유효하지 않은 W&B run URL 형식",
                )
            )
            continue

        try:
            base_run = api.run(f"{run_info['entity']}/{run_info['project']}/{run_info['run_id']}")
            discovered_by_reward_enum = dict(
                _discover_reward_enum_runs(
                    api,
                    run_info["entity"],
                    run_info["project"],
                    base_run,
                    reward_enums,
                )
            )
        except Exception as e:
            for reward_enum in reward_enums:
                run_results.append(
                    RunResult(
                        method=method,
                        run_url=original_run_url,
                        reward_enum=reward_enum,
                        artifact_name=artifact_name,
                        error=f"run discovery 실패: {e}",
                    )
                )
            continue

        for reward_enum in reward_enums:
            run = discovered_by_reward_enum.get(reward_enum)
            if run is None:
                run_results.append(
                    RunResult(
                        method=method,
                        run_url=original_run_url,
                        reward_enum=reward_enum,
                        artifact_name=artifact_name,
                        error=f"ev_re-{reward_enum} run을 찾을 수 없습니다",
                    )
                )
                continue

            target_dir = (
                download_root
                / safe_slug(run_info["entity"])
                / safe_slug(run_info["project"])
                / safe_slug(run.id)
            )
            current_run_url = run_url(run_info["entity"], run_info["project"], run.id)
            try:
                h5_local, err, selected_artifact_name = download_eval_h5_from_run(
                    run,
                    target_dir,
                    artifact_name=artifact_name,
                )
                csv_dir, csv_err, selected_csv_artifact_name = download_eval_csv_from_run(run, target_dir)
                if err is not None:
                    run_results.append(
                        RunResult(
                            method=method,
                            run_url=current_run_url,
                            reward_enum=reward_enum,
                            run_name=run.name,
                            artifact_name=selected_artifact_name or artifact_name,
                            csv_artifact_name=selected_csv_artifact_name,
                            csv_dir=csv_dir,
                            error=err,
                        )
                    )
                    continue

                try:
                    h5_stats = analyze_h5_file(h5_local)
                    run_results.append(
                        RunResult(
                            method=method,
                            run_url=current_run_url,
                            reward_enum=reward_enum,
                            run_name=run.name,
                            artifact_name=selected_artifact_name,
                            csv_artifact_name=selected_csv_artifact_name,
                            h5_path=h5_local,
                            csv_dir=csv_dir,
                            h5_stats=h5_stats,
                            error=csv_err,
                        )
                    )
                except Exception as e:
                    run_results.append(
                        RunResult(
                            method=method,
                            run_url=current_run_url,
                            reward_enum=reward_enum,
                            run_name=run.name,
                            artifact_name=selected_artifact_name,
                            csv_artifact_name=selected_csv_artifact_name,
                            h5_path=h5_local,
                            csv_dir=csv_dir,
                            error=f"H5 분석 실패: {e}",
                        )
                    )
            except Exception as e:
                run_results.append(
                    RunResult(
                        method=method,
                        run_url=current_run_url,
                        reward_enum=reward_enum,
                        artifact_name=artifact_name,
                        error=f"run 처리 실패: {e}",
                    )
                )

    return run_results

