"""
eval_downloader.py
==================
Downloads eval artifacts uploaded by runner.py from W&B to local storage.

Artifact types
  - eval_csv  (type=dataset): ctrl_sim.csv / results.csv / diversity.csv / summary.csv
  - eval_h5_* (type=dataset): eval.h5

Run config
  - run_config.json: 각 run의 W&B config를 필터링하여 저장
    불필요한 키는 results/run_config_exclude_keys.json 에서 관리

Examples
--------
    python results/eval_downloader.py
    python results/eval_downloader.py --experiment allseen
    python results/eval_downloader.py --experiment unseen
    python results/eval_downloader.py --h5
    python results/eval_downloader.py --output wandb_projects
    python results/eval_downloader.py --finished-only --workers 4
    python results/eval_downloader.py --no-run-config
"""

import argparse
import json
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

from tqdm import tqdm

# ── Add results/ and project root to sys.path ────────────────────────────────
_HERE        = os.path.dirname(os.path.abspath(__file__))               # results/utils/wandb/
_RESULTS_DIR = os.path.abspath(os.path.join(_HERE, "..", ".."))         # results/
_ROOT        = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))   # project root
if _RESULTS_DIR not in sys.path:
    sys.path.insert(0, _RESULTS_DIR)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from sweep.wandb_utils.config import DEFAULT_ENTITY, DEFAULT_NUM_WORKERS
from sweep.wandb_utils.downloader import get_api

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _load_cfg() -> dict:
    cfg_path = os.path.join(_RESULTS_DIR, "config.json")
    if os.path.isfile(cfg_path):
        with open(cfg_path, encoding="utf-8") as f:
            return json.load(f)
    return {}

_CFG = _load_cfg()


def _load_exclude_keys() -> set[str]:
    """run_config_exclude_keys.json 에서 제외 키 목록을 읽어 set 으로 반환한다.

    JSON 파일의 ``exclude_keys`` 배열에서 ``#`` 으로 시작하는 주석 문자열은
    자동으로 건너뛴다.
    """
    path = os.path.join(_RESULTS_DIR, "run_config_exclude_keys.json")
    if not os.path.isfile(path):
        logger.warning(
            "run_config_exclude_keys.json 을 찾을 수 없습니다 — 필터 없이 전체 config 를 저장합니다."
        )
        return set()
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    raw = data.get("exclude_keys", [])
    return {k for k in raw if isinstance(k, str) and not k.startswith("#")}


_EXCLUDE_KEYS: set[str] = _load_exclude_keys()


_ALL_GAMES: list[str] = ["doom", "dungeon", "pokemon", "sokoban", "zelda"]


def _derive_game_lists(raw_cfg: dict) -> dict:
    """raw run.config 의 ``include_*`` 플래그로 seen_games / unseen_games 를 파생한다.

    doom / doom2 는 표시 게임 기준 "doom" 하나로 통합한다.
    두 플래그가 모두 없으면 게임이 특정되지 않으므로 빈 리스트를 반환한다.
    """
    # include_* 플래그가 하나도 없는 config 는 게임 구분 불가
    flag_keys = {"include_doom", "include_doom2", "include_dungeon",
                 "include_pokemon", "include_sokoban", "include_zelda"}
    if not flag_keys.intersection(raw_cfg):
        return {}

    seen_flags = {
        "doom":    bool(raw_cfg.get("include_doom", False) or raw_cfg.get("include_doom2", False)),
        "dungeon": bool(raw_cfg.get("include_dungeon", False)),
        "pokemon": bool(raw_cfg.get("include_pokemon", False)),
        "sokoban": bool(raw_cfg.get("include_sokoban", False)),
        "zelda":   bool(raw_cfg.get("include_zelda", False)),
    }
    return {
        "seen_games":   [g for g in _ALL_GAMES if     seen_flags[g]],
        "unseen_games": [g for g in _ALL_GAMES if not seen_flags[g]],
    }


def _flatten_encoder(cfg: dict) -> dict:
    """``encoder`` 값이 문자열 Python dict 이면 ``encoder.<subkey>`` 로 펼쳐서 반환한다.

    원본 ``encoder`` 키는 제거되고, 예를 들어 ``encoder.model``, ``encoder.ckpt_path``
    등의 최상위 키로 대체된다. 값이 없거나 dict 가 아니면 그대로 반환한다.
    """
    import ast

    result = dict(cfg)
    enc = result.pop("encoder", None)
    if enc is None:
        return result

    # W&B 는 종종 dict 를 Python repr 문자열로 저장함 — ast.literal_eval 로 파싱
    if isinstance(enc, str):
        try:
            enc = ast.literal_eval(enc)
        except (ValueError, SyntaxError):
            # 파싱 실패 시 원본 문자열 그대로 보존
            result["encoder"] = enc
            return result

    if isinstance(enc, dict):
        for sub_key, sub_val in enc.items():
            result[f"encoder.{sub_key}"] = sub_val
    else:
        result["encoder"] = enc

    return result


def _filter_run_config(cfg: dict, exclude_keys: set[str] | None = None) -> dict:
    """run.config 에서 불필요한 키를 제거하고 ``encoder`` 를 dot notation 으로 펼친다.

    처리 순서:
    1. raw config 에서 seen_games / unseen_games 파생
    2. 최상위 exclude_keys 제거
    3. ``encoder`` 문자열 dict → ``encoder.<subkey>`` 로 펼치기
    4. 펼쳐진 ``encoder.*`` 키 중 exclude_keys 에 포함된 것도 제거
    """
    keys = exclude_keys if exclude_keys is not None else _EXCLUDE_KEYS
    # 1) raw config 기준 게임 리스트 파생 (필터 전)
    game_lists = _derive_game_lists(cfg)
    # 2) 최상위 필터 (encoder 자체는 아직 남김)
    filtered = {k: v for k, v in cfg.items() if k not in keys}
    # 3) encoder 펼치기
    expanded = _flatten_encoder(filtered)
    # 4) 펼쳐진 encoder.* 키에도 exclude 필터 적용
    result = {k: v for k, v in expanded.items() if k not in keys}
    # seen_games / unseen_games 주입 (기존 키 우선 — 이미 있으면 덮어쓰지 않음)
    for k, v in game_lists.items():
        result.setdefault(k, v)
    # dataset_reward_enum → re 로 축약
    if "dataset_reward_enum" in result:
        result["re"] = result.pop("dataset_reward_enum")
    return result


@dataclass
class _RunResult:
    run_name: str
    status: str          # "ok" | "skipped" | "error"
    error: str = ""


@dataclass
class _ProjectSummary:
    project: str
    ok: int = 0
    skipped: int = 0
    errors: list[_RunResult] = field(default_factory=list)

# Project list to download
# Priority: --experiment > --projects > all experiments combined (config.json fallback)
_EXPERIMENTS: dict[str, list[str]] = {
    name: exp.get("target_projects", [])
    for name, exp in _CFG.get("experiments", {}).items()
}
_ALL_EXPERIMENT_PROJECTS: list[str] = [
    p for projects in _EXPERIMENTS.values() for p in projects
]
# Unique, preserving order
_seen: set[str] = set()
_ALL_EXPERIMENT_PROJECTS_DEDUP: list[str] = []
for _p in _ALL_EXPERIMENT_PROJECTS:
    if _p not in _seen:
        _seen.add(_p)
        _ALL_EXPERIMENT_PROJECTS_DEDUP.append(_p)

TARGET_PROJECTS: list[str] = _ALL_EXPERIMENT_PROJECTS_DEDUP or [
    "aaai27_eval_cpcgrl",
    "aaai27_eval_cpcgrl_all",
]
_DEFAULT_NUM_WORKERS: int = _CFG.get("wandb", {}).get("num_workers", DEFAULT_NUM_WORKERS)
_DEFAULT_OUTPUT: str = "wandb_projects"
_PROJECT_DISPLAY_NAMES: dict[str, str] = _CFG.get("project_display_names", {})


def _resolve_output_dir(output_arg: str) -> str:
    """Resolve the downloader cache root.

    The rest of results/ reads the default ``wandb_projects`` relative to
    results/, so keep downloader writes in that same cache directory regardless
    of the process cwd.
    """
    raw = os.path.expanduser(output_arg)
    if os.path.isabs(raw):
        return os.path.abspath(raw)
    if os.path.normpath(raw) == "wandb_projects":
        return os.path.join(_RESULTS_DIR, "wandb_projects")
    return os.path.abspath(raw)

# ---------------------------------------------------------------------------
# Download a single run
# ---------------------------------------------------------------------------


def _download_run(
    run,
    output_dir: str,
    project: str = "",
    download_csv: bool = True,
    download_h5: bool = True,
    download_run_config: bool = True,
    skip_if_exists: bool = True,
) -> _RunResult:
    """Download eval artifacts from a single W&B run.
    Returns
    -------
    _RunResult  with status "ok" | "skipped" | "error"
    """
    train_dir, eval_dir = run.name.split("--")
    run_dir = os.path.join(output_dir, train_dir, eval_dir)
    ctrl_sim = os.path.join(run_dir, "ctrl_sim.csv")
    results = os.path.join(run_dir, "results.csv")
    h5_local = os.path.join(run_dir, "eval.h5")
    config_path = os.path.join(run_dir, "run_config.json")

    # ── skip check ────────────────────────────────────────────────────────
    if skip_if_exists:
        expected: list[str] = []
        if download_csv:
            expected.extend([ctrl_sim, results])
        if download_h5:
            expected.append(h5_local)
        if download_run_config:
            expected.append(config_path)
        if expected and all(os.path.isfile(p) for p in expected):
            return _RunResult(run_name=run.name, status="skipped")

    os.makedirs(run_dir, exist_ok=True)

    # ── list artifacts ────────────────────────────────────────────────────
    try:
        artifacts = list(run.logged_artifacts())
    except Exception as e:
        return _RunResult(run_name=run.name, status="error", error=f"failed to list artifacts: {e}")

    csv_artifact = None
    h5_artifacts = []

    for art in artifacts:
        if art.name.startswith("eval_csv") and art.type == "dataset":
            csv_artifact = art
        elif art.name.startswith("eval_h5") and art.type == "dataset":
            h5_artifacts.append(art)

    errors = []

    # ── download eval_csv ─────────────────────────────────────────────────
    csv_complete = os.path.isfile(ctrl_sim) and os.path.isfile(results)
    if download_csv:
        if skip_if_exists and csv_complete:
            pass
        elif csv_artifact is None:
            errors.append("eval_csv artifact not found")
        else:
            try:
                for f in csv_artifact.files():
                    local_path = os.path.join(run_dir, f.name)
                    if skip_if_exists and os.path.isfile(local_path):
                        continue
                    f.download(root=run_dir, replace=True)
            except Exception as e:
                errors.append(f"eval_csv error: {e}")

    # ── download eval_h5 ──────────────────────────────────────────────────
    if download_h5 and h5_artifacts:
        latest_h5 = h5_artifacts[-1]
        if not (skip_if_exists and os.path.isfile(h5_local)):
            try:
                for f in latest_h5.files():
                    f.download(root=run_dir, replace=True)
            except Exception as e:
                errors.append(f"eval_h5 error: {e}")

    # ── save run config ───────────────────────────────────────────────────
    if download_run_config:
        if not (skip_if_exists and os.path.isfile(config_path)):
            try:
                filtered = _filter_run_config(run.config)
                # method: config.json project_display_names 기반 축약 이름 주입
                method = _PROJECT_DISPLAY_NAMES.get(project, project) if project else None
                if method:
                    filtered = {"method": method, **filtered}
                with open(config_path, "w", encoding="utf-8") as fp:
                    json.dump(filtered, fp, indent=2, ensure_ascii=False)
            except Exception as e:
                errors.append(f"run_config error: {e}")

    if errors:
        return _RunResult(run_name=run.name, status="error", error=" | ".join(errors))
    return _RunResult(run_name=run.name, status="ok")



# Download all runs in a project
# ---------------------------------------------------------------------------


def download_eval_project(
    project: str,
    entity: str = DEFAULT_ENTITY,
    output_dir: str = "wandb_projects",
    download_csv: bool = True,
    download_h5: bool = True,
    download_run_config: bool = True,
    skip_if_exists: bool = True,
    n_workers: int = DEFAULT_NUM_WORKERS,
    filters: dict | None = None,
    per_page: int = 200,
) -> _ProjectSummary:
    """Download eval artifacts from all runs in a project."""
    api = get_api()
    proj_output_dir = os.path.join(output_dir, project)
    os.makedirs(proj_output_dir, exist_ok=True)

    runs = list(api.runs(f"{entity}/{project}", filters=filters or {}, per_page=per_page))
    summary = _ProjectSummary(project=project)

    def _worker(run):
        return _download_run(
            run,
            output_dir=proj_output_dir,
            project=project,
            download_csv=download_csv,
            download_h5=download_h5,
            download_run_config=download_run_config,
            skip_if_exists=skip_if_exists,
        )

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_worker, r): r.name for r in runs}
        with tqdm(total=len(futures), desc=project, unit="run") as pbar:
            for fut in as_completed(futures):
                try:
                    res: _RunResult = fut.result()
                except Exception as e:
                    res = _RunResult(run_name=futures[fut], status="error", error=str(e))

                if res.status == "ok":
                    summary.ok += 1
                elif res.status == "skipped":
                    summary.skipped += 1
                else:
                    summary.errors.append(res)

                pbar.update(1)

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    _exp_names = list(_EXPERIMENTS.keys())
    _exp_choices_str = ", ".join(_exp_names) if _exp_names else "none defined"
    parser = argparse.ArgumentParser(
        description="W&B eval artifact downloader (eval_csv / eval_h5 / run_config)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
examples:
  python results/eval_downloader.py
  python results/eval_downloader.py --experiment allseen
  python results/eval_downloader.py --experiment unseen
  python results/eval_downloader.py --no-h5
  python results/eval_downloader.py --no-run-config
  python results/eval_downloader.py --output wandb_projects --finished-only
  python results/eval_downloader.py --projects aaai27_eval_cpcgrl --workers 4

available experiments: {_exp_choices_str}
        """,
    )
    parser.add_argument(
        "--experiment",
        choices=_exp_names if _exp_names else None,
        default=None,
        metavar="EXPERIMENT",
        help=f"Experiment group to download (choices: {_exp_choices_str}). "
             "Overrides --projects.",
    )
    parser.add_argument(
        "--projects",
        nargs="+",
        default=None,
        metavar="PROJECT",
        help="Explicit W&B project list. Ignored when --experiment is set. "
             "Defaults to all projects across all experiments.",
    )
    parser.add_argument(
        "--entity",
        default=DEFAULT_ENTITY,
        help=f"W&B entity (default: {DEFAULT_ENTITY})",
    )
    parser.add_argument(
        "--output",
        default=_DEFAULT_OUTPUT,
        help="Local root path to save downloads (default: results/wandb_projects)",
    )
    parser.add_argument(
        "--no-csv",
        action="store_true",
        help="Skip downloading eval_csv artifact (ctrl_sim/results/diversity/summary CSVs)",
    )
    parser.add_argument(
        "--no-h5",
        action="store_true",
        default=True,
        help="Skip downloading eval_h5 artifact / eval.h5 (default)",
    )
    parser.add_argument(
        "--h5",
        action="store_false",
        dest="no_h5",
        help="Include eval_h5 artifact (eval.h5) in download",
    )
    parser.add_argument(
        "--no-run-config",
        action="store_true",
        help="Skip saving run_config.json (filtered W&B run config)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing files",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=_DEFAULT_NUM_WORKERS,
        help="Number of parallel download threads (default: config.json wandb.num_workers)",
    )
    parser.add_argument(
        "--finished-only",
        action="store_true",
        help="Only download runs with state=finished",
    )
    return parser.parse_args()


def _print_summary(summaries: list[_ProjectSummary]) -> None:
    """Log per-project download results as a table."""
    COL_W = [40, 8, 10, 8]
    headers = ["Project", "OK", "Skipped", "Error"]
    sep = "+" + "+".join("-" * (w + 2) for w in COL_W) + "+"
    fmt = "| " + " | ".join(f"{{:<{w}}}" for w in COL_W) + " |"

    lines = [sep, fmt.format(*headers), sep]
    for s in summaries:
        lines.append(fmt.format(s.project, s.ok, s.skipped, len(s.errors)))
    lines.append(sep)
    logger.info("\n" + "\n".join(lines))

    # Error details
    any_error = any(s.errors for s in summaries)
    if any_error:
        logger.warning("[!] Error details:")
        for s in summaries:
            for r in s.errors:
                logger.warning("  [%s] %s — %s", s.project, r.run_name, r.error)


def main():
    args = parse_args()
    output_dir = _resolve_output_dir(args.output)
    logger.info("output cache: %s", output_dir)

    # Resolve project list: --experiment > --projects > all experiments
    if args.experiment:
        projects = _EXPERIMENTS.get(args.experiment, [])
        if not projects:
            logger.error("Experiment '%s' has no target_projects defined.", args.experiment)
            sys.exit(1)
        logger.info("Experiment: %s  (%d projects)", args.experiment, len(projects))
    elif args.projects:
        projects = args.projects
    else:
        projects = TARGET_PROJECTS

    filters = {"state": "finished"} if args.finished_only else None

    summaries: list[_ProjectSummary] = []
    for project in projects:
        summary = download_eval_project(
            project=project,
            entity=args.entity,
            output_dir=output_dir,
            download_csv=not args.no_csv,
            download_h5=not args.no_h5,
            download_run_config=not args.no_run_config,
            skip_if_exists=not args.force,
            n_workers=args.workers,
            filters=filters,
        )
        summaries.append(summary)

    _print_summary(summaries)


if __name__ == "__main__":
    main()
