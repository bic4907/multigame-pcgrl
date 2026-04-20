"""W&B 테이블/파일 다운로더 클래스."""

import os
import json
import shutil
import tempfile
import uuid
import re
from os.path import basename
from copy import deepcopy
from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Callable

import pandas as pd
from tqdm import tqdm
import wandb

import logging

log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logger = logging.getLogger(basename(__file__))
logger.setLevel(getattr(logging, log_level, logging.INFO))

from .config import (
    FLATTEN_KEYS,
    REMOVE_KEYS,
    DEFAULT_ENTITY,
    API_TIMEOUT,
    DEFAULT_NUM_WORKERS,
)

# ---------------------------------------------------------------------------
# 모듈 레벨 API 인스턴스
# ---------------------------------------------------------------------------
_api: Optional[wandb.Api] = None


def get_api(timeout: int = API_TIMEOUT) -> wandb.Api:
    """싱글턴 W&B API 인스턴스를 반환한다."""
    global _api
    if _api is None:
        logger.info(f"Loading W&B API with timeout={timeout}s")
        _api = wandb.Api(timeout=timeout)
        logger.info("W&B API loaded")
    return _api


# ---------------------------------------------------------------------------
# Config 처리 헬퍼
# ---------------------------------------------------------------------------

def _process_config(
    config: dict,
    run,
    flatten_keys: list[str] = FLATTEN_KEYS,
    remove_keys: list[str] = REMOVE_KEYS,
) -> dict:
    """run.config 를 평탄화하고 불필요한 키를 제거한 dict 를 반환한다."""
    config_dict = deepcopy(config)
    config_dict["run_id"] = run.id

    for key in flatten_keys:
        sub_dict = config_dict.get(key)
        if sub_dict is None:
            continue
        if isinstance(sub_dict, str):
            sub_dict = eval(sub_dict)
        for sub_key, value in sub_dict.items():
            config_dict[f"{key}.{sub_key}"] = value

    for key in remove_keys:
        config_dict.pop(key, None)

    return {k: str(v) for k, v in config_dict.items()}


# ---------------------------------------------------------------------------
# 멀티프로세스 워커 (기존 download() 메서드용)
# ---------------------------------------------------------------------------


def _run_worker(args: tuple) -> None:
    """멀티프로세싱 Pool 에서 실행되는 단일 run 다운로드 워커.

    Parameters
    ----------
    args : tuple
        (run_id, project_name, ctx) 형태.
        ctx 는 entity, output_dir, target_files 등을 담은 dict.
    """
    run_id, project_name, ctx = args

    api = get_api()
    entity = ctx.get("entity", DEFAULT_ENTITY)
    output_dir_base = ctx.get("output_dir", "results")
    target_files = ctx.get("target_files", [])
    flatten_keys = ctx.get("flatten_keys", FLATTEN_KEYS)
    remove_keys = ctx.get("remove_keys", REMOVE_KEYS)
    tmp_root = ctx.get("tmp_root", tempfile.gettempdir())

    full_run_path = f"{entity}/{project_name}/{run_id}"
    run = api.run(full_run_path)
    run_name = run.name

    # 폴더명: config.exp_dir 의 basename, 없으면 run_id
    folder_name = os.path.basename(run.config.get("exp_dir", run_id))
    run_output_dir = os.path.join(output_dir_base, project_name, folder_name)

    if not target_files:
        logger.warning(f"[{run_id}] target_files 가 비어 있습니다 — skipping")
        return

    if all(
        os.path.exists(os.path.join(run_output_dir, f"{t}.csv"))
        for t in target_files
    ):
        logger.debug(f"{run_output_dir} already complete — skipping")
        return

    temp_download_dir = os.path.join(tmp_root, run_output_dir, f"tmp_{uuid.uuid4().hex}")

    try:
        config_dict = _process_config(run.config, run, flatten_keys, remove_keys)
        os.makedirs(run_output_dir, exist_ok=True)

        file_map: dict = {name: None for name in target_files}
        for f in run.files():
            if f.name.endswith(".table.json"):
                fname = os.path.basename(f.name)  # e.g. results_2709_abc.table.json
                for target_name in target_files:
                    if fname.startswith(target_name):
                        file_map[target_name] = f

        os.makedirs(temp_download_dir, exist_ok=True)

        for key, f in file_map.items():
            if f is None:
                logger.warning(f"[{run_name}] '{key}' table file not found — skipping")
                continue

            local_json = os.path.join(run_output_dir, f"{key}.json")
            f.download(root=temp_download_dir, replace=True)
            downloaded = os.path.join(temp_download_dir, f.name)
            os.rename(downloaded, local_json)

            with open(local_json, "r") as fp:
                data_dict = json.load(fp)

            df = pd.DataFrame(data_dict["data"], columns=data_dict["columns"])
            for k, v in config_dict.items():
                df[f"config.{k}"] = v
            df.to_csv(os.path.join(run_output_dir, f"{key}.csv"), index=False)

        with open(os.path.join(run_output_dir, "config.json"), "w") as fp:
            json.dump(run.config, fp, indent=2, ensure_ascii=False)

    except Exception as e:
        logger.error(f"[{run_name}] error: {e}")
    finally:
        shutil.rmtree(temp_download_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# 메인 다운로더 클래스
# ---------------------------------------------------------------------------


class WandbTableDownloader:
    """W&B 프로젝트에서 테이블 파일을 다운로드하는 클래스.

    Parameters
    ----------
    entity : str
        W&B 엔티티(팀/유저). (기본값: ``DEFAULT_ENTITY``)
    output_dir : str
        기본 결과물이 저장될 디렉토리 경로. (기본값: ``"results"``)
    target_files : list[str], optional
        ``download()`` 에서 사용할 테이블 이름 목록.
        ``download_project()`` 를 쓸 때는 불필요.
    num_workers : int
        병렬 다운로드 워커 수. (기본값: ``DEFAULT_NUM_WORKERS``)
    tmp_root : str
        임시 다운로드 폴더 루트. (기본값: OS 임시 디렉토리)
    flatten_keys : list[str], optional
        config 에서 평탄화할 키. (기본값: ``FLATTEN_KEYS``)
    remove_keys : list[str], optional
        config 에서 제거할 키. (기본값: ``REMOVE_KEYS``)
    """

    def __init__(
        self,
        entity: str = DEFAULT_ENTITY,
        output_dir: str = "results",
        target_files: Optional[list[str]] = None,
        num_workers: int = DEFAULT_NUM_WORKERS,
        tmp_root: str = None,
        flatten_keys: Optional[list[str]] = None,
        remove_keys: Optional[list[str]] = None,
    ):
        self.entity = entity
        self.output_dir = output_dir
        self.target_files = target_files
        self.num_workers = num_workers
        self.tmp_root = tmp_root or os.path.join(tempfile.gettempdir(), "wandb_download")
        self.flatten_keys = flatten_keys if flatten_keys is not None else FLATTEN_KEYS
        self.remove_keys = remove_keys if remove_keys is not None else REMOVE_KEYS

    # ------------------------------------------------------------------ #
    #  내부 헬퍼
    # ------------------------------------------------------------------ #

    def _build_ctx(self) -> dict:
        """기존 download() 워커에 전달할 컨텍스트 dict."""
        return {
            "entity": self.entity,
            "output_dir": self.output_dir,
            "target_files": self.target_files or [],
            "flatten_keys": self.flatten_keys,
            "remove_keys": self.remove_keys,
            "tmp_root": self.tmp_root,
        }

    # ------------------------------------------------------------------ #
    #  download_project  (download_unseen_games 등에서 사용)
    # ------------------------------------------------------------------ #

    def download_project(
        self,
        project: str,
        table_patterns: dict[str, str],
        output_dir: Optional[str] = None,
        extra_cols_fn: Optional[Callable] = None,
        dir_name_fn: Optional[Callable] = None,
        n_workers: int = DEFAULT_NUM_WORKERS,
        filters: Optional[dict] = None,
        per_page: int = 200,
        skip_if_exists: bool = True,
    ) -> Optional[str]:
        """프로젝트의 모든 run 에서 테이블을 다운로드하고 병합 CSV 를 반환한다.

        Parameters
        ----------
        project : str
            W&B 프로젝트 이름.
        table_patterns : dict[str, str]
            ``{출력이름: 검색패턴}`` 매핑.
            예: ``{"results": "results"}``
            → ``media/table/results*.table.json`` 에 매칭.
        output_dir : str, optional
            결과 저장 경로. 미지정 시 ``self.output_dir`` 사용.
        extra_cols_fn : callable, optional
            ``(config: dict, run) -> dict`` 형태.
            반환된 dict 가 각 행에 컬럼으로 추가된다.
        dir_name_fn : callable, optional
            ``(run) -> str`` 형태. run 별 저장 폴더명을 결정한다.
            미지정 시 ``config.exp_dir`` 의 basename 을 사용한다.
        n_workers : int
            병렬 다운로드 스레드 수.
        filters : dict, optional
            W&B run 필터. 예: ``{"state": "finished"}``.
        per_page : int
            W&B API 페이지당 run 수.
        skip_if_exists : bool
            True 면 이미 CSV 가 있는 run 은 건너뛴다.

        Returns
        -------
        str or None
            병합된 CSV 경로. 없으면 None.
        """


        if dir_name_fn is None:
            dir_name_fn = lambda run: os.path.basename(
                run.config.get("exp_dir", run.name)
            )
        output_dir = output_dir or self.output_dir
        os.makedirs(output_dir, exist_ok=True)
        tmp_root = self.tmp_root
        os.makedirs(tmp_root, exist_ok=True)

        api = get_api()
        run_path = f"{self.entity}/{project}"
        runs = api.runs(run_path, filters=filters or {}, per_page=per_page)
        run_list = list(runs)

        logger.info(
            f"[{project}] {len(run_list)}개 run 다운로드 시작 "
            f"(workers={n_workers})"
        )

        def _process_one_run(run):
            """단일 run 을 처리하는 내부 함수 (ThreadPoolExecutor 용)."""
            folder_name = dir_name_fn(run)
            run_dir = os.path.join(output_dir, folder_name)

            # 스킵 체크
            if skip_if_exists and all(
                os.path.isfile(os.path.join(run_dir, f"{name}.csv"))
                for name in table_patterns
            ):
                logger.debug(f"[{run.name}] already exists — skipping")
                return

            os.makedirs(run_dir, exist_ok=True)
            tmp_dir = os.path.join(
                tmp_root, project, run.id, f"tmp_{uuid.uuid4().hex}"
            )
            os.makedirs(tmp_dir, exist_ok=True)

            try:
                # extra columns
                extra: dict = {}
                if extra_cols_fn is not None:
                    extra = extra_cols_fn(run.config, run)

                # W&B 파일 탐색
                wandb_files: dict = {name: None for name in table_patterns}
                for f in run.files():
                    if f.name.endswith(".table.json"):
                        fname = os.path.basename(f.name)  # e.g. results_2709_abc.table.json
                        for name, pattern in table_patterns.items():
                            if fname.startswith(pattern):
                                wandb_files[name] = f

                for name, wf in wandb_files.items():
                    if wf is None:
                        logger.warning(
                            f"[{run.name}] '{name}' table not found — skipping"
                        )
                        continue

                    wf.download(root=tmp_dir, replace=True)
                    downloaded = os.path.join(tmp_dir, wf.name)
                    local_json = os.path.join(run_dir, f"{name}.json")
                    os.rename(downloaded, local_json)

                    with open(local_json, "r") as fp:
                        data = json.load(fp)

                    df = pd.DataFrame(data["data"], columns=data["columns"])
                    for k, v in extra.items():
                        df[k] = v
                    df.to_csv(os.path.join(run_dir, f"{name}.csv"), index=False)

                # config 저장
                with open(os.path.join(run_dir, "config.json"), "w") as fp:
                    json.dump(run.config, fp, indent=2, ensure_ascii=False)

            except Exception as e:
                logger.error(f"[{run.name}] error: {e}")
            finally:
                shutil.rmtree(tmp_dir, ignore_errors=True)

        # ThreadPoolExecutor 사용 (extra_cols_fn 직렬화 문제 없음)
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = [executor.submit(_process_one_run, r) for r in run_list]
            for fut in tqdm(
                as_completed(futures), total=len(futures), desc=project
            ):
                fut.result()  # 예외 전파

        logger.info(f"[{project}] 다운로드 완료")

        return self.combine_csvs(
            output_dir=output_dir,
            table_patterns=table_patterns,
            project=project,
        )

    # ------------------------------------------------------------------ #
    #  combine_csvs
    # ------------------------------------------------------------------ #

    def combine_csvs(
        self,
        output_dir: Optional[str] = None,
        table_patterns: Optional[dict[str, str]] = None,
        project: str = "",
    ) -> Optional[str]:
        """output_dir 아래의 run 별 CSV 를 하나의 병합 CSV 로 합친다.

        Parameters
        ----------
        output_dir : str, optional
            run 별 CSV 가 저장된 디렉토리. 미지정 시 ``self.output_dir``.
        table_patterns : dict[str, str], optional
            ``{출력이름: 패턴}`` 매핑. 키만 사용한다.
        project : str
            로그용 프로젝트 이름.

        Returns
        -------
        str or None
            마지막으로 생성된 병합 CSV 경로. 없으면 None.
        """
        output_dir = output_dir or self.output_dir
        if table_patterns is None:
            logger.warning("table_patterns 가 지정되지 않았습니다.")
            return None

        combined_path = None

        for name in table_patterns:
            all_dfs: list[pd.DataFrame] = []

            for entry in os.listdir(output_dir):
                csv_path = os.path.join(output_dir, entry, f"{name}.csv")
                if os.path.isfile(csv_path):
                    try:
                        all_dfs.append(pd.read_csv(csv_path))
                    except Exception as e:
                        logger.warning(f"CSV 읽기 실패 {csv_path}: {e}")

            if all_dfs:
                merged = pd.concat(all_dfs, ignore_index=True)
                combined_path = os.path.join(output_dir, f"combined_{name}.csv")
                merged.to_csv(combined_path, index=False)
                logger.info(
                    f"[{project}] 병합 완료: {combined_path} "
                    f"({len(merged)} rows, {len(all_dfs)} runs)"
                )
            else:
                logger.warning(f"[{project}] '{name}' CSV 가 없어 병합을 건너뜁니다.")

        return combined_path

    # ------------------------------------------------------------------ #
    #  download  (기존 인터페이스 — multiprocessing Pool 사용)
    # ------------------------------------------------------------------ #

    def download(self, project_names: str | list[str]) -> None:
        """주어진 프로젝트(들)의 모든 run 을 다운로드한다.

        ``target_files`` 가 ``__init__`` 에서 지정되어 있어야 한다.
        """
        if not self.target_files:
            raise ValueError(
                "download() 를 사용하려면 target_files 를 지정해야 합니다. "
                "(예: WandbTableDownloader(target_files=['results']))"
            )

        if isinstance(project_names, str):
            project_names = [project_names]

        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.tmp_root, exist_ok=True)

        ctx = self._build_ctx()
        api = get_api()

        for project_name in project_names:
            runs = api.runs(f"{self.entity}/{project_name}", per_page=20)
            run_args = [(run.id, project_name, ctx) for run in runs]

            logger.info(
                f"[{project_name}] {len(run_args)}개 run 다운로드 시작 "
                f"(workers={self.num_workers})"
            )

            with Pool(processes=self.num_workers) as pool:
                list(
                    tqdm(
                        pool.imap_unordered(_run_worker, run_args),
                        total=len(run_args),
                        desc=project_name,
                    )
                )

        logger.info("모든 프로젝트 다운로드 완료")

    def download_single_run(self, project_name: str, run_id: str) -> None:
        """단일 run 을 다운로드한다 (기존 인터페이스)."""
        if not self.target_files:
            raise ValueError("download_single_run() 를 사용하려면 target_files 를 지정해야 합니다.")
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.tmp_root, exist_ok=True)
        _run_worker((run_id, project_name, self._build_ctx()))

    # ------------------------------------------------------------------ #
    #  유틸리티
    # ------------------------------------------------------------------ #

    @staticmethod
    def organize_ablation_run_folders(
        ablation_folder_name: str,
        source_root: str = "results",
        target_pattern: str = "results/eval_ablation_{modality}_vipcgrl",
        modality_regex: str = r"_(md-[^_]+)",
    ) -> None:
        """모달리티별로 ablation run 폴더를 재구성한다."""
        source_dir = os.path.join(source_root, ablation_folder_name)

        for root, dirs, _files in os.walk(source_dir):
            for exp_name in dirs:
                match = re.search(modality_regex, exp_name)
                if match:
                    modality_option = match.group(1)
                    new_target_base = target_pattern.format(modality=modality_option)
                    source_path = os.path.join(root, exp_name)
                    target_path = os.path.join(new_target_base, exp_name)
                    os.makedirs(new_target_base, exist_ok=True)
                    try:
                        shutil.move(source_path, target_path)
                    except Exception as e:
                        logger.error(f"폴더 이동 실패 '{exp_name}': {e}")
                else:
                    logger.warning(
                        f"'{exp_name}' 에서 모달리티 옵션을 찾을 수 없습니다."
                    )
            dirs.clear()

        logger.info("폴더 재구성 완료")


# ---------------------------------------------------------------------------
# CLI 진입점
# ---------------------------------------------------------------------------

def main():
    downloader = WandbTableDownloader(
        output_dir="results",
        target_files=["raw", "diversity"],
    )
    downloader.download(
        [
            "eval_cpcgrl",
            "eval_ipcgrl",
            "0722_eval_ablation_vipcgrl",
        ]
    )


if __name__ == "__main__":
    main()
