"""W&B text text/file text to text class."""

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
# text level API text
# ---------------------------------------------------------------------------
_api: Optional[wandb.Api] = None


def get_api(timeout: int = API_TIMEOUT) -> wandb.Api:
    """text W&B API text  returntext."""
    global _api
    if _api is None:
        logger.info(f"Loading W&B API with timeout={timeout}s")
        _api = wandb.Api(timeout=timeout)
        logger.info("W&B API loaded")
    return _api


# ---------------------------------------------------------------------------
# Config process text
# ---------------------------------------------------------------------------

def _process_config(
    config: dict,
    run,
    flatten_keys: list[str] = FLATTEN_KEYS,
    remove_keys: list[str] = REMOVE_KEYS,
) -> dict:
    """run.config   text text text  removetext dict   returntext."""
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
# textprocess text (existing download() text for )
# ---------------------------------------------------------------------------


def _run_worker(args: tuple) -> None:
    """text to text Pool  in  Usagetext  text run download text.

    Parameters
    ----------
    args : tuple
        (run_id, project_name, ctx) form.
        ctx   entity, output_dir, target_files text  text  dict.
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

    # foldertext: config.exp_dir  of  basename, if missing run_id
    folder_name = os.path.basename(run.config.get("exp_dir", run_id))
    run_output_dir = os.path.join(output_dir_base, project_name, folder_name)

    if not target_files:
        logger.warning(f"[{run_id}] target_files   text text — skipping")
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
# text text to text class
# ---------------------------------------------------------------------------


class WandbTableDownloader:
    """W&B text to text in  text text file  downloadtext  class.

    Parameters
    ----------
    entity : str
        W&B text(text/text). (default value: ``DEFAULT_ENTITY``)
    output_dir : str
        default resultwater  savetext directory path. (default value: ``"results"``)
    target_files : list[str], optional
        ``download()``  in  text for text text text name list.
        ``download_project()``   text text  text.
    num_workers : int
        parallel download text text. (default value: ``DEFAULT_NUM_WORKERS``)
    tmp_root : str
        text download folder text. (default value: OS text directory)
    flatten_keys : list[str], optional
        config  in  text text. (default value: ``FLATTEN_KEYS``)
    remove_keys : list[str], optional
        config  in  removetext text. (default value: ``REMOVE_KEYS``)
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
    #  internal text
    # ------------------------------------------------------------------ #

    def _build_ctx(self) -> dict:
        """existing download() text in   before text texttext dict."""
        return {
            "entity": self.entity,
            "output_dir": self.output_dir,
            "target_files": self.target_files or [],
            "flatten_keys": self.flatten_keys,
            "remove_keys": self.remove_keys,
            "tmp_root": self.tmp_root,
        }

    # ------------------------------------------------------------------ #
    #  download_project  (download_unseen_games text in  text for )
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
        """text to text of  text run  in  text text  downloadtext merge CSV   returntext.

        Parameters
        ----------
        project : str
            W&B text to text name.
        table_patterns : dict[str, str]
            ``{textname: text}`` text.
            text: ``{"results": "results"}``
            → ``media/table/results*.table.json``  in  text.
        output_dir : str, optional
            result save path. text text ``self.output_dir`` text for .
        extra_cols_fn : callable, optional
            ``(config: dict, run) -> dict`` form.
            returntext dict   each row in  text as  text text.
        dir_name_fn : callable, optional
            ``(run) -> str`` form. run text save foldertext  text.
            text text ``config.exp_dir``  of  basename   text for text.
        n_workers : int
            parallel download text text.
        filters : dict, optional
            W&B run filter. text: ``{"state": "finished"}``.
        per_page : int
            W&B API text text run text.
        skip_if_exists : bool
            True text  text CSV   with run   text.

        Returns
        -------
        str or None
            mergetext CSV path. if missing None.
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
            f"[{project}] {len(run_list)}text run download start "
            f"(workers={n_workers})"
        )

        def _process_one_run(run):
            """text run   processtext  internal function (ThreadPoolExecutor  for )."""
            folder_name = dir_name_fn(run)
            run_dir = os.path.join(output_dir, folder_name)

            # text text
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

                # W&B file text
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

                # config save
                with open(os.path.join(run_dir, "config.json"), "w") as fp:
                    json.dump(run.config, fp, indent=2, ensure_ascii=False)

            except Exception as e:
                logger.error(f"[{run.name}] error: {e}")
            finally:
                shutil.rmtree(tmp_dir, ignore_errors=True)

        # ThreadPoolExecutor text for  (extra_cols_fn text issue none)
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = [executor.submit(_process_one_run, r) for r in run_list]
            for fut in tqdm(
                as_completed(futures), total=len(futures), desc=project
            ):
                fut.result()  # exception  before text

        logger.info(f"[{project}] download finish")

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
        """output_dir below of  run text CSV   text of  merge CSV  to  text.

        Parameters
        ----------
        output_dir : str, optional
            run text CSV   savetext directory. text text ``self.output_dir``.
        table_patterns : dict[str, str], optional
            ``{textname: text}`` text. text text for text.
        project : str
             to text for  text to text name.

        Returns
        -------
        str or None
            text as  createtext merge CSV path. if missing None.
        """
        output_dir = output_dir or self.output_dir
        if table_patterns is None:
            logger.warning("table_patterns   text text.")
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
                        logger.warning(f"CSV read failure {csv_path}: {e}")

            if all_dfs:
                merged = pd.concat(all_dfs, ignore_index=True)
                combined_path = os.path.join(output_dir, f"combined_{name}.csv")
                merged.to_csv(combined_path, index=False)
                logger.info(
                    f"[{project}] merge finish: {combined_path} "
                    f"({len(merged)} rows, {len(all_dfs)} runs)"
                )
            else:
                logger.warning(f"[{project}] '{name}' CSV   text merge  text.")

        return combined_path

    # ------------------------------------------------------------------ #
    #  download  (existing interface — multiprocessing Pool text for )
    # ------------------------------------------------------------------ #

    def download(self, project_names: str | list[str]) -> None:
        """text text to text(text) of  text run   downloadtext.

        ``target_files``   ``__init__``  in  text text text.
        """
        if not self.target_files:
            raise ValueError(
                "download()   text for text target_files   text text. "
                "(text: WandbTableDownloader(target_files=['results']))"
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
                f"[{project_name}] {len(run_args)}text run download start "
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

        logger.info("text text to text download finish")

    def download_single_run(self, project_name: str, run_id: str) -> None:
        """text run   downloadtext (existing interface)."""
        if not self.target_files:
            raise ValueError("download_single_run()   text for text target_files   text text.")
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.tmp_root, exist_ok=True)
        _run_worker((run_id, project_name, self._build_ctx()))

    # ------------------------------------------------------------------ #
    #  utility
    # ------------------------------------------------------------------ #

    @staticmethod
    def organize_ablation_run_folders(
        ablation_folder_name: str,
        source_root: str = "results",
        target_pattern: str = "results/eval_ablation_{modality}_vipcgrl",
        modality_regex: str = r"_(md-[^_]+)",
    ) -> None:
        """textby ablation run folder  text."""
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
                        logger.error(f"folder move failure '{exp_name}': {e}")
                else:
                    logger.warning(
                        f"'{exp_name}'  in  text text  text  text text."
                    )
            dirs.clear()

        logger.info("folder text finish")


# ---------------------------------------------------------------------------
# CLI text
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
