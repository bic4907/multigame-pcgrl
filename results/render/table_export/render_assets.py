from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from .csv_rows import (
    condition_bucket_key,
    condition_contrast_pair_rows_from_run_csv,
    h5_folder_name,
    reward_enum_condition_rows_from_run_csv,
    row_key,
)
from .models import RunResult
from .utils import reward_enum_value, safe_slug, unique_methods


def _render_state_png(
    h5_path: Path,
    row: dict[str, str],
    seed_i: int,
    out_path: Path,
    tile_size: int,
    renderer,
) -> Optional[Path]:
    import h5py

    folder_name = h5_folder_name(row)
    seed_name = f"seed_{seed_i}"
    with h5py.File(str(h5_path), "r") as h5:
        state_path = f"{folder_name}/{seed_name}/state"
        if state_path not in h5:
            return None
        state = h5[state_path][()]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    renderer.render(
        game=row.get("game", ""),
        level=state,
        tile_size=tile_size,
        show_tile_numbers=False,
    ).save(str(out_path))
    return out_path


def _load_dataset_samples_for_gt():
    from dataset.multigame import MultiGameDataset
    from instruct_rl.utils.dataset_loader_helpers.preprocessing import apply_tile_offset, preprocess_samples

    ds = MultiGameDataset(use_tile_mapping=True)
    samples = list(ds)
    samples = preprocess_samples(samples, longtail_cut=True)
    return apply_tile_offset(samples, 1)


def _find_dataset_sample(row: dict[str, str], samples: list) -> Optional[Any]:
    game = row.get("game")
    instruction = row.get("instruction")
    reward_enum_raw = row.get("reward_enum")
    reward_enum = int(float(reward_enum_raw)) if reward_enum_raw not in (None, "") else None

    for sample in samples:
        if sample.game != game:
            continue
        if reward_enum is not None and sample.meta.get("reward_enum") != reward_enum:
            continue
        if instruction and sample.instruction == instruction:
            return sample

    for sample in samples:
        if sample.game == game and (reward_enum is None or sample.meta.get("reward_enum") == reward_enum):
            return sample
    return None


def _render_dataset_png(
    row: dict[str, str],
    samples: list,
    out_path: Path,
    tile_size: int,
    renderer,
) -> Optional[Path]:
    sample = _find_dataset_sample(row, samples)
    if sample is None:
        return None

    out_path.parent.mkdir(parents=True, exist_ok=True)
    renderer.render(
        game=row.get("game", ""),
        level=sample.array,
        tile_size=tile_size,
        show_tile_numbers=False,
    ).save(str(out_path))
    return out_path


def _combine_pngs(paths: list[Path], out_path: Path, gap: int = 8) -> Optional[Path]:
    from PIL import Image

    images = [Image.open(path).convert("RGBA") for path in paths if path is not None and path.exists()]
    if not images:
        return None
    width = sum(image.width for image in images) + gap * (len(images) - 1)
    height = max(image.height for image in images)
    canvas = Image.new("RGBA", (width, height), (255, 255, 255, 0))
    x = 0
    for image in images:
        y = (height - image.height) // 2
        canvas.alpha_composite(image, (x, y))
        x += image.width + gap
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.convert("RGB").save(str(out_path))
    return out_path


def _render_pair_state_png(
    h5_path: Path,
    row: dict,
    seed_i: int,
    out_path: Path,
    tile_size: int,
    renderer,
) -> Optional[Path]:
    side_paths = []
    for member_i, member in enumerate(row.get("_pair_members", [])):
        side_path = out_path.parent / f"{out_path.stem}_{member_i}.png"
        rendered = _render_state_png(h5_path, member, seed_i, side_path, tile_size, renderer)
        if rendered is not None:
            side_paths.append(rendered)
    return _combine_pngs(side_paths, out_path)


def _render_pair_dataset_png(
    row: dict,
    samples: list,
    out_path: Path,
    tile_size: int,
    renderer,
) -> Optional[Path]:
    side_paths = []
    for member_i, member in enumerate(row.get("_pair_members", [])):
        side_path = out_path.parent / f"{out_path.stem}_{member_i}.png"
        rendered = _render_dataset_png(member, samples, side_path, tile_size, renderer)
        if rendered is not None:
            side_paths.append(rendered)
    return _combine_pngs(side_paths, out_path)


def render_image_table_assets(
    run_results: list[RunResult],
    output_dir: Path,
    *,
    max_rows_per_condition: int = 4,
    seed_i: int = 0,
    tile_size: int = 12,
    condition_targets: Optional[list[dict]] = None,
    num_episodes: int = 10,
) -> tuple[list[dict[str, str]], dict[tuple[str, str], Path], dict[str, Path]]:
    rows_by_reward_enum_condition_game = {}
    if condition_targets:
        indexed_targets = [
            {**target, "_target_i": target_i}
            for target_i, target in enumerate(condition_targets, start=1)
        ]
        base_runs_by_reward: dict[int, RunResult] = {}
        for run_result in run_results:
            if run_result.csv_dir is None or run_result.reward_enum is None:
                continue
            base_runs_by_reward.setdefault(int(run_result.reward_enum), run_result)
        candidate_rows = []
        for reward_enum, base_run in sorted(base_runs_by_reward.items()):
            reward_targets = [
                target for target in indexed_targets
                if int(target["reward_enum"]) == reward_enum
            ]
            if reward_targets:
                candidate_rows.extend(
                    condition_contrast_pair_rows_from_run_csv(base_run, reward_targets, num_episodes)
                )
        for row in candidate_rows:
            key = (reward_enum_value(row), condition_bucket_key(row), row.get("row_i"))
            rows_by_reward_enum_condition_game.setdefault(key, row)
    else:
        for run_result in run_results:
            if run_result.csv_dir is None:
                continue
            candidate_rows = reward_enum_condition_rows_from_run_csv(run_result, max_rows_per_condition)
            for row in candidate_rows:
                key = (
                    reward_enum_value(row),
                    condition_bucket_key(row),
                    row.get("_condition_bucket_label", ""),
                    row.get("game"),
                    row.get("row_i"),
                )
                rows_by_reward_enum_condition_game.setdefault(key, row)
    rows = list(rows_by_reward_enum_condition_game.values())
    if not rows:
        return [], {}, {}

    from dataset.multigame.render import GameLevelRenderer

    renderer = GameLevelRenderer()
    images_dir = output_dir / "images"
    method_images: dict[tuple[str, str], Path] = {}
    dataset_images: dict[str, Path] = {}
    run_by_method_reward_enum = {
        (run_result.method, run_result.reward_enum): run_result
        for run_result in run_results
        if run_result.reward_enum is not None
    }

    dataset_samples = None
    for row in rows:
        key = row_key(row)
        reward_enum = reward_enum_value(row)

        for method in unique_methods(run_results):
            run_result = run_by_method_reward_enum.get((method, reward_enum))
            if run_result is None or run_result.h5_path is None:
                continue
            out_path = images_dir / key / f"{safe_slug(run_result.method)}.png"
            episode_seed = seed_i
            if row.get("_pair_members"):
                rendered = _render_pair_state_png(
                    run_result.h5_path,
                    row,
                    episode_seed,
                    out_path,
                    tile_size,
                    renderer,
                )
            else:
                rendered = _render_state_png(
                    run_result.h5_path,
                    row,
                    episode_seed,
                    out_path,
                    tile_size,
                    renderer,
                )
            if rendered is not None:
                method_images[(key, run_result.method)] = rendered.relative_to(output_dir)

        if dataset_samples is None:
            dataset_samples = _load_dataset_samples_for_gt()
        gt_path = images_dir / key / "dataset.png"
        if row.get("_pair_members"):
            rendered_gt = _render_pair_dataset_png(row, dataset_samples, gt_path, tile_size, renderer)
        else:
            rendered_gt = _render_dataset_png(row, dataset_samples, gt_path, tile_size, renderer)
        if rendered_gt is not None:
            dataset_images[key] = rendered_gt.relative_to(output_dir)

    return rows, method_images, dataset_images
