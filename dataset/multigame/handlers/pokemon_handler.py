"""
dataset/multigame/handlers/pokemon_handler.py
==============================================
POKEMON dataset handler.

POKEMON stores all maps and labels in a single NPY file.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional

import numpy as np

from ..base import BaseGameHandler, GameSample, GameTag, TileLegend
from .fdm_game.pokemon import POKEMONPreprocessor, make_legend, POKEMON_PALETTE  # noqa: F401

_DEFAULT_POKEMON_ROOT = Path(__file__).parent.parent.parent / "five-dollar-model"



class POKEMONHandler(BaseGameHandler):
    """
    POKEMON handler.

    Parameters
    ----------
    root : root path of the POKEMON dataset (default: dataset/five-dollar-model)
    npy_name : NPY filename (default: datasets/maps_noaug.npy)
    """

    def __init__(
        self,
        root: Path | str = _DEFAULT_POKEMON_ROOT,
        npy_name: str = "datasets/maps_noaug.npy",
        handler_config: Optional[Any] = None,
    ) -> None:
        self._root = Path(root)
        self._handler_config = handler_config
        npy_path = self._root / npy_name

        if not npy_path.exists():
            raise FileNotFoundError(f"POKEMON NPY not found: {npy_path}")

        # NPY file load
        data = np.load(npy_path, allow_pickle=True)
        if data.ndim == 0:
            data = data.item()

        if not isinstance(data, dict):
            raise ValueError(f"Expected dict in NPY, got {type(data)}")

        self._images: List[np.ndarray] = data.get("images", [])
        self._labels: List[str] = data.get("labels", [])
        self._preprocessor = POKEMONPreprocessor()
        self._legend: TileLegend = make_legend()

        if len(self._images) != len(self._labels):
            raise ValueError(
                f"Mismatch: {len(self._images)} images, {len(self._labels)} labels"
            )

    @property
    def game_tag(self) -> str:
        return GameTag.POKEMON

    @property
    def game_dir(self) -> Path:
        return self._root

    def list_entries(self) -> List[str]:
        """Return an NPY index as source_id, limited to 1,000 entries."""
        max_samples = 1000
        total = len(self._images)
        limit = min(total, max_samples)
        return [f"pokemon_{i:04d}" for i in range(limit)]

    def load_sample(self, source_id: str, order: Optional[int] = None) -> GameSample:
        """
        Return the GameSample for a source_id (for example, "pokemon_0000").
        """
        # source_id in  index extract
        try:
            idx = int(source_id.split("_")[1])
        except (ValueError, IndexError):
            raise KeyError(f"Invalid source_id format: {source_id!r}")

        if idx < 0 or idx >= len(self._images):
            raise KeyError(f"Index out of range: {idx}")

        onehot_map = self._images[idx]
        instruction = self._labels[idx]

        sample = self._preprocessor.process_pokemon_sample(
            onehot_map=onehot_map,
            source_id=source_id,
            instruction=instruction,
        )
        if order is not None:
            sample.order = order

        return sample

    def list_entries_with_filtering(self, max_tile_ratio: Optional[float] = None, max_tile_count: Optional[int] = None) -> tuple[List[str], int, int]:
        """
        Apply filtering and return only valid entries.

        Parameters
        ----------
        max_tile_ratio : Optional[float]
            Maximum ratio that one tile may occupy in the unpadded 10x10 map.
            If None, use the value from config.
        max_tile_count : Optional[int]
            Maximum count of one tile in the padded 16x16 map.
            If None, use the value from config.

        Returns
        -------
        tuple[List[str], int, int]
            (valid source IDs, count removed by max_tile_ratio, count removed by max_tile_count)
        """
        # Read defaults from config
        if max_tile_ratio is None:
            max_tile_ratio = self._handler_config.pokemon.max_tile_ratio if self._handler_config else 1.0
        if max_tile_count is None:
            max_tile_count = self._handler_config.pokemon.max_tile_count if self._handler_config else 256

        valid_ids = []
        filtered_by_ratio = 0
        filtered_by_count = 0
        max_samples = 1000  # Limit to at most 1,000 samples

        # Remove duplicate "house on the beach" samples: exclude the final seven (indices 874-880)
        excluded_duplicates = set(range(874, 881))

        for i in range(len(self._images)):
            # max_samples reach check
            if len(valid_ids) >= max_samples:
                break

            if i in excluded_duplicates:
                continue

            onehot_map = self._images[i]

            # Step 1: max_tile_ratio filtering on the unpadded 10x10 map
            if not self._preprocessor.is_valid_pokemon_map(onehot_map, max_tile_ratio):
                filtered_by_ratio += 1
                continue

            # Step 2: tile-count filtering on the padded 16x16 map
            map_10x10 = self._preprocessor.transform_pokemon_onehot(onehot_map)
            padded_map = self._preprocessor.pad_to_16x16(map_10x10)

            # Check tile counts in the padded map
            tile_counts = {}
            for val in padded_map.flatten():
                tile_counts[val] = tile_counts.get(val, 0) + 1
            max_count = max(tile_counts.values()) if tile_counts else 0
            if max_count >= max_tile_count:
                filtered_by_count += 1
                continue

            valid_ids.append(f"pokemon_{i:04d}")
            # max_samples reach check
            if len(valid_ids) >= max_samples:
                break

        return valid_ids, filtered_by_ratio, filtered_by_count

    def __iter__(self):
        """Iterate over all samples."""
        for i, source_id in enumerate(self.list_entries()):
            yield self.load_sample(source_id, order=i)

    def __len__(self) -> int:
        return len(self._images)

    def __repr__(self) -> str:
        return f"POKEMONHandler(samples={len(self._images)})"
