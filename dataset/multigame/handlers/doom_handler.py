"""
dataset/multigame/handlers/doom_handler.py
==========================================
DOOM level dataset handler (TheVGLC based).
Doom map preprocessing handler.
- Automatic file discovery
- Slicing large maps into 16x16 regions
- Tile mapping and conversion
"""
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Optional
from .vglc_games.doom import DoomPreprocessor, DOOM_PALETTE, make_legend
from ..base import (
    BaseGameHandler,
    GameSample,
    GameTag,
    TileLegend,
    enforce_char_grid_top_left_16x16,
    enforce_top_left_16x16,
)
_DEFAULT_VGLC_ROOT = Path(__file__).parent.parent.parent / "TheVGLC"
_DEFAULT_DOOM_ROOT = _DEFAULT_VGLC_ROOT / "Doom"
_DEFAULT_DOOM2_ROOT = _DEFAULT_VGLC_ROOT / "Doom2"
class DoomHandler(BaseGameHandler):
    """
    Doom level handler.
    Automatically discover, slice, and convert levels from the TheVGLC Doom dataset.
    Parameters
    ----------
    root : Path | str
        Doom level directory containing *.txt files.
    handler_config : Optional[Any]
        HandlerConfig object containing the doom_slicing settings.
    """
    def __init__(
        self,
        root: Path | str = _DEFAULT_DOOM_ROOT,
        handler_config: Optional[Any] = None,
    ) -> None:
        self._root = Path(root)
        self._preprocessor = DoomPreprocessor()
        self._legend: TileLegend = make_legend()
        self._handler_config = handler_config
        self._entries: Optional[List[str]] = None  # lazy
        self._sliced_cache: Dict[str, GameSample] = {}
    @property
    def game_tag(self) -> str:
        return GameTag.DOOM
    @property
    def game_dir(self) -> Path:
        return self._root
    def _discover(self) -> List[str]:
        """Discover and slice Doom level files."""
        if not self._root.exists():
            return []
        # VGLC layout: prefer the Processed directory, otherwise use the root
        processed = self._root / "Processed"
        if processed.exists():
            txt_files = sorted(processed.glob("*.txt"))
        else:
            txt_files = sorted(self._root.glob("*.txt"))
        txt_files = [p for p in txt_files if not p.name.lower().startswith("readme")]
        # Doom  before  for : discover_and_process call
        if hasattr(self._preprocessor, "discover_and_process"):
            return self._preprocessor.discover_and_process(
                files=txt_files,
                config=self._handler_config,
                game_tag=self.game_tag,
                legend=self._legend,
                cache=self._sliced_cache,
            )
        return [str(p) for p in txt_files]
    def list_entries(self) -> List[str]:
        if self._entries is None:
            self._entries = self._discover()
        return self._entries
    def load_sample(self, source_id: str, order: Optional[int] = None) -> GameSample:
        # Return cached data when available (including sliced data)
        if source_id in self._sliced_cache:
            sample = self._sliced_cache[source_id]
            if order is not None:
                sample.order = order
            return sample
        # Parse source_id when the sample is not cached
        # source_id has the form "path/to/file.txt|slice_idx"
        if "|" in source_id:
            file_path, slice_idx_str = source_id.rsplit("|", 1)
            try:
                slice_idx = int(slice_idx_str)
            except ValueError:
                raise ValueError(
                    f"Invalid source_id format: {source_id!r}. "
                    f"Expected 'path|slice_idx'"
                )
        else:
            file_path = source_id
            slice_idx = 0
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Doom level file not found: {file_path}")
        text = path.read_text(encoding="utf-8", errors="replace")
        char_grid = self._preprocessor.parse_txt(text)
        # Apply configured slicing, or use the complete map
        if self._handler_config and hasattr(self._handler_config, "doom"):
            sliced_maps = self._preprocessor.slice_large_map(
                char_grid,
                empty_max=self._handler_config.doom.empty_max,
                floor_empty_max=self._handler_config.doom.floor_empty_max,
            )
            if slice_idx >= len(sliced_maps):
                raise IndexError(
                    f"slice index {slice_idx} out of range for "
                    f"{path.name} ({len(sliced_maps)} slices)"
                )
            sliced_data = sliced_maps[slice_idx]
            char_grid = sliced_data["map"]
        else:
            # No slicing configuration: pad or crop the complete map to 16x16
            if slice_idx != 0:
                raise IndexError(f"slice_idx {slice_idx} invalid without slicing config")
        array = self._preprocessor.transform(char_grid)
        array = enforce_top_left_16x16(
            array, game=self.game_tag, source_id=source_id
        )
        char_grid = enforce_char_grid_top_left_16x16(char_grid)
        sample = GameSample(
            game=self.game_tag,
            source_id=source_id,
            array=array,
            char_grid=char_grid,
            legend=self._legend,
            instruction=None,
            order=order,
            meta={"file": str(path.name), "game_dir": str(self._root)},
        )
        # cache in  save
        self._sliced_cache[source_id] = sample
        return sample
    def __repr__(self) -> str:
        return f"DoomHandler(levels={len(self.list_entries())})"
DOOM_PALETTE_DICT = DOOM_PALETTE
