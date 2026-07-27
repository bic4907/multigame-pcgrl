"""
dataset/multigame/handlers/vglc_handler.py
==========================================
TheVGLC dataset handler.

- Supports game selection through the selected_games list
- Automatically discovers Processed/*.txt files in each game directory
- Convert characters to integers with game-specific preprocessing
- Also supports root-level *.txt files for games such as MegaMan that lack a Processed/ directory

Has no external package dependencies beyond NumPy.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from ..base import (
    BaseGameHandler,
    GameSample,
    GameTag,
    TileLegend,
    enforce_char_grid_top_left_16x16,
    enforce_top_left_16x16,
)
from .vglc_games import PREPROCESSORS, LEGEND_FACTORIES, SUPPORTED_GAMES

# ── VGLC game-name-to-directory mapping ────────────────────────────────────────
_GAME_DIR: Dict[str, str] = {
    GameTag.ZELDA:       "The Legend of Zelda",
    GameTag.MARIO:       "Super Mario Bros",
    GameTag.LODE_RUNNER: "Lode Runner",
    GameTag.KID_ICARUS:  "Kid Icarus",
    GameTag.DOOM:        "Doom",
    GameTag.MEGA_MAN:    "MegaMan",
}

# ── Games without a Processed directory (txt files are in the root) ────────────
_ROOT_TXT_GAMES = {GameTag.MEGA_MAN}

_DEFAULT_VGLC_ROOT = Path(__file__).parent.parent.parent / "TheVGLC"


class VGLCGameHandler(BaseGameHandler):
    """
    Handler for a single VGLC game.

    Parameters
    ----------
    game_tag  : GameTag constant (e.g. GameTag.ZELDA)
    vglc_root : root path of the TheVGLC repository
    split     : subdirectory to use (default: "Processed")
    handler_config : HandlerConfig object containing doom_slicing settings
    """

    def __init__(
        self,
        game_tag: str,
        vglc_root: Path | str = _DEFAULT_VGLC_ROOT,
        split: str = "Processed",
        handler_config: Optional[Any] = None,
    ) -> None:
        if game_tag not in SUPPORTED_GAMES:
            raise ValueError(
                f"Unsupported game: {game_tag!r}. "
                f"Supported: {SUPPORTED_GAMES}"
            )
        self._game_tag = game_tag
        self._root = Path(vglc_root) / _GAME_DIR[game_tag]
        self._split = split
        self._preprocessor = PREPROCESSORS[game_tag]()
        self._legend: TileLegend = LEGEND_FACTORIES[game_tag]()
        self._handler_config = handler_config
        self._entries: Optional[List[str]] = None  # lazy
        self._sliced_cache: Dict[str, GameSample] = {}

    @property
    def game_tag(self) -> str:
        return self._game_tag

    @property
    def game_dir(self) -> Path:
        return self._root

    def _discover(self) -> List[str]:
        if self._game_tag in _ROOT_TXT_GAMES:
            txt_files = sorted(self._root.glob("*.txt"))
        else:
            processed = self._root / self._split
            if not processed.exists():
                txt_files = sorted(self._root.glob("*.txt"))
            else:
                txt_files = sorted(processed.glob("*.txt"))
        txt_files = [p for p in txt_files if not p.name.lower().startswith("readme")]

        # Delegate when the preprocessor provides its own discovery/slicing logic
        if hasattr(self._preprocessor, 'discover_and_process'):
            return self._preprocessor.discover_and_process(
                files=txt_files,
                config=self._handler_config,
                game_tag=self._game_tag,
                legend=self._legend,
                cache=self._sliced_cache
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
                # Note: because the object is reused, changing order also changes the cached object.
                # A copy would avoid this, although overwriting is generally harmless during sequential access.
                # Returning a shallow copy or separating order assignment would be safer.
                # Preserve the existing mutation behavior here while keeping caching and retrieval focused.
                sample.order = order
            return sample

        # Standard VGLC game
        path = Path(source_id)
        text = path.read_text(encoding="utf-8", errors="replace")
        char_grid = self._preprocessor.parse_txt(text)
        array = self._preprocessor.transform(char_grid)
        array = enforce_top_left_16x16(array, game=self._game_tag, source_id=source_id)
        char_grid = enforce_char_grid_top_left_16x16(char_grid)

        if hasattr(self._preprocessor, 'postprocess_array'):
            array = self._preprocessor.postprocess_array(array)

        sample = GameSample(
            game=self._game_tag,
            source_id=source_id,
            array=array,
            char_grid=char_grid,
            legend=self._legend,
            instruction=None,
            order=order,
            meta={"file": str(path.name), "game_dir": str(self._root)},
        )

        # Cache standard games after the first load as well
        self._sliced_cache[source_id] = sample

        return sample

    def __repr__(self) -> str:
        return (
            f"VGLCGameHandler(game={self._game_tag!r}, "
            f"levels={len(self.list_entries())})"
        )


class VGLCHandler:
    """
    Combined TheVGLC handler for multiple games.

    Parameters
    ----------
    vglc_root      : root path of the TheVGLC repository
    selected_games : game tags to load (all games when None)
    split          : subdirectory to use (default: "Processed")
    handler_config : HandlerConfig object containing doom_slicing and related settings

    Example
    -------
        handler = VGLCHandler(selected_games=["zelda", "mario"], handler_config=config)
        for sample in handler:
            print(sample.game, sample.shape)
    """

    def __init__(
        self,
        vglc_root: Path | str = _DEFAULT_VGLC_ROOT,
        selected_games: Optional[List[str]] = None,
        split: str = "Processed",
        handler_config: Optional[Any] = None,
    ) -> None:
        self._root = Path(vglc_root)
        if selected_games is None:
            selected_games = list(_GAME_DIR.keys())
        invalid = [g for g in selected_games if g not in SUPPORTED_GAMES]
        if invalid:
            raise ValueError(
                f"Unsupported games: {invalid}. "
                f"Supported: {SUPPORTED_GAMES}"
            )
        self._selected_games = selected_games
        self._split = split
        self._game_handlers: Dict[str, VGLCGameHandler] = {
            g: VGLCGameHandler(
                g,
                vglc_root=self._root,
                split=split,
                handler_config=handler_config,
            )
            for g in selected_games
        }

    @property
    def selected_games(self) -> List[str]:
        return list(self._selected_games)

    def game_handler(self, game_tag: str) -> VGLCGameHandler:
        if game_tag not in self._game_handlers:
            raise KeyError(
                f"Game {game_tag!r} not in selected games: {self._selected_games}"
            )
        return self._game_handlers[game_tag]

    def list_entries(self, game_tag: Optional[str] = None) -> List[str]:
        """Return source IDs for a specific game or for all games."""
        if game_tag:
            return self.game_handler(game_tag).list_entries()
        entries = []
        for g in self._selected_games:
            entries.extend(self._game_handlers[g].list_entries())
        return entries

    def load_sample(self, source_id: str, order: Optional[int] = None) -> GameSample:
        """Delegate to the appropriate game handler based on the source_id path."""
        p = Path(source_id)
        for g, h in self._game_handlers.items():
            if _GAME_DIR[g] in str(p):
                return h.load_sample(source_id, order=order)
        # Fallback: check every handler
        for h in self._game_handlers.values():
            if source_id in h.list_entries():
                return h.load_sample(source_id, order=order)
        raise KeyError(f"source_id not found in any handler: {source_id!r}")

    def __iter__(self):
        order = 0
        for g in self._selected_games:
            for sample in self._game_handlers[g]:
                sample.order = order
                order += 1
                yield sample

    def __len__(self) -> int:
        return sum(len(h) for h in self._game_handlers.values())

    def all_samples(self) -> List[GameSample]:
        return list(self)

    def __repr__(self) -> str:
        counts = {g: len(self._game_handlers[g]) for g in self._selected_games}
        return f"VGLCHandler(games={counts})"
