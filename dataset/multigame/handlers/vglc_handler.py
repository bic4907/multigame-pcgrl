"""
dataset/multigame/handlers/vglc_handler.py
==========================================
TheVGLC 데이터셋 핸들러.

- 게임 선택 가능 (selected_games 리스트로 필터링)
- 각 게임 폴더의 Processed/*.txt 파일을 자동 탐색
- 게임별 전처리기로 char → int 변환
- MegaMan 처럼 Processed/ 폴더가 없는 경우 루트 *.txt 파일도 지원

외부 패키지 의존 없음 (numpy만 사용).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

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

# ── VGLC 게임 이름 → 폴더명 매핑 ───────────────────────────────────────────────
_GAME_DIR: Dict[str, str] = {
    GameTag.ZELDA:       "The Legend of Zelda",
    GameTag.MARIO:       "Super Mario Bros",
    GameTag.LODE_RUNNER: "Lode Runner",
    GameTag.KID_ICARUS:  "Kid Icarus",
    GameTag.DOOM:        "Doom",
    GameTag.MEGA_MAN:    "MegaMan",
}

# ── Processed 폴더가 없는 게임(루트에 txt가 있는 경우) ──────────────────────────
_ROOT_TXT_GAMES = {GameTag.MEGA_MAN}

_DEFAULT_VGLC_ROOT = Path(__file__).parent.parent.parent / "TheVGLC"


class VGLCGameHandler(BaseGameHandler):
    """
    단일 VGLC 게임 핸들러.

    Parameters
    ----------
    game_tag  : GameTag 상수 (e.g. GameTag.ZELDA)
    vglc_root : TheVGLC 저장소 루트 경로
    split     : 사용할 하위 폴더 (기본 "Processed")
    """

    def __init__(
        self,
        game_tag: str,
        vglc_root: Path | str = _DEFAULT_VGLC_ROOT,
        split: str = "Processed",
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
        self._entries: Optional[List[str]] = None  # lazy

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
                # fallback: 루트 txt
                txt_files = sorted(self._root.glob("*.txt"))
            else:
                txt_files = sorted(processed.glob("*.txt"))
        # README 계열 비레벨 텍스트는 제외
        txt_files = [p for p in txt_files if not p.name.lower().startswith("readme")]
        return [str(p) for p in txt_files]

    def list_entries(self) -> List[str]:
        if self._entries is None:
            self._entries = self._discover()
        return self._entries

    def load_sample(self, source_id: str, order: Optional[int] = None) -> GameSample:
        path = Path(source_id)
        text = path.read_text(encoding="utf-8", errors="replace")
        char_grid = self._preprocessor.parse_txt(text)
        array = self._preprocessor.transform(char_grid)
        array = enforce_top_left_16x16(array, game=self._game_tag, source_id=source_id)
        char_grid = enforce_char_grid_top_left_16x16(char_grid)
        return GameSample(
            game=self._game_tag,
            source_id=source_id,
            array=array,
            char_grid=char_grid,
            legend=self._legend,
            instruction=None,
            order=order,
            meta={"file": str(path.name), "game_dir": str(self._root)},
        )

    def __repr__(self) -> str:
        return (
            f"VGLCGameHandler(game={self._game_tag!r}, "
            f"levels={len(self.list_entries())})"
        )


class VGLCHandler:
    """
    TheVGLC 전체 핸들러 (여러 게임 통합).

    Parameters
    ----------
    vglc_root      : TheVGLC 저장소 루트 경로
    selected_games : 불러올 게임 태그 리스트 (None이면 전체)
    split          : 사용할 하위 폴더 (기본 "Processed")

    Example
    -------
        handler = VGLCHandler(selected_games=["zelda", "mario"])
        for sample in handler:
            print(sample.game, sample.shape)
    """

    def __init__(
        self,
        vglc_root: Path | str = _DEFAULT_VGLC_ROOT,
        selected_games: Optional[List[str]] = None,
        split: str = "Processed",
    ) -> None:
        self._root = Path(vglc_root)
        if selected_games is None:
            selected_games = list(_GAME_DIR.keys())
        # 지원 게임만 필터링
        invalid = [g for g in selected_games if g not in SUPPORTED_GAMES]
        if invalid:
            raise ValueError(
                f"Unsupported games: {invalid}. "
                f"Supported: {SUPPORTED_GAMES}"
            )
        self._selected_games = selected_games
        self._split = split
        self._game_handlers: Dict[str, VGLCGameHandler] = {
            g: VGLCGameHandler(g, vglc_root=self._root, split=split)
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
        """특정 게임 또는 전체 게임의 source_id 목록 반환."""
        if game_tag:
            return self.game_handler(game_tag).list_entries()
        entries = []
        for g in self._selected_games:
            entries.extend(self._game_handlers[g].list_entries())
        return entries

    def load_sample(self, source_id: str, order: Optional[int] = None) -> GameSample:
        """source_id 경로를 보고 자동으로 해당 게임 핸들러로 위임."""
        p = Path(source_id)
        for g, h in self._game_handlers.items():
            if _GAME_DIR[g] in str(p):
                return h.load_sample(source_id, order=order)
        # fallback: 모든 핸들러에서 확인
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

