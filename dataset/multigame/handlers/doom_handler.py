"""
dataset/multigame/handlers/doom_handler.py
==========================================
DOOM 레벨 데이터셋 핸들러 (TheVGLC 기반).
Doom 맵을 처리하기 위한 핸들러.
- 파일 자동 탐색
- 대형 맵 슬라이싱 (16x16 단위)
- 타일 매핑 및 변환
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
    Doom 레벨 핸들러.
    TheVGLC Doom 데이터셋에서 레벨을 자동 탐색, 슬라이싱, 변환합니다.
    Parameters
    ----------
    root : Path | str
        Doom 레벨 디렉토리 (*.txt 파일 포함)
    handler_config : Optional[Any]
        HandlerConfig 객체 (doom_slicing 설정)
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
        """Doom 레벨 파일 탐색 및 슬라이싱."""
        if not self._root.exists():
            return []
        # VGLC 구조: Processed 폴더 우선, 없으면 루트
        processed = self._root / "Processed"
        if processed.exists():
            txt_files = sorted(processed.glob("*.txt"))
        else:
            txt_files = sorted(self._root.glob("*.txt"))
        txt_files = [p for p in txt_files if not p.name.lower().startswith("readme")]
        # Doom 전용: discover_and_process 호출
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
        # 캐시에 있으면 반환 (슬라이싱된 데이터 등)
        if source_id in self._sliced_cache:
            sample = self._sliced_cache[source_id]
            if order is not None:
                sample.order = order
            return sample
        # 캐시에 없는 경우: source_id 파싱
        # source_id는 "path/to/file.txt|slice_idx" 형식
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
        # 슬라이싱 (설정이 있으면 적용, 없으면 전체)
        if self._handler_config and hasattr(self._handler_config, "doom_slicing"):
            sliced_maps = self._preprocessor.slice_large_map(
                char_grid,
                empty_max=self._handler_config.doom_slicing.empty_max,
                floor_empty_max=self._handler_config.doom_slicing.floor_empty_max,
            )
            if slice_idx >= len(sliced_maps):
                raise IndexError(
                    f"slice index {slice_idx} out of range for "
                    f"{path.name} ({len(sliced_maps)} slices)"
                )
            sliced_data = sliced_maps[slice_idx]
            char_grid = sliced_data["map"]
        else:
            # 슬라이싱 설정 없음: 전체 맵을 16x16으로 패딩/잘라냄
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
        # 캐시에 저장
        self._sliced_cache[source_id] = sample
        return sample
    def __repr__(self) -> str:
        return f"DoomHandler(levels={len(self.list_entries())})"
DOOM_PALETTE_DICT = DOOM_PALETTE
