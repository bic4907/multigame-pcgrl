"""
dataset/multigame/handlers/zelda_handler.py
===========================================
The Legend of Zelda (TheVGLC) 핸들러.

TheVGLC의 젤다 데이터는 전체 던전 맵이 하나의 텍스트 파일로 제공된다.
각 문자가 타일 하나를 의미하며, 방(room)은 11×16(W×H) 크기의 패치로 분할된다.
전부 void('-')로 이루어진 패치는 빈 공간이므로 제거한다.

방 구조 (NES The Legend of Zelda)
----------------------------------
  - 한 방 = 가로 11문자 × 세로 16줄
  - 구성: 상하 벽(WW…) 2줄씩 + 좌우 벽(WW) 2문자씩 + 내부 7×12
  - 인접 방은 벽을 나란히 놓음 (WWWW = 방1 오른벽 + 방2 왼벽)
  - 분리된 방은 11문자 대시('-----------')로 구분

전처리:
  1. 11×16 패치에서 외곽 벽 1줄/1칸씩 제거 → 9×14 (내부 + 벽 1겹)
  2. 짧은 축(가로 9)을 nearest-neighbor stretch → 14×14 정사각형
  3. 14×14를 16×16 가운데 정렬 (WALL 패딩)
  4. 90도 회전 증강으로 데이터 2배
  5. 일부 맵(50%)에 FLOOR/EMPTY 위치에 MOB·OBJECT를 1~5개 랜덤 드롭 (seed=42 고정)

타일 매핑 (vglc_games/zelda.py 기준)
-------------------------------------
0  : EMPTY   (-, 공백)
1  : WALL    (W)
2  : FLOOR   (F)
3  : DOOR    (D)
4  : BLOCK   (B)
5  : START   (S)
6  : MOB     (M)
7  : OBJECT  (O, I, P)
99 : UNKNOWN
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np

from ..base import (
    BaseGameHandler,
    GameSample,
    GameTag,
    TileLegend,
)
from .vglc_games.zelda import (
    ZeldaTile,
    ZeldaPreprocessor,
    ZELDA_PALETTE,
    make_legend,
)

# ── 경로 기본값 ─────────────────────────────────────────────────────────────────
_DEFAULT_ZELDA_ROOT = (
    Path(__file__).parent.parent.parent / "TheVGLC" / "The Legend of Zelda"
)

# ── 패치 크기 (NES 젤다 방 크기) ─────────────────────────────────────────────────
PATCH_W = 11   # 가로 (문자 수)
PATCH_H = 16   # 세로 (줄 수)
TARGET_SIZE = 16  # 최종 출력 크기 (16×16)

# 외곽 벽 제거 후 크기
TRIMMED_W = PATCH_W - 2   # 11 - 2 = 9
TRIMMED_H = PATCH_H - 2   # 16 - 2 = 14


def _read_map_text(path: Path) -> List[str]:
    """텍스트 파일을 읽어 줄 목록을 반환한다. 끝의 빈 줄은 제거."""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    while lines and not lines[-1].strip():
        lines.pop()
    return lines


def _text_to_int_grid(
    lines: List[str],
    preprocessor: ZeldaPreprocessor,
) -> np.ndarray:
    """
    전체 텍스트 맵을 정수 2D 배열로 변환한다.
    모든 줄을 동일 너비로 맞추고(단축 줄은 EMPTY 패딩), 문자→정수 변환.
    """
    if not lines:
        return np.zeros((0, 0), dtype=np.int32)
    max_w = max(len(l) for l in lines)
    H = len(lines)
    grid = np.full((H, max_w), ZeldaTile.EMPTY, dtype=np.int32)
    for r, line in enumerate(lines):
        for c, ch in enumerate(line):
            grid[r, c] = preprocessor.char_to_int(ch)
    return grid


def _extract_rooms(
    grid: np.ndarray,
    patch_h: int = PATCH_H,
    patch_w: int = PATCH_W,
) -> List[Tuple[np.ndarray, int, int]]:
    """
    전체 맵 그리드를 patch_h × patch_w 패치로 분할하고,
    전부 EMPTY(0)인 패치는 제외한다.

    Returns
    -------
    list of (patch_array, row_idx, col_idx)
        patch_array : (patch_h, patch_w) int32
        row_idx     : 패치 행 인덱스 (0-based)
        col_idx     : 패치 열 인덱스 (0-based)
    """
    H, W = grid.shape
    rooms = []

    for iy, y in enumerate(range(0, H - patch_h + 1, patch_h)):
        for ix, x in enumerate(range(0, W - patch_w + 1, patch_w)):
            patch = grid[y : y + patch_h, x : x + patch_w]
            # 전부 empty (void) 인 패치는 제외
            if np.all(patch == ZeldaTile.EMPTY):
                continue
            rooms.append((patch.copy(), iy, ix))

    return rooms


def _trim_outer_wall(patch: np.ndarray) -> np.ndarray:
    """
    11×16 패치에서 외곽 벽 1줄/1칸 제거.

    원본 (16H × 11W):
        row 0  : WWWWWWWWWWW  ← 제거 (벽 행)
        row 1  : WWWWDDDWWWW  ← 유지 (벽+문)
        ...
        row 14 : WWWWDDDWWWW  ← 유지
        row 15 : WWWWWWWWWWW  ← 제거 (벽 행)
        col 0  : W (제거)
        col 10 : W (제거)

    결과: 9W × 14H
    """
    return patch[1:-1, 1:-1].copy()


def _stretch_to_square(patch: np.ndarray) -> np.ndarray:
    """
    직사각형 패치의 짧은 축을 nearest-neighbor로 늘려 정사각형으로 만든다.

    예: 14H × 9W → 14 × 14  (가로 9→14 stretch)
        9H × 14W → 14 × 14  (세로 9→14 stretch)

    정수 타일 ID를 유지하기 위해 nearest-neighbor 인덱스 매핑을 사용한다.
    """
    h, w = patch.shape
    if h == w:
        return patch

    target = max(h, w)

    if w < h:
        # 가로가 짧음 → 가로를 h 크기로 늘림
        col_indices = np.round(np.linspace(0, w - 1, target)).astype(int)
        return patch[:, col_indices].copy()
    else:
        # 세로가 짧음 → 세로를 w 크기로 늘림
        row_indices = np.round(np.linspace(0, h - 1, target)).astype(int)
        return patch[row_indices, :].copy()


def _center_pad_to_16x16(patch: np.ndarray) -> np.ndarray:
    """
    임의 크기 패치를 16×16 중앙 정렬로 패딩한다.
    빈 영역은 WALL(1)로 채운다 (외곽이므로).
    """
    h, w = patch.shape
    if h == TARGET_SIZE and w == TARGET_SIZE:
        return patch
    out = np.full((TARGET_SIZE, TARGET_SIZE), ZeldaTile.WALL, dtype=np.int32)
    y0 = (TARGET_SIZE - h) // 2
    x0 = (TARGET_SIZE - w) // 2
    out[y0 : y0 + h, x0 : x0 + w] = patch
    return out


def _rotate_90(patch: np.ndarray) -> np.ndarray:
    """시계방향 90도 회전."""
    return np.rot90(patch, k=-1).copy()


def _is_uniform_center_12x12(padded: np.ndarray) -> bool:
    """
    16x16 맵의 중앙 12x12 영역(테두리 2줄 제외)이 모두 같은 타일인지 확인.
    
    Parameters
    ----------
    padded : np.ndarray
        16x16 맵
    
    Returns
    -------
    bool
        중앙 12x12가 모두 같은 값이면 True
    """
    center = padded[2:14, 2:14]  # 중앙 12x12 추출
    return bool(np.all(center == center[0, 0]))


# 드롭 가능한 타일 종류
_DROP_TILES = [ZeldaTile.MOB, ZeldaTile.OBJECT]

# 드롭 대상이 되는 빈 타일 (FLOOR, EMPTY 모두 가능)
_DROPPABLE_TILES = {ZeldaTile.FLOOR, ZeldaTile.EMPTY}

# 드롭 증강 비율 (원본 중 몇 %에 적용할지)
DROP_AUG_RATIO = 0.5  # 원본의 50%에만 드롭 적용

# 드롭 개수 범위
DROP_COUNT_MIN = 1
DROP_COUNT_MAX = 5


def _augment_random_drop(
    padded: np.ndarray,
    rng: np.random.Generator,
) -> Optional[np.ndarray]:
    """
    16×16 패치의 FLOOR 또는 EMPTY 타일 중 1~5개를 랜덤으로
    MOB 또는 OBJECT로 교체한다.

    드롭 가능한 위치가 없으면 None을 반환한다.
    """
    droppable = np.argwhere(
        np.isin(padded, list(_DROPPABLE_TILES))
    )
    if len(droppable) == 0:
        return None

    # 드롭 개수: 1~5 (가용 위치 수 이내)
    n_drop = rng.integers(DROP_COUNT_MIN, DROP_COUNT_MAX + 1)
    n_drop = min(n_drop, len(droppable))

    aug = padded.copy()
    chosen = droppable[rng.choice(len(droppable), size=n_drop, replace=False)]
    for pos in chosen:
        tile = rng.choice(_DROP_TILES)
        aug[pos[0], pos[1]] = tile
    return aug


# ── 정수 → 대표 문자 역매핑 ─────────────────────────────────────────────────────
_INT_TO_CHAR: Dict[int, str] = {
    ZeldaTile.EMPTY:   "-",
    ZeldaTile.WALL:    "W",
    ZeldaTile.FLOOR:   "F",
    ZeldaTile.DOOR:    "D",
    ZeldaTile.BLOCK:   "B",
    ZeldaTile.START:   "S",
    ZeldaTile.MOB:     "M",
    ZeldaTile.OBJECT:  "O",
    ZeldaTile.UNKNOWN: "?",
}


def _int_to_char(val: int) -> str:
    return _INT_TO_CHAR.get(val, "?")


def _array_to_char_grid(arr: np.ndarray) -> List[List[str]]:
    """정수 배열 → 문자 그리드."""
    return [[_int_to_char(int(val)) for val in row] for row in arr]


class ZeldaHandler(BaseGameHandler):
    """
    The Legend of Zelda (TheVGLC) 핸들러.

    전처리 과정:
      1. Processed/ 폴더의 텍스트 맵을 11×16 패치로 분할
      2. 빈 방(전부 EMPTY) 제거
      3. 외곽 벽 1줄/1칸 제거 → 9×14
      4. 짧은 축(가로) nearest-neighbor stretch → 14×14
      5. 16×16 가운데 정렬 (WALL 패딩)
      6. 90도 회전 증강 → 데이터 2배

    Parameters
    ----------
    root  : TheVGLC/The Legend of Zelda 폴더 경로
    split : 하위 폴더명 (기본 "Processed")
    """

    def __init__(
        self,
        root: Path | str = _DEFAULT_ZELDA_ROOT,
        split: str = "Processed",
        handler_config: Optional[Any] = None,
    ) -> None:
        self._root = Path(root)
        self._split = split
        self._handler_config = handler_config
        self._preprocessor = ZeldaPreprocessor()
        self._legend = make_legend()
        self._samples: Optional[List[GameSample]] = None  # lazy

    @property
    def game_tag(self) -> str:
        return GameTag.ZELDA

    # ── 파일 탐색 ────────────────────────────────────────────────────────────────

    def _discover_files(self) -> List[Path]:
        """Processed/ 폴더에서 텍스트 파일 목록을 반환한다."""
        processed = self._root / self._split
        if not processed.exists():
            raise FileNotFoundError(
                f"Zelda Processed directory not found: {processed}"
            )
        files = sorted(processed.glob("*.txt"))
        files = [p for p in files if not p.name.lower().startswith("readme")]
        return files

    # ── 전체 로드 ────────────────────────────────────────────────────────────────

    def _load_all(self) -> List[GameSample]:
        files = self._discover_files()
        if not files:
            raise FileNotFoundError(
                f"[zelda] 레벨 파일을 찾을 수 없습니다: {self._root / self._split}"
            )

        rng = np.random.default_rng(seed=42)
        samples: List[GameSample] = []
        # 드롭 증강 후보 (원본 padded + 메타)
        originals_for_drop: List[tuple] = []

        for fpath in files:
            fname = fpath.stem  # e.g. "tloz1_1"
            lines = _read_map_text(fpath)
            grid = _text_to_int_grid(lines, self._preprocessor)

            if grid.size == 0:
                continue

            rooms = _extract_rooms(grid)
            for patch, ry, rx in rooms:
                # 1) 외곽 벽 제거 → 14H × 9W
                trimmed = _trim_outer_wall(patch)

                # 2) 짧은 축 stretch → 14 × 14 정사각형
                squared = _stretch_to_square(trimmed)

                # 3) 16×16 가운데 정렬 (상하좌우 1칸 WALL 패딩)
                padded = _center_pad_to_16x16(squared)

                # 이 코드 주석처리하면 zelda arguement 비활성화
                # padded = self._preprocessor.postprocess_array(padded)

                source_id = f"{fname}_r{ry}_c{rx}"
                base_meta = {
                    "file": fname,
                    "room_row": ry,
                    "room_col": rx,
                    "original_size": (PATCH_H, PATCH_W),
                    "trimmed_size": (TRIMMED_H, TRIMMED_W),
                    "stretched_size": squared.shape,
                    "output_size": (TARGET_SIZE, TARGET_SIZE),
                }

                # 원본
                samples.append(GameSample(
                    game=GameTag.ZELDA,
                    source_id=source_id,
                    array=padded,
                    char_grid=_array_to_char_grid(padded),
                    legend=self._legend,
                    instruction=None,
                    order=len(samples),
                    meta={**base_meta, "augmented": False},
                ))

                # 90도 회전 증강
                rotated = _rotate_90(padded)
                samples.append(GameSample(
                    game=GameTag.ZELDA,
                    source_id=f"{source_id}_rot90",
                    array=rotated,
                    char_grid=_array_to_char_grid(rotated),
                    legend=self._legend,
                    instruction=None,
                    order=len(samples),
                    meta={**base_meta, "augmented": True, "augmentation": "rot90"},
                ))

                # 드롭 증강 후보 등록
                originals_for_drop.append((padded, source_id, base_meta))

        # ── 랜덤 몹/오브젝트 드롭 증강 (원본의 50%) ─────────────────────────
        n_drop = int(len(originals_for_drop) * DROP_AUG_RATIO)
        drop_indices = rng.choice(
            len(originals_for_drop), size=n_drop, replace=False,
        )
        for idx in drop_indices:
            padded, source_id, base_meta = originals_for_drop[idx]
            dropped = _augment_random_drop(padded, rng)
            if dropped is None:
                continue
            samples.append(GameSample(
                game=GameTag.ZELDA,
                source_id=f"{source_id}_drop",
                array=dropped,
                char_grid=_array_to_char_grid(dropped),
                legend=self._legend,
                instruction=None,
                order=len(samples),
                meta={**base_meta, "augmented": True, "augmentation": "random_drop"},
            ))

        # ── 필터링: uniform center 맵의 증강 버전만 제거 ─────────────────────────
        samples_before_filter = len(samples)
        
        # 1단계: uniform center 맵의 증강 버전만 제거 (원본은 유지)
        uniform_source_ids = set()
        for sample in samples:
            # 원본이 uniform인지 확인 (rot90, drop이 아닌 원본만 체크)
            if not sample.source_id.endswith("_rot90") and not sample.source_id.endswith("_drop"):
                padded_array = sample.array
                if _is_uniform_center_12x12(padded_array):
                    # 이 원본의 증강 버전만 제거 (_rot90, _drop)
                    base_id = sample.source_id
                    uniform_source_ids.add(f"{base_id}_rot90")
                    uniform_source_ids.add(f"{base_id}_drop")
        
        # 2단계: uniform center 맵의 증강 버전 제거
        filtered_samples = [s for s in samples if s.source_id not in uniform_source_ids]
        
        # 3단계: order 재설정
        for i, sample in enumerate(filtered_samples):
            sample.order = i

        return filtered_samples

    # ── BaseGameHandler 인터페이스 ────────────────────────────────────────────────

    def _ensure_loaded(self) -> None:
        if self._samples is None:
            self._samples = self._load_all()

    def list_entries(self) -> List[str]:
        self._ensure_loaded()
        return [s.source_id for s in self._samples]

    def load_sample(self, source_id: str, order: Optional[int] = None) -> GameSample:
        self._ensure_loaded()
        for s in self._samples:
            if s.source_id == source_id:
                if order is not None:
                    s.order = order
                return s
        raise KeyError(f"[zelda] source_id not found: {source_id}")

    def __len__(self) -> int:
        self._ensure_loaded()
        return len(self._samples)

    def __iter__(self) -> Iterator[GameSample]:
        self._ensure_loaded()
        yield from self._samples

    def __repr__(self) -> str:
        n = len(self._samples) if self._samples is not None else "?"
        return f"ZeldaHandler(root={str(self._root)!r}, rooms={n})"

