"""
dataset/multigame/handlers/boxoban_handler.py
=============================================
Google DeepMind Boxoban 데이터셋 핸들러.

출처: https://github.com/google-deepmind/boxoban-levels
폴더 구조:
    boxoban_levels/
        hard/           000.txt … 003.txt
        medium/train/   000.txt … 449.txt
        medium/valid/   000.txt … 009.txt

레벨 파일 형식
--------------
- 레벨은 `;` 주석 줄로 구분
- 레벨 크기: 10×10 고정
- 문자 의미:
    '#' : wall
    ' ' : floor/empty
    '.' : target (goal square)
    '$' : box
    '@' : player
    '*' : box on target
    '+' : player on target

타일 ID (tile_mapping.json 의 통합 카테고리로 매핑 가능)
---------------------------------------------------------
0  EMPTY  – floor/empty (' ', '.')
1  WALL   – wall ('#')
2  FLOOR  – floor (같은 empty 계열, 구분용)
3  ENEMY  – (없음, 사용 안 함)
4  OBJECT – box ('$', '*')
5  SPAWN  – player ('@', '+')

16×16 정규화
------------
10×10 레벨을 중앙(top-left)에 배치하고 나머지는 WALL(1)로 패딩.

Diversity 필터링
----------------
- 타일 히스토그램(floor/wall/box/player 비율)을 feature vector 로 삼음
- 전체 레벨을 feature space에서 greedy max-min sampling 으로 다양성 확보
"""
from __future__ import annotations

import hashlib
import re
import warnings
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import numpy as np

from ..base import BaseGameHandler, GameSample, GameTag, TileLegend

# ── 경로 기본값 ─────────────────────────────────────────────────────────────────
_DEFAULT_BOXOBAN_ROOT = Path(__file__).parent.parent.parent / "boxoban_levels"

# ── 타일 상수 ────────────────────────────────────────────────────────────────────
class BoxobanTile:
    EMPTY  = 0   # floor / empty (  '.' 포함)
    WALL   = 1   # wall  ('#')
    OBJECT = 4   # box   ('$', '*')
    SPAWN  = 5   # player('@', '+')


# 문자 → 정수 타일 ID
_CHAR_MAP: dict[str, int] = {
    " ": BoxobanTile.EMPTY,
    ".": BoxobanTile.EMPTY,   # target square → empty (구조 상 floor)
    "#": BoxobanTile.WALL,
    "$": BoxobanTile.OBJECT,  # box
    "*": BoxobanTile.OBJECT,  # box on target
    "@": BoxobanTile.SPAWN,   # player
    "+": BoxobanTile.SPAWN,   # player on target
}

BOXOBAN_PALETTE: dict[int, Tuple[int, int, int]] = {
    BoxobanTile.EMPTY:  (200, 180, 120),
    BoxobanTile.WALL:   (80,  80,  80),
    BoxobanTile.OBJECT: (255, 215, 0),
    BoxobanTile.SPAWN:  (0,   200, 0),
}

# Boxoban 원본 크기
_LEVEL_SIZE = 10
_TARGET_SIZE = 16


def _make_legend() -> TileLegend:
    return TileLegend(char_to_attrs={
        " ": ["passable", "floor"],
        ".": ["passable", "floor", "target"],
        "#": ["solid", "wall"],
        "$": ["passable", "object", "box"],
        "*": ["passable", "object", "box", "target"],
        "@": ["passable", "spawn", "player"],
        "+": ["passable", "spawn", "player", "target"],
    })


# ── 레벨 파싱 ────────────────────────────────────────────────────────────────────

def _parse_levels_from_file(path: Path) -> List[List[str]]:
    """
    txt 파일 하나에서 레벨 목록을 파싱한다.
    반환값: 각 레벨 = 원본 문자열 행의 리스트
    """
    levels: List[List[str]] = []
    current: List[str] = []

    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.rstrip()
        if line.startswith(";"):
            # 구분자 → 이전 레벨 저장
            if current:
                levels.append(current)
                current = []
        elif line == "" and current:
            # 빈 줄이 레벨 뒤에 오는 경우
            levels.append(current)
            current = []
        else:
            current.append(line)

    if current:
        levels.append(current)

    return [lvl for lvl in levels if lvl]


def _lines_to_array(lines: List[str]) -> Optional[np.ndarray]:
    """
    문자열 행 목록 → (H, W) int32 ndarray.
    행 길이가 다르면 ' '(EMPTY)로 오른쪽 패딩.
    10×10 이 아닌 레벨은 None 반환.
    """
    if not lines:
        return None

    H = len(lines)
    W = max(len(l) for l in lines)

    if H != _LEVEL_SIZE or W != _LEVEL_SIZE:
        return None          # 비정형 레벨 제외

    grid = np.zeros((H, W), dtype=np.int32)
    for r, line in enumerate(lines):
        padded = line.ljust(W)
        for c, ch in enumerate(padded):
            grid[r, c] = _CHAR_MAP.get(ch, BoxobanTile.WALL)

    return grid


def _strip_wall_border(arr: np.ndarray) -> np.ndarray:
    """
    가장자리에서 WALL(1) 로만 이루어진 행/열을 제거해 유효 영역을 반환한다.

    예) 10×10 레벨에서 첫/마지막 행이 전부 WALL이고
        첫/마지막 열이 전부 WALL이면 → 8×8 반환.
    non-WALL 셀이 하나도 없으면 원본 그대로 반환.
    """
    non_wall = (arr != BoxobanTile.WALL)
    rows = np.where(non_wall.any(axis=1))[0]
    cols = np.where(non_wall.any(axis=0))[0]
    if rows.size == 0 or cols.size == 0:
        return arr  # 전부 WALL인 경우 원본 반환
    r0, r1 = int(rows[0]), int(rows[-1]) + 1
    c0, c1 = int(cols[0]), int(cols[-1]) + 1
    return arr[r0:r1, c0:c1]


def _fit_to_target(arr: np.ndarray, target: int = _TARGET_SIZE) -> np.ndarray:
    """
    원본 레벨을 target×target 크기로 변환한다.

    전처리 방식
    -----------
    1. 구조(WALL / EMPTY) 레이어를 target×target 으로 nearest-neighbor 보간 → 꽉 채움
    2. 오브젝트(BOX / PLAYER) 는 원본 (r, c) 위치 비율을 target 좌표로 환산해 재배치
       - 개수는 원본 동일하게 유지 (늘리지 않음)
    3. 결과는 target×target 로 정확히 반환
    """
    src_h, src_w = arr.shape

    # ── 1. 구조 레이어 nearest-neighbor 리사이즈 ──────────────────────────────
    structure = np.where(arr == BoxobanTile.WALL,
                         BoxobanTile.WALL, BoxobanTile.EMPTY).astype(np.int32)
    row_idx = (np.arange(target) * src_h / target).astype(np.int32)
    col_idx = (np.arange(target) * src_w / target).astype(np.int32)
    out = structure[np.ix_(row_idx, col_idx)]   # (target, target)

    # ── 2. 오브젝트 재배치 ────────────────────────────────────────────────────
    object_tiles = {BoxobanTile.OBJECT, BoxobanTile.SPAWN}
    for r in range(src_h):
        for c in range(src_w):
            tid = int(arr[r, c])
            if tid not in object_tiles:
                continue
            nr = int(round((r + 0.5) / src_h * target - 0.5))
            nc = int(round((c + 0.5) / src_w * target - 0.5))
            nr = max(0, min(target - 1, nr))
            nc = max(0, min(target - 1, nc))
            out[nr, nc] = tid

    return out.astype(np.int32)

# 하위 호환 alias
_scale2x_to_16x16 = _fit_to_target


# ── Diversity 필터링 ─────────────────────────────────────────────────────────────

def _feature_vector(arr: np.ndarray) -> np.ndarray:
    """
    레벨 배열 → 다양성 측정용 특징 벡터. (16×16 기준)

    특징:
      0  : wall 비율
      1  : empty/floor 비율
      2  : object(box) 수
      3  : spawn(player) 수
      4  : 상위 절반 wall 비율
      5  : 하위 절반 wall 비율
      6  : 좌측 절반 wall 비율
      7  : 우측 절반 wall 비율
    """
    total  = arr.size
    region = arr   # 전체 16×16 사용

    wall_r  = (region == BoxobanTile.WALL).sum()  / total
    empty_r = (region == BoxobanTile.EMPTY).sum() / total
    n_box   = float((region == BoxobanTile.OBJECT).sum())
    n_player= float((region == BoxobanTile.SPAWN).sum())

    h, w    = region.shape
    mid_r, mid_c = h // 2, w // 2
    half_size    = mid_r * w

    top_wall  = (region[:mid_r, :]  == BoxobanTile.WALL).sum() / max(half_size, 1)
    bot_wall  = (region[mid_r:, :]  == BoxobanTile.WALL).sum() / max(half_size, 1)
    left_wall = (region[:, :mid_c]  == BoxobanTile.WALL).sum() / max(mid_c * h, 1)
    right_wall= (region[:, mid_c:]  == BoxobanTile.WALL).sum() / max(mid_c * h, 1)

    return np.array([
        wall_r, empty_r, n_box, n_player,
        top_wall, bot_wall, left_wall, right_wall,
    ], dtype=np.float32)



def _diversity_sample(
    arrays: List[np.ndarray],
    n: int,
    seed: int = 42,
) -> List[int]:
    """
    Greedy max-min distance sampling (farthest point sampling).
    특징 공간에서 서로 가장 먼 n 개의 인덱스를 반환.
    """
    if len(arrays) <= n:
        return list(range(len(arrays)))

    rng = np.random.default_rng(seed)
    features = np.stack([_feature_vector(a) for a in arrays])   # (N, D)

    # 정규화
    std = features.std(axis=0) + 1e-8
    features = features / std

    chosen = [int(rng.integers(len(arrays)))]
    dists = np.full(len(arrays), np.inf)

    for _ in range(n - 1):
        last = features[chosen[-1]]
        d = np.linalg.norm(features - last, axis=1)
        dists = np.minimum(dists, d)
        chosen.append(int(np.argmax(dists)))

    return chosen


# ── 오브젝트 증강 ────────────────────────────────────────────────────────────────

def _is_corner(array: np.ndarray, r: int, c: int) -> bool:
    """
    (r, c)가 구석인지 판단한다.
    구석 = EMPTY 타일이면서 상하좌우 중 수직으로 인접한 두 방향이 모두 WALL인 경우.
    경계 밖은 WALL로 간주한다.
    """
    H, W = array.shape

    def is_wall(rr, cc):
        if rr < 0 or rr >= H or cc < 0 or cc >= W:
            return True
        return array[rr, cc] == BoxobanTile.WALL

    top    = is_wall(r - 1, c)
    bottom = is_wall(r + 1, c)
    left   = is_wall(r, c - 1)
    right  = is_wall(r, c + 1)

    return (top and left) or (top and right) or (bottom and left) or (bottom and right)


def _placeable_positions(array: np.ndarray) -> np.ndarray:
    """
    OBJECT를 놓을 수 있는 위치를 반환한다.
    조건: EMPTY 타일이고 구석이 아닌 위치.
    """
    candidates = []
    for r, c in np.argwhere(array == BoxobanTile.EMPTY):
        if not _is_corner(array, r, c):
            candidates.append([r, c])
    return np.array(candidates, dtype=np.int32) if candidates else np.empty((0, 2), dtype=np.int32)


def _augment_objects(array: np.ndarray) -> np.ndarray:
    """
    SOKOBAN ARGUEMENT
    Sokoban 맵의 오브젝트(box) 수를 확률적으로 증감한다.

    - 40% : 변경 없음 (4개 유지)
    - 20% : 랜덤 1개 제거 → 3개
    - 20% : EMPTY(비구석) 위치에 1개 추가 → 5개
    - 20% : EMPTY(비구석) 위치에 2개 추가 → 6개

    시드는 배열 내용의 MD5 해시 → 동일 입력이면 항상 동일 결과.
    """
    seed = int.from_bytes(
        hashlib.md5(array.tobytes()).digest()[:4], byteorder='big'
    )
    rng = np.random.default_rng(seed)

    choice = rng.choice([0, 1, 2, 3], p=[0.4, 0.2, 0.2, 0.2])
    result = array.copy()

    if choice == 0:
        # 변경 없음
        return result

    elif choice == 1:
        # 1개 제거
        obj_positions = np.argwhere(result == BoxobanTile.OBJECT)
        if len(obj_positions) == 0:
            return result
        idx = rng.integers(len(obj_positions))
        r, c = obj_positions[idx]
        result[r, c] = BoxobanTile.EMPTY

    else:
        # 1개(choice==2) 또는 2개(choice==3) 추가
        n_add = 1 if choice == 2 else 2
        placeable = _placeable_positions(result)
        if len(placeable) == 0:
            return result
        n_add = min(n_add, len(placeable))
        chosen = rng.choice(len(placeable), size=n_add, replace=False)
        for idx in chosen:
            r, c = placeable[idx]
            result[r, c] = BoxobanTile.OBJECT

    return result


# ── 핸들러 클래스 ────────────────────────────────────────────────────────────────

class BoxobanHandler(BaseGameHandler):
    """
    Google DeepMind Boxoban 핸들러.

    Parameters
    ----------
    root        : boxoban_levels 폴더 경로
    difficulty  : "hard" | "medium" | "both"
    split       : medium 전용 - "train" | "valid" | "all"
    n_sample    : diversity sampling 으로 추출할 수 (None = 전체)
    seed        : diversity sampling 시드

    Example
    -------
        handler = BoxobanHandler(n_sample=1000)
        for sample in handler:
            print(sample.array.shape, sample.source_id)

        samples = handler.sample(500)
    """

    game_tag = GameTag.SOKOBAN

    def __init__(
        self,
        root: Path | str = _DEFAULT_BOXOBAN_ROOT,
        difficulty: str = "both",
        split: str = "train",
        n_sample: Optional[int] = 1000,
        seed: int = 42,
    ) -> None:
        self._root = Path(root)
        self._difficulty = difficulty
        self._split = split
        self._n_sample = n_sample
        self._seed = seed
        self._samples: Optional[List[GameSample]] = None  # lazy

    @property
    def game_tag(self) -> str:
        return GameTag.SOKOBAN

    # ── 파일 목록 수집 ────────────────────────────────────────────────────────────

    def _collect_files(self) -> List[Path]:
        files: List[Path] = []
        d = self._difficulty

        if d in ("hard", "both"):
            hard_dir = self._root / "hard"
            if hard_dir.exists():
                files += sorted(hard_dir.glob("*.txt"))
            else:
                warnings.warn(f"[boxoban] hard 폴더 없음: {hard_dir}")

        if d in ("medium", "both"):
            splits = ["train", "valid"] if self._split == "all" else [self._split]
            for sp in splits:
                med_dir = self._root / "medium" / sp
                if med_dir.exists():
                    files += sorted(med_dir.glob("*.txt"))
                else:
                    warnings.warn(f"[boxoban] medium/{sp} 폴더 없음: {med_dir}")

        return files

    # ── 전체 레벨 로드 ────────────────────────────────────────────────────────────

    def _load_all(self) -> List[GameSample]:
        files = self._collect_files()
        if not files:
            raise FileNotFoundError(
                f"[boxoban] 레벨 파일을 찾을 수 없습니다: {self._root}"
            )

        legend = _make_legend()
        all_arrays: List[np.ndarray] = []
        all_ids:    List[str]        = []

        for fpath in files:
            rel = fpath.relative_to(self._root)
            level_lines = _parse_levels_from_file(fpath)
            for lvl_idx, lines in enumerate(level_lines):
                arr = _lines_to_array(lines)
                if arr is None:
                    continue
                processed = _fit_to_target(arr, _TARGET_SIZE)

                #이 라인 주석처리하면 sokoban arguement 비활성화
                #processed = _augment_objects(processed)

                source_id = f"{rel}#{lvl_idx}"
                all_arrays.append(processed)
                all_ids.append(source_id)

        if not all_arrays:
            raise ValueError("[boxoban] 파싱 가능한 레벨이 없습니다.")

        # ── Diversity sampling ────────────────────────────────────────────────
        n = self._n_sample
        if n is not None and n < len(all_arrays):
            chosen_idxs = _diversity_sample(all_arrays, n, seed=self._seed)
        else:
            chosen_idxs = list(range(len(all_arrays)))

        samples: List[GameSample] = []
        for order, idx in enumerate(chosen_idxs):
            samples.append(GameSample(
                game=GameTag.SOKOBAN,
                source_id=all_ids[idx],
                array=all_arrays[idx],
                legend=legend,
                order=order,
                meta={
                    "difficulty":    self._difficulty,
                    "original_size": (_LEVEL_SIZE, _LEVEL_SIZE),
                    "output_size":   (_TARGET_SIZE, _TARGET_SIZE),
                    "scale_method":  "fit_to_target_center",
                    "scale":         max(1, _TARGET_SIZE // _LEVEL_SIZE),
                },
            ))

        return samples

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
                return s
        raise KeyError(f"[boxoban] source_id not found: {source_id}")

    def __len__(self) -> int:
        self._ensure_loaded()
        return len(self._samples)

    def __iter__(self) -> Iterator[GameSample]:
        self._ensure_loaded()
        yield from self._samples

    def sample(self, n: int, seed: int = 0) -> List[GameSample]:
        """전체 로드된 샘플 중 n 개를 랜덤 추출."""
        self._ensure_loaded()
        rng = np.random.default_rng(seed)
        idxs = rng.choice(len(self._samples), size=min(n, len(self._samples)), replace=False)
        return [self._samples[i] for i in idxs]

    def filter_by_difficulty(self, difficulty: str) -> List[GameSample]:
        self._ensure_loaded()
        return [s for s in self._samples if s.meta.get("difficulty") == difficulty]

    def stats(self) -> dict:
        """로드된 샘플에 대한 간단한 통계."""
        self._ensure_loaded()
        arrays = [s.array for s in self._samples]
        wall_ratios = [(a == BoxobanTile.WALL).mean() for a in arrays]
        return {
            "n_samples":       len(arrays),
            "difficulty":      self._difficulty,
            "scale_method":    "2x_structure_object_reposition",
            "wall_ratio_mean": float(np.mean(wall_ratios)),
            "wall_ratio_std":  float(np.std(wall_ratios)),
            "wall_ratio_min":  float(np.min(wall_ratios)),
            "wall_ratio_max":  float(np.max(wall_ratios)),
        }

