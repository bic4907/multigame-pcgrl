"""
dataset/cpcgrl_buffer/dataset.py
================================
CPCGRL pair dataset 로더.

Usage:
    from dataset.cpcgrl_buffer import CPCGRLBufferDataset

    ds = CPCGRLBufferDataset()              # 기본 경로에서 로드
    print(len(ds))                          # 12655
    pair = ds[0]                            # MapTransitionPair
    print(pair.before.shape)                # (16, 16)
    print(pair.after.shape)                 # (16, 16)
    print(pair.reward_enum)                 # 3
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Union

import numpy as np


# ── 데이터 클래스 ────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class MapTransitionPair:
    """env_map 의 (before, after) 전이 쌍 1개.

    Attributes
    ----------
    before : np.ndarray  (H, W) int32
        t 시점의 env_map.
    after : np.ndarray  (H, W) int32
        t+1 시점의 env_map.
    reward_enum : int
        이 쌍이 수집된 reward_enum (1~5).
    timestep : int
        before 시점의 total_timestep.
    """
    before: np.ndarray
    after: np.ndarray
    reward_enum: int
    timestep: int

    @property
    def pair(self) -> np.ndarray:
        """(2, H, W) 형태로 반환."""
        return np.stack([self.before, self.after], axis=0)

    @property
    def diff(self) -> np.ndarray:
        """after - before 변화 맵. 타일이 바뀐 위치만 non-zero."""
        return self.after.astype(np.int16) - self.before.astype(np.int16)

    @property
    def changed_mask(self) -> np.ndarray:
        """변경된 타일 위치 boolean mask (H, W)."""
        return self.before != self.after

    @property
    def n_changes(self) -> int:
        """변경된 타일 수."""
        return int(self.changed_mask.sum())

    def __repr__(self) -> str:
        h, w = self.before.shape
        return (
            f"MapTransitionPair(re={self.reward_enum}, ts={self.timestep}, "
            f"map={h}x{w}, changes={self.n_changes})"
        )


# ── 데이터셋 클래스 ──────────────────────────────────────────────────────────

_DEFAULT_NPZ = Path(__file__).parent / "cpcgrl_pair_dataset.npz"
_DEFAULT_META = Path(__file__).parent / "metadata.json"


@dataclass
class CPCGRLBufferDataset:
    """CPCGRL pair dataset 로더.

    Parameters
    ----------
    npz_path : str or Path, optional
        .npz 파일 경로. 기본값은 같은 폴더의 cpcgrl_pair_dataset.npz.
    reward_enums : list[int], optional
        특정 reward_enum 만 필터. None 이면 전체 로드.

    Examples
    --------
    >>> ds = CPCGRLBufferDataset()
    >>> len(ds)
    12655
    >>> ds[0]
    MapTransitionPair(re=3, ts=..., map=16x16, changes=1)
    >>> ds.by_reward_enum(1)
    CPCGRLBufferDataset(n=736, reward_enums=[1])
    """
    npz_path: Union[str, Path] = field(default_factory=lambda: _DEFAULT_NPZ)
    reward_enums: Optional[List[int]] = None

    # ── 내부 상태 (post_init 에서 로드) ──
    _pairs: np.ndarray = field(init=False, repr=False)
    _reward_enums_arr: np.ndarray = field(init=False, repr=False)
    _timesteps: np.ndarray = field(init=False, repr=False)
    _metadata: dict = field(init=False, repr=False)

    def __post_init__(self):
        npz_path = Path(self.npz_path)
        assert npz_path.exists(), f"Dataset not found: {npz_path}"

        data = np.load(npz_path)
        pairs = data["env_map_pairs"]          # (N, 2, H, W)
        re_arr = data["reward_enums"]          # (N,)
        ts_arr = data["timesteps"]             # (N,)
        data.close()

        # reward_enum 필터
        if self.reward_enums is not None:
            mask = np.isin(re_arr, self.reward_enums)
            pairs = pairs[mask]
            re_arr = re_arr[mask]
            ts_arr = ts_arr[mask]

        object.__setattr__(self, "_pairs", pairs)
        object.__setattr__(self, "_reward_enums_arr", re_arr)
        object.__setattr__(self, "_timesteps", ts_arr)

        # metadata
        meta_path = npz_path.parent / "metadata.json"
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                object.__setattr__(self, "_metadata", json.load(f))
        else:
            object.__setattr__(self, "_metadata", {})

    # ── 기본 접근 ────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return self._pairs.shape[0]

    def __getitem__(self, idx) -> Union[MapTransitionPair, "CPCGRLBufferDataset"]:
        if isinstance(idx, (int, np.integer)):
            return MapTransitionPair(
                before=self._pairs[idx, 0],
                after=self._pairs[idx, 1],
                reward_enum=int(self._reward_enums_arr[idx]),
                timestep=int(self._timesteps[idx]),
            )
        # slice / fancy indexing → 새 dataset 반환
        return self._subset(idx)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __repr__(self) -> str:
        re_list = sorted(set(self._reward_enums_arr.tolist()))
        return f"CPCGRLBufferDataset(n={len(self)}, reward_enums={re_list})"

    # ── 필터링 ───────────────────────────────────────────────────────────

    def by_reward_enum(self, *enums: int) -> "CPCGRLBufferDataset":
        """특정 reward_enum 만 필터링한 서브셋 반환."""
        mask = np.isin(self._reward_enums_arr, enums)
        return self._subset(mask)

    # ── 배치 접근 (numpy 배열) ───────────────────────────────────────────

    @property
    def pairs(self) -> np.ndarray:
        """전체 (N, 2, H, W) 배열."""
        return self._pairs

    @property
    def before_maps(self) -> np.ndarray:
        """(N, H, W) — 모든 before 맵."""
        return self._pairs[:, 0]

    @property
    def after_maps(self) -> np.ndarray:
        """(N, H, W) — 모든 after 맵."""
        return self._pairs[:, 1]

    @property
    def reward_enums_array(self) -> np.ndarray:
        """(N,) — reward_enum 배열."""
        return self._reward_enums_arr

    @property
    def timesteps_array(self) -> np.ndarray:
        """(N,) — timestep 배열."""
        return self._timesteps

    @property
    def metadata(self) -> dict:
        """빌드 메타데이터."""
        return self._metadata

    @property
    def map_shape(self) -> tuple:
        """단일 맵 shape (H, W)."""
        return tuple(self._pairs.shape[2:])

    @property
    def available_reward_enums(self) -> List[int]:
        """데이터에 존재하는 reward_enum 목록."""
        return sorted(set(self._reward_enums_arr.tolist()))

    # ── 통계 ─────────────────────────────────────────────────────────────

    def summary(self) -> dict:
        """데이터셋 요약 통계."""
        re_arr = self._reward_enums_arr
        return {
            "total_pairs": len(self),
            "map_shape": self.map_shape,
            "tile_min": int(self._pairs.min()),
            "tile_max": int(self._pairs.max()),
            "reward_enum_distribution": {
                int(rn): int((re_arr == rn).sum())
                for rn in sorted(set(re_arr.tolist()))
            },
        }

    # ── 샘플링 ───────────────────────────────────────────────────────────

    def sample(self, n: int = 1, seed: int | None = None) -> Union[MapTransitionPair, list]:
        """랜덤 샘플링. n=1 이면 단일 Pair, n>1 이면 리스트."""
        rng = np.random.default_rng(seed)
        indices = rng.choice(len(self), size=n, replace=False)
        if n == 1:
            return self[int(indices[0])]
        return [self[int(i)] for i in indices]

    # ── 내부 유틸 ────────────────────────────────────────────────────────

    def _subset(self, idx) -> "CPCGRLBufferDataset":
        """인덱스/마스크로 서브셋 생성 (npz 재로드 없이)."""
        new = object.__new__(CPCGRLBufferDataset)
        object.__setattr__(new, "npz_path", self.npz_path)
        object.__setattr__(new, "reward_enums", None)
        object.__setattr__(new, "_pairs", self._pairs[idx])
        object.__setattr__(new, "_reward_enums_arr", self._reward_enums_arr[idx])
        object.__setattr__(new, "_timesteps", self._timesteps[idx])
        object.__setattr__(new, "_metadata", self._metadata)
        return new

