"""
dataset/cpcgrl_buffer/dataset.py
================================
CPCGRL pair dataset 로더.
단일 .npz 파일에서 (game, reward_enum) 별 키로 저장된 데이터를 읽어
통합 접근을 제공한다.

Usage:
    from dataset.cpcgrl_buffer import CPCGRLBufferDataset

    # 전체 로드
    ds = CPCGRLBufferDataset()
    print(len(ds))                          # 총 쌍 수
    pair = ds[0]                            # MapTransitionPair
    print(pair.before.shape)                # (16, 16)
    print(pair.game)                        # 'doom'
    print(pair.reward_enum)                 # 3

    # 특정 게임만
    ds_doom = CPCGRLBufferDataset(games=["doom"])

    # 특정 게임 + reward_enum
    ds_doom_r1 = CPCGRLBufferDataset(games=["doom"], reward_enums=[1])

    # 이미 로드된 데이터셋에서 필터링
    ds_zelda = ds.by_game("zelda")
    ds_re3 = ds.by_reward_enum(3)
"""
from __future__ import annotations

import json
import re as _re
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
    game : str
        게임 이름 (doom, dungeon, pokemon, sokoban, zelda 등).
    reward_enum : int
        이 쌍이 수집된 reward_enum (0~4).
    timestep : int
        before 시점의 total_timestep.
    """
    before: np.ndarray
    after: np.ndarray
    game: str
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
            f"MapTransitionPair(game={self.game!r}, re={self.reward_enum}, "
            f"ts={self.timestep}, map={h}x{w}, changes={self.n_changes})"
        )


# ── 데이터셋 클래스 ──────────────────────────────────────────────────────────

_DEFAULT_NPZ = Path(__file__).parent / "cpcgrl_pair_dataset.npz"

# {game}_re{rn} 키 패턴 (타임스탬프 키 _ts 제외)
_KEY_PATTERN = _re.compile(r"^(\w+)_re(\d+)$")


@dataclass
class CPCGRLBufferDataset:
    """CPCGRL pair dataset 로더.

    단일 .npz 파일에서 {game}_re{rn} 키로 저장된 데이터를 읽는다.
    game, reward_enum 으로 필터링 가능.

    Parameters
    ----------
    npz_path : str or Path, optional
        .npz 파일 경로. 기본값은 같은 폴더의 cpcgrl_pair_dataset.npz.
    games : list[str], optional
        특정 게임만 로드. None 이면 전체.
    reward_enums : list[int], optional
        특정 reward_enum 만 로드. None 이면 전체.

    Examples
    --------
    >>> ds = CPCGRLBufferDataset()
    >>> len(ds)
    54187
    >>> ds[0]
    MapTransitionPair(game='doom', re=0, ts=..., map=16x16, changes=1)
    >>> ds.by_game("doom")
    CPCGRLBufferDataset(n=15069, games=['doom'], reward_enums=[0,1,2,3,4])
    """
    npz_path: Union[str, Path] = field(default_factory=lambda: _DEFAULT_NPZ)
    games: Optional[List[str]] = None
    reward_enums: Optional[List[int]] = None

    # ── 내부 상태 (post_init 에서 로드) ──
    _pairs: np.ndarray = field(init=False, repr=False)
    _games_arr: np.ndarray = field(init=False, repr=False)
    _reward_enums_arr: np.ndarray = field(init=False, repr=False)
    _timesteps: np.ndarray = field(init=False, repr=False)
    _metadata: dict = field(init=False, repr=False)

    def __post_init__(self):
        npz_path = Path(self.npz_path)
        assert npz_path.exists(), f"Dataset not found: {npz_path}"

        data = np.load(npz_path, allow_pickle=True)

        # 메타데이터 로드
        if "_metadata" in data:
            metadata = json.loads(str(data["_metadata"]))
        else:
            metadata = {}

        # 키 파싱: {game}_re{rn} 형태의 키만 추출
        group_keys = []
        for key in data.files:
            m = _KEY_PATTERN.match(key)
            if m:
                game, rn = m.group(1), int(m.group(2))
                group_keys.append((key, game, rn))

        # game / reward_enum 필터
        if self.games is not None:
            group_keys = [(k, g, r) for k, g, r in group_keys if g in self.games]
        if self.reward_enums is not None:
            group_keys = [(k, g, r) for k, g, r in group_keys if r in self.reward_enums]

        assert group_keys, (
            f"No matching groups for games={self.games}, "
            f"reward_enums={self.reward_enums} in {npz_path}"
        )

        # 데이터 로드 & 병합
        all_pairs, all_games, all_re, all_ts = [], [], [], []
        for key, game, rn in sorted(group_keys):
            pairs = data[key]                      # (N, 2, H, W)
            ts_key = f"{key}_ts"
            ts = data[ts_key] if ts_key in data else np.zeros(pairs.shape[0], dtype=np.int64)
            n = pairs.shape[0]

            all_pairs.append(pairs)
            all_ts.append(ts)
            all_re.append(np.full(n, rn, dtype=np.int32))
            all_games.append(np.full(n, game, dtype=object))

        data.close()

        object.__setattr__(self, "_pairs", np.concatenate(all_pairs, axis=0))
        object.__setattr__(self, "_games_arr", np.concatenate(all_games, axis=0))
        object.__setattr__(self, "_reward_enums_arr", np.concatenate(all_re, axis=0))
        object.__setattr__(self, "_timesteps", np.concatenate(all_ts, axis=0))
        object.__setattr__(self, "_metadata", metadata)

    # ── 기본 접근 ────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return self._pairs.shape[0]

    def __getitem__(self, idx) -> Union[MapTransitionPair, "CPCGRLBufferDataset"]:
        if isinstance(idx, (int, np.integer)):
            return MapTransitionPair(
                before=self._pairs[idx, 0],
                after=self._pairs[idx, 1],
                game=str(self._games_arr[idx]),
                reward_enum=int(self._reward_enums_arr[idx]),
                timestep=int(self._timesteps[idx]),
            )
        return self._subset(idx)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __repr__(self) -> str:
        games = sorted(set(self._games_arr.tolist()))
        re_list = sorted(set(self._reward_enums_arr.tolist()))
        return (
            f"CPCGRLBufferDataset(n={len(self)}, "
            f"games={games}, reward_enums={re_list})"
        )

    # ── 필터링 ───────────────────────────────────────────────────────────

    def by_game(self, *game_names: str) -> "CPCGRLBufferDataset":
        """특정 게임만 필터링한 서브셋 반환."""
        mask = np.isin(self._games_arr, game_names)
        return self._subset(mask)

    def by_reward_enum(self, *enums: int) -> "CPCGRLBufferDataset":
        """특정 reward_enum 만 필터링한 서브셋 반환."""
        mask = np.isin(self._reward_enums_arr, enums)
        return self._subset(mask)

    def by_game_and_re(self, game: str, reward_enum: int) -> "CPCGRLBufferDataset":
        """특정 (game, reward_enum) 조합만 필터링."""
        mask = (self._games_arr == game) & (self._reward_enums_arr == reward_enum)
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
    def games_array(self) -> np.ndarray:
        """(N,) — game 이름 배열 (object dtype)."""
        return self._games_arr

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
    def available_games(self) -> List[str]:
        """데이터에 존재하는 게임 목록."""
        return sorted(set(self._games_arr.tolist()))

    @property
    def available_reward_enums(self) -> List[int]:
        """데이터에 존재하는 reward_enum 목록."""
        return sorted(set(self._reward_enums_arr.tolist()))

    # ── 통계 ─────────────────────────────────────────────────────────────

    def summary(self) -> dict:
        """데이터셋 요약 통계."""
        re_arr = self._reward_enums_arr
        games_arr = self._games_arr
        return {
            "total_pairs": len(self),
            "map_shape": self.map_shape,
            "tile_min": int(self._pairs.min()),
            "tile_max": int(self._pairs.max()),
            "games": self.available_games,
            "game_distribution": {
                g: int((games_arr == g).sum())
                for g in self.available_games
            },
            "reward_enum_distribution": {
                int(rn): int((re_arr == rn).sum())
                for rn in self.available_reward_enums
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
        object.__setattr__(new, "games", None)
        object.__setattr__(new, "reward_enums", None)
        object.__setattr__(new, "_pairs", self._pairs[idx])
        object.__setattr__(new, "_games_arr", self._games_arr[idx])
        object.__setattr__(new, "_reward_enums_arr", self._reward_enums_arr[idx])
        object.__setattr__(new, "_timesteps", self._timesteps[idx])
        object.__setattr__(new, "_metadata", self._metadata)
        return new

