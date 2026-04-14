"""
dataset/cpcgrl_buffer/dataset.py
================================
CPCGRL pair dataset 로더.
(game, reward_enum) 별 개별 .npz 파일을 읽어 통합 접근을 제공한다.

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

_DEFAULT_PAIRS_DIR = Path(__file__).parent / "pairs"


@dataclass
class CPCGRLBufferDataset:
    """CPCGRL pair dataset 로더.

    dataset/cpcgrl_buffer/pairs/ 폴더의 개별 .npz 파일들을 읽는다.
    game, reward_enum 으로 필터링 가능.

    Parameters
    ----------
    pairs_dir : str or Path, optional
        pairs 폴더 경로. 기본값은 같은 폴더의 pairs/.
    games : list[str], optional
        특정 게임만 로드. None 이면 전체.
    reward_enums : list[int], optional
        특정 reward_enum 만 로드. None 이면 전체.

    Examples
    --------
    >>> ds = CPCGRLBufferDataset()
    >>> len(ds)
    70000
    >>> ds[0]
    MapTransitionPair(game='doom', re=0, ts=..., map=16x16, changes=1)
    >>> ds.by_game("doom")
    CPCGRLBufferDataset(n=14000, games=['doom'], reward_enums=[0,1,2,3,4])
    >>> ds.by_reward_enum(1, 2)
    CPCGRLBufferDataset(n=28000, games=[...], reward_enums=[1,2])
    """
    pairs_dir: Union[str, Path] = field(default_factory=lambda: _DEFAULT_PAIRS_DIR)
    games: Optional[List[str]] = None
    reward_enums: Optional[List[int]] = None

    # ── 내부 상태 (post_init 에서 로드) ──
    _pairs: np.ndarray = field(init=False, repr=False)
    _games_arr: np.ndarray = field(init=False, repr=False)       # object array of str
    _reward_enums_arr: np.ndarray = field(init=False, repr=False)
    _timesteps: np.ndarray = field(init=False, repr=False)
    _metadata: dict = field(init=False, repr=False)

    def __post_init__(self):
        pairs_dir = Path(self.pairs_dir)
        assert pairs_dir.exists(), f"Pairs dir not found: {pairs_dir}"

        # metadata 로드
        meta_path = pairs_dir / "metadata.json"
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
        else:
            metadata = {}

        # 파일 매니페스트에서 필터링할 대상 결정
        file_list = metadata.get("files", [])
        if not file_list:
            # metadata 없으면 파일명 패턴으로 탐색
            import re as _re
            for p in sorted(pairs_dir.glob("pairs_*_re*.npz")):
                m = _re.match(r"pairs_(\w+)_re(\d+)\.npz", p.name)
                if m:
                    file_list.append({
                        "file": p.name,
                        "game": m.group(1),
                        "reward_enum": int(m.group(2)),
                    })

        # game / reward_enum 필터
        targets = file_list
        if self.games is not None:
            targets = [f for f in targets if f["game"] in self.games]
        if self.reward_enums is not None:
            targets = [f for f in targets if f["reward_enum"] in self.reward_enums]

        assert targets, (
            f"No matching files for games={self.games}, reward_enums={self.reward_enums} "
            f"in {pairs_dir}"
        )

        # npz 파일들 로드 & 병합
        all_pairs, all_games, all_re, all_ts = [], [], [], []
        for entry in targets:
            fpath = pairs_dir / entry["file"]
            if not fpath.exists():
                print(f"  WARNING: {fpath} not found, skipping")
                continue
            data = np.load(fpath)
            n = data["env_map_pairs"].shape[0]
            all_pairs.append(data["env_map_pairs"])
            all_re.append(data["reward_enums"])
            all_ts.append(data["timesteps"])
            all_games.append(np.full(n, entry["game"], dtype=object))
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
        # slice / fancy indexing → 새 dataset 반환
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
        object.__setattr__(new, "pairs_dir", self.pairs_dir)
        object.__setattr__(new, "games", None)
        object.__setattr__(new, "reward_enums", None)
        object.__setattr__(new, "_pairs", self._pairs[idx])
        object.__setattr__(new, "_games_arr", self._games_arr[idx])
        object.__setattr__(new, "_reward_enums_arr", self._reward_enums_arr[idx])
        object.__setattr__(new, "_timesteps", self._timesteps[idx])
        object.__setattr__(new, "_metadata", self._metadata)
        return new

