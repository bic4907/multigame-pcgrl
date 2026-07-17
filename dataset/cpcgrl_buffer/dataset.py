"""
dataset/cpcgrl_buffer/dataset.py
================================
CPCGRL pair dataset  to text.
text .npz file in  (game, reward_enum) text text to  savetext data  text
text text  text.

Usage:
    from dataset.cpcgrl_buffer import CPCGRLBufferDataset

    # all load
    ds = CPCGRLBufferDataset()
    print(len(ds))                          # total text text
    pair = ds[0]                            # MapTransitionPair
    print(pair.before.shape)                # (16, 16)
    print(pair.game)                        # 'doom'
    print(pair.reward_enum)                 # 3

    # text gametext
    ds_doom = CPCGRLBufferDataset(games=["doom"])

    # text game + reward_enum
    ds_doom_r1 = CPCGRLBufferDataset(games=["doom"], reward_enums=[1])

    #  text loadtext dataset in  filtering
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


# ── data class ────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class MapTransitionPair:
    """env_map  of  (before, after)  before   text 1text.

    Attributes
    ----------
    before : np.ndarray  (H, W) int32
        t text of  env_map.
    after : np.ndarray  (H, W) int32
        t+1 text of  env_map.
    game : str
        game name (doom, dungeon, pokemon, sokoban, zelda text).
    reward_enum : int
          text  text reward_enum (0~4).
    timestep : int
        before text of  total_timestep.
    """
    before: np.ndarray
    after: np.ndarray
    game: str
    reward_enum: int
    timestep: int

    @property
    def pair(self) -> np.ndarray:
        """(2, H, W) form to  return."""
        return np.stack([self.before, self.after], axis=0)

    @property
    def diff(self) -> np.ndarray:
        """after - before text map. tile  text abovetext non-zero."""
        return self.after.astype(np.int16) - self.before.astype(np.int16)

    @property
    def changed_mask(self) -> np.ndarray:
        """text tile abovetext boolean mask (H, W)."""
        return self.before != self.after

    @property
    def n_changes(self) -> int:
        """text tile text."""
        return int(self.changed_mask.sum())

    def __repr__(self) -> str:
        h, w = self.before.shape
        return (
            f"MapTransitionPair(game={self.game!r}, re={self.reward_enum}, "
            f"ts={self.timestep}, map={h}x{w}, changes={self.n_changes})"
        )


# ── dataset class ──────────────────────────────────────────────────────────

_DEFAULT_NPZ = Path(__file__).parent / "cpcgrl_pair_dataset.npz"

# {game}_re{rn} text text (text text _ts text)
_KEY_PATTERN = _re.compile(r"^(\w+)_re(\d+)$")


@dataclass
class CPCGRLBufferDataset:
    """CPCGRL pair dataset  to text.

    text .npz file in  {game}_re{rn} text to  savetext data  text text.
    game, reward_enum  as  filtering available.

    Parameters
    ----------
    npz_path : str or Path, optional
        .npz file path. default value  same folder of  cpcgrl_pair_dataset.npz.
    games : list[str], optional
        text gametext load. None  text all.
    reward_enums : list[int], optional
        text reward_enum text load. None  text all.

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

    # ── internal text (post_init  in  load) ──
    _pairs: np.ndarray = field(init=False, repr=False)
    _games_arr: np.ndarray = field(init=False, repr=False)
    _reward_enums_arr: np.ndarray = field(init=False, repr=False)
    _timesteps: np.ndarray = field(init=False, repr=False)
    _metadata: dict = field(init=False, repr=False)

    def __post_init__(self):
        npz_path = Path(self.npz_path)
        assert npz_path.exists(), f"Dataset not found: {npz_path}"

        data = np.load(npz_path, allow_pickle=True)

        # metadata load
        if "_metadata" in data:
            metadata = json.loads(str(data["_metadata"]))
        else:
            metadata = {}

        # text parsing: {game}_re{rn} form of  text extract
        group_keys = []
        for key in data.files:
            m = _KEY_PATTERN.match(key)
            if m:
                game, rn = m.group(1), int(m.group(2))
                group_keys.append((key, game, rn))

        # game / reward_enum filter
        if self.games is not None:
            group_keys = [(k, g, r) for k, g, r in group_keys if g in self.games]
        if self.reward_enums is not None:
            group_keys = [(k, g, r) for k, g, r in group_keys if r in self.reward_enums]

        assert group_keys, (
            f"No matching groups for games={self.games}, "
            f"reward_enums={self.reward_enums} in {npz_path}"
        )

        # data load & merge
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

    # ── default text ────────────────────────────────────────────────────────

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

    # ── filtering ───────────────────────────────────────────────────────────

    def by_game(self, *game_names: str) -> "CPCGRLBufferDataset":
        """text gametext filteringtext text return."""
        mask = np.isin(self._games_arr, game_names)
        return self._subset(mask)

    def by_reward_enum(self, *enums: int) -> "CPCGRLBufferDataset":
        """text reward_enum text filteringtext text return."""
        mask = np.isin(self._reward_enums_arr, enums)
        return self._subset(mask)

    def by_game_and_re(self, game: str, reward_enum: int) -> "CPCGRLBufferDataset":
        """text (game, reward_enum) text filtering."""
        mask = (self._games_arr == game) & (self._reward_enums_arr == reward_enum)
        return self._subset(mask)

    # ── batch text (numpy array) ───────────────────────────────────────────

    @property
    def pairs(self) -> np.ndarray:
        """all (N, 2, H, W) array."""
        return self._pairs

    @property
    def before_maps(self) -> np.ndarray:
        """(N, H, W) — text before map."""
        return self._pairs[:, 0]

    @property
    def after_maps(self) -> np.ndarray:
        """(N, H, W) — text after map."""
        return self._pairs[:, 1]

    @property
    def games_array(self) -> np.ndarray:
        """(N,) — game name array (object dtype)."""
        return self._games_arr

    @property
    def reward_enums_array(self) -> np.ndarray:
        """(N,) — reward_enum array."""
        return self._reward_enums_arr

    @property
    def timesteps_array(self) -> np.ndarray:
        """(N,) — timestep array."""
        return self._timesteps

    @property
    def metadata(self) -> dict:
        """build metadata."""
        return self._metadata

    @property
    def map_shape(self) -> tuple:
        """text map shape (H, W)."""
        return tuple(self._pairs.shape[2:])

    @property
    def available_games(self) -> List[str]:
        """data in  text  game list."""
        return sorted(set(self._games_arr.tolist()))

    @property
    def available_reward_enums(self) -> List[int]:
        """data in  text  reward_enum list."""
        return sorted(set(self._reward_enums_arr.tolist()))

    # ── text ─────────────────────────────────────────────────────────────

    def summary(self) -> dict:
        """dataset summary text."""
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

    # ── sampletext ───────────────────────────────────────────────────────────

    def sample(self, n: int = 1, seed: int | None = None) -> Union[MapTransitionPair, list]:
        """random sampletext. n=1  text text Pair, n>1  text text."""
        rng = np.random.default_rng(seed)
        indices = rng.choice(len(self), size=n, replace=False)
        if n == 1:
            return self[int(indices[0])]
        return [self[int(i)] for i in indices]

    # ── internal utility ────────────────────────────────────────────────────────

    def _subset(self, idx) -> "CPCGRLBufferDataset":
        """index/text to  text create (npz textload text )."""
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

