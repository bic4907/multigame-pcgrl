from __future__ import annotations

from collections import deque
from typing import Iterable

import numpy as np

from .constants import COUNT_TILE_ID_BY_REWARD_ENUM, PASSABLE_TILE_IDS


def _neighbors(y: int, x: int, height: int, width: int) -> Iterable[tuple[int, int]]:
    if y > 0:
        yield y - 1, x
    if y + 1 < height:
        yield y + 1, x
    if x > 0:
        yield y, x - 1
    if x + 1 < width:
        yield y, x + 1

def _path_metric_and_coords(level: np.ndarray) -> tuple[float, list[tuple[int, int]]]:
    passable = np.isin(level, list(PASSABLE_TILE_IDS))
    coords = list(zip(*np.where(passable)))
    if not coords:
        return 0.0, []

    height, width = level.shape
    best_dist = -1
    best_path: list[tuple[int, int]] = []

    for start in coords:
        dist = {start: 0}
        parent: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
        q = deque([start])
        while q:
            y, x = q.popleft()
            for ny, nx in _neighbors(y, x, height, width):
                nxt = (ny, nx)
                if not passable[ny, nx] or nxt in dist:
                    continue
                dist[nxt] = dist[(y, x)] + 1
                parent[nxt] = (y, x)
                q.append(nxt)

        if not dist:
            continue
        end, dist_value = max(dist.items(), key=lambda item: item[1])
        if dist_value > best_dist:
            path = []
            cur: tuple[int, int] | None = end
            while cur is not None:
                path.append(cur)
                cur = parent[cur]
            best_dist = dist_value
            best_path = list(reversed(path))

    return float(max(best_dist, 0)), best_path

def _compute_metric(level: np.ndarray, reward_enum: int) -> float:
    if reward_enum == 1:
        metric, _ = _path_metric_and_coords(level)
        return metric
    if reward_enum in COUNT_TILE_ID_BY_REWARD_ENUM:
        return float(np.sum(level == COUNT_TILE_ID_BY_REWARD_ENUM[reward_enum]))
    if reward_enum == 0:
        return float(_count_regions(level))
    return float("nan")

def _count_regions(level: np.ndarray) -> int:
    passable = np.isin(level, list(PASSABLE_TILE_IDS))
    visited = np.zeros(passable.shape, dtype=bool)
    height, width = level.shape
    regions = 0
    for y, x in zip(*np.where(passable)):
        if visited[y, x]:
            continue
        regions += 1
        visited[y, x] = True
        q = deque([(int(y), int(x))])
        while q:
            cy, cx = q.popleft()
            for ny, nx in _neighbors(cy, cx, height, width):
                if passable[ny, nx] and not visited[ny, nx]:
                    visited[ny, nx] = True
                    q.append((ny, nx))
    return regions

