from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any


def analyze_h5_file(h5_path: Path):
    import h5py

    stats = defaultdict(lambda: {
        "levels": 0,
        "seeds": set(),
        "tile_min": float("inf"),
        "tile_max": float("-inf"),
        "shapes": set(),
    })

    with h5py.File(str(h5_path), "r") as f:
        for key in f.keys():
            game = key.split("_")[0]
            stats[game]["levels"] += 1
            seeds = list(f[key].keys())
            stats[game]["seeds"].update(seeds)
            if not seeds:
                continue

            state = f[key][seeds[0]]["state"][:]
            stats[game]["tile_min"] = min(stats[game]["tile_min"], state.min())
            stats[game]["tile_max"] = max(stats[game]["tile_max"], state.max())
            stats[game]["shapes"].add(state.shape)

    result: dict[str, dict[str, Any]] = {}
    for game, data in stats.items():
        result[game] = {
            "levels": data["levels"],
            "seeds": len(data["seeds"]),
            "tile_range": (int(data["tile_min"]), int(data["tile_max"])),
            "shapes": list(data["shapes"]),
        }
    return result

