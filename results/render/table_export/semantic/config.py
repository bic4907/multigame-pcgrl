from __future__ import annotations

import json
from pathlib import Path

from ..models import CandidateRow, RenderConfigPanels


def _load_render_config(path_value: str | Path) -> tuple[RenderConfigPanels, dict]:
    path = Path(path_value)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"render_config must be a JSON object: {path}")
    raw_panels = payload.get("panels", {})
    if not isinstance(raw_panels, dict):
        raise ValueError(f"render_config.panels must be a nested object: {path}")
    scope = payload.get("scope", {})
    if not isinstance(scope, dict):
        scope = {}

    panels: RenderConfigPanels = {}
    for method, method_panels in raw_panels.items():
        if not isinstance(method_panels, dict):
            continue
        for game, game_panels in method_panels.items():
            if not isinstance(game_panels, dict):
                continue
            for feature, feature_panels in game_panels.items():
                if not isinstance(feature_panels, dict):
                    continue
                for side, panel in feature_panels.items():
                    if str(side).lower() not in {"low", "mid", "high"} or not isinstance(panel, dict):
                        continue
                    row_i = panel.get("row_i")
                    seed = panel.get("seed")
                    if row_i is None or seed is None:
                        continue
                    panels[(str(method), str(game).lower(), str(feature), str(side).lower())] = {
                        "row_i": str(row_i),
                        "seed": int(float(seed)),
                    }
    return panels, {"scope": scope}

def _render_config_candidate(
    panels: RenderConfigPanels,
    method_candidates: dict[tuple[str, str], CandidateRow],
    method: str,
    game: str,
    feature: str,
    side: str,
) -> tuple[CandidateRow, int] | None:
    panel = panels.get((method, game, feature, side))
    if panel is None:
        return None
    candidate = method_candidates.get((game, panel["row_i"]))
    if candidate is None:
        raise RuntimeError(
            f"render_config requested missing candidate: {method}/{game}/{feature}/{side} row_i={panel['row_i']}"
        )
    seed = int(panel["seed"])
    if seed not in candidate.seed_metrics:
        raise RuntimeError(
            f"render_config requested missing seed: {method}/{game}/{feature}/{side} "
            f"row_i={panel['row_i']} seed={seed}"
        )
    return candidate, seed
