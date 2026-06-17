from __future__ import annotations

from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from conf.config import CLIPDecoderTrainConfig


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GAMES = ("dungeon", "pokemon", "sokoban", "doom", "zelda")
GAME_ALIASES = {
    "doom2": "doom",
}
GAME_COLORS = {
    "dungeon": "#1f77b4",
    "pokemon": "#2ca02c",
    "sokoban": "#9467bd",
    "doom": "#17becf",
    "zelda": "#d62728",
}
TYPE_MARKERS = {"state": "o", "text": "x"}


def cfg_get(config: DictConfig | CLIPDecoderTrainConfig, path: str, default=None):
    if isinstance(config, DictConfig):
        value = OmegaConf.select(config, path, default=default)
        return default if value is None else value

    cur = config
    for part in path.split("."):
        if not hasattr(cur, part):
            return default
        cur = getattr(cur, part)
        if cur is None:
            return default
    return cur


def repo_relative_path(raw) -> Path:
    path = Path(str(raw)).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def canonical_game_name(game: str) -> str:
    return GAME_ALIASES.get(str(game).strip(), str(game).strip())


def normalize_games(raw) -> list[str]:
    if raw is None:
        return list(DEFAULT_GAMES)
    if isinstance(raw, str):
        if raw.strip().lower() == "all":
            return list(DEFAULT_GAMES)
        raw_games = [g.strip() for g in raw.split(",") if g.strip()]
    else:
        raw_games = [str(g).strip() for g in raw if str(g).strip()]

    games = []
    seen = set()
    for game in raw_games:
        canonical = canonical_game_name(game)
        if canonical not in seen:
            games.append(canonical)
            seen.add(canonical)
    return games
