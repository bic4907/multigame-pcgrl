from __future__ import annotations

import os
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_EVAL_CACHE_DIR = _REPO_ROOT / ".eval_cache"

EVAL_CACHE_ROOT = Path(
    os.environ.get("EVAL_CACHE_DIR", str(_DEFAULT_EVAL_CACHE_DIR))
).resolve()
CLIP_EMBED_CACHE_DIR = EVAL_CACHE_ROOT / "clip_latent_embeddings"
DECODER_REWARD_CACHE_DIR = EVAL_CACHE_ROOT / "decoder_reward_predictions"

# reward_enum -> readable name
REWARD_ENUM_NAMES = {
    0: "region",
    1: "path_length",
    2: "interactable",
    3: "hazard",
    4: "collectable",
}

