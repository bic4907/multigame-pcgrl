"""Static configuration for the transferability analysis.

Everything that is *not* computed from data lives here so the other modules stay
free of magic strings and hard-coded paths.
"""
from __future__ import annotations

from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
# results/transferbility/analysis/config.py -> repo root is parents[3].
HERE = Path(__file__).resolve().parent
TRANSFER_DIR = HERE.parent                       # results/transferbility
REPO_ROOT = HERE.parents[2]                      # multigame-pcgrl

RESULT_CSV = TRANSFER_DIR / "src" / "source_target_table_5seed.csv"
CACHE_DIR = REPO_ROOT / "dataset" / "multigame" / "cache" / "artifacts"
OUTPUT_DIR = TRANSFER_DIR / "output"

# ── Games ──────────────────────────────────────────────────────────────────────
GAMES = ["dungeon", "pokemon", "sokoban", "doom", "zelda"]
BASELINE_SOURCE = "none"                          # source==none row == no mixing baseline

# ── Reward enum mapping ─────────────────────────────────────────────────────────
# The result CSV labels rewards with human names; annotations use integer enums.
# ``overall`` is the aggregate score and has no single condition distribution.
REWARD_LABEL_TO_ENUM = {
    "Region": 0,
    "Path Length": 1,
    "Interactable": 2,
    "Hazard": 3,
    "Collectable": 4,
}
ENUM_TO_REWARD_LABEL = {v: k for k, v in REWARD_LABEL_TO_ENUM.items()}
OVERALL_LABEL = "overall"

# Ordered list of the five per-feature reward labels.
FEATURE_LABELS = list(REWARD_LABEL_TO_ENUM.keys())

# ── Feature presence ────────────────────────────────────────────────────────────
# Some games structurally lack a feature (its condition is identically zero).
# These are excluded from "does source distribution similarity help?" tests
# because the source cannot carry information about a feature it never contains.
# Derived from dataset/reward_annotations/README.md sub_condition table and
# verified empirically in data.py (all-zero / missing distributions).
ABSENT_FEATURES = {
    ("dungeon", 2),   # dungeon has no interactable tiles
    ("sokoban", 3),   # sokoban has no hazard tiles
    ("sokoban", 4),   # sokoban has no collectable tiles
}


def feature_present(game: str, enum: int) -> bool:
    """Return True if ``game`` actually contains reward feature ``enum``."""
    return (game, enum) not in ABSENT_FEATURES
