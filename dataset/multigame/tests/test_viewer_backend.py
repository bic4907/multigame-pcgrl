from __future__ import annotations

import sys
from pathlib import Path

import pytest

_DATASET_ROOT = Path(__file__).parent.parent.parent
if str(_DATASET_ROOT) not in sys.path:
    sys.path.insert(0, str(_DATASET_ROOT))

from ..handlers.boxoban_handler import BoxobanTile
from ..viewer.backend import DatasetViewerBackend


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


@pytest.fixture()
def mini_roots(tmp_path: Path):
    vglc_root = tmp_path / "TheVGLC"
    boxoban_root = tmp_path / "boxoban_levels"
    dungeon_root = tmp_path / "dungeon_level_dataset_missing"

    # Minimal VGLC Zelda sample.
    _write(
        vglc_root / "The Legend of Zelda" / "Processed" / "z1.txt",
        "WWWW\nW--W\nW--W\nWWWW\n",
    )

    # Boxoban with 2 valid 10x10 levels + 1 invalid level.
    _write(
        boxoban_root / "hard" / "000.txt",
        "\n".join([
            "; level 0",
            "##########",
            "#   $    #",
            "#   .    #",
            "#   @    #",
            "#        #",
            "#        #",
            "#        #",
            "#        #",
            "#        #",
            "##########",
            "; level 1",
            "##########",
            "# $   .  #",
            "#   @    #",
            "#        #",
            "#        #",
            "#        #",
            "#        #",
            "#        #",
            "#        #",
            "##########",
            "; invalid",
            "#####",
            "# @ #",
            "#####",
        ])
        + "\n",
    )

    return vglc_root, dungeon_root, boxoban_root


def test_games_with_counts(mini_roots):
    vglc_root, dungeon_root, boxoban_root = mini_roots
    backend = DatasetViewerBackend(
        vglc_root=vglc_root,
        dungeon_root=dungeon_root,
        boxoban_root=boxoban_root,
    )

    rows = {row["game"]: row["count"] for row in backend.games_with_counts()}
    assert rows["zelda"] == 1
    assert rows["boxoban"] == 2
    assert "dungeon" not in rows


def test_boxoban_sample_payload(mini_roots):
    vglc_root, dungeon_root, boxoban_root = mini_roots
    backend = DatasetViewerBackend(
        vglc_root=vglc_root,
        dungeon_root=dungeon_root,
        boxoban_root=boxoban_root,
    )

    sample = backend.get_sample(game="boxoban", index=0)

    assert sample["count"] == 2
    assert sample["shape"] == [16, 16]
    assert sample["array"][0][0] == BoxobanTile.WALL
    assert sample["array"][15][15] == BoxobanTile.WALL
    assert "1" in sample["palette"]

    # ── 새 필드 검증: unified + tile_names ───────────────────────────────
    assert "unified_array" in sample
    assert "unified_palette" in sample
    assert "unified_names" in sample
    assert "tile_names" in sample

    assert isinstance(sample["unified_array"], list)
    assert len(sample["unified_array"]) == 16
    assert isinstance(sample["unified_palette"], dict)
    assert isinstance(sample["tile_names"], dict)  # wall color


def test_boxoban_index_out_of_range(mini_roots):
    vglc_root, dungeon_root, boxoban_root = mini_roots
    backend = DatasetViewerBackend(
        vglc_root=vglc_root,
        dungeon_root=dungeon_root,
        boxoban_root=boxoban_root,
    )

    with pytest.raises(IndexError):
        backend.get_sample(game="boxoban", index=9)
