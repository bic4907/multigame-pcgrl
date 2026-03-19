#!/usr/bin/env python3
"""
Quick integration test for viewer rendering modes.
"""
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dataset.multigame.viewer.backend import DatasetViewerBackend


def test_payload_fields():
    """Verify that get_sample() includes unified_array, unified_palette, tile_names."""
    td = Path(tempfile.mkdtemp())
    vglc_root = td / "vglc"
    zelda_dir = vglc_root / "The Legend of Zelda" / "Processed"
    zelda_dir.mkdir(parents=True, exist_ok=True)
    (zelda_dir / "z1.txt").write_text("WWWW\nW--W\nW--W\nWWWW\n", encoding="utf-8")

    backend = DatasetViewerBackend(
        vglc_root=vglc_root,
        dungeon_root=td / "dungeon_missing",
        boxoban_root=td / "boxoban_missing",
    )

    sample = backend.get_sample("zelda", 0)

    required = [
        "game", "index", "count", "source_id", "shape",
        "array", "palette", "tile_names",
        "unified_array", "unified_palette", "unified_names",
        "instruction", "meta",
    ]
    missing = [k for k in required if k not in sample]
    if missing:
        print(f"FAIL: Missing keys: {missing}")
        return False

    print("✓ All required keys present")
    print(f"  array shape: {len(sample['array'])} × {len(sample['array'][0]) if sample['array'] else 0}")
    print(f"  unified_array shape: {len(sample['unified_array'])} × {len(sample['unified_array'][0]) if sample['unified_array'] else 0}")
    print(f"  palette entries: {len(sample['palette'])}")
    print(f"  unified_palette entries: {len(sample['unified_palette'])}")
    print(f"  tile_names entries: {len(sample['tile_names'])}")
    print(f"  unified_names entries: {len(sample['unified_names'])}")
    return True


if __name__ == "__main__":
    success = test_payload_fields()
    sys.exit(0 if success else 1)

