#!/usr/bin/env python3
"""
Unified Dataset Viewer Launcher

Run the dataset viewer with support for Dungeon, POKEMON, Boxoban, and Doom datasets.

Usage:
    python -m dataset.multigame.viewer.server [--host HOST] [--port PORT]

Example:
    python -m dataset.multigame.viewer.server --host 127.0.0.1 --port 8765
"""
import argparse
import sys
from pathlib import Path

# Ensure we can import from the project root
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from dataset.multigame.viewer.server import run_server


def main():
    parser = argparse.ArgumentParser(
        description="Unified Dataset Viewer - Browse Dungeon, POKEMON, Boxoban, and Doom levels"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="Port to bind to (default: 8765)",
    )
    parser.add_argument(
        "--dungeon-root",
        type=str,
        default=None,
        help="Dungeon dataset root path (auto-detected if not provided)",
    )
    parser.add_argument(
        "--pokemon-root",
        type=str,
        default=None,
        help="POKEMON (Five-Dollar-Model) dataset root path (auto-detected if not provided)",
    )
    parser.add_argument(
        "--boxoban-root",
        type=str,
        default=None,
        help="Boxoban dataset root path (auto-detected if not provided)",
    )
    parser.add_argument(
        "--doom-root",
        type=str,
        default=None,
        help="Doom dataset root path (auto-detected if not provided)",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("🎮 Unified Dataset Viewer")
    print("=" * 70)
    print(f"🔗 Starting server on http://{args.host}:{args.port}")
    print()
    print("📂 Supported datasets:")
    print("  ✓ Dungeon (dungeon_level_dataset)")
    print("  ✓ POKEMON (Five-Dollar-Model)")
    print("  ✓ Boxoban (boxoban_levels)")
    print("  ✓ Doom (doom_levels)")
    print()
    print("💡 Tips:")
    print("  - Use arrow keys (← →) to navigate samples")
    print("  - Click rendering mode tabs to switch views (Raw/Unified/Symbol)")
    print("  - Check the legend panel for tile mappings")
    print("  - Album view lets you browse multiple samples at once")
    print()
    print("Press Ctrl+C to stop the server")
    print("=" * 70)

    kwargs = {
        "host": args.host,
        "port": args.port,
    }

    if args.dungeon_root:
        kwargs["dungeon_root"] = args.dungeon_root
    if args.pokemon_root:
        kwargs["pokemon_root"] = args.pokemon_root
    if args.boxoban_root:
        kwargs["boxoban_root"] = args.boxoban_root
    if args.doom_root:
        kwargs["doom_root"] = args.doom_root

    run_server(**kwargs)


if __name__ == "__main__":
    main()


