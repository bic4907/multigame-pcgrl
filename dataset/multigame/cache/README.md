# MultiGame Cache

This directory stores local preprocessing cache files for `MultiGameDataset`.

- Location used by default: `dataset/multigame/cache/artifacts/`
- Cache key includes:
  - dataset init args (`vglc_root`, `dungeon_root`, `vglc_games`, etc.)
  - hash of `dataset/multigame/*.py` source code
  - schema version

This means:
- same code + same args -> cache hit
- changed code or args -> cache miss (rebuild)

You can commit cache files to git if desired.

