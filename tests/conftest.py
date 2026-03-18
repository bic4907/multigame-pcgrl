"""tests/conftest.py — tests/ 전용 fixtures."""
import pytest


@pytest.fixture()
def multigame_env_info() -> dict:
    """make_multigame_env() 로 생성한 env의 tile 정보."""
    from envs.probs.multigame import make_multigame_env
    env, _ = make_multigame_env()
    return {
        "all_tiles":        [t.name for t in env.tile_enum],
        "n_all_tiles":      len(env.tile_enum),
        "editable_tiles":   [t.name for t in env.rep.editable_tile_enum],
        "n_editable_tiles": env.rep.n_editable_tiles,
        "unavailable_tiles": env.unavailable_tiles,
    }

