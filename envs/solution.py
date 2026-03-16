import warnings
from functools import partial

from flax import struct
import jax
import chex
import jax.numpy as jnp
from jax import jit

from envs.pathfinding import check_event, check_event_jit, get_max_path_length_static
from envs.probs.dungeon3 import Dungeon3Tiles, Dungeon3Problem
from envs.utils import generate_color_palette, generate_offset_palette


@struct.dataclass
class Solutions:
    n: chex.Array
    index: chex.Array
    color: chex.Array
    path: chex.Array
    offset: chex.Array
    layover: chex.Array
    enemy_encounter: chex.Array


def get_solution(env_map) -> Solutions:
    # NOTE: EvaluatorSolution
    """
    Generate solutions for a scenario and store them in a Solutions object.
    """

    p_xy = jnp.argwhere(env_map == Dungeon3Tiles.PLAYER, size=1, fill_value=-1)[0]
    d_xy = jnp.argwhere(env_map == Dungeon3Tiles.DOOR, size=1, fill_value=-1)[0]

    passable_tiles = Dungeon3Problem.passable_tiles

    color_palette = generate_color_palette(30)
    offset_palette = generate_offset_palette(9)

    exist_keys = jnp.argwhere(env_map == Dungeon3Tiles.KEY, size=30, fill_value=-1)

    # Initialize maximum solution size (e.g., 30)
    max_solutions = 30

    max_path_len = get_max_path_length_static(map_shape=env_map.shape)

    # Initialize arrays for solutions
    indices = jnp.full((max_solutions,), -1, dtype=jnp.int32)
    colors = jnp.zeros((max_solutions, 4), dtype=jnp.uint8)  # Assuming RGBA colors
    paths = jnp.full((max_solutions, max_path_len * 2, 2), -1, dtype=jnp.int32)
    offsets = jnp.zeros((max_solutions, 2), dtype=jnp.int32)
    layovers = jnp.full((max_solutions, 2), -1, dtype=jnp.int32)
    enemy_encounters = jnp.zeros((max_solutions, 3), dtype=jnp.int32)

    sol_cnt = 0

    # Loop over keys and compute solutions
    for i, key in enumerate(exist_keys):
        # Skip invalid keys
        if jnp.all(key == jnp.array([-1, -1])):
            continue

        # Check for valid solutions
        has_solution, enemy_encounter, path = check_event_jit(
            env_map=env_map,
            passable_tiles=passable_tiles,
            src=p_xy,
            key=key,
            trg=d_xy,
            exist_keys=exist_keys,
        )

        if has_solution:
            color = color_palette[sol_cnt % len(color_palette)]
            offset = offset_palette[sol_cnt % len(offset_palette)]

            # Update solution properties
            indices = indices.at[sol_cnt].set(i)
            colors = colors.at[sol_cnt].set(color)
            paths = paths.at[sol_cnt].set(path)
            offsets = offsets.at[sol_cnt].set(offset)
            layovers = layovers.at[sol_cnt].set(key)
            enemy_encounters = enemy_encounters.at[sol_cnt].set(enemy_encounter)

            sol_cnt += 1

        # Break if max solutions are reached
        if sol_cnt >= max_solutions:
            break

    # Trim the arrays to the actual number of solutions
    solutions = Solutions(
        n=sol_cnt,
        index=indices,
        color=colors,
        path=paths,
        offset=offsets,
        layover=layovers,
        enemy_encounter=enemy_encounters,
    )

    return solutions


@jax.jit
def get_solution_jit(env_map) -> Solutions:
    """
    Generate solutions for a scenario in a JAX-compatible way.
    """

    p_xy = jnp.argwhere(env_map == Dungeon3Tiles.PLAYER, size=1, fill_value=-1)[0]
    d_xy = jnp.argwhere(env_map == Dungeon3Tiles.DOOR, size=1, fill_value=-1)[0]

    passable_tiles = Dungeon3Problem.passable_tiles

    color_palette = generate_color_palette(30)
    offset_palette = generate_offset_palette(9)

    exist_keys = jnp.argwhere(env_map == Dungeon3Tiles.KEY, size=30, fill_value=-1)

    max_solutions = 30
    max_path_len = get_max_path_length_static(map_shape=env_map.shape)

    # Initialize arrays for solutions
    indices = jnp.full((max_solutions,), -1, dtype=jnp.int32)
    colors = jnp.zeros((max_solutions, 4), dtype=jnp.uint8)  # Assuming RGBA colors
    paths = jnp.full((max_solutions, max_path_len * 2, 2), -1, dtype=jnp.int32)
    offsets = jnp.zeros((max_solutions, 2), dtype=jnp.int32)
    layovers = jnp.full((max_solutions, 2), -1, dtype=jnp.int32)
    enemy_encounters = jnp.zeros((max_solutions, 3), dtype=jnp.int32)

    def process_key(i, carry):
        sol_cnt, indices, colors, paths, offsets, layovers, enemy_encounters = carry
        key = exist_keys[i]

        # Skip invalid keys
        def skip_invalid(_):
            return sol_cnt, indices, colors, paths, offsets, layovers, enemy_encounters

        # Process valid keys
        def process_valid(_):
            has_solution, enemy_encounter, path = check_event_jit(
                env_map=env_map,
                passable_tiles=passable_tiles,
                src=p_xy,
                key=key,
                trg=d_xy,
                exist_keys=exist_keys,
            )

            def add_solution(_):
                color = color_palette[jnp.mod(sol_cnt, color_palette.shape[0])]
                offset = offset_palette[jnp.mod(sol_cnt, offset_palette.shape[0])]

                indices_updated = indices.at[sol_cnt].set(i)
                colors_updated = colors.at[sol_cnt].set(color)
                paths_updated = paths.at[sol_cnt].set(path)
                offsets_updated = offsets.at[sol_cnt].set(offset)
                layovers_updated = layovers.at[sol_cnt].set(key)
                enemy_encounters_updated = enemy_encounters.at[sol_cnt].set(
                    enemy_encounter
                )

                return (
                    sol_cnt + 1,
                    indices_updated,
                    colors_updated,
                    paths_updated,
                    offsets_updated,
                    layovers_updated,
                    enemy_encounters_updated,
                )

            return jax.lax.cond(
                has_solution,
                add_solution,
                lambda _: (
                    sol_cnt,
                    indices,
                    colors,
                    paths,
                    offsets,
                    layovers,
                    enemy_encounters,
                ),
                operand=None,
            )

        return jax.lax.cond(
            jnp.all(key == jnp.array([-1, -1])),
            skip_invalid,
            process_valid,
            operand=None,
        )

    # Initial state for the loop
    initial_state = (0, indices, colors, paths, offsets, layovers, enemy_encounters)

    # Iterate over exist_keys using JAX
    sol_cnt, indices, colors, paths, offsets, layovers, enemy_encounters = (
        jax.lax.fori_loop(
            0,
            exist_keys.shape[0],
            process_key,
            initial_state,
        )
    )

    # Trim the arrays to the actual number of solutions
    solutions = Solutions(
        n=sol_cnt,
        index=indices,
        color=colors,
        path=paths,
        offset=offsets,
        layover=layovers,
        enemy_encounter=enemy_encounters,
    )

    return solutions


if __name__ == "__main__":
    from debug.scenario_levels import AllLevels

    level = AllLevels[3]

    # solutions = get_solution(level)

    print(get_solution_jit(level).n)
