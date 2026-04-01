from functools import partial

import jax
from jax import vmap
import chex
import jax.numpy as jnp

from .fitnesses import *
from .fitnesses.multigame_amount import get_multigame_amount_fitness
from .weights import FitnessWeight

@partial(jax.jit, static_argnums=())
def get_fitness_batch(
    reward_i: chex.Array,
    condition: chex.Array,
    curr_env_map: chex.Array,
    normal_weights: chex.Array,
) -> chex.Array:
    """Compute batch fitness.

    인덱스 매핑은 reward.py 와 동일:
    0=none, 1=region, 2=path_length,
    3=interactive, 4=hazard, 5=collectable, 6+=none

    fitness 에서는 placement(배치품질)는 사용하지 않고
    amount(개수 달성도)만 평가한다.
    """
    fitness_funcs = [
        lambda cond, curr_map: 0.0,  # 0: no-op
        lambda cond, curr_map: get_region_fitness(curr_map, cond[0]) * FitnessWeight.REGION,  # 1: region
        lambda cond, curr_map: get_path_length_fitness(curr_map, cond[1]) * FitnessWeight.PATH_LENGTH,  # 2: path_length
        lambda cond, curr_map: get_multigame_amount_fitness(curr_map, cond[2], tile_name="interactive") * FitnessWeight.MONSTER,  # 3: interactive
        lambda cond, curr_map: get_multigame_amount_fitness(curr_map, cond[3], tile_name="hazard") * FitnessWeight.MONSTER,  # 4: hazard
        lambda cond, curr_map: get_multigame_amount_fitness(curr_map, cond[4], tile_name="collectable") * FitnessWeight.MONSTER,  # 5: collectable
        lambda cond, curr_map: 0.0,  # 6+: no-op
    ]

    # Map indices to functions using `switch`
    def compute_value(func_idx, cond_value, _curr_env_map):
        reward_values = jax.vmap(lambda idx: jax.lax.switch(idx, fitness_funcs, cond_value, _curr_env_map))(
            func_idx)
        return jnp.sum(reward_values)

    compute_reward_vmap = vmap(compute_value, in_axes=(0, 0, 0))
    rewards = compute_reward_vmap(reward_i, condition, curr_env_map)

    # multiply by weights
    rewards = jnp.divide(rewards, normal_weights)

    return jax.lax.stop_gradient(rewards)
