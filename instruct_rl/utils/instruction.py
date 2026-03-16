from functools import partial

import jax
import jax.numpy as jnp
from instruct_rl.utils.augmentation import augment_levels


def update_instruction(inst_batch, inst_dataset, mask, rng, n_envs):
    random_indices = jax.random.randint(rng, (n_envs,), 0, inst_dataset.reward_i.shape[0])
    instruct_sample = jax.tree.map(lambda x: x[random_indices], inst_dataset)

    def apply_mask(new_val, old_val):
        expand_dims = len(new_val.shape) - 1
        broadcast_mask = mask.reshape((n_envs,) + (1,) * expand_dims)
        return jnp.where(broadcast_mask, new_val, old_val)

    updated_inst = jax.tree.map(apply_mask, instruct_sample, inst_batch)

    return updated_inst


def sample_levels(level_db, instruction_batch, rng, augment):
    """
    Args:
        level_db: jnp.ndarray of shape (num_conditions, num_levels, 16, 16)
        condition_id: jnp.ndarray of shape (num_envs,) or (num_envs, 1)
        rng: jax.random.PRNGKey
    Returns:
        sampled_levels: jnp.ndarray of shape (num_envs, 16, 16)
    """
    condition_id = instruction_batch.condition_id
    num_envs = jnp.shape(condition_id)[0]
    num_levels = jnp.shape(level_db)[1]

    # flatten condition_id if needed
    condition_id = condition_id.squeeze()

    rng, subkey = jax.random.split(rng)
    level_indices = jax.random.randint(subkey, (num_envs,), 0, num_levels)

    def select_level(cond_id, lvl_idx):
        return level_db[cond_id, lvl_idx]

    sampled_levels = jax.vmap(select_level)(condition_id, level_indices)

    sampled_levels = jax.lax.cond(
        augment,
        lambda x: augment_levels(x, instruction_batch, rng),
        lambda x: x,
        sampled_levels,
    )

    return sampled_levels


@partial(jax.jit, static_argnums=(5))
def update_level_sample(level_batch, level_db, instruction_batch, mask, rng, augment):
    condition_id = instruction_batch.condition_id
    if condition_id.ndim == 2 and condition_id.shape[1] == 1:
        condition_id = condition_id[:, 0]  # (n_envs,)

    n_envs = jnp.shape(condition_id)[0]

    level_sample = sample_levels(level_db, instruction_batch, rng, augment)

    # masking
    def apply_mask(new_val, old_val):
        expand_dims = len(new_val.shape) - 1
        broadcast_mask = mask.reshape((n_envs,) + (1,) * expand_dims)
        return jnp.where(broadcast_mask, new_val, old_val)

    updated_level = jax.tree.map(apply_mask, level_sample, level_batch)


    return updated_level

