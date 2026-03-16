import jax
import jax.numpy as jnp
from jax import lax, jit


@jit
def rot90_jit(x, k):
    return jax.lax.switch(
        k,
        [
            lambda x: x,  # 0
            lambda x: jnp.flip(jnp.swapaxes(x, 0, 1), axis=0),  # 90
            lambda x: jnp.flip(jnp.flip(x, axis=0), axis=1),    # 180
            lambda x: jnp.swapaxes(jnp.flip(x, axis=0), 0, 1),  # 270
        ],
        x
    )

def augment_levels(levels, instruction_batch, rng):
    reward_enum = instruction_batch.reward_i.squeeze(-1)      # (B,)
    condition_enum = instruction_batch.condition[:, 4]        # (B,)
    keys = jax.random.split(rng, levels.shape[0])             # (B,)

    def augment_one(level, reward, condition, key):
        def random_aug(level):
            k = jax.random.randint(key, (), 0, 4)
            level = rot90_jit(level, k)
            key1, key2 = jax.random.split(key)
            level = jnp.where(jax.random.uniform(key1) > 0.5, jnp.flip(level, axis=0), level)
            level = jnp.where(jax.random.uniform(key2) > 0.5, jnp.flip(level, axis=1), level)
            return level

        def rule_based_aug(level):
            v_flip = jnp.flip(level, axis=0)
            h_flip = jnp.flip(level, axis=1)
            flipped = jnp.where(jnp.logical_or(condition == 0, condition == 2), h_flip, v_flip)
            return jnp.where(jax.random.uniform(key) > 0.5, flipped, level)

        return lax.cond(
            reward != 5,
            lambda _: random_aug(level),
            lambda _: rule_based_aug(level),
            operand=None
        )

    return jax.vmap(augment_one)(levels, reward_enum, condition_enum, keys)