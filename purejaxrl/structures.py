import jax.numpy as jnp
from flax import struct
from flax.training.train_state import TrainState
from typing import NamedTuple


class RunnerState(struct.PyTreeNode):
    train_state: TrainState
    env_state: jnp.ndarray
    last_obs: jnp.ndarray
    rng: jnp.ndarray
    update_i: int


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray
    env_map: jnp.ndarray
    human_level: jnp.ndarray
    
class ReturnInfo(NamedTuple):
    cond_return: jnp.ndarray
    sim_return: jnp.ndarray
    coef_sim_return: jnp.ndarray
    total_return: jnp.ndarray
    prev_done: jnp.ndarray
    
    
class LossInfo(NamedTuple):
    total_loss: jnp.ndarray
    value_loss: jnp.ndarray
    actor_loss: jnp.ndarray
    entropy: jnp.ndarray
    