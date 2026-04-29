import os
import chex
from flax import struct


__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))


@struct.dataclass
class Instruct:
    reward_i: int
    condition: chex.Array
    embedding: chex.Array
    condition_id: int
    goal_sim: chex.Array          # cos_sim(goal_state_embed, text_embed), precomputed offline


@struct.dataclass
class EmbeddingBufferReward:
    embedding: chex.Array
    buffer: chex.Array
    reward: chex.Array


