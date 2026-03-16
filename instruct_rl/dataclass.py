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


@struct.dataclass
class EmbeddingBufferReward:
    embedding: chex.Array
    buffer: chex.Array
    reward: chex.Array


