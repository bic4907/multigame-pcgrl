from os.path import abspath, dirname, join
from sysconfig import get_path

import jax.numpy as jnp
import chex
import jax
from flax.struct import dataclass
from jax import vmap

from envs.probs.dungeon3 import Dungeon3Tiles
from evaluator.losses import region_loss, path_length_loss
from evaluator.losses.multigame_amount_loss import multigame_amount_loss
from evaluator.measures import get_region, get_path_length
from evaluator.measures.multigame_amount import (
    get_interactive_count,
    get_hazard_count,
    get_collectable_count,
)


@dataclass
class LossResult:
    reward_enum: chex.Array
    value: chex.Array
    loss: chex.Array
    feature: chex.Array


default_loss = 0.0


def get_loss_batch(
    reward_i: chex.Array,
    condition: chex.Array,
    env_maps: chex.Array,
) -> chex.Array:
    """
    Compute batch rewards by mapping indices to reward functions and executing them in parallel.

    Args:
        reward_i: Array of indices mapping to functions in call_reward.
        condition: Array of conditions corresponding to each reward calculation.
        prev_env_map: Previous environment map.
        curr_env_map: Current environment map.

    Returns:
        rewards: Array of computed rewards.
    """

    # map size

    aggregate_funcs = [
        lambda cond, env_map: get_region(env_map),  # 1 (region)
        lambda cond, env_map: get_path_length(env_map),  # 2 (path_length)
        lambda cond, env_map: get_interactive_count(env_map),  # 3 (interactive)
        lambda cond, env_map: get_hazard_count(env_map),  # 4 (hazard)
        lambda cond, env_map: get_collectable_count(env_map),  # 5 (collectable)
    ]

    loss_funcs = [
        lambda cond, env_map: region_loss(env_map, cond[0]),  # 1 (region)
        lambda cond, env_map: path_length_loss(
            env_map, cond[1]
        ),  # 2 (path_length)
        lambda cond, env_map: multigame_amount_loss(
            env_map, "interactive", cond[2], absolute=True
        ),  # 3 (interactive)
        lambda cond, env_map: multigame_amount_loss(
            env_map, "hazard", cond[3], absolute=True
        ),  # 4 (hazard)
        lambda cond, env_map: multigame_amount_loss(
            env_map, "collectable", cond[4], absolute=True
        ),  # 5 (collectable)
    ]


    def aggregate(func_idx, cond_value, env_map):
        reward_values = jax.vmap(
            lambda idx: jax.lax.switch(idx, aggregate_funcs, cond_value, env_map).ravel())(func_idx)
        return jnp.sum(reward_values)

    # Map indices to functions using `switch`
    def compute_loss(func_idx, cond_value, env_map):
        return jax.lax.switch(
            func_idx,
            loss_funcs,
            cond_value,
            env_map,
        ).ravel()

    def measure_all(env_map):
        n_region = get_region(env_map)
        n_path_length = get_path_length(env_map)
        n_interactive = get_interactive_count(env_map)
        n_hazard = get_hazard_count(env_map)
        n_collectable = get_collectable_count(env_map)

        return jnp.array([
            n_region, n_path_length,
            n_interactive, n_hazard, n_collectable,
        ])


    values = vmap(aggregate, in_axes=(0, 0, 0))(reward_i, condition, env_maps)
    values = jnp.expand_dims(values, axis=1)

    losses = vmap(lambda reward_idx: vmap(compute_loss, in_axes=(0, 0, 0))(reward_idx, condition, env_maps).sum(
        axis=1
    ))(jnp.transpose(reward_i))
    losses = jnp.transpose(losses)
    losses = jnp.sum(losses, axis=1)

    feature = vmap(measure_all, in_axes=(0))(env_maps)

    result = LossResult(
        reward_enum=vmap(process_row)(reward_i),
        value=values,
        loss=losses,
        feature=feature,
    )

    return jax.lax.stop_gradient(result)


def process_row(row):

    def body(carry, x):
        carry = jnp.where(x > 0, carry * 10 + x, carry)
        return carry, carry

    _, result = jax.lax.scan(body, init=0, xs=row)
    return result[-1]



if __name__ == "__main__":
    import pandas as pd

    instruct_csv = abspath(
        join(dirname(__file__), "..", "instruct", "test", "test", "combined2.csv")
    )
    instruct_df = pd.read_csv(instruct_csv)
    instruct_df = instruct_df[instruct_df["train"] == True]

    # sort with reward_enum
    instruct_df = instruct_df.sort_values(by="reward_enum")

    cond_df = instruct_df.filter(regex="condition_*")
    cond_df = cond_df.reindex(
        sorted(cond_df.columns, key=lambda x: int(x.split("_")[-1])), axis=1
    )

    from instruct_rl.dataclass import Instruct

    reward_enum_list = [[int(digit) for digit in str(num)] for num in instruct_df["reward_enum"].to_list()]
    max_len = max(len(x) for x in reward_enum_list)
    split_result = jnp.array([
        x + [0] * (max_len - len(x)) for x in reward_enum_list
    ])

    instruct = Instruct(
        reward_i=split_result,
        condition=jnp.array(cond_df.to_numpy()),
        embedding=jnp.zeros((len(instruct_df), 128)),
    )

    # Dummy environment maps
    # randonly initialize the 16x16 with number 1~6
    prev_env_map = jax.random.randint(
        minval=1, maxval=7, shape=(len(instruct_df), 16, 16), key=jax.random.PRNGKey(0)
    )

    # Compute rewards
    result = get_loss_batch(instruct.reward_i, instruct.condition, prev_env_map)

    instruct_df["reward"] = result.loss
    instruct_df = instruct_df.loc[:, ~instruct_df.columns.str.startswith("embed")]
    instruct_df = instruct_df.drop(columns=["train"])
    # remove columns starts with 'condition_5' to 'condition_8'
    instruct_df = instruct_df.loc[:, ~instruct_df.columns.str.startswith("condition_5")]
    instruct_df = instruct_df.loc[:, ~instruct_df.columns.str.startswith("condition_6")]
    instruct_df = instruct_df.loc[:, ~instruct_df.columns.str.startswith("condition_7")]
    instruct_df = instruct_df.loc[:, ~instruct_df.columns.str.startswith("condition_8")]

    # tabulate print the dataframe (show all columns)
    from tabulate import tabulate

    print(tabulate(instruct_df, headers="keys", tablefmt="psql"))

    result_df = pd.DataFrame({"reward_enum": result.reward_enum, "loss": result.loss})
    mean_loss = result_df.groupby("reward_enum").agg({"loss": ["mean"]})
    mean_loss.columns = mean_loss.columns.droplevel(0)
    mean_loss = mean_loss.reset_index()

    print(tabulate(mean_loss, headers="keys", tablefmt="psql"))
    for _, row in mean_loss.iterrows():
        reward_enum, mean = row

        print(f"reward_enum: {reward_enum}, mean: {mean}")
