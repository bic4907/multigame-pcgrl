"""
instruct_rl/utils/checkpointer.py
==================================
체크포인트 초기화 및 로딩 유틸.
train_cpcgrl.py 에서 분리.
"""
import logging
import os
from glob import glob
from os.path import basename, join
from typing import Any, Tuple

import jax
import jax.numpy as jnp
import optax
import orbax.checkpoint as ocp
from flax.training.train_state import TrainState

from conf.config import Config
from envs.pcgrl_env import (
    OldQueuedState,
    gen_dummy_queued_state,
    gen_dummy_queued_state_old,
)
from instruct_rl.utils.path_utils import (
    get_ckpt_dir,
    gymnax_pcgrl_make,
    init_network,
)
from purejaxrl.experimental.s5.wrappers import LogWrapper
from purejaxrl.structures import RunnerState

logger = logging.getLogger(basename(__file__))


def init_checkpointer(config: Config) -> Tuple[Any, dict, Any]:
    """체크포인트 매니저를 초기화하고 기존 체크포인트를 복원한다.

    Returns
    -------
    (checkpoint_manager, restored_ckpt, enc_param)
    """
    # This will not affect training, just for initializing dummy env etc. to load checkpoint.
    rng = jax.random.PRNGKey(30)
    # Set up checkpointing
    ckpt_dir = get_ckpt_dir(config)

    # Create a dummy checkpoint so we can restore it to the correct dataclasses
    env, env_params = gymnax_pcgrl_make(config.env_name, config=config)

    env = LogWrapper(env)

    rng, _rng = jax.random.split(rng)
    network = init_network(env, env_params, config)
    init_x = env.gen_dummy_obs(env_params)
    network_params = network.init(_rng, init_x)

    tx = optax.chain(
        optax.clip_by_global_norm(config.MAX_GRAD_NORM),
        optax.adam(config.lr, eps=1e-5),
    )
    train_state = TrainState.create(
        apply_fn=network.apply,
        params=network_params,
        tx=tx,
    )
    rng, _rng = jax.random.split(rng)
    reset_rng = jax.random.split(_rng, config.n_envs)

    vmap_reset_fn = jax.vmap(env.reset, in_axes=(0, None, None))
    obsv, env_state = vmap_reset_fn(reset_rng, env_params, gen_dummy_queued_state(env))
    runner_state = RunnerState(
        train_state=train_state,
        env_state=env_state,
        last_obs=obsv,
        rng=rng,
        update_i=0,
    )
    target = {"runner_state": runner_state, "step_i": 0}
    # Get absolute path
    ckpt_dir = os.path.abspath(ckpt_dir)

    options = ocp.CheckpointManagerOptions(max_to_keep=2, create=True)
    checkpoint_manager = ocp.CheckpointManager(
        ckpt_dir,
        options=options,
    )

    def try_load_ckpt(steps_prev_complete, target):
        runner_state = target["runner_state"]
        try:
            restored_ckpt = checkpoint_manager.restore(
                steps_prev_complete,
                args=ocp.args.StandardRestore(target),
            )
        except KeyError:
            runner_state = runner_state.replace(
                env_state=runner_state.env_state.replace(
                    env_state=runner_state.env_state.env_state.replace(
                        queued_state=gen_dummy_queued_state_old(env)
                    )
                )
            )
            target = {"runner_state": runner_state, "step_i": 0}
            restored_ckpt = checkpoint_manager.restore(
                steps_prev_complete, items=target
            )

        restored_ckpt["steps_prev_complete"] = steps_prev_complete
        if restored_ckpt is None:
            raise TypeError("Restored checkpoint is None")

        if isinstance(runner_state.env_state.env_state.queued_state, OldQueuedState):
            dummy_queued_state = gen_dummy_queued_state(env)

            # Now add leading dimension with size to match the shape of the original queued_state
            dummy_queued_state = jax.tree_map(
                lambda x: jnp.array(x, dtype=bool) if isinstance(x, bool) else x,
                dummy_queued_state,
            )
            dummy_queued_state = jax.tree_map(
                lambda x: jnp.repeat(x[None], config.n_envs, axis=0), dummy_queued_state
            )

            runner_state = restored_ckpt["runner_state"]
            runner_state = runner_state.replace(
                env_state=runner_state.env_state.replace(
                    env_state=runner_state.env_state.env_state.replace(
                        queued_state=dummy_queued_state,
                    )
                )
            )
            restored_ckpt["runner_state"] = runner_state

        return restored_ckpt

    if checkpoint_manager.latest_step() is None:
        restored_ckpt = None
    else:
        ckpt_subdirs = os.listdir(ckpt_dir)
        ckpt_steps = [int(cs) for cs in ckpt_subdirs if cs.isdigit()]

        # Sort in decreasing order
        ckpt_steps.sort(reverse=True)
        for steps_prev_complete in ckpt_steps:
            try:
                restored_ckpt = try_load_ckpt(steps_prev_complete, target)
                if restored_ckpt is None:
                    raise TypeError("Restored checkpoint is None")
                break
            except TypeError as e:
                print(
                    f"Failed to load checkpoint at step {steps_prev_complete}. Error: {e}"
                )
                continue

    if config.encoder.ckpt_path is not None:
        logger.info(f"Restoring encoder checkpoint from {config.encoder.ckpt_path}")

        ckpt_subdirs = glob(join(config.encoder.ckpt_path, '*'))

        ckpt_steps = [int(basename(cs)) for cs in ckpt_subdirs if basename(cs).isdigit()]

        assert len(ckpt_steps) > 0, (
            "No checkpoint found in encoder checkpoint path. Set 'encoder.ckpt_path' to the paths ends with '**/ckpts'"
        )

        # Sort in decreasing order
        ckpt_steps.sort(reverse=True)
        for steps_prev_complete in ckpt_steps:
            ckpt_dir = os.path.join(config.encoder.ckpt_path, str(steps_prev_complete))

            try:
                from flax.training import checkpoints

                enc_state = checkpoints.restore_checkpoint(
                    ckpt_dir=ckpt_dir, target=None, prefix="",
                )

                assert enc_state is not None, "Restored params are None ({})".format(
                    ckpt_dir
                )

                enc_param = enc_state["params"]["params"]
                if config.encoder.model in ["clip", "cnnclip"]:
                    enc_param = enc_param
                else:
                    def get_encoder_params_recursive(params, key):
                        if key in params:
                            return params[key]
                        for v in params.values():
                            if isinstance(v, dict):
                                result = get_encoder_params_recursive(v, key)
                                if result is not None:
                                    return result
                        return None

                    enc_param = get_encoder_params_recursive(enc_param, "encoder")
                    assert enc_param is not None, "Encoder not found in checkpoint"

                break

            except TypeError as e:
                logging.error(
                    f"Failed to load checkpoint at step {steps_prev_complete}. Error: {e}"
                )
                continue
    else:
        enc_param = None

    return checkpoint_manager, restored_ckpt, enc_param

