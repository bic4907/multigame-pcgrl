"""
instruct_rl/utils/checkpointer.py
==================================
체크포인트 초기화 및 로딩 유틸.
train_cpcgrl.py 에서 분리.
"""
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
from instruct_rl.utils.log_utils import get_logger
from instruct_rl.utils.path_utils import (
    get_ckpt_dir,
    gymnax_pcgrl_make,
    init_network,
)
from purejaxrl.experimental.s5.wrappers import LogWrapper
from purejaxrl.structures import RunnerState

logger = get_logger(__file__)


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
                logger.warning(
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

        logger.info(f"  Available checkpoint steps: {sorted(ckpt_steps)}")

        # Sort in decreasing order
        ckpt_steps.sort(reverse=True)
        enc_param = None
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

                # ── 로딩 성공 로그 ──
                log_encoder_ckpt_loaded(enc_param, ckpt_dir, steps_prev_complete)
                break

            except TypeError as e:
                logger.error(
                    f"Failed to load checkpoint at step {steps_prev_complete}. Error: {e}"
                )
                continue

        if enc_param is None:
            logger.error(f"  ❌ Failed to load any encoder checkpoint from {config.encoder.ckpt_path}")
    elif config.encoder.model == "clip":
        from encoder.clip_model import get_clip_hf_pretrained_params
        logger.info("No encoder ckpt_path — loading pretrained CLIP weights from HuggingFace (openai/clip-vit-base-patch32)")
        enc_param = get_clip_hf_pretrained_params(config.encoder)
    else:
        enc_param = None

    return checkpoint_manager, restored_ckpt, enc_param


# ── train_cpcgrl.py 에서 분리한 체크포인트 유틸 ─────────────────────────────


def init_checkpoint_step(runner_state, checkpoint_manager):
    """학습 시작 시 step=0 체크포인트를 저장한다. (jax.debug.callback 용)"""
    ckpt = {"runner_state": runner_state, "step_i": 0}
    ckpt = jax.device_get(ckpt)
    try:
        checkpoint_manager.save(0, args=ocp.args.StandardSave(ckpt))
        checkpoint_manager.wait_until_finished()
    except Exception as e:
        logger.warning(f"init_checkpoint_step failed: {e}")


def save_checkpoint_step(runner_state, info, steps_prev_complete,
                         checkpoint_manager, config):
    """에피소드 완료 시점 기준으로 체크포인트를 저장한다. (jax.debug.callback 용)"""
    timesteps = info["timestep"][info["returned_episode"]] * config.n_envs

    if len(timesteps) > 0:
        t = timesteps[-1].item()
        latest_ckpt_step = checkpoint_manager.latest_step()
        if latest_ckpt_step is None or t - latest_ckpt_step >= config.ckpt_freq:
            logger.info(f"Saving checkpoint at step {t}")
            ckpt = {"runner_state": runner_state, "step_i": t}
            ckpt = jax.device_get(ckpt)
            try:
                checkpoint_manager.save(t, args=ocp.args.StandardSave(ckpt))
                checkpoint_manager.wait_until_finished()
            except Exception as e:
                logger.warning(f"save_checkpoint_step failed at step {t}: {e}")


def _deep_merge(base: dict, update: dict) -> dict:
    """update의 키만 base에 재귀적으로 덮어씀 (없는 키는 base 값 유지)."""
    for k, v in update.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v
    return base


def apply_encoder_params(runner_state, encoder_params, config):
    """인코더 파라미터를 runner_state 에 적용하고 메모리 사용량을 로깅한다.

    Parameters
    ----------
    runner_state : RunnerState
    encoder_params : dict
    config : Config

    Returns
    -------
    runner_state : RunnerState (파라미터 주입 완료)
    """
    from flax.traverse_util import flatten_dict

    source = config.encoder.ckpt_path if config.encoder.ckpt_path else "HuggingFace pretrained"
    logger.info(f"Parameters loaded from encoder ({source})")
    _deep_merge(runner_state.train_state.params["params"]["subnet"]["encoder"], encoder_params)

    logger.info("-" * 80)
    for key, enc_param in encoder_params.items():
        if "temperature" in key:
            continue

        flat_enc_params = flatten_dict(enc_param, sep="/")
        total_bytes = 0
        for path, array in flat_enc_params.items():
            numel = array.size
            bpe = array.dtype.itemsize
            mem = int(numel * bpe)
            total_bytes += mem

        if total_bytes < 1024 ** 2:
            total_hr = f"{total_bytes / 1024:,.1f} KB"
        else:
            total_hr = f"{total_bytes / 1024 ** 2:,.2f} MB"
        logger.info(f"{key} parameters memory: {total_bytes:,d} bytes ({total_hr})")

    return runner_state


# ── 인코더 체크포인트 로깅 유틸 ──────────────────────────────────────────────


def log_encoder_ckpt_loaded(enc_param, ckpt_dir: str, step: int):
    """인코더 체크포인트 로딩 성공 시 요약 로그를 출력한다."""
    logger.info(f"  ✅ Encoder checkpoint loaded successfully (step={step})")
    logger.info(f"     ckpt_dir: {ckpt_dir}")
    if isinstance(enc_param, dict):
        from flax.traverse_util import flatten_dict
        logger.info(f"     top-level keys: {list(enc_param.keys())}")
        _total_params = 0
        for _k, _v in enc_param.items():
            if isinstance(_v, dict):
                _flat = flatten_dict(_v, sep="/")
                _n = sum(a.size for a in _flat.values())
            else:
                _n = _v.size if hasattr(_v, 'size') else 0
            _total_params += _n
        logger.info(f"     total encoder params: {_total_params:,d}")


def log_encoder_params_summary(encoder_params, config):
    """encoder_params 존재 여부와 내용물 상세 정보를 로그로 출력한다."""
    if encoder_params is not None:
        from flax.traverse_util import flatten_dict
        logger.info("=" * 80)
        logger.info("✅ Encoder checkpoint found — applying pretrained encoder params")
        logger.info(f"   ckpt_path : {config.encoder.ckpt_path}")
        logger.info(f"   ckpt_name : {getattr(config.encoder, 'ckpt_name', None)}")
        logger.info(f"   encoder   : {config.encoder.model}")
        if isinstance(encoder_params, dict):
            logger.info(f"   top-level keys: {list(encoder_params.keys())}")
            # for k, v in encoder_params.items():
            #     if isinstance(v, dict):
            #         _flat = flatten_dict(v, sep="/")
            #         _n = sum(a.size for a in _flat.values())
            #         logger.info(f"     [{k}] sub-keys={len(_flat)}, params={_n:,d}")
            #         for _path, _arr in list(_flat.items())[:5]:
            #             logger.info(f"       {_path}: shape={_arr.shape}, dtype={_arr.dtype}")
            #         if len(_flat) > 5:
            #             logger.info(f"       ... and {len(_flat) - 5} more")
            #     elif hasattr(v, 'shape'):
            #         logger.info(f"     [{k}] shape={v.shape}, dtype={v.dtype}")
            #     else:
            #         logger.info(f"     [{k}] type={type(v).__name__}")
        else:
            logger.info(f"   encoder_params type: {type(encoder_params).__name__}")
        logger.info("=" * 80)
    else:
        logger.info("=" * 80)
        logger.info("⚠️  No encoder checkpoint — encoder_params is None")
        logger.info(f"   ckpt_path : {getattr(config.encoder, 'ckpt_path', None)}")
        logger.info(f"   ckpt_name : {getattr(config.encoder, 'ckpt_name', None)}")
        logger.info("=" * 80)
