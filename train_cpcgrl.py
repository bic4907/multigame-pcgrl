from datetime import datetime
import os
import wandb
import shutil
from functools import partial
from timeit import default_timer as timer
import hydra
import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState
from tensorboardX import SummaryWriter

from conf.config import CPCGRLConfig
from envs.pcgrl_env import gen_dummy_queued_state
from instruct_rl.evaluate import get_loss_batch
from evaluator import get_reward_batch
from instruct_rl.utils.instruction import update_instruction
from instruct_rl.utils.callbacks import log_callback, eval_callback, loss_callback, create_log_handler
from instruct_rl.utils.checkpointer import (
    init_checkpointer,
    init_checkpoint_step,
    save_checkpoint_step,
    apply_encoder_params,
)
from instruct_rl.utils.dataset_loader import load_dataset_instruct
from instruct_rl.utils.log_handler import (
    CSVLoggingHandler,
    TensorBoardLoggingHandler,
    WandbLoggingHandler,
)
from instruct_rl.utils.log_utils import get_logger, suppress_jax_debug_logs
from instruct_rl.utils.logger import get_wandb_name, get_group_name
from instruct_rl.utils.path_utils import (
    gymnax_pcgrl_make,
    init_config,
    init_network,
)
from purejaxrl.experimental.s5.wrappers import LogWrapper
from purejaxrl.structures import RunnerState, Transition, LossInfo, ReturnInfo

suppress_jax_debug_logs()

logger = get_logger(__file__)


def make_train(config, restored_ckpt, checkpoint_manager, encoder_params, train_inst=None, test_inst=None):
    config.NUM_UPDATES = config.total_timesteps // config.num_steps // config.n_envs
    config.MINIBATCH_SIZE = config.n_envs * config.num_steps // config.NUM_MINIBATCHES

    env, env_params = gymnax_pcgrl_make(config.env_name, config=config)

    latest_update_step = checkpoint_manager.latest_step()
    if latest_update_step is None:
        latest_update_step = 0

    env = LogWrapper(env)
    env.init_graphics()

    def linear_schedule(count):
        frac = (
            1.0
            - (count // (config.NUM_MINIBATCHES * config.update_epochs))
            / config.NUM_UPDATES
        )
        return config["LR"] * frac

    def train(rng, runner_state):
        train_start_time = timer()

        # INIT NETWORK
        network = init_network(env, env_params, config)

        rng, _rng = jax.random.split(rng)
        init_x = env.gen_dummy_obs(env_params)

        network_params = network.init(_rng, init_x)

        if config.ANNEAL_LR:
            tx = optax.chain(
                optax.clip_by_global_norm(config.MAX_GRAD_NORM),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(config.MAX_GRAD_NORM),
                optax.adam(config.lr, eps=1e-5),
            )
        train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
        )

        # INIT ENV FOR TRAIN
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config.n_envs)

        dummy_queued_state = gen_dummy_queued_state(env)

        # Apply pmap
        vmap_reset_fn = jax.vmap(env.reset, in_axes=(0, None, None))
        obsv, env_state = vmap_reset_fn(reset_rng, env_params, dummy_queued_state)

        rng, _rng = jax.random.split(rng)

        steps_prev_complete = 0
        runner_state = RunnerState(train_state, env_state, obsv, rng, update_i=0)

        if restored_ckpt is not None:
            steps_prev_complete = restored_ckpt["steps_prev_complete"]
            runner_state = restored_ckpt["runner_state"]
            steps_remaining = config.total_timesteps - steps_prev_complete
            config.NUM_UPDATES = int(
                steps_remaining // config.num_steps // config.n_envs
            )

        if encoder_params is not None:
            runner_state = apply_encoder_params(runner_state, encoder_params, config)


        multiple_handler = create_log_handler(
            config,
            handler_classes=[TensorBoardLoggingHandler, WandbLoggingHandler, CSVLoggingHandler],
            train_start_time=train_start_time,
            steps_prev_complete=steps_prev_complete,
        )

        _log_callback = partial(
            log_callback,
            config=config,
            writer=multiple_handler,
            train_start_time=train_start_time,
            steps_prev_complete=steps_prev_complete,
        )

        if config.representation == "narrow":
            runner_state = runner_state.replace(
                env_state=runner_state.env_state.replace(
                    env_state=runner_state.env_state.env_state.replace(
                        rep_state=runner_state.env_state.env_state.rep_state.replace(
                            agent_coords=runner_state.env_state.env_state.rep_state.agent_coords[
                                :, : config.map_width**2
                            ]
                        )
                    )
                )
            )

        init_checkpoint = partial(init_checkpoint_step, checkpoint_manager=checkpoint_manager)
        save_checkpoint = partial(
            save_checkpoint_step,
            checkpoint_manager=checkpoint_manager,
            config=config,
        )

        # TRAIN LOOP
        def _update_step_with_render(update_runner_state, _):
            # COLLECT TRAJECTORIES

            runner_state, update_steps, instruct_sample, level_sample, return_info = update_runner_state
            
            def _env_step(carry, _):
                runner_state, instruct_sample, level_sample, return_info = carry

                train_state, env_state, last_obs, rng, update_i = (
                    runner_state.train_state,
                    runner_state.env_state,
                    runner_state.last_obs,
                    runner_state.rng,
                    runner_state.update_i,
                )

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)

                if config.use_nlp and train_inst is not None:
                    last_obs = last_obs.replace(nlp_obs=instruct_sample.embedding)

                if config.vec_cont and train_inst is not None:
                    vmap_state_fn = jax.vmap(env.prob.get_cont_obs, in_axes=(0, 0, None))
                    cont_obs = vmap_state_fn(env_state.env_state.env_map, instruct_sample.condition, config.raw_obs)
                    last_obs = last_obs.replace(nlp_obs=cont_obs)
                
                if config.use_clip and train_inst is not None:
                    last_obs = last_obs.replace(nlp_obs=instruct_sample.embedding)


                # Squash the gpu dimension (network only takes one batch dimension)
                pi, value, _, _ , _, _ = network.apply(train_state.params, last_obs, rng=_rng, return_text_embed=False, return_state_embed=False, return_sketch_embed=False)

                rng, _rng = jax.random.split(rng)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config.n_envs)

                vmap_step_fn = jax.vmap(env.step, in_axes=(0, 0, 0, None))

                prev_env_state = env_state

                obsv, env_state, reward_env, done, info = vmap_step_fn(
                    rng_step, env_state, action, env_params
                )

                if train_inst is not None:
                    cond_reward_batch = get_reward_batch(
                        instruct_sample.reward_i,
                        instruct_sample.condition,
                        prev_env_state.env_state.env_map,
                        env_state.env_state.env_map,
                        map_size=config.map_width,
                    )

                    reward_batch = cond_reward_batch
                        
                else:
                    reward_batch = reward_env


                reward = jnp.where(done, reward_env, reward_batch)

                env_state = env_state.replace(
                    returned_episode_returns=(
                        env_state.returned_episode_returns - reward_env + reward
                    )
                )

                if train_inst is not None:
                    instruct_sample = update_instruction(instruct_sample, train_inst, done, rng, config.n_envs)

                info["returned_episode_returns"] = env_state.returned_episode_returns

                transition = Transition(
                    done, action, value, reward, log_prob, last_obs, info,
                    env_state.env_state.env_map if config.use_sim_reward else None,
                    level_sample if config.human_demo  else None,
                )
                runner_state = RunnerState(
                    train_state, env_state, obsv, rng, update_i=update_i
                )
                return (runner_state, instruct_sample, level_sample, return_info), transition

            (runner_state, instruct_sample, level_sample, return_info), traj_batch = jax.lax.scan(
                _env_step, (runner_state, instruct_sample, level_sample, return_info), (None, None), config.num_steps
            )

            # CALCULATE ADVANTAGE
            train_state, env_state, last_obs, rng = (
                runner_state.train_state,
                runner_state.env_state,
                runner_state.last_obs,
                runner_state.rng,
            )

            _, last_val, _, _, _, _ = network.apply(train_state.params, last_obs, return_text_embed=False, return_state_embed=False, return_sketch_embed=False)

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.done,
                        transition.value,
                        transition.reward,
                    )
                    delta = reward + config.GAMMA * next_value * (1 - done) - value
                    gae = delta + config.GAMMA * config.GAE_LAMBDA * (1 - done) * gae
                    return (gae, value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value

            advantages, targets = _calculate_gae(traj_batch, last_val)

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):

                def _update_minbatch(carry, batch_info):
                    train_state, rng, loss_sum = carry
                    traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, traj_batch, gae, targets, mutation_key):
                        # RERUN NETWORK
                        # obs = traj_batch.obs[None]

                        pi, value, _, _, _, _ = network.apply(params, traj_batch.obs, return_text_embed=False, return_state_embed=False, return_sketch_embed=False)
                        # action = traj_batch.action.reshape(pi.logits.shape[:-1])
                        log_prob = pi.log_prob(traj_batch.action)

                        # jax.debug.print("{}", traj_batch.obs.nlp_obs)

                        # CALCULATE VALUE LOSS
                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-config.CLIP_EPS, config.CLIP_EPS)
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = (
                            0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                        )

                        # CALCULATE ACTOR LOSS
                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)

                        # Some reshaping to accomodate player, x, and y dimensions to action output. (Not used often...)
                        gae = gae[..., None, None, None]

                        loss_actor1 = ratio * gae
                        loss_actor2 = (
                            jnp.clip(
                                ratio,
                                1.0 - config.CLIP_EPS,
                                1.0 + config.CLIP_EPS,
                            )
                            * gae
                        )
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                        loss_actor = loss_actor.mean()
                        entropy = pi.entropy().mean()

                        total_loss = (
                            loss_actor
                            + config.VF_COEF * value_loss
                            - config.ENT_COEF * entropy
                        )
                        return total_loss, (value_loss, loss_actor, entropy)
                    rng, mutation_key = jax.random.split(rng)
                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    (total_loss, (value_loss, loss_actor, entropy)), grads = grad_fn(
                        train_state.params, traj_batch, advantages, targets, mutation_key
                    )
                    train_state = train_state.apply_gradients(grads=grads)
                    
                    new_loss_sum = LossInfo(
                        total_loss = loss_sum.total_loss + total_loss,
                        value_loss = loss_sum.value_loss + value_loss,
                        actor_loss = loss_sum.actor_loss + loss_actor,
                        entropy = loss_sum.entropy + entropy,
                    )
                    return (train_state, rng, new_loss_sum), None 

                train_state, traj_batch, advantages, targets, loss_sum, rng = update_state
                rng, _rng = jax.random.split(rng)
                batch_size = config.MINIBATCH_SIZE * config.NUM_MINIBATCHES
                assert batch_size == config.num_steps * config.n_envs, (
                    "batch size must be equal to number of steps * number " + "of envs"
                )
                permutation = jax.random.permutation(_rng, batch_size)
                batch = (traj_batch, advantages, targets)
                batch = jax.tree_util.tree_map(
                    lambda x: x.reshape((batch_size,) + x.shape[2:]), batch
                )
                shuffled_batch = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=0), batch
                )
                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.reshape(
                        x, [config.NUM_MINIBATCHES, -1] + list(x.shape[1:])
                    ),
                    shuffled_batch,
                )
                
                (train_state, rng, loss_sum), _ = jax.lax.scan(
                    _update_minbatch, (train_state, rng, loss_sum), minibatches
                )
                update_state = (train_state, traj_batch, advantages, targets, loss_sum, rng) 
                return update_state, None 

            # Save initial weight

            loss_sum = LossInfo(0.0, 0.0, 0.0, 0.0)
            update_state = (train_state, traj_batch, advantages, targets, loss_sum, rng)
            update_state, _= jax.lax.scan(
                _update_epoch, update_state, None, config.update_epochs
            )
            train_state = update_state[0]
            metric = traj_batch.info
            
            num_loss = config.NUM_MINIBATCHES * config.update_epochs
            loss_sum = update_state[4]
            loss_mean = LossInfo(
                total_loss = loss_sum.total_loss / num_loss,
                value_loss = loss_sum.value_loss / num_loss,
                actor_loss = loss_sum.actor_loss / num_loss,
                entropy = loss_sum.entropy / num_loss,
            )
            rng = update_state[-1]

            # Save weight to checkpoint
            jax.debug.callback(
                save_checkpoint, runner_state, metric, steps_prev_complete
            )
            jax.debug.callback(_log_callback, metric, loss_mean, return_info)

            runner_state = RunnerState(
                train_state,
                env_state,
                last_obs,
                rng,
                update_i=runner_state.update_i + 1,
            )

            update_steps = update_steps + 1

            def _evaluate_step():
                nonlocal rng

                rng, _rng = jax.random.split(rng)
                reset_rng = jax.random.split(_rng, config.n_envs)

                # sample n_envs rows from the instruct struct
                if test_inst is not None:
                    random_indices = jax.random.permutation(
                        rng,
                        jnp.arange(config.n_envs),
                    )[0 : config.n_envs]
                    instruct_sample = jax.tree.map(
                        lambda x: x[random_indices], test_inst
                    )
                else:
                    instruct_sample = jnp.zeros((config.n_envs, config.nlp_input_dim))

                def _env_step(carry, _):
                    rng, last_obs, state, done = carry

                    if config.use_nlp and test_inst is not None:
                        last_obs = last_obs.replace(nlp_obs=instruct_sample.embedding)

                    if config.vec_cont and test_inst is not None:
                        vmap_state_fn = jax.vmap(env.prob.get_cont_obs, in_axes=(0, 0, None))
                        cont_obs = vmap_state_fn(env_state.env_state.env_map, instruct_sample.condition, config.raw_obs)
                        last_obs = last_obs.replace(nlp_obs=cont_obs)

                    if config.use_clip and test_inst is not None:
                        last_obs = last_obs.replace(nlp_obs=instruct_sample.embedding)

                    # SELECT ACTION
                    rng, _rng = jax.random.split(rng)
                    # Squash the gpu dimension (network only takes one batch dimension)

                    pi, value, _, _, _, _ = network.apply(train_state.params, last_obs, return_text_embed=False, return_state_embed=False, return_sketch_embed=False)
                    action = pi.sample(seed=_rng)
                    log_prob = pi.log_prob(action)

                    # STEP ENV
                    rng, _rng = jax.random.split(rng)
                    rng_step = jax.random.split(_rng, config.n_envs)

                    vmap_step_fn = jax.vmap(env.step, in_axes=(0, 0, 0, None))

                    obsv, next_state, reward_env, done, info = vmap_step_fn(
                        rng_step, state, action, env_params
                    )

                    if test_inst is not None:
                        cond_reward_batch = get_reward_batch(
                            instruct_sample.reward_i,
                            instruct_sample.condition,
                            state.env_state.env_map,
                            next_state.env_state.env_map,
                            map_size=config.map_width,
                        )
                        reward_batch = cond_reward_batch
                    else:
                        reward_batch = reward_env

                    reward = jnp.where(done, reward_env, reward_batch)

                    next_state = next_state.replace(
                        returned_episode_returns=next_state.returned_episode_returns
                        - reward_env
                        + reward
                    )
                    info["returned_episode_returns"] = (
                        next_state.returned_episode_returns
                    )

                    transition = Transition(
                        done, action, value, reward, log_prob, obsv, info,
                        next_state.env_state, level_sample
                    )

                    return (rng, obsv, next_state, done), (transition, next_state)

                vmap_reset_fn = jax.vmap(env.reset, in_axes=(0, None, None))
                init_obs, init_state = vmap_reset_fn(
                    reset_rng, env_params, gen_dummy_queued_state(env)
                )
                done = jnp.zeros((config.n_envs,), dtype=bool)

                _, (traj_batch, states) = jax.lax.scan(
                    _env_step,
                    (rng, init_obs, init_state, done),
                    None,
                    length=int(
                        # config <- TrainConfig
                        # map_width: 16 -> 256
                        (config.map_width**2)
                        # max_board_scans: 3
                        * config.max_board_scans
                        # representation: "turtle" -> 2
                        * (2 if config.representation == "turtle" else 1)
                    ),
                )

                eval_metric = traj_batch.info

                states = jax.tree.map(
                    lambda x, y: jnp.concatenate([x[None], y], axis=0),
                    init_state,
                    states,
                )
                env0_state = jax.tree.map(lambda x: x[:, 0], states.env_state)

                frames = jax.vmap(env.render)(env0_state)

                _eval_callback = partial(
                    eval_callback,
                    config=config,
                    writer=multiple_handler,
                    train_start_time=train_start_time,
                    steps_prev_complete=steps_prev_complete,
                )

                jax.debug.callback(_eval_callback, eval_metric, metric, states, frames)

                if test_inst is not None:
                    loss = get_loss_batch(
                        reward_i=instruct_sample.reward_i,
                        condition=instruct_sample.condition,
                        env_maps=states.env_state.env_map[-2],
                        map_size=config.map_width
                    )

                    _loss_callback = partial(
                        loss_callback,
                        config=config,
                        writer=multiple_handler,
                    )

                    jax.debug.callback(_loss_callback, metric, loss)

                return None

            do_eval = (config.eval_freq != -1) and (
                update_steps % config.eval_freq == 0
            )
            _eval_step = _evaluate_step

            jax.lax.cond(
                do_eval,
                lambda _: _eval_step(),
                lambda _: None,
                operand=None,
            )

            return (runner_state, update_steps, instruct_sample, level_sample, return_info), metric

        # Initialize the checkpoint at step 0
        jax.debug.callback(init_checkpoint, runner_state)

        _update_step = _update_step_with_render
        # Begin train

        # sample n_envs rows from the instruct struct

        random_indices = jax.random.randint(
            runner_state.rng, (config.n_envs,), 0, train_inst.reward_i.shape[0]
        )
        instruct_sample = jax.tree.map(lambda x: x[random_indices], train_inst)
        level_sample = None
        logger.info(f"Instruction: {instruct_sample}")


        return_info = ReturnInfo(
                jnp.zeros((config.n_envs, )), 
                jnp.zeros((config.n_envs, )), 
                jnp.zeros((config.n_envs, )),
                jnp.zeros((config.n_envs, )), 
                jnp.zeros((config.n_envs,), dtype=jnp.bool_)
                )
            
        runner_state, metric = jax.lax.scan(
            _update_step,
            (runner_state, latest_update_step, instruct_sample, level_sample, return_info),
            None,
            config.NUM_UPDATES - latest_update_step,
        )

        return {"runner_state": runner_state, "metrics": metric}

    return lambda rng: train(rng, config)



def main_chunk(config, rng, exp_dir):
    """When jax jits the training loop, it pre-allocates an array with size equal to number of training steps. So, when training for a very long time, we sometimes need to break training up into multiple
    chunks to save on VRAM.
    """

    checkpoint_manager, restored_ckpt, encoder_param = init_checkpointer(config)

    if restored_ckpt is None:
        progress_csv_path = os.path.join(exp_dir, "progress.csv")
        assert not os.path.exists(progress_csv_path), (
            "Progress csv already exists, but have no checkpoint to restore "
            + "from. Run with `overwrite=True` to delete the progress csv."
        )
        # Create csv for logging progress
        with open(os.path.join(exp_dir, "progress.csv"), "w") as f:
            f.write("timestep,ep_return\n")

    # ── MultiGameDataset 기반 CPCGRL: jax.jit 밖에서 데이터셋 로드 ────────
    train_inst, test_inst = None, None

    if hasattr(config, 'dataset_game') and config.dataset_game is not None:
        train_inst, test_inst = load_dataset_instruct(config)

    train_jit = jax.jit(
        make_train(config, restored_ckpt, checkpoint_manager, encoder_param,
                   train_inst=train_inst, test_inst=test_inst)
    )
    out = train_jit(rng)

    jax.block_until_ready(out)

    return out


@hydra.main(version_base=None, config_path="./conf", config_name="train_cpcgrl")
def main(config: CPCGRLConfig):

    if config.initialize is None or config.initialize:
        config = init_config(config)

    rng = jax.random.PRNGKey(config.seed)

    exp_dir = config.exp_dir
    logger.info(f"running experiment at {exp_dir}")

    if config.wandb_key:
        dt = datetime.now().strftime("%Y%m%d%H%M%S")
        wandb_id = f"{get_wandb_name(config)}-{dt}"
        wandb.login(key=config.wandb_key)
        wandb.init(
            project=config.wandb_project,
            group=get_group_name(config),
            entity=config.wandb_entity,
            name=get_wandb_name(config),
            id=wandb_id,
            save_code=True,
            config_exclude_keys=[
                "wandb_key",
                "_vid_dir",
                "_img_dir",
                "_numpy_dir",
                "_traj_dir",
                "overwrite",
                "initialize",
            ],
        )
        wandb.config.update(dict(config), allow_val_change=True)

    # Need to do this before setting up checkpoint manager so that it doesn't refer to old checkpoints.
    if config.overwrite and os.path.exists(exp_dir):
        shutil.rmtree(exp_dir)

    if config.timestep_chunk_size != -1:
        n_chunks = config.total_timesteps // config.timestep_chunk_size
        for i in range(n_chunks):
            config.total_timesteps = config.timestep_chunk_size + (
                i * config.timestep_chunk_size
            )
            logger.info(f"Running chunk {i + 1}/{n_chunks}")
            out = main_chunk(config, rng, exp_dir)

    else:
        out = main_chunk(config, rng, exp_dir)


if __name__ == "__main__":
    main()
