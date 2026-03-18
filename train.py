from datetime import datetime
import logging
import os
from glob import glob
import numpy as np
import wandb
import shutil
from functools import partial
from os.path import abspath, basename, dirname, join
from timeit import default_timer as timer
from typing import Any, Tuple
import hydra
import jax
import jax.numpy as jnp
import optax
import orbax.checkpoint as ocp
import pandas as pd
from transformers import CLIPProcessor
from flax.training.train_state import TrainState
from flax.traverse_util import flatten_dict
from jax.experimental.array_serialization.serialization import logger
from tensorboardX import SummaryWriter

from conf.config import Config, TrainConfig
from envs.pcgrl_env import (
    OldQueuedState,
    gen_dummy_queued_state,
    gen_dummy_queued_state_old,
)
from envs.pcgrl_env import PCGRLObs
from instruct_rl.dataclass import Instruct
from instruct_rl.evaluate import get_loss_batch
from evaluator import get_reward_batch
from instruct_rl.human_data.dataset import DatasetManager
from instruct_rl.utils.instruction import update_level_sample, sample_levels, update_instruction
from instruct_rl.utils.log_handler import (
    CSVLoggingHandler,
    MultipleLoggingHandler,
    TensorBoardLoggingHandler,
    WandbLoggingHandler,
)
from instruct_rl.utils.logger import get_wandb_name, get_group_name
from instruct_rl.utils.path_utils import (
    get_ckpt_dir,
    gymnax_pcgrl_make,
    init_config,
    init_network,
)
from instruct_rl.utils.level_processing_utils import add_coord_channel_batch, map2onehot_batch
from purejaxrl.experimental.s5.wrappers import LogWrapper
from purejaxrl.structures import RunnerState, Transition, LossInfo, ReturnInfo

log_level = os.getenv(
    "LOG_LEVEL", "INFO"
).upper()  # Add the environment variable ;LOG_LEVEL=DEBUG
from utils import render_callback

logger = logging.getLogger(basename(__file__))
logger.setLevel(getattr(logging, log_level, logging.INFO))


def log_callback(metric, loss_mean, return_info, steps_prev_complete, config, writer, train_start_time):
    timesteps = metric["timestep"][metric["returned_episode"]] * config.n_envs
    return_values = metric["returned_episode_returns"][metric["returned_episode"]]
    
    
    if len(timesteps) > 0:
        t = timesteps[-1].item()
        ep_return_mean = return_values.mean()
        ep_return_max = return_values.max()
        ep_return_min = return_values.min()
        
        ep_length = metric["returned_episode_lengths"][
            metric["returned_episode"]
        ].mean()
        fps = (t - steps_prev_complete) / (timer() - train_start_time)

        prefix = (
            f"Iteration_{config.current_iteration}/train/"
            if config.current_iteration > 0
            else ""
        )

        metric = {
            f"Train/{prefix}ep_return": ep_return_mean,
            f"Train/{prefix}ep_return_max": ep_return_max,
            f"Train/{prefix}ep_return_min": ep_return_min,
            f"Train/{prefix}ep_length": ep_length,
            f"Train/{prefix}fps": fps,
            f"Train/{prefix}total_loss": loss_mean.total_loss,
            f"Train/{prefix}value_loss": loss_mean.value_loss,
            f"Train/{prefix}actor_loss": loss_mean.actor_loss,
            f"Train/{prefix}entropy": loss_mean.entropy,
            f"Train/{prefix}cond_return": jnp.mean(return_info.cond_return),
            f"Train/{prefix}sim_return": jnp.mean(return_info.sim_return),
            f"Train/{prefix}coef_sim_return": jnp.mean(return_info.coef_sim_return),
            f"Train/{prefix}total_return": jnp.mean(return_info.total_return),
            f"Train/Step": t,
        }

        # log metrics
        writer.log(metric, t)
        
        print(
            f"[train] global step={t}; episodic return mean: {ep_return_mean:.02f}, "
            + f"max: {ep_return_max:.02f}, min: {ep_return_min:.02f}, fps: {fps:.02f}, "
            + f"loss: {loss_mean.total_loss:.02f}, "
            + f"actor_loss: {loss_mean.actor_loss:.02f}, "
            + f"value_loss: {loss_mean.value_loss:.02f}, "
            + f"entropy: {loss_mean.entropy:.02f}"
        )

    

def eval_callback(
    eval_metric,
    train_metric,
    states,
    frames,
    steps_prev_complete,
    config,
    writer,
    train_start_time,
):
    timesteps = (
        train_metric["timestep"][train_metric["returned_episode"]] * config.n_envs
    )
    return_values = eval_metric["returned_episode_returns"][
        eval_metric["returned_episode"]
    ]

    # jax.debug.print("{}", return_values)

    if len(timesteps) > 0:
        t = timesteps[-1].item()
        ep_return_mean = return_values.mean()
        ep_return_max = return_values.max()
        ep_return_min = return_values.min()

        ep_length = eval_metric["returned_episode_lengths"][
            eval_metric["returned_episode"]
        ].mean()
        prefix = (
            f"Iteration_{config.current_iteration}/train/"
            if config.current_iteration > 0
            else ""
        )

        metric = {
            f"Eval/{prefix}ep_return": ep_return_mean,
            f"Eval/{prefix}ep_return_max": ep_return_max,
            f"Eval/{prefix}ep_return_min": ep_return_min,
            f"Eval/{prefix}ep_length": ep_length,
            "Train/Step": t,
        }

        # log metrics
        writer.log(metric, t)
        render_callback(
            frames=frames,
            states=states,
            video_dir=config._vid_dir,
            image_dir=config._img_dir,
            numpy_dir=config._numpy_dir,
            traj_dir=config._traj_dir,
            logger=logger,
            config=config,
            t=t,
        )

        print(
            f"[eval] global step={t}; episodic return mean: {ep_return_mean} "
            + f"max: {ep_return_max}, min: {ep_return_min}"
        )


def loss_callback(metric, loss, config, writer):
    timesteps = metric["timestep"][metric["returned_episode"]] * config.n_envs

    if len(timesteps) > 0:
        t = timesteps[-1].item()

        result_df = pd.DataFrame({"reward_enum": loss.reward_enum, "loss": loss.loss})

        mean_loss = result_df.groupby("reward_enum").agg({"loss": ["mean"]})
        mean_loss.columns = mean_loss.columns.droplevel(0)
        mean_loss = mean_loss.reset_index()

        dict_loss = dict()
        for _, row in mean_loss.iterrows():
            reward_enum, mean = row
            dict_loss[f"Loss/{str(int(reward_enum))}"] = mean

        writer.log(dict_loss, t)
        dict_str = ", ".join([f"{k}: {v}" for k, v in dict_loss.items()])
        print(f"[eval] global step={t}; loss: {dict_str}")


def make_train(config, restored_ckpt, checkpoint_manager, encoder_params):
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

        # Create a tensorboard writer
        writer = SummaryWriter(config.exp_dir)

        # INIT NETWORK
        network = init_network(env, env_params, config)

        rng, _rng = jax.random.split(rng)
        init_x = env.gen_dummy_obs(env_params)

        network_params = network.init(_rng, init_x)

        if config.human_demo:
            human_level_db_path = abspath(join(dirname(__file__), "instruct_rl", "human_data", f"{config.human_level}.npz"))
            human_level_db = np.load(human_level_db_path, allow_pickle=True)['arr_0'].item()['levels']
            human_level_db = jnp.array(human_level_db)

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
            logger.info(
                f"Parameters loaded from encoder checkpoint ({config.encoder.ckpt_path})"
            )
            runner_state.train_state.params["params"]["subnet"]["encoder"] = (
                encoder_params
            )

            logger.info("-" * 80)
            for key, enc_param in encoder_params.items():
                if "temperature" in key:
                    continue

                flat_enc_params = flatten_dict(enc_param, sep="/")
                total_bytes = 0
                for path, array in flat_enc_params.items():
                    numel = array.size
                    bpe   = array.dtype.itemsize   # bytes per element
                    mem   = int(numel * bpe)
                    total_bytes += mem

                if total_bytes < 1024**2:
                    total_hr = f"{total_bytes/1024:,.1f} KB"
                else:
                    total_hr = f"{total_bytes/1024**2:,.2f} MB"
                logger.info(f"{key} parameters memory: {total_bytes:,d} bytes ({total_hr})")
        
          
        if config.instruct_csv:
            csv_path = abspath(
                join(dirname(__file__), "instruct", f"{config.instruct_csv}.csv")
            )

            instruct_df = pd.read_csv(csv_path)            
            instruct_df['cond_id'] = (instruct_df.index // 4) + (instruct_df['reward_enum']-1)*8
            processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            
            def get_train_test(df, is_train=True):
                df = df[df["train"] == is_train]

                cond_id = jnp.array(df["cond_id"].to_list()).reshape(-1, 1)

                embedding_df = df.filter(regex="embed_*")
                embedding_df = embedding_df.reindex(
                    sorted(embedding_df.columns, key=lambda x: int(x.split("_")[-1])),
                    axis=1,
                )
                embedding = jnp.array(embedding_df.to_numpy())

                if config.nlp_input_dim > embedding.shape[1]:
                    embedding = jnp.pad(
                        embedding,
                        ((0, 0), (0, config.nlp_input_dim - embedding.shape[1])),
                        mode="constant",
                    )

                condition_df = df.filter(regex=r'(?<!sub_)condition_')
                condition_df = condition_df.reindex(
                    sorted(condition_df.columns, key=lambda x: int(x.split("_")[-1])),
                    axis=1,
                )
                condition = jnp.array(condition_df.to_numpy())

                reward_enum_list = [[int(digit) for digit in str(num)] for num in df["reward_enum"].to_list()]
                max_len = max(len(x) for x in reward_enum_list)

                reward_enum = jnp.array([
                    x + [0] * (max_len - len(x)) for x in reward_enum_list
                ])

                if config.multimodal_condition:
                    dataset_mgr = DatasetManager(config.human_demo_path)
                else:
                    dataset_mgr = None

                if config.encoder.model == 'clip':
                    language_instr_list = df["instruction"].to_list()
                    tokenized_instrs = processor(
                        text = language_instr_list,
                        return_tensors="jax",
                        padding="max_length",
                        truncation=True,
                        max_length=77
                    )
                    input_ids, attention_mask = tokenized_instrs['input_ids'], tokenized_instrs['attention_mask']

                    instr_x = PCGRLObs(
                        map_obs=jnp.repeat(init_x.map_obs, input_ids.shape[0], axis=0),
                        past_map_obs=None,
                        flat_obs=jnp.repeat(init_x.flat_obs, input_ids.shape[0], axis=0),
                        nlp_obs=jnp.repeat(init_x.nlp_obs, input_ids.shape[0], axis=0),
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        pixel_values=jnp.zeros((input_ids.shape[0], 224, 224, config.clip_input_channel), dtype=jnp.float32),
                    )
                    _, _, _, embedding, _, _ = network.apply(runner_state.train_state.params, x=instr_x,
                                                             return_text_embed=True,
                                                             return_state_embed=False,
                                                             return_sketch_embed=False)
                    logger.info(
                        f"Generated clip text embeddings for {input_ids.shape[0]} instructions. is_train: {is_train}"
                    )
                elif config.encoder.model == 'cnnclip':
                    language_instr_list = df["instruction"].to_list()
                    tokenized_instrs = processor(
                        text = language_instr_list,
                        return_tensors="jax",
                        padding="max_length",
                        truncation=True,
                        max_length=77
                    )
                    input_ids, attention_mask = tokenized_instrs['input_ids'], tokenized_instrs['attention_mask']

                    if config.multimodal_condition:
                        levels = dataset_mgr.get_levels(language_instr_list, n=10, to_jax=True, squeeze_n=False, coord_channel=True) # (5, 10, 16, 16, 5)
                        sketches = dataset_mgr.get_sketches(language_instr_list, n=10, to_jax=True, squeeze_n=False, coord_channel=True) # (5, 10, 224, 224, 3)

                        n_inst, n_samples, H, W, C = levels.shape
                        n_inst, n_samples, H_, W_, C_ = sketches.shape

                        levels = jnp.reshape(levels, (n_inst * n_samples, H, W, C))  # (50, 16, 16, 5)
                        sketches = jnp.reshape(sketches, (n_inst * n_samples, H_, W_, C_))

                        input_ids = jnp.repeat(input_ids, n_samples, axis=0)
                        attention_mask = jnp.repeat(attention_mask, n_samples, axis=0)

                    else:
                        levels = jnp.zeros((input_ids.shape[0], 16, 16, 5), dtype=jnp.float32)
                        sketches = jnp.zeros((input_ids.shape[0], 224, 224, 3), dtype=jnp.float32)

                    instr_x = PCGRLObs(
                        map_obs=jnp.repeat(init_x.map_obs, input_ids.shape[0], axis=0),
                        past_map_obs=None,
                        flat_obs=jnp.repeat(init_x.flat_obs, input_ids.shape[0], axis=0),
                        nlp_obs=jnp.repeat(init_x.nlp_obs, input_ids.shape[0], axis=0),
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        pixel_values=levels,
                        sketch_values=sketches,
                    )

                    _, _, _, embedding_t, embedding_s, embedding_k = network.apply(runner_state.train_state.params, x=instr_x,
                                                             return_text_embed=True,
                                                             return_state_embed=True if config.multimodal_condition else False,
                                                             return_sketch_embed=True if config.multimodal_condition else False)

                    if config.multimodal_condition:
                        embedding = jnp.concatenate([embedding_t, embedding_s, embedding_k], axis=0)
                    else:
                        embedding = embedding_t

                    logger.info(
                        f"Generated cnnclip {embedding.shape} embeddings for {input_ids.shape[0]} instructions. (is_train: {is_train}, multimodal: {config.multimodal_condition})"
                    )

                return Instruct(
                    reward_i=reward_enum,
                    condition=condition,
                    embedding=embedding,
                    condition_id=cond_id,
                )

            train_inst = get_train_test(instruct_df, is_train=True)
            test_inst = get_train_test(instruct_df, is_train=False)
        else:
            train_inst, test_inst = None, None
                    
 

        handler_classes = [
            TensorBoardLoggingHandler,
            WandbLoggingHandler,
            CSVLoggingHandler,
        ]
        multiple_handler = MultipleLoggingHandler(
            config=config, handler_classes=handler_classes, logger=logger
        )

        # Set the start time and previous steps
        multiple_handler.set_start_time(train_start_time)
        multiple_handler.set_steps_prev_complete(steps_prev_complete)

        multiple_handler.add_text("Train/Config", f"```{str(config)}```")
        # if reward_function is in this scope

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

        def init_checkpoint(runner_state):
            ckpt = {"runner_state": runner_state, "step_i": 0}
            ckpt = jax.device_get(ckpt)
            checkpoint_manager.save(0, args=ocp.args.StandardSave(ckpt))

        def save_checkpoint(runner_state, info, steps_prev_complete):
            # Get the global env timestep numbers corresponding to the points at which different episodes were finished
            timesteps = info["timestep"][info["returned_episode"]] * config.n_envs

            if len(timesteps) > 0:
                # Get the latest global timestep at which some episode was finished
                t = timesteps[-1].item()
                latest_ckpt_step = checkpoint_manager.latest_step()
                if latest_ckpt_step is None or t - latest_ckpt_step >= config.ckpt_freq:
                    print(f"Saving checkpoint at step {t}")
                    ckpt = {"runner_state": runner_state, "step_i": t}
                    ckpt = jax.device_get(ckpt)
                    checkpoint_manager.save(t, args=ocp.args.StandardSave(ckpt))

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

                # change the instruct sample and level demo data
                # TODO: refactor this part to be cleanerly separated from the rest of the code
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
        if train_inst is not None:

            random_indices = jax.random.randint(
                runner_state.rng, (config.n_envs,), 0, train_inst.reward_i.shape[0]
            )
            instruct_sample = jax.tree.map(lambda x: x[random_indices], train_inst)

            if config.human_demo:
                level_sample = sample_levels(
                    human_level_db, instruct_sample, runner_state.rng, config.human_augment
                )
            else:
                level_sample = None
            logger.info(f"Instruction: {instruct_sample}")
        else:
            instruct_sample = None
            level_sample = None
            logger.info("Instruction: None")

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


def init_checkpointer(config: Config) -> Tuple[Any, dict]:
    # This will not affect training, just for initializing dummy env etc. to load checkpoint.
    rng = jax.random.PRNGKey(30)
    # Set up checkpointing
    ckpt_dir = get_ckpt_dir(config)

    # Create a dummy checkpoint so we can restore it to the correct dataclasses
    env, env_params = gymnax_pcgrl_make(config.env_name, config=config)
    # env = FlattenObservationWrapper(env)

    env = LogWrapper(env)

    rng, _rng = jax.random.split(rng)
    network = init_network(env, env_params, config)
    init_x = env.gen_dummy_obs(env_params)
    network_params = network.init(_rng, init_x,)

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

    # reset_rng_r = reset_rng.reshape((config.n_gpus, -1) + reset_rng.shape[1:])
    vmap_reset_fn = jax.vmap(env.reset, in_axes=(0, None, None))
    # pmap_reset_fn = jax.pmap(vmap_reset_fn, in_axes=(0, None))
    obsv, env_state = vmap_reset_fn(reset_rng, env_params, gen_dummy_queued_state(env))
    runner_state = RunnerState(
        train_state=train_state,
        env_state=env_state,
        last_obs=obsv,
        # ep_returns=jnp.full(config.num_envs, jnp.nan),
        rng=rng,
        update_i=0,
    )
    target = {"runner_state": runner_state, "step_i": 0}
    # Get absolute path
    ckpt_dir = os.path.abspath(ckpt_dir)

    options = ocp.CheckpointManagerOptions(max_to_keep=2, create=True)
    # checkpoint_manager = orbax.checkpoint.CheckpointManager(
    #     ckpt_dir, orbax.checkpoint.PyTreeCheckpointer(), options)
    checkpoint_manager = ocp.CheckpointManager(
        # ocp.test_utils.erase_and_create_empty(ckpt_dir),
        ckpt_dir,
        options=options,
    )

    def try_load_ckpt(steps_prev_complete, target):
        runner_state = target["runner_state"]
        try:
            restored_ckpt = checkpoint_manager.restore(
                # steps_prev_complete, items=target)
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

            # Now add leading dimension with sizeto match the shape of the original queued_state
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
        # print(f"Restoring checkpoint from {ckpt_dir}")
        # steps_prev_complete = checkpoint_manager.latest_step()

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

    train_jit = jax.jit(
        make_train(config, restored_ckpt, checkpoint_manager, encoder_param)
    )
    out = train_jit(rng)

    jax.block_until_ready(out)

    return out


@hydra.main(version_base=None, config_path="./conf", config_name="train_pcgrl")
def main(config: TrainConfig):

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
            print(f"Running chunk {i + 1}/{n_chunks}")
            out = main_chunk(config, rng, exp_dir)

    else:
        out = main_chunk(config, rng, exp_dir)


if __name__ == "__main__":
    main()
