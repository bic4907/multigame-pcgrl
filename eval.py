import math
from datetime import datetime
from typing import Tuple, Any
import imageio
import numpy as np
import optax
import pandas as pd
import os
import cv2
from os.path import basename, dirname, join, abspath
import hydra
import jax
import jax.numpy as jnp
import wandb
from flax.training.train_state import TrainState
import orbax.checkpoint as ocp
from tqdm import tqdm
from transformers import CLIPProcessor

from conf.config import EvalConfig, Config
from envs.pcgrl_env import gen_dummy_queued_state, PCGRLObs
from instruct_rl import NUM_FEATURES, FEATURE_NAMES
from instruct_rl.dataclass import Instruct
from instruct_rl.evaluate import get_loss_batch
from instruct_rl.evaluation.metrics.tpkldiv import TPKLEvaluator
from instruct_rl.evaluation.metrics.vit import ViTEvaluator
from instruct_rl.evaluation.hamming import compute_hamming_distance
from instruct_rl.human_data.dataset import DatasetManager
from instruct_rl.utils.level_processing_utils import add_coord_channel_batch, map2onehot_batch
from instruct_rl.utils.logger import get_wandb_name_eval
from instruct_rl.vision.data.render import render_array_batch
from purejaxrl.experimental.s5.wrappers import LogWrapper
from purejaxrl.structures import Transition, RunnerState
from train import init_checkpointer
from instruct_rl.utils.path_utils import (
    get_ckpt_dir,
    gymnax_pcgrl_make,
    init_config,
    init_network,
)
from evaluator import get_reward_batch

import logging
log_level = os.getenv('LOG_LEVEL', 'INFO').upper()  # Add the environment variable ;LOG_LEVEL=DEBUG
logger = logging.getLogger(basename(__file__))
logger.setLevel(getattr(logging, log_level, logging.INFO))



def make_eval(config, restored_ckpt, encoder_params):
    env, env_params = gymnax_pcgrl_make(config.env_name, config=config)
    env = LogWrapper(env)
    env.init_graphics()

    def eval(rng, runner_state):
        # INIT NETWORK
        network = init_network(env, env_params, config)

        rng, _rng = jax.random.split(rng)
        init_x = env.gen_dummy_obs(env_params)

        if restored_ckpt is not None:
            network_params = restored_ckpt['runner_state'].train_state.params
        else:
            network_params = network.init(_rng, init_x)

        if config.ANNEAL_LR:
            def linear_schedule(count):
                frac = (
                    1.0 - (count // (config.NUM_MINIBATCHES * config.update_epochs))
                    / config.NUM_UPDATES
                )
                return config.LR * frac
            tx = optax.chain(
                optax.clip_by_global_norm(config.MAX_GRAD_NORM),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(config.MAX_GRAD_NORM),
                optax.adam(config.lr, eps=1e-5),
            )
        train_state = TrainState.create(apply_fn=network.apply, params=network_params, tx=tx)

        # INIT ENV FOR TRAIN
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config.n_envs)

        dummy_queued_state = gen_dummy_queued_state(env)

        vmap_reset_fn = jax.vmap(env.reset, in_axes=(0, None, None))
        obsv, env_state = vmap_reset_fn(reset_rng, env_params, dummy_queued_state)

        rng, _rng = jax.random.split(rng)

        if restored_ckpt is not None:
            runner_state = restored_ckpt["runner_state"]
        else:
            runner_state = RunnerState(train_state, env_state, obsv, rng, update_i=0)

        if encoder_params is not None:
            logger.info(f"Parameters loaded from encoder checkpoint ({config.encoder.ckpt_path})")
            runner_state.train_state.params['params']['subnet']['encoder'] = encoder_params

        csv_path = abspath(join(dirname(__file__), "instruct", f"{config.eval_instruct_csv}.csv"))
        instruct_df = pd.read_csv(csv_path)
        # index to 'row_i'
        instruct_df = instruct_df.reset_index()
        instruct_df = instruct_df.rename(columns={'index': 'row_i'})

        instruct_df.to_csv(join(config.eval_dir, 'input.csv'), index=False)

        embedding_df = instruct_df.filter(regex="embed_*")
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

        condition_df = instruct_df.filter(regex=r'(?<!sub_)condition_')
        condition_df = condition_df.reindex(
            sorted(condition_df.columns, key=lambda x: int(x.split("_")[-1])),
            axis=1,
        )
        condition = jnp.array(condition_df.to_numpy())

        reward_enum_list = [[int(digit) for digit in str(num)] for num in instruct_df["reward_enum"].to_list()]

        max_len = max(len(x) for x in reward_enum_list)

        reward_enum = jnp.array([
            x + [0] * (max_len - len(x)) for x in reward_enum_list
        ])
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

        pixel_values, sketch_values = None, None

        if config.encoder.model == 'clip':
            language_instr_list = instruct_df["instruction"].to_list()
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
            _, _, _, embedding, _, _ = network.apply(runner_state.train_state.params, x=instr_x, return_text_embed=True, return_state_embed=False, return_sketch_embed=False)
            logger.info(
                f"Generated clip text embeddings for {input_ids.shape[0]} instructions."
            )

        elif config.encoder.model == 'cnnclip':

            logger.info(f'Generating cnnclip text embeddings with `{config.eval_modality}` modality.')

            if config.eval_modality == 'text':
                processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
                language_instr_list = instruct_df["instruction"].to_list()
                tokenized_instrs = processor(
                    text = language_instr_list,
                    return_tensors="jax",
                    padding="max_length",
                    truncation=True,
                    max_length=77
                )
                input_ids, attention_mask = tokenized_instrs['input_ids'], tokenized_instrs['attention_mask']

                pixel_values = jnp.zeros((input_ids.shape[0], 16, 16, config.clip_input_channel), dtype=jnp.float32)

                logger.info(
                    f"Generated cnnclip text embeddings for {input_ids.shape[0]} instructions."
                )

            elif config.eval_modality == 'state':

                input_ids = jnp.zeros((1, 77), dtype=jnp.int32)  # Dummy input_ids for state modality
                attention_mask = jnp.ones((1, 77), dtype=jnp.int32)

                language_instr_list = instruct_df["instruction"].to_list()
                dataset_mgr = DatasetManager(config.eval_human_demo_path)

                input_level_raw = dataset_mgr.get_levels(instructions=language_instr_list, n=1, squeeze_n=True)

                input_level = map2onehot_batch(input_level_raw)
                input_level = add_coord_channel_batch(input_level)
                pixel_values = input_level

                assert input_level.shape[-1] == config.clip_input_channel, \
                          f"Expected {config.clip_input_channel} channels, but got {input_level.shape[-1]} channels."

                rendered_values = render_array_batch(input_level_raw)

                if wandb.run:
                    for i, image_arr in enumerate(rendered_values):
                        wandb.log({f"CondState/reward_{i}": wandb.Image(image_arr)})

                logger.info(f"Generated cnnclip state embeddings for {input_level.shape[0]} instructions.")


            elif config.eval_modality == 'sketch':

                input_ids = jnp.zeros((1, 77), dtype=jnp.int32)  # Dummy input_ids for state modality
                attention_mask = jnp.ones((1, 77), dtype=jnp.int32)

                language_instr_list = instruct_df["instruction"].to_list()
                dataset_mgr = DatasetManager(config.eval_human_demo_path)

                sketch_values_raw = dataset_mgr.get_sketches(instructions=language_instr_list, n=1, squeeze_n=True)
                sketch_values = add_coord_channel_batch(sketch_values_raw)

                if wandb.run:
                    for i, sketch_arr in enumerate(sketch_values_raw):
                        sketch_arr = np.array(sketch_arr, dtype=np.float32)
                        wandb.log({f"CondSketch/reward_{i}": wandb.Image(sketch_arr)})

                logger.info(f"Generated cnnclip sketch embeddings for {sketch_values.shape[0]} instructions.")
            else:
                raise ValueError(f"Unknown eval_modality: {config.eval_modality}")


            instr_x = PCGRLObs(
                map_obs=jnp.repeat(init_x.map_obs, input_ids.shape[0], axis=0),
                past_map_obs=None,
                flat_obs=jnp.repeat(init_x.flat_obs, input_ids.shape[0], axis=0),
                nlp_obs=jnp.repeat(init_x.nlp_obs, input_ids.shape[0], axis=0),
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                sketch_values=sketch_values
            )

            _, _, _, text_embed, state_embed, sketch_embed = network.apply(runner_state.train_state.params, x=instr_x,
                                                  return_text_embed=True if config.eval_modality == 'text' else False,
                                                  return_state_embed=True if config.eval_modality == 'state' else False,
                                                  return_sketch_embed=True if config.eval_modality == 'sketch' else False)


            if config.eval_modality == 'text':
                embedding = text_embed
            elif config.eval_modality == 'state':
                embedding = state_embed
            elif config.eval_modality == 'sketch':
                embedding = sketch_embed

        instruct = Instruct(
                    reward_i=reward_enum,
                    condition=condition,
                    embedding=embedding,
                    condition_id=None
                )

        # Expand instruct for n_eps
        n_envs = config.n_envs
        n_eps = config.n_eps

        eval_batches = sorted(np.tile(list(range(0, len(instruct_df), 1)), n_eps))
        eval_batches = jnp.array(eval_batches)

        eval_rendered = list()

        n_rows = len(eval_batches)
        repetitions = np.tile(list(range(1, n_eps + 1, 1)), len(instruct_df))

        if len(eval_batches) != len(repetitions):
            raise Exception(f"Length of eval_batches and repetitions do not match {len(eval_batches)} != {len(repetitions)}")

        n_batches = math.ceil(n_rows / n_envs)

        losses, values, features = list(), list(), list()

        with tqdm(total=n_batches, desc="Evaluating Batches") as pbar:
            for batch_i in range(n_batches):
                # Get current batch
                start_idx = batch_i * n_envs
                end_idx = min((batch_i + 1) * n_envs, n_rows)
                idxes = eval_batches[start_idx:end_idx]
                batch_valid_size = len(idxes)

                batch_embedding = instruct.embedding[idxes]
                batch_reward_i = instruct.reward_i[idxes]
                batch_condition = instruct.condition[idxes]
                batch_repetition = repetitions[start_idx:end_idx]

                if len(batch_embedding) < n_envs:
                    batch_embedding = jnp.pad(batch_embedding,((0, n_envs - len(batch_embedding)), (0, 0)), mode="constant",)
                    batch_condition = jnp.pad(batch_condition,((0, n_envs - len(batch_condition)), (0, 0)), mode="constant",)
                    batch_reward_i = jnp.pad(batch_reward_i,((0, n_envs - len(batch_reward_i))), mode="constant",)
                    batch_repetition = jnp.pad(batch_repetition,((0, n_envs - len(batch_repetition))), mode="constant",)

                batch_instruct = Instruct(
                    reward_i=batch_reward_i,
                    condition=batch_condition,
                    embedding=batch_embedding,
                    condition_id=None
                )


                reset_rng = jnp.stack([jax.random.PRNGKey(seed) for seed in batch_repetition])

                init_obs, init_state = vmap_reset_fn(
                    reset_rng, env_params, gen_dummy_queued_state(env)
                )

                done = jnp.zeros((n_envs,), dtype=bool)

                def _env_step(carry, _):
                    rng, last_obs, state, done = carry

                    if config.use_nlp:
                        last_obs = last_obs.replace(nlp_obs=batch_instruct.embedding)

                    if config.vec_cont:
                        vmap_state_fn = jax.vmap(env.prob.get_cont_obs, in_axes=(0, 0, None))
                        cont_obs = vmap_state_fn(env_state.env_state.env_map, batch_instruct.condition, config.raw_obs)
                        last_obs = last_obs.replace(nlp_obs=cont_obs)

                    if config.use_clip:
                        last_obs = last_obs.replace(nlp_obs=batch_instruct.embedding)

                    rng, _rng = jax.random.split(rng)

                    # SELECT ACTION
                    pi, value, _, _ ,_, _ = network.apply(runner_state.train_state.params, last_obs,
                                                       return_text_embed=False,
                                                       return_state_embed=False,
                                                       return_sketch_embed=False)

                    action = pi.sample(seed=_rng)

                    log_prob = pi.log_prob(action)

                    # STEP ENV
                    rng, _rng = jax.random.split(rng)
                    rng_step = jax.random.split(_rng, config.n_envs)

                    # STEP ENV
                    vmap_step_fn = jax.vmap(env.step, in_axes=(0, 0, 0, None))

                    obsv, next_state, reward_env, done, info = vmap_step_fn(
                        rng_step, state, action, env_params
                    )

                    cond_reward_batch = get_reward_batch(
                        batch_instruct.reward_i,
                        batch_instruct.condition,
                        state.env_state.env_map,
                        next_state.env_state.env_map,
                    )
                    reward = jnp.where(done, reward_env, cond_reward_batch)

                    next_state = next_state.replace(
                        returned_episode_returns=next_state.returned_episode_returns
                        - reward_env
                        + reward
                    )
                    info["returned_episode_returns"] = next_state.returned_episode_returns

                    transition = Transition(
                        done, action, value, reward, log_prob, obsv, info,
                        next_state.env_state, None
                    )

                    return (rng, obsv, next_state, done), (transition, state)

                @jax.jit
                def run_eval_step(rng, init_obs, init_state, done):
                    env_step_len = int(
                        (config.map_width**2)
                        * config.max_board_scans
                        * (2 if config.representation == "turtle" else 1)
                    )
                    carray0 = (rng, init_obs, init_state, done)
                    _, (_, states) = jax.lax.scan(
                        _env_step,
                        carray0,
                        None,
                        length=env_step_len,
                    )
                    states = jax.tree.map(
                        lambda x, y: jnp.concatenate([x[None], y], axis=0),
                        init_state,
                        states,
                    )

                    def _env_render(env_state):
                        return jax.vmap(env.render)(env_state.env_state)
                    def _env_render_env_map(env_state):
                        return jax.vmap(env.render_env_map)(env_state.env_state.env_map)
                    last_states = jax.tree.map(lambda x: x[[-1], ...], states)
                    rendered = jax.vmap(_env_render)(last_states)
                    rendered_raw = jax.vmap(_env_render_env_map)(last_states)

                    rendered = rendered.transpose(1, 0, 2, 3, 4)  # (1, 4, 288, 288, 4)
                    _n_row, _n_eps, _n_height, _n_width, _n_channel = rendered_raw.shape
                    rendered_raw = rendered_raw.reshape(-1, _n_height, _n_width, _n_channel)

                    result = get_loss_batch(
                        reward_i=batch_instruct.reward_i,
                        condition=batch_instruct.condition,
                        env_maps=states.env_state.env_map[-1, :, :, :],
                    )
                    return result, rendered, rendered_raw, last_states

                rng = jax.random.PRNGKey(30)
                result, rendered, raw_rendered, last_states = run_eval_step(rng, init_obs, init_state, done)

                result = jax.device_get(result)
                losses.append(result.loss)
                values.append(result.value)
                features.append(result.feature)

                rendered = jax.device_get(rendered)
                raw_rendered = jax.device_get(raw_rendered)
                eval_rendered.append(raw_rendered)


                for idx, (row_i, reward_i, repeat_i, feature, state) in enumerate(zip(
                        idxes,
                        batch_reward_i[:batch_valid_size],
                        batch_repetition[:batch_valid_size],
                        result.feature[:batch_valid_size],
                        last_states.env_state.env_map[0, :][:batch_valid_size])
                ):
                    os.makedirs(f"{config.eval_dir}/reward_{row_i}/seed_{repeat_i}", exist_ok=True)

                    frames = rendered[idx]
                    if wandb.run:
                        raw_frame = raw_rendered[idx]
                        wandb.log({f"RawImage/reward_{row_i}/seed_{repeat_i}": wandb.Image(raw_frame)})

                    for i, frame in enumerate(frames):
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
                        task_text = ""
                        if 1 in reward_i:
                            task_text += f"RG: {int(feature[0])} | "
                        if 2 in reward_i:
                            task_text += f"PL: {int(feature[1])} | "
                        if 3 in reward_i:
                            task_text += f"WC: {int(feature[2])} | "
                        if 4 in reward_i:
                            task_text += f"BC: {int(feature[3])} | "
                        if 5 in reward_i:
                            task_text += f"BD | "
                        frame = cv2.putText(
                            frame,
                            task_text,
                            (20, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (255, 255, 255),
                            1,
                            cv2.LINE_AA,
                        )
                        imageio.imwrite(f"{config.eval_dir}/reward_{row_i}/seed_{repeat_i}/frame_{i}.png", frame)

                        np.save(f"{config.eval_dir}/reward_{row_i}/seed_{repeat_i}/state_{i}.npy", state)

                        if wandb.run:
                            wandb.log({f"Image/reward_{row_i}/seed_{repeat_i}": wandb.Image(frame)})

                            if config.flush:
                                os.system(f"rm -r {config.eval_dir}/reward_{row_i}/seed_{repeat_i}/frame_{i}.png")

                pbar.update(1)


        losses = np.stack(losses, axis=0).reshape(-1)[:n_rows]
        features = np.stack(features, axis=0).reshape(-1, NUM_FEATURES)[:n_rows]

        # get rows by index
        df_output = instruct_df.iloc[eval_batches]
        df_output = df_output.loc[:, ~df_output.columns.str.startswith("embed")]

        df_output['seed'] = repetitions
        df_output["loss"] = losses
        df_output = df_output.reset_index()

        # features for visualization
        feat_df = pd.DataFrame(features, columns=[f"feat_{i}" for i in FEATURE_NAMES]).reset_index()
        df_output = pd.concat([df_output, feat_df], axis=1)

        mean_loss = df_output.groupby('reward_enum').agg({'loss': ['mean']})
        mean_loss.columns = mean_loss.columns.droplevel(0)
        mean_loss = mean_loss.reset_index()

        dict_loss = dict()
        for _, row in mean_loss.iterrows():
            reward_enum, mean = row
            dict_loss[f'Loss/{str(int(reward_enum))}'] = mean


        # Start of diversity evaluation
        if config.diversity:
            scores = list()

            for row_i, row in tqdm(instruct_df.iterrows(), desc="Computing Diversity"):
                states = list()
                for seed_i in range(1, config.n_eps + 1):
                    state = np.load(f"{config.eval_dir}/reward_{row_i}/seed_{seed_i}/state_0.npy")
                    states.append(state)
                states = np.array(states)
                score = compute_hamming_distance(states)
                scores.append(score)

            diversity_df = instruct_df.copy()
            diversity_df = diversity_df.loc[:, ~diversity_df.columns.str.startswith('embed')]
            diversity_df['diversity'] = scores

            if wandb.run:
                diversity_table = wandb.Table(dataframe=diversity_df)
                wandb.log({'diversity': diversity_table})
        # End of diversity evaluation

        # Start of human-likeness evaluation
        if config.human_likeness:
            eval_rendered = np.concatenate(eval_rendered, axis=0)
            eval_rendered = eval_rendered[:n_rows]

            evaluator = ViTEvaluator(normalized_vector=config.vit_normalize)

            index_ids = instruct_df.index.to_numpy()
            task_ids = (index_ids // 4).astype(int)
            task_ids = np.repeat(task_ids, n_eps)

            human_scores = evaluator.run(eval_rendered, task_ids)
            df_output['human_likeness'] = human_scores

            # End of human-likeness evaluation

        if config.tpkldiv:

            states = list()
            for row_i, row in tqdm(instruct_df.iterrows(), desc="Computing TPKLDiv"):
                for seed_i in range(1, config.n_eps + 1):
                    state = np.load(f"{config.eval_dir}/reward_{row_i}/seed_{seed_i}/state_0.npy")
                    states.append(state)

            states = np.array(states)

            index_ids = instruct_df.index.to_numpy()
            task_ids = (index_ids // 4).astype(int)
            task_ids = np.repeat(task_ids, n_eps)

            evaluator = TPKLEvaluator()
            scores = evaluator.run(states, task_ids, show_progress=True)

            tpkl_score = np.array(scores).reshape(-1)

            df_output['tpkldiv'] = tpkl_score

        df_output.to_csv(f"{config.eval_dir}/loss.csv", index=False)

        if wandb.run:
            wandb.log(dict_loss)
            raw_table = wandb.Table(dataframe=df_output)
            wandb.log({'raw': raw_table})

        if wandb.run and config.flush:
            for row_i, _ in instruct_df.iterrows():
                os.system(f"rm -r {config.eval_dir}/reward_{row_i}")

        return losses


    return lambda rng: eval(rng, config)


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
    # init_x = env.observation_space(env_params).sample(_rng)[None, ]
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

    # reset_rng_r = reset_rng.reshape((config.n_gpus, -1) + reset_rng.shape[1:])
    vmap_reset_fn = jax.vmap(env.reset, in_axes=(0, None, None))
    # pmap_reset_fn = jax.pmap(vmap_reset_fn, in_axes=(0, None))
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

    options = ocp.CheckpointManagerOptions(max_to_keep=2, create=False)
    # checkpoint_manager = orbax.checkpoint.CheckpointManager(
    #     ckpt_dir, orbax.checkpoint.PyTreeCheckpointer(), options)
    checkpoint_manager = ocp.CheckpointManager(
        ckpt_dir,
        ocp.PyTreeCheckpointer(),
        options=options,
    )

    def try_load_ckpt(steps_prev_complete, target):
        try:
            restored_ckpt = checkpoint_manager.restore(
                steps_prev_complete,
                items=target,
                # args=ocp.args.StandardRestore(target, strict=False),
            )
        except Exception as e:
            restored_ckpt = checkpoint_manager.restore(
                steps_prev_complete,
                args=ocp.args.StandardRestore(target),

            )

        restored_ckpt["steps_prev_complete"] = steps_prev_complete
        if restored_ckpt is None:
            raise TypeError("Restored checkpoint is None")

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

        ckpt_subdirs = os.listdir(config.encoder.ckpt_path)
        ckpt_steps = [int(cs) for cs in ckpt_subdirs if cs.isdigit()]

        # Sort in decreasing order
        ckpt_steps.sort(reverse=True)
        for steps_prev_complete in ckpt_steps:

            ckpt_dir = os.path.join(config.encoder.ckpt_path, str(steps_prev_complete))

            try:
                from flax.training import checkpoints

                enc_state = checkpoints.restore_checkpoint(ckpt_dir=ckpt_dir, target=None)
                assert enc_state is not None, "Restored params are None ({})".format(ckpt_dir)

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

                    enc_param = get_encoder_params_recursive(enc_param, 'encoder')
                    assert enc_param is not None, "Encoder not found in checkpoint"

                break


            except TypeError as e:
                logging.error(f"Failed to load checkpoint at step {steps_prev_complete}. Error: {e}")
                continue
    else:
        enc_param = None


    return checkpoint_manager, restored_ckpt, enc_param

def main_chunk(config, rng):
    """When jax jits the training loop, it pre-allocates an array with size equal to number of training steps. So, when training for a very long time, we sometimes need to break training up into multiple
    chunks to save on VRAM.
    """

    if not config.random_agent:
        _, restored_ckpt, encoder_param = init_checkpointer(config)
    else:
        restored_ckpt, encoder_param = None, None

    eval_jit = make_eval(config, restored_ckpt, encoder_param)
    out = eval_jit(rng)
    jax.block_until_ready(out)

    return out

@hydra.main(version_base=None, config_path="./conf", config_name="eval_pcgrl")
def main(config: EvalConfig):
    config = init_config(config)

    if config.eval_aug_type is not None and config.eval_embed_type is not None and config.eval_instruct is not None:
        config.eval_instruct_csv = f'{config.eval_aug_type}/{config.eval_embed_type}/{config.eval_instruct}'

    if config.n_eps < 2 and config.diversity:
        raise Exception("Diversity evaluation requires n_eps > 1")

    rng = jax.random.PRNGKey(config.seed)

    exp_dir = config.exp_dir
    logger.info(f"running experiment at {exp_dir}")

    eval_dir = os.path.join(exp_dir, f"ev_{config.eval_instruct}_{config.eval_modality}"
                                     f"{f'_{config.eval_exp_name}' if config.eval_exp_name else ''}")
    config.eval_dir = eval_dir

    if config.reevaluate:
        if os.path.exists(eval_dir):
            logger.info(f"Removing existing evaluation directory at {eval_dir}")
            os.system(f"rm -r {eval_dir}")
        else:
            logger.info(f"No existing evaluation directory found at {eval_dir}")
    else:
        if os.path.exists(eval_dir):
            raise Exception(f"Evaluation directory already exists at {eval_dir}. Set reevaluate=True to overwrite.")

    os.makedirs(eval_dir, exist_ok=True)

    logger.info(f"running evaluation at {eval_dir}")

    if config.wandb_key:
        dt = datetime.now().strftime("%Y%m%d%H%M%S")
        wandb_id = f"{get_wandb_name_eval(config)}-{dt}"
        wandb.login(key=config.wandb_key)
        wandb.init(
            project=config.wandb_project,
            entity=config.wandb_entity,
            name=get_wandb_name_eval(config),
            id=wandb_id,
            save_code=True,
            config_exclude_keys=[
                "wandb_key",
                "_vid_dir",
                "_img_dir",
                "_numpy_dir",
                "overwrite",
                "initialize",
            ],
        )
        wandb.config.update(dict(config), allow_val_change=True)


    main_chunk(config, rng)

if __name__ == '__main__':
    main()