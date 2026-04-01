import os
import warnings

import gymnax
import jax
import numpy as np
import jax.numpy as jnp
import yaml
import imageio

import wandb
from conf.config import Config
from conf.game_utils import parse_game_str
from envs.candy import Candy, CandyParams
from envs.pcgrl_env import PROB_CLASSES, PCGRLEnvParams, PCGRLEnv, ProbEnum, RepEnum, \
    gen_dummy_queued_state
from envs.play_pcgrl_env import PlayPCGRLEnv, PlayPCGRLEnvParams
from models import ActorCritic, ActorCriticPCGRL, AutoEncoder, ConvForward, \
    ConvForward2, Dense, NCA, SeqNCA


def is_default_hiddims(config: Config):
    return tuple(config.hidden_dims) == (64, 256)[:len(config.hidden_dims)]


def get_exp_dir(config: Config):
    if config.env_name == 'PCGRL':
        ctrl_str = '_ctrl_' + '_'.join(config.ctrl_metrics) if len(config.ctrl_metrics) > 0 else ''
        game_str = f'game-{config.game}_' if hasattr(config, 'game') and config.game else ''
        exp_dir = os.path.join(
            'saves',
            f'{config.problem}{ctrl_str}_{config.representation}_{config.model}-' +
            f'{config.activation}_w-{config.map_width}_' + \
            ('random-shape_' if config.randomize_map_shape else '') + \
            f'{game_str}' +
            f'vrf-{config.vrf_size}_' + \
            (f'cp-{config.change_pct}_' if config.change_pct > 0 else '') +
            f'arf-{config.arf_size}_' + \
            (f"hd-{'-'.join((str(hd) for hd in config.hidden_dims))}_" if not is_default_hiddims(config) else '') + \
            f'sp-{config.static_tile_prob}_'
            f'bs-{config.max_board_scans}_' + \
            f'fz-{config.n_freezies}_' + \
            f'act-{"x".join([str(e) for e in config.act_shape])}_' + \
            f'nag-{config.n_agents}_' + \
            ('empty-start_' if config.empty_start else '') + \
            ('pinpoints_' if config.pinpoints else '') + \
            (f'{config.n_envs}-envs_' if config.profile_fps else '') + \
            f'{config.seed}_{config.exp_name}')
    elif config.env_name == 'PlayPCGRL':
        exp_dir = os.path.join(
            'saves',
            f'play_w-{config.map_width}_' + \
            f'{config.model}-{config.activation}_' + \
            f'vrf-{config.vrf_size}_arf-{config.arf_size}_' + \
            f'{config.seed}_{config.exp_name}',
        )
    elif config.env_name == 'Candy':
        exp_dir = os.path.join(
            'saves',
            'candy_' + \
            f'{config.seed}_{config.exp_name}',
        )
    else:
        exp_dir = os.path.join(
            'saves',
            config.env_name,
        )
    return exp_dir


def init_config(config: Config):
    config.n_gpus = jax.local_device_count()

    # ── game 약어 → include_* 자동 동기화 ──
    if hasattr(config, 'game') and config.game:
        includes = parse_game_str(config.game)
        for key, val in includes.items():
            setattr(config, key, val)

    if config.env_name == 'Candy':
        config.exp_dir = get_exp_dir(config)
        return config

    if config.representation in set({'wide', 'nca'}):
        config.arf_size = config.vrf_size = config.map_width

    if config.representation == 'nca':
        config.act_shape = (config.map_width, config.map_width)

    else:
        config.arf_size = (2 * config.map_width -
                           1 if config.arf_size == -1 else config.arf_size)

        config.vrf_size = (2 * config.map_width -
                           1 if config.vrf_size == -1 else config.vrf_size)

    if hasattr(config, 'evo_pop_size') and hasattr(config, 'n_envs'):
        assert config.n_envs % (config.evo_pop_size * 2) == 0, "n_envs must be divisible by evo_pop_size * 2"
    if config.model == 'conv2':
        config.arf_size = config.vrf_size = min([config.arf_size, config.vrf_size])

    config.exp_dir = get_exp_dir(config)

    if config.model == 'seqnca':
        config.hidden_dims = config.hidden_dims[:1]

    return config



def get_ckpt_dir(config: Config):
    return os.path.join(config.exp_dir, 'ckpts')


def init_network(env: PCGRLEnv, env_params: PCGRLEnvParams, config: Config):
    if config.env_name == 'Candy':
        # In the candy-player environment, action space is flat discrete space over all candy-direction combos.
        action_dim = env.action_space(env_params).n

    elif 'PCGRL' in config.env_name:
        action_dim = env.rep.action_space.n
        # First consider number of possible tiles
        # action_dim = env.action_space(env_params).n
        # action_dim = env.rep.per_tile_action_dim

    else:
        action_dim = env.num_actions

    if config.model == "dense":
        network = Dense(
            action_dim, activation=config.activation,
            arf_size=config.arf_size, vrf_size=config.vrf_size,
        )
    elif config.model == "conv":
        network = ConvForward(
            action_dim=action_dim, activation=config.activation,
            arf_size=config.arf_size, act_shape=config.act_shape,
            vrf_size=config.vrf_size,
            hidden_dims=config.hidden_dims,
        )
    elif config.model == "conv2":
        network = ConvForward2(
            action_dim=action_dim, activation=config.activation,
            act_shape=config.act_shape,
            hidden_dims=config.hidden_dims,
        )
    elif config.model == "seqnca":
        network = SeqNCA(
            action_dim, activation=config.activation,
            arf_size=config.arf_size, act_shape=config.act_shape,
            vrf_size=config.vrf_size,
            hidden_dims=config.hidden_dims,
        )
    elif config.model in {"nca", "autoencoder"}:
        if config.model == "nca":
            network = NCA(
                representation=config.representation,
                tile_action_dim=env.rep.tile_action_dim,
                activation=config.activation,
            )
        elif config.model == "autoencoder":
            network = AutoEncoder(
                representation=config.representation,
                action_dim=action_dim,
                activation=config.activation,
            )
    else:
        raise Exception(f"Unknown model {config.model}")
    # if config.env_name == 'PCGRL':
    if 'PCGRL' in config.env_name:
        network = ActorCriticPCGRL(network, act_shape=config.act_shape,
                                   n_agents=config.n_agents, n_ctrl_metrics=len(config.ctrl_metrics))
    # elif config.env_name == 'PlayPCGRL':
    #     network = ActorCriticPlayPCGRL(network)
    else:
        network = ActorCritic(network)
    return network


def get_env_params_from_config(config: Config):
    map_shape = ((config.map_width, config.map_width) if not config.is_3d
                 else (config.map_width, config.map_width, config.map_width))
    rf_size = max(config.arf_size, config.vrf_size)
    rf_shape = (rf_size, rf_size) if not config.is_3d else (rf_size, rf_size, rf_size)

    act_shape = tuple(config.act_shape)
    if config.is_3d:
        assert len(config.act_shape) == 3

    # Convert strings to enum ints
    problem = ProbEnum[config.problem.upper()]
    prob_cls = PROB_CLASSES[problem]
    ctrl_metrics = tuple([int(prob_cls.metrics_enum[c.upper()]) for c in config.ctrl_metrics])

    env_params = PCGRLEnvParams(
        problem=problem,
        representation=int(RepEnum[config.representation.upper()]),
        map_shape=map_shape,
        rf_shape=rf_shape,
        act_shape=act_shape,
        static_tile_prob=config.static_tile_prob,
        n_freezies=config.n_freezies,
        n_agents=config.n_agents,
        max_board_scans=config.max_board_scans,
        ctrl_metrics=ctrl_metrics,
        change_pct=config.change_pct,
        randomize_map_shape=config.randomize_map_shape,
        empty_start=config.empty_start,
        pinpoints=config.pinpoints,
        multiagent=config.multiagent or config.n_agents > 1,
    )
    return env_params


def get_play_env_params_from_config(config: Config):
    map_shape = (config.map_width, config.map_width)
    rf_size = max(config.arf_size, config.vrf_size)
    rf_shape = (rf_size, rf_size) if not config.is_3d else (rf_size, rf_size, rf_size)

    return PlayPCGRLEnvParams(
        map_shape=map_shape,
        rf_shape=rf_shape,
    )


def gymnax_pcgrl_make(env_name, config: Config, **env_kwargs):
    if env_name in gymnax.registered_envs:
        return gymnax.make(env_name)

    elif env_name == 'PCGRL':
        env_params = get_env_params_from_config(config)
        env = PCGRLEnv(env_params)

    elif env_name == 'PlayPCGRL':
        env_params = get_play_env_params_from_config(config)
        env = PlayPCGRLEnv(env_params)

    elif env_name == 'Candy':
        env_params = CandyParams()
        env = Candy(env_params)

    return env, env_params



def write_sweep_confs(_hypers: dict, eval_hypers: dict):
    conf_sweeps_dir = os.path.join('conf', 'sweeps')
    os.makedirs(conf_sweeps_dir, exist_ok=True)
    for grid_hypers in _hypers:
        name = grid_hypers['NAME']
        save_grid_hypers = grid_hypers.copy()
        save_grid_hypers['eval_hypers'] = eval_hypers
        with open(os.path.join(conf_sweeps_dir, f'{name}.yaml'), 'w') as f:
            f.write(yaml.dump(save_grid_hypers))
        # with open(os.path.join(conf_sweeps_dir, f'{name}.json'), 'w') as f:
        #     f.write(json.dumps(grid_hypers, indent=4))


def make_sim_render_episode_single(config: Config, network, env: PCGRLEnv, env_params: PCGRLEnvParams, runner_state):
    max_episode_len = env.max_steps  # Maximum steps per episode

    def sim_render_episode(actor_params):
        def step_env(carry, _):
            rng, obs, state, done = carry

            # SELECT ACTION
            # # Squash the gpu dimension (network only takes one batch dimension)
            #
            rng, _rng = jax.random.split(rng)

            pi, value = network.apply(actor_params, obs)
            action = pi.sample(seed=rng)

            # # STEP ENV
            rng, _rng = jax.random.split(rng)
            rng_step = jax.random.split(_rng, config.n_envs)

            # rng_step = rng_step.reshape((config.n_gpus, -1) + rng_step.shape[1:])
            vmap_step_fn = jax.vmap(env.step, in_axes=(0, 0, 0, None))
            # pmap_step_fn = jax.pmap(vmap_step_fn, in_axes=(0, 0, 0, None))
            obsv, next_state, reward, done, info = vmap_step_fn(
                rng_step, state, action, env_params
            )
            # next_state = state

            return (rng, obsv, next_state, done), next_state

        # Initialize the `done` flag
        done = jnp.zeros((config.n_envs,), dtype=bool)

        rng = jax.random.PRNGKey(0)

        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config.n_envs)

        vmap_reset_fn = jax.vmap(env.reset, in_axes=(0, None, None))
        init_obs, init_state = vmap_reset_fn(
            reset_rng,
            env_params,
            gen_dummy_queued_state(env)
        )
        _, states = jax.lax.scan(step_env, (rng, init_obs, init_state, done), None, length=max_episode_len)
        # Concatenate the init_state to the states
        states = jax.tree.map(lambda x, y: jnp.concatenate([x[None], y], axis=0), init_state, states)
        first_env_state = jax.tree_map(lambda x: x[:, 0], states.env_state)

        frames = jax.vmap(env.render)(first_env_state)

        first_env_last_state = first_env_state.env_map[-2]
        final_frame = frames[-2]

        # pass states to save them as "npy" files
        return frames, states

    # JIT compile the simulation function for better performance
    return jax.jit(sim_render_episode)


def render_callback(frames,
                    states,
                    video_dir: str = None,
                    image_dir: str = None,
                    numpy_dir: str = None,
                    traj_dir: str = None,
                    t: int = 0,
                    logger=None,
                    config: Config = None):
    fps = 120

    # save video
    if video_dir is None:
        warnings.warn("video_dir is not set. Skipping video save.")
    else:
        if not os.path.exists(video_dir):
            os.makedirs(video_dir, exist_ok=True)

        video_path = os.path.join(video_dir, f"video_{t}.gif")
        imageio.mimsave(video_path, np.array(frames[:-2]), duration=1 / fps, loop=1)

        if logger is not None:
            logger.info(f"Saved gif to {video_path}")
        else:
            print(f"Saved gif to {video_path}")

        # logging video at Wandb
        if wandb.run is not None:
            key_name = f"Iteration_{config.current_iteration}/Eval/video" if config.current_iteration > 0 else "Eval/video"

            # convert to t to int

            wandb.log({key_name: wandb.Video(video_path, fps=fps, format="gif"), 'train/step': t})

    if numpy_dir is None:
        warnings.warn("numpy_dir is not set. Skipping image numpy.")
    else:
        if not os.path.exists(numpy_dir):
            os.makedirs(numpy_dir, exist_ok=True)

        levels = states.env_state.env_map[-2, :10, ...]

        # save with level_0.npy, level_1.npy, ...
        for idx, level in enumerate(levels):
            numpy_path = os.path.join(numpy_dir, f"level_{t}_{idx}.npy")
            np.save(numpy_path, level)

        if logger is not None:
            logger.info(f"Saved {len(levels)} npy files to {numpy_dir}/level_{t}_*.npy")
        else:
            print(f"Saved {len(levels)} npy files to {numpy_dir}/level_{t}_*.npy")
    
    if traj_dir is None:
        warnings.warn("traj_dir is not set. Skipping trajectory save.")
    else:
        if not os.path.exists(traj_dir):
            os.makedirs(traj_dir, exist_ok=True)
        env0_traj = states.env_state.env_map[:, 0, ...]
        traj_path = os.path.join(traj_dir, f"traj_{t}.npy")
        np.save(traj_path, env0_traj)
        if logger is not None:
            logger.info(f"Saved trajectory to {traj_path}")
        else:
            print(f"Saved trajectory to {traj_path}")

    # save last image to PNG
    if image_dir is None:
        warnings.warn("image_dir is not set. Skipping image save.")
    else:
        if not os.path.exists(image_dir):
            os.makedirs(image_dir, exist_ok=True)

        image_path = os.path.join(image_dir, f"image_{t}.png")
        imageio.imwrite(image_path, frames[-2])

        if logger is not None:
            logger.info(f"Saved png to {image_path}")
        else:
            print(f"Saved png png {image_path}")

        # logging image on wandb
        if wandb.run is not None:
            key_name = f"Iteration_{config.current_iteration}/Eval/image" if config.current_iteration > 0 else "Eval/image"

            wandb.log({key_name: wandb.Image(image_path), 'train/step': t})
