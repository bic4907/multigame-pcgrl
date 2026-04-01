import os
import logging
import gymnax
import jax
import yaml
from os.path import basename, dirname

from conf.config import Config, EncoderConfig
from conf.game_utils import parse_game_str
from envs.candy import Candy, CandyParams
from envs.pcgrl_env import PROB_CLASSES, PCGRLEnvParams, PCGRLEnv, ProbEnum, RepEnum
from envs.play_pcgrl_env import PlayPCGRLEnv, PlayPCGRLEnvParams

log_level = os.getenv('LOG_LEVEL', 'INFO').upper()  # Add the environment variable ;LOG_LEVEL=DEBUG
logger = logging.getLogger(basename(__file__))
logger.setLevel(getattr(logging, log_level, logging.INFO))

DATASET_DIR = os.path.abspath(os.path.join(dirname(__file__), '..', "pcgrl_buffer"))



def is_default_hiddims(config: Config):
    return tuple(config.hidden_dims) == (64, 256)[:len(config.hidden_dims)]


def get_dataset_dir(config):
    filename = f"embed-{config.embed_type}"

    if config.buffer_ratio < 1:
        filename = f"{filename}_bufratio-{config.buffer_ratio}"

    filename = f"{filename}.npz"

    filename = os.path.join(DATASET_DIR, filename)

    return filename


def get_exp_group(config):
    if config.encoder.model in ['clip', 'cnnclip']:

        text_ratio_str = 't' if config.text_ratio==1.0 else f"t.{str(config.text_ratio).split('.')[1]}"
        modality = [text_ratio_str]
        
        if config.encoder.state:
            state_ratio_str = 's' if config.state_ratio==1.0 else f"s.{str(config.state_ratio).split('.')[1]}"
            modality.append(state_ratio_str)

        modality = ''.join(modality)

        config_dict = {
            'game': getattr(config, 'game', 'dg'),
            'md': modality,
            'exp': config.exp_name,
        }
    else:
        config_dict = {
            'game': getattr(config, 'game', 'dg'),
            'exp': config.exp_name,
        }

    exp_group = os.path.join(
        '_'.join([f'{key}-{value}' for key, value in config_dict.items()])
    )

    flags_dict = {}
    # Append suffixes for enabled flags
    for flag, suffix in flags_dict.items():
        if getattr(config, flag, False):  # Check if the flag exists and is True
            exp_group += f'_{suffix}'

    return exp_group

def get_short_target(target: str) -> str:
    # Split the target string into words
    words = target.split()

    # If there's only one word, return it with the length
    if len(words) == 1:
        return f"{words[0]}_{len(target)}"

    # Otherwise, take the first and last words and include the length
    return f"{words[0]}X{words[-1]}{len(target)}"


def get_exp_name(config):
    exp_group = get_exp_group(config)

    # target_character = get_short_target(config.target_character) if config.task == 'scenario' else config.target_character
    return f'{config.dir_prefix}{exp_group}_{config.seed}'


def get_exp_dir(config):
    return os.path.join('saves', get_exp_name(config))


def init_config(config: Config):
    config.n_gpus = jax.local_device_count()

    if config.aug_type is not None and config.embed_type is not None and config.instruct is not None:
        config.instruct_csv = f'{config.aug_type}/{config.embed_type}/{config.instruct}'

    config.text_ratio = min([0.25,0.5,0.75,1.0], key=lambda x: abs(x - config.text_ratio))
    
    # For coord Channel(x,y)
    config.clip_input_channel = config.clip_input_channel + 2

    # ── game 약어 → include_* 자동 동기화 ──
    if hasattr(config, 'game') and config.game:
        includes = parse_game_str(config.game)
        for key, val in includes.items():
            setattr(config, key, val)

    config.exp_group = get_exp_group(config)
    config.exp_dir = get_exp_dir(config)

    config.arf_size = (2 * config.map_width -
                       1 if config.arf_size == -1 else config.arf_size)

    config.vrf_size = (2 * config.map_width -
                       1 if config.vrf_size == -1 else config.vrf_size)

    if config.model == 'seqnca':
        config.hidden_dims = config.hidden_dims[:1]

    return config


def init_clip_config(config: Config):
    config.n_gpus = jax.local_device_count()

    if config.aug_type is not None and config.embed_type is not None and config.instruct is not None:
        config.instruct_csv = f'{config.aug_type}/{config.embed_type}/{config.instruct}'

    # ── game 약어 → include_* 자동 동기화 ──
    if hasattr(config, 'game') and config.game:
        includes = parse_game_str(config.game)
        for key, val in includes.items():
            setattr(config, key, val)

    config.exp_group = get_exp_group(config)
    config.exp_dir = get_exp_dir(config)

    config.arf_size = (2 * config.map_width -
                       1 if config.arf_size == -1 else config.arf_size)

    config.vrf_size = (2 * config.map_width -
                       1 if config.vrf_size == -1 else config.vrf_size)

    if config.model == 'seqnca':
        config.hidden_dims = config.hidden_dims[:1]

    return config


def get_ckpt_dir(config: Config):
    return os.path.join(config.exp_dir, 'ckpts')


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
        nlp_input_dim=config.nlp_input_dim if config.use_nlp else -1,
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
