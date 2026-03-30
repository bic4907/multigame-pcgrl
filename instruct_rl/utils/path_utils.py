import os
import gymnax
import jax
from glob import glob
import yaml
from os.path import abspath, join

from encoder.model import apply_encoder_model
from encoder.clip_model import get_clip_encoder, get_cnnclip_encoder
from conf.config import Config, EvoMapConfig, SweepConfig, TrainConfig, EncoderConfig
from envs.candy import Candy, CandyParams
from envs.pcgrl_env import PROB_CLASSES, PCGRLEnvParams, PCGRLEnv, ProbEnum, RepEnum
from envs.play_pcgrl_env import PlayPCGRLEnv, PlayPCGRLEnvParams
from models import ActorCritic, ActorCriticPCGRL, AutoEncoder, ConvForward, ConvForward2, Dense, \
    NCA, SeqNCA, NLPConvForward, EncoderNLPConvForward, EncoderCLIPConvForward

from instruct_rl.utils.log_utils import get_logger

logger = get_logger(__file__)


def get_exp_dir_evo_map(config: EvoMapConfig):
    exp_dir = os.path.join(
        'saves_evo_map',
        config.problem,
        f'pop-{config.evo_pop_size}_' +
        f'parents-{config.n_parents}_' +
        f'mut-{config.mut_rate}_' +
        f'{config.seed}_{config.exp_name}',
        )
    return exp_dir


def is_default_hiddims(config: Config):
    return tuple(config.hidden_dims) == (64, 256)[:len(config.hidden_dims)]


def get_exp_group(config):
    if config.env_name == 'PCGRL':

        # ── MultiGameDataset 기반 CPCGRL / IPCGRL / VIPCGRL 모드 ──
        if hasattr(config, 'dataset_game') and config.dataset_game is not None:
            config_dict = {
                'model': config.model,
                'exp': config.exp_name,
                'game': config.dataset_game,
            }
            if hasattr(config, 'dataset_reward_enum') and config.dataset_reward_enum is not None:
                config_dict['re'] = config.dataset_reward_enum
            exp_group = '_'.join([f'{key}-{value}' for key, value in config_dict.items()])
            if config.use_clip:
                exp_group += '_clip'
            elif config.use_nlp:
                exp_group += '_nlp'
            else:
                exp_group += '_vec_ro'
            return exp_group

        if config.use_nlp or config.vec_cont or config.use_clip:
            nlp_dict = {
                'embed': config.embed_type,
                'inst': config.instruct,
            }

            if config.encoder.model:
                nlp_dict['enc'] = config.encoder.model
                nlp_dict['tr'] = str(config.encoder.trainable).lower()[0]
            if config.encoder.model == 'clip':
                nlp_dict['hu'] = str(config.human_demo).lower()[0]
            if config.encoder.model == 'cnnclip':
                if config.use_sim_reward:
                    if config.only_sim_reward:
                        nlp_dict['os'] = str(config.only_sim_reward).lower()[0]
                    else:
                        nlp_dict['s'] = str(config.use_sim_reward).lower()[0]
                    nlp_dict['co'] = str(config.SIM_COEF)
                    nlp_dict['hu'] = str(config.human_demo).lower()[0]
        else:
            nlp_dict = {}

        # task
        config_dict = {
            'model': config.model,
            'exp': config.exp_name,
        }

        enc_def_setting = EncoderConfig()
        tr_def_setting = TrainConfig()

        # RQ4 parameters
        if config.buffer_ratio != tr_def_setting.buffer_ratio:
            config_dict['br'] = config.buffer_ratio
        if config.encoder.output_dim != enc_def_setting.output_dim:
            config_dict['es'] = config.encoder.output_dim

        if hasattr(config, 'random_agent') and config.random_agent:
            config_dict['model'] = 'rand'

        encoder_dict = dict()

        if config.encoder.model in ['clip', 'cnnclip']:
            text_ratio_str = 't' if config.text_ratio == 1.0 else f"t.{str(config.text_ratio).split('.')[1]}"
            modality = [text_ratio_str]
            if config.encoder.state:
                state_ratio_str = 's' if config.state_ratio == 1.0 else f"s.{str(config.state_ratio).split('.')[1]}"
                modality.append(state_ratio_str)
            if config.encoder.sketch:
                sketch_ratio_str = 'k' if config.sketch_ratio == 1.0 else f"k.{str(config.sketch_ratio).split('.')[1]}"
                modality.append(sketch_ratio_str)
            modality = ''.join(modality)
            encoder_dict['md'] = modality

            encoder_dict['cf'] = config.SIM_COEF

        config_dict = {**nlp_dict, **config_dict, **encoder_dict}


        exp_group = os.path.join(
            '_'.join([f'{key}-{value}' for key, value in config_dict.items()])
        )

        flags_dict = {
            'vec_cont': 'vec',
            'raw_obs': 'ro',
        }
        # Append suffixes for enabled flags

        for flag, suffix in flags_dict.items():
            if getattr(config, flag, False):  # Check if the flag exists and is True
                exp_group += f'_{suffix}'

    elif config.env_name == 'PlayPCGRL':
        exp_group = os.path.join(
            'saves',
            f'play_w-{config.map_width}_' + \
            f'{config.model}-{config.activation}_' + \
            f'vrf-{config.vrf_size}_arf-{config.arf_size}_' + \
            f'{config.exp_name}'
        )
    elif config.env_name == 'Candy':
        exp_group = os.path.join(
            'candy_' + \
            f'{config.exp_name}'
        )
    else:
        exp_group = os.path.join(
            config.env_name
        )
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

    return f'{exp_group}_s-{config.seed}'


def get_exp_dir(config):
    return os.path.join('saves', get_exp_name(config))


def init_config(config: Config):
    config.n_gpus = jax.local_device_count()

    # ── MultiGameDataset 기반 CPCGRL / IPCGRL / VIPCGRL 모드 ─────────────
    if hasattr(config, 'dataset_game') and config.dataset_game is not None:
        config.raw_obs = True
        # instruct_csv는 사용하지 않음
        config.instruct_csv = None

        if config.use_clip:
            # ── VIPCGRL 모드: pretrained CLIP 임베딩을 입력으로 사용 ──
            config.vec_cont = False
            config.use_nlp = False
            if config.encoder.model is None:
                config.encoder.model = 'cnnclip'
            if config.nlp_input_dim <= 0:
                config.nlp_input_dim = 512  # CLIP text feature dim
            config.vec_input_dim = config.nlp_input_dim
            config.use_sim_reward = True
            # dataset 기반 VIPCGRL: CLIP 임베딩이 사전 계산되어 nlp_obs에 직접 입력
            # → cnnclipconv 가 아닌 nlpconv 로 설정 (내부 CLIP encoder 불필요)
            if config.model not in ('nlpconv',):
                config.model = 'nlpconv'
            logger.info(f"[VIPCGRL] dataset_game={config.dataset_game}, "
                        f"dataset_reward_enum={getattr(config, 'dataset_reward_enum', None)}, "
                        f"nlp_input_dim={config.nlp_input_dim}, "
                        f"encoder={config.encoder.model}")
        elif config.use_nlp:
            # ── IPCGRL 모드: BERT → MLP 인코더 피처를 입력으로 사용 ──
            config.vec_cont = False
            if config.nlp_input_dim <= 0:
                config.nlp_input_dim = 768  # BERT base dim
            config.vec_input_dim = config.nlp_input_dim
            if config.model not in ('nlpconv',):
                config.model = 'nlpconv'
            # IPCGRL 은 MLP 인코더를 기본으로 사용
            if config.encoder.model is None:
                config.encoder.model = 'mlp'
            logger.info(f"[IPCGRL] dataset_game={config.dataset_game}, "
                        f"dataset_reward_enum={getattr(config, 'dataset_reward_enum', None)}, "
                        f"nlp_input_dim={config.nlp_input_dim}, "
                        f"encoder={config.encoder.model}")
        else:
            # ── CPCGRL 모드: raw condition 벡터를 사용 ──
            config.vec_cont = True
            config.use_nlp = False
            config.vec_input_dim = 9
            config.nlp_input_dim = 0
            logger.info(f"[CPCGRL] dataset_game={config.dataset_game}, "
                        f"dataset_reward_enum={getattr(config, 'dataset_reward_enum', None)}")

            if config.vec_cont is True and config.model != 'contconv':
                config.model = 'contconv'
                logger.info("[CPCGRL] Setting model to `contconv` due to the vec_cont flag")

        # exp_dir 등 공통 설정은 아래에서 계속 처리

    elif config.aug_type is not None and config.embed_type is not None and config.instruct is not None:
        config.instruct_csv = f'{config.aug_type}/{config.embed_type}/{config.instruct}'

    if config.use_sim_reward == False and config.only_sim_reward == True:
        logger.warning("Setting use_sim_reward to True due to the only_sim_reward flag")
        config.use_sim_reward = True

    if config.encoder.model == 'cnnclip':
        config.use_clip = True
        config.use_sim_reward = True

    if config.instruct_csv is not None:
        if hasattr(config, 'vec_cont') and config.vec_cont is True:
            config.use_nlp = False
            config.vec_input_dim = 9
            config.nlp_input_dim = 0
        elif hasattr(config, 'use_clip') and config.use_clip is True:
            config.use_nlp = False
            if config.model == 'conv':
                if config.encoder.model == 'clip':
                    config.model = 'clipconv'
                    logger.info("Setting model to `clipconv` due to the instruct set")
                elif config.encoder.model == 'cnnclip':
                    config.model = 'cnnclipconv'
                    logger.info("Setting model to `cnnclipconv` due to the instruct set")

        else:
            config.use_nlp = True
            if config.model == 'conv':
                config.model = 'nlpconv'
                logger.info("Setting model to `nlpconv` due to the instruct set")

    if config.vec_cont is True and config.model != 'contconv':
        config.model = 'contconv'
        logger.warning("Setting model to `contconv` due to the vec_cont flag")

    if config.encoder.model is not None:
        logger.info(f'Loading checkpoint for the encoder model: {config.encoder.model} '
                    f'(embed size: {config.encoder.output_dim}, buffer_ratio: {config.buffer_ratio})')

        # For coord Channel(x,y)
        config.clip_input_channel = config.clip_input_channel + 2
        config.text_ratio = min([0.25, 0.5, 0.75, 1.0], key=lambda x: abs(x - config.text_ratio))

        # encoder.ckpt 가 지정되지 않은 경우(dataset 기반 IPCGRL 등) 체크포인트 탐색 스킵
        if config.encoder.ckpt is None and hasattr(config, 'dataset_game') and config.dataset_game is not None:
            logger.info("[IPCGRL] encoder.ckpt not specified — MLP encoder will be trained from scratch")
        else:
            try:
                ckpt_dir = abspath(config.encoder.ckpt_dir)

                exp_dirs = glob(join(ckpt_dir, '*'))

                conditions = {
                    'embed_type': f'enc-{config.encoder.model}',
                    'embed_size': f'es-{config.encoder.output_dim}',
                    'buffer_ratio': f'br-{config.buffer_ratio}',
                }

                if config.encoder.model in ['cnnclip', 'clip']:
                    text_ratio_str = 't' if config.text_ratio == 1.0 else f"t.{str(config.text_ratio).split('.')[1]}"
                    modality = [text_ratio_str]
                    if config.encoder.state:
                        state_ratio_str = 's' if config.state_ratio == 1.0 else f"s.{str(config.state_ratio).split('.')[1]}"
                        modality.append(state_ratio_str)
                    if config.encoder.sketch:
                        sketch_ratio_str = 'k' if config.sketch_ratio == 1.0 else f"k.{str(config.sketch_ratio).split('.')[1]}"
                        modality.append(sketch_ratio_str)

                    modality = ''.join(modality)
                    conditions['md'] = modality

                exp_dirs = [
                    d for d in exp_dirs
                    if all(keyword in d for keyword in conditions.values())
                ]

                if len(exp_dirs) == 0:
                    raise FileNotFoundError(f"Could not find encoder checkpoint for the condition: {conditions}")
                elif len(exp_dirs) > 1:
                    raise FileExistsError(f"Multiple encoder checkpoints found for the condition: {conditions}")

                config.encoder.ckpt_path = join(exp_dirs[0], 'ckpts')

                logger.info(f"Encoder checkpoint set to [{config.encoder.ckpt_path}]")
            except Exception as e:
                logger.error(f"Error loading encoder checkpoint: {e}")
                exit(-1)

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

    config.exp_group = get_exp_group(config)
    config.exp_dir = get_exp_dir(config)

    config._vid_dir = os.path.join(config.exp_dir, 'videos')
    config._img_dir = os.path.join(config.exp_dir, 'images')
    config._numpy_dir = os.path.join(config.exp_dir, 'numpy')
    config._traj_dir = os.path.join(config.exp_dir, 'traj')

    if config.model == 'seqnca':
        config.hidden_dims = config.hidden_dims[:1]

    return config


def init_config_evo_map(config: EvoMapConfig):
    config.arf_size = (2 * config.map_width -
                       1 if config.arf_size == -1 else config.arf_size)

    config.vrf_size = (2 * config.map_width -
                       1 if config.vrf_size == -1 else config.vrf_size)

    config.n_gpus = jax.local_device_count()
    config.exp_dir = get_exp_dir_evo_map(config)
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

    if config.vec_cont is True and config.model != 'contconv':
        logger.warning("Setting model to `contconv` due to the vec_cont flag")
        config.model = 'contconv'

    if config.encoder.model is not None:
        # dataset 기반 모드에서는 model이 이미 설정되어 있으므로 스킵
        _is_dataset_mode = hasattr(config, 'dataset_game') and config.dataset_game is not None
        if not _is_dataset_mode:
            if config.encoder.model == 'clip':
                config.model = 'clipconv'
                logger.info(f"Setting model to `clipconv` due to the `clip.encoder.model={config.encoder.model}`")
            elif config.encoder.model == 'cnnclip':
                config.model = 'cnnclipconv'
                logger.info(f"Setting model to `cnnclipconv` due to the `clip.encoder.model={config.encoder.model}`")

    if config.model == "dense":
        network = Dense(
            action_dim, activation=config.activation,
            arf_size=config.arf_size, vrf_size=config.vrf_size,
        )

    elif config.model == "nlpconv" or config.model == 'contconv':

        # dataset 기반 VIPCGRL: encoder 불필요 (CLIP 임베딩이 사전 계산됨)
        _skip_encoder = (
                hasattr(config, 'dataset_game') and config.dataset_game is not None
                and config.encoder.model in ('cnnclip', 'clip')
        )
        network = EncoderNLPConvForward(
            config=config.encoder,
            encoder=None if _skip_encoder else (apply_encoder_model(config.encoder) if config.encoder.model else None),
            train_encoder=config.encoder.trainable,
            nlp_conv_forward=NLPConvForward(
                action_dim=action_dim, activation=config.activation,
                arf_size=config.arf_size, act_shape=config.act_shape,
                vrf_size=config.vrf_size,
                nlp_input_dim=config.nlp_input_dim,
                hidden_dims=config.hidden_dims
            )
        )

    elif config.model == "clipconv":
        network = EncoderCLIPConvForward(
            config=config.encoder,
            encoder=get_clip_encoder(config.encoder) if config.encoder.model else None,
            train_encoder=config.encoder.trainable,
            nlp_conv_forward=NLPConvForward(
                action_dim=action_dim, activation=config.activation,
                arf_size=config.arf_size, act_shape=config.act_shape,
                vrf_size=config.vrf_size,
                nlp_input_dim=config.nlp_input_dim,
                hidden_dims=config.hidden_dims
            ),
            action_dim=action_dim,
            act_shape=config.act_shape,
        )
    elif config.model == "cnnclipconv":
        network = EncoderCLIPConvForward(
            config=config.encoder,
            encoder=get_cnnclip_encoder(config.encoder)[0] if config.encoder.model else None,
            train_encoder=config.encoder.trainable,
            nlp_conv_forward=NLPConvForward(
                action_dim=action_dim, activation=config.activation,
                arf_size=config.arf_size, act_shape=config.act_shape,
                vrf_size=config.vrf_size,
                nlp_input_dim=config.nlp_input_dim,
                hidden_dims=config.hidden_dims
            ),
            action_dim=action_dim,
            act_shape=config.act_shape,
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

    if 'PCGRL' in config.env_name:
        network = ActorCriticPCGRL(network, act_shape=config.act_shape,
                                   n_agents=config.n_agents, n_ctrl_metrics=len(config.ctrl_metrics),
                                   nlp_input_dim=env_params.nlp_input_dim, model_type=config.model)
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

    # dataset 기반 VIPCGRL 은 nlp_input_dim 으로 CLIP embedding 차원을 전달
    _use_nlp_dim = config.use_nlp or (
            config.use_clip and hasattr(config, 'dataset_game') and config.dataset_game is not None
    )

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
        nlp_input_dim=config.nlp_input_dim if _use_nlp_dim else -1,
        vec_input_dim=config.vec_input_dim if config.vec_cont else -1,
        clip_input_channel=config.clip_input_channel if (config.use_clip and not _use_nlp_dim) else -1,
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


def get_sweep_conf_path(cfg: SweepConfig):
    conf_sweeps_dir = os.path.join('conf', 'sweeps')
    # sweep_conf_path_json = os.path.join(conf_sweeps_dir, f'{cfg.name}.json')
    sweep_conf_path_yaml = os.path.join(conf_sweeps_dir, f'{cfg.name}.yaml')
    return sweep_conf_path_yaml


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


def make_directory_recursive(path):
    if path is None:
        return

    if not os.path.exists(path):
        make_directory_recursive(os.path.dirname(path))
    else:
        return
    os.makedirs(path, exist_ok=True)
