import hashlib
import os
import uuid
import gymnax
import jax
from glob import glob
import yaml
from os.path import abspath, join

from encoder.model import apply_encoder_model
from encoder.clip_model import get_clip_encoder, get_cnnclip_encoder, get_cnnclip_decoder_encoder
from conf.config import Config, TrainConfig, EncoderConfig
from conf.game_utils import (
    parse_game_str,
    GAME_ABBR,
    infer_seen_games_from_ckpt_name,
    unseen_abbr_from_seen_games,
)
from encoder.pretrained_clip_model import get_pretrained_clip_encoder
from encoder.finetuned_clip_model import get_finetuned_clip_encoder
from envs.candy import Candy, CandyParams
from envs.pcgrl_env import PROB_CLASSES, PCGRLEnvParams, PCGRLEnv, ProbEnum, RepEnum
from envs.play_pcgrl_env import PlayPCGRLEnv, PlayPCGRLEnvParams
from models import ActorCritic, ActorCriticPCGRL, AutoEncoder, ConvForward, ConvForward2, Dense, \
    NCA, SeqNCA, NLPConvForward, EncoderNLPConvForward, EncoderCLIPConvForward

from instruct_rl.utils.log_utils import get_logger

logger = get_logger(__file__)


def is_default_hiddims(config: Config):
    return tuple(config.hidden_dims) == (64, 256)[:len(config.hidden_dims)]


def _game_abbr(dataset_game: str) -> str:
    """Convert a dataset_game name or abbreviation to a two-letter abbreviation."""
    from conf.game_utils import GAME_ABBR_INV
    full = GAME_ABBR[dataset_game][0] if dataset_game in GAME_ABBR else dataset_game
    return GAME_ABBR_INV.get(full, full)


def _enc_str(encoder_config) -> str:
    """Return a six-character hash of an encoder checkpoint name for non-ReWARD models."""
    ckpt = getattr(encoder_config, 'ckpt_name', None) or getattr(encoder_config, 'ckpt_path', None) or ""
    h = hashlib.md5(ckpt.encode()).hexdigest()[:6] if ckpt else "scratch"
    return f'_enc-{h}'


def _to_pstr(v: float) -> str:
    """0.05 → '0p05', 1.0 → '1' form to  convert."""
    return f"{v:g}".replace('.', 'p')


def _build_unseen_suffix(un_abbr, ur, sr) -> str:
    """un_abbr/ur/sr → '_un-XX_ur-YY_sr-ZZ' form suffix.

    Return an empty string when every value is None/empty.
    Omit sr from experiment names when sr == 1.0.
    """
    parts = []
    if un_abbr:
        parts.append(f'un-{un_abbr}')
    if ur is not None:
        parts.append(f'ur-{_to_pstr(ur)}')
    if sr is not None and sr != 1.0:
        parts.append(f'sr-{_to_pstr(sr)}')
    return ('_' + '_'.join(parts)) if parts else ''


def _parse_unseen_from_ckpt(ckpt_name: str):
    """encoder ckpt name in  (un_abbr, ur, sr) extract. if missing None."""
    if not ckpt_name:
        return None, None, None
    import re
    m_abbr = re.search(r'_unseen-([^_]+)_', ckpt_name)
    m_ur   = re.search(r'_ur-([\d.]+)_', ckpt_name)
    m_sr   = re.search(r'_sr-([\d.]+)_', ckpt_name)
    return (
        m_abbr.group(1) if m_abbr else None,
        float(m_ur.group(1)) if m_ur else None,
        float(m_sr.group(1)) if m_sr else None,
    )


def _unseen_abbr_from_seen_games(seen_games):
    """Create a canonical unseen-game abbreviation from a seen-game list."""
    return unseen_abbr_from_seen_games(seen_games)



def _unseen_suffix(config) -> str:
    """Build a common unseen suffix for VIPCGRL and ReWARD.

    Precedence:
      1. config.train_unseen_abbr / train_unseen_ratio / train_seen_ratio
         (explicit parameters, ReWARD only)
      2. config.encoder.ckpt_name  in  parsing (VIPCGRL / ReWARD common)
      3. config.reward_seen_games/seen_games  in  automatic compute (un_abbr only)

    Return an empty string and omit the suffix when no unseen information exists.
    Format: '_un-XX_ur-YY_sr-ZZ'
    """
    # ── 1. Explicit parameters available only in ReWARD config ──────────────
    un_abbr = getattr(config, 'train_unseen_abbr', None)   # e.g. "zd"
    ur      = getattr(config, 'train_unseen_ratio', None)  # e.g. 0.05
    # train_seen_ratio is an RL-training parameter, so do not use it in sr paths
    # Prefer sr parsed from the encoder ckpt_name
    sr      = None

    # ── 2. encoder.ckpt_name  in  parsing ──────────────────────────────────────
    if un_abbr is None or ur is None or sr is None:
        enc_cfg = getattr(config, 'encoder', None)
        ckpt_name = (getattr(enc_cfg, 'ckpt_name', None)
                     or getattr(enc_cfg, 'ckpt_path', None) or "")
        c_un, c_ur, c_sr = _parse_unseen_from_ckpt(ckpt_name)
        if un_abbr is None: un_abbr = c_un
        if ur is None:      ur      = c_ur
        if sr is None:      sr      = c_sr

        # Older full-shot subset encoder names do not include ``_unseen-XX``.
        # Example: ``clip-game-dgpk_exp-def_0`` means seen={dg,pk}, so the
        # downstream VIPCGRL/IPCGRL/ReWARD run still needs an unseen suffix to
        # avoid folder collisions across encoder choices.
        if un_abbr is None:
            un_abbr = _unseen_abbr_from_seen_games(
                infer_seen_games_from_ckpt_name(ckpt_name)
            )

    # ── train_seen_ratio fallback only without encoder sr, excluding 1.0 ─────
    if sr is None:
        train_sr = getattr(config, 'train_seen_ratio', None)
        if train_sr is not None and train_sr != 1.0:
            sr = train_sr

    # ── 3. seen-game metadata  in  automatic compute (un_abbr only) ───────────────
    # Train configs populate reward_seen_games from encoder dataset_setting.json.
    # Eval configs populate seen_games from the same source. Treat both as the
    # same path identity signal so eval looks under the trained exp_dir.
    if un_abbr is None:
        seen_games = (
            getattr(config, 'reward_seen_games', None)
            or getattr(config, 'seen_games', None)
            or []
        )
        un_abbr = _unseen_abbr_from_seen_games(seen_games)

    # Omit the suffix when there is no unseen information
    return _build_unseen_suffix(un_abbr, ur, sr)


def get_exp_group(config) -> str:
    """Return the experiment group name without the seed.

    Used as the WandB group and exp_dir path prefix.
    """
    exp_name = getattr(config, 'exp_name', None) or 'def'

    # ── Random policy ──────────────────────────────────────────────────────────
    if getattr(config, 'random_agent', False):
        return f'random_exp-{exp_name}'

    # ── MultiGameDataset based mode (CPCGRL / IPCGRL / VIPCGRL / ReWARD) ──────
    if not (hasattr(config, 'dataset_game') and config.dataset_game is not None):
        return config.env_name  # fallback for non-dataset configs

    game  = _game_abbr(config.dataset_game)
    re    = getattr(config, 'dataset_reward_enum', None)
    re_s  = f'_re-{re}' if re is not None else ''
    exp_s = f'_exp-{exp_name}'

    # CPCGRL: raw condition vector
    if getattr(config, 'vec_cont', False):
        return f'cpcgrl_game-{game}{re_s}{exp_s}'

    # IPCGRL / MIPCGRL: BERT embedding
    # Parse unseen information from the encoder checkpoint and add a suffix,
    # following the same rule as VIPCGRL/ReWARD. Omit it when absent.
    # MIPCGRL shares the use_nlp=True branch, but is_mipcgrl gives it a distinct
    # prefix to prevent disk/WandB collisions with IPCGRL checkpoints.
    if getattr(config, 'use_nlp', False):
        enc = _unseen_suffix(config)
        kind = 'mipcgrl' if getattr(config, 'is_mipcgrl', False) else 'ipcgrl'
        return f'{kind}_game-{game}{re_s}{exp_s}{enc}'

    # PretrainedCLIP: model=pretrained_clip, enc suffix none
    if getattr(config, 'model', None) == 'pretrained_clip':
        return f'preclip_pcgrl_game-{game}{re_s}{exp_s}'

    # FinetunedCLIP: model=finetuned_clip with an encoder-checkpoint hash suffix.
    # Separate exp_dir by injected fine-tuned checkpoint even for identical
    # games/experiment names, matching ReWARD semantics.
    if getattr(config, 'model', None) == 'finetuned_clip':
        enc = _enc_str(config.encoder)
        return f'finclip_pcgrl_game-{game}{re_s}{exp_s}{enc}'

    # ReWARD: explicit param-based path (un-XX / ur-XX / sr-XX)
    if hasattr(config, 'decoder'):
        rdm = getattr(config, 'reward_decoder_mode', 'unseen')
        enc = _unseen_suffix(config)

        # encoder delta_weight suffix (if non-zero)
        encoder_delta_w = getattr(config, 'encoder_delta_weight', 0.0)
        delta_s = f'_dw-{_to_pstr(encoder_delta_w)}' if encoder_delta_w != 0.0 else ''

        # dataset_unseen_ratio suffix (only when not default 1.0)
        dur = getattr(config, 'dataset_unseen_ratio', 1.0)
        uro_s = f'_uro-{_to_pstr(dur)}' if dur != 1.0 else ''

        return f'reward_game-{game}{re_s}_rdm-{rdm}{enc}{delta_s}{uro_s}{exp_s}'

    # VIPCGRL: parse unseen information from the encoder checkpoint and append
    # the same suffix as ReWARD; omit it when absent.
    enc = _unseen_suffix(config)
    if getattr(config, 'use_clip', False):
        return f'vipcgrl_game-{game}{re_s}{exp_s}{enc}'

    raise ValueError(f"[get_exp_group] Unknown model type for config: {config}")



def get_short_target(target: str) -> str:
    # Split the target string into words
    words = target.split()

    # If there's only one word, return it with the length
    if len(words) == 1:
        return f"{words[0]}_{len(target)}"

    # Otherwise, take the first and last words and include the length
    return f"{words[0]}X{words[-1]}{len(target)}"


def encoder_hash(config):
    _ckpt_name = config.encoder.ckpt_name or config.encoder.ckpt_path or ""
    enc_hash = hashlib.md5(_ckpt_name.encode()).hexdigest()[:6] if _ckpt_name else "scratch"
    config.exp_name = f"{config.exp_name}-{enc_hash}"

    return config

def get_exp_name(config):
    exp_group = get_exp_group(config)

    prefix = getattr(config, 'dir_prefix', '')
    return f'{prefix}{exp_group}_s-{config.seed}'


def get_exp_dir(config):
    saves_dir = getattr(config, 'saves_dir', 'saves')
    return os.path.join(saves_dir, get_exp_name(config))


def init_config(config: Config):
    config.n_gpus = jax.local_device_count()

    # ── random_expname: choose exp_name randomly from a UUID ──
    if getattr(config, 'random_exp_name', False):
        config.exp_name = uuid.uuid4().hex[:8]
        logger.info(f"[random_exp_name] exp_name randomly set to: {config.exp_name}")

    # ── game abbreviation → include_* automatic sync ──
    if hasattr(config, 'game') and config.game:
        includes = parse_game_str(config.game)
        for key, val in includes.items():
            setattr(config, key, val)

        # ── Synchronize dataset_game when game is explicitly provided ─────────
        # Replace dataset_game with game when it is None or the default "all".
        if hasattr(config, 'dataset_game'):
            _dg_val = getattr(config, 'dataset_game', None)
            if _dg_val is None or _dg_val == 'all':
                # Convert abbreviations such as dg to full game names
                _game_key = config.game.lower()
                if _game_key in GAME_ABBR:
                    config.dataset_game = GAME_ABBR[_game_key][0]
                else:
                    config.dataset_game = config.game
                logger.debug(f"[init_config] dataset_game overridden by game='{config.game}' → '{config.dataset_game}'")

    # ── MultiGameDataset based CPCGRL / IPCGRL / VIPCGRL mode ─────────────
    if hasattr(config, 'dataset_game') and config.dataset_game is not None:
        config.raw_obs = True
        # instruct_csv is unused
        config.instruct_csv = None

        if config.use_clip:
            # ── VIPCGRL / PretrainedCLIP: use CLIP latent embeddings as input ──
            config.vec_cont = False
            config.use_nlp = False
            if config.encoder.model is None:
                config.encoder.model = 'cnnclip'
            if config.nlp_input_dim <= 0:
                config.nlp_input_dim = config.encoder.output_dim  # encoder output dim (e.g. 64)
            config.vec_input_dim = config.nlp_input_dim
            # Dataset-based VIPCGRL: preserve an existing cnnclipconv/clipconv setting
            if config.model not in ('nlpconv', 'cnnclipconv', 'clipconv', 'pretrained_clip', 'finetuned_clip'):
                config.model = 'nlpconv'
            _mode_tag = (
                "FinetunedCLIP" if config.model == 'finetuned_clip'
                else ("PretrainedCLIP" if config.model == 'pretrained_clip' else "VIPCGRL")
            )
            logger.info(f"[{_mode_tag}] dataset_game={config.dataset_game}, "
                        f"dataset_reward_enum={getattr(config, 'dataset_reward_enum', None)}, "
                        f"nlp_input_dim={config.nlp_input_dim}, "
                        f"encoder={config.encoder.model}")
        elif config.use_nlp:
            # ── IPCGRL: use BERT-to-MLP encoder features as input ──
            config.vec_cont = False
            if config.nlp_input_dim <= 0:
                config.nlp_input_dim = 768  # BERT base dim
            config.vec_input_dim = config.nlp_input_dim
            if config.model not in ('nlpconv',):
                config.model = 'nlpconv'
            # IPCGRL uses the MLP encoder by default
            if config.encoder.model is None:
                config.encoder.model = 'mlp'
            logger.info(f"[IPCGRL] dataset_game={config.dataset_game}, "
                        f"dataset_reward_enum={getattr(config, 'dataset_reward_enum', None)}, "
                        f"nlp_input_dim={config.nlp_input_dim}, "
                        f"encoder={config.encoder.model}")
        else:
            # ── CPCGRL: use raw condition vectors ──
            config.vec_cont = True
            config.use_nlp = False
            config.vec_input_dim = 5
            config.nlp_input_dim = 0
            logger.info(f"[CPCGRL] dataset_game={config.dataset_game}, "
                        f"dataset_reward_enum={getattr(config, 'dataset_reward_enum', None)}")

            if config.vec_cont is True and config.model != 'contconv':
                config.model = 'contconv'
                logger.info("[CPCGRL] Setting model to `contconv` due to the vec_cont flag")

        # Continue with common settings such as exp_dir below

    elif config.aug_type is not None and config.embed_type is not None and config.instruct is not None:
        config.instruct_csv = f'{config.aug_type}/{config.embed_type}/{config.instruct}'

    if config.encoder.model == 'cnnclip':
        config.use_clip = True

    if hasattr(config, 'vec_cont') and config.vec_cont is True:
        config.use_nlp = False
        config.vec_input_dim = 5
        config.nlp_input_dim = 0
    elif hasattr(config, 'use_clip') and config.use_clip is True:
        config.use_nlp = False

        if hasattr(config, 'decoder'):
            config.model = 'cnnclipconv'

        elif config.model == 'conv':
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
        if getattr(config, 'use_pretrained_clip_reward', False):
            config.clip_input_channel = config.clip_input_channel
        else:
            config.clip_input_channel = config.clip_input_channel + 2

        config.text_ratio = min([0.25, 0.5, 0.75, 1.0], key=lambda x: abs(x - config.text_ratio))

        # ── Skip when encoder.ckpt_path is already specified ──
        if config.encoder.ckpt_path is not None:
            logger.info(f"Encoder checkpoint path already set: [{config.encoder.ckpt_path}]")

        # ── encoder.ckpt_name  as  pretrained_encoders/  in  direct load ──
        elif config.encoder.ckpt_name is not None:
            _project_root = os.path.dirname(os.path.dirname(os.path.dirname(abspath(__file__))))
            _pretrained_dir = join(_project_root, config.encoder.ckpt_dir, config.encoder.ckpt_name, "ckpts")
            if not os.path.isdir(_pretrained_dir):
                logger.error(f"Pretrained encoder checkpoint not found: {_pretrained_dir}")
                exit(-1)
            config.encoder.ckpt_path = _pretrained_dir
            logger.info(f"Encoder checkpoint set from ckpt_name='{config.encoder.ckpt_name}' → [{config.encoder.ckpt_path}]")

        # Skip checkpoint discovery when encoder.ckpt is unspecified (e.g. dataset-based IPCGRL)
        elif config.encoder.ckpt is None and hasattr(config, 'dataset_game') and config.dataset_game is not None:
            logger.info("[IPCGRL] encoder.ckpt not specified — MLP encoder will be trained from scratch")
        else:
            try:
                ckpt_dir = abspath(config.encoder.ckpt_dir)

                exp_dirs = glob(join(ckpt_dir, '*'))

                conditions = {
                    'embed_type': f'enc-{config.encoder.model}',
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

def get_ckpt_dir(config: Config):
    return os.path.join(config.exp_dir, 'ckpts')


def init_network(env: PCGRLEnv, env_params: PCGRLEnvParams, config: Config):
    if config.env_name == 'Candy':
        # In the candy-player environment, action space is flat discrete space over all candy-direction combos.
        action_dim = env.action_space(env_params).n

    elif 'PCGRL' in config.env_name:
        action_dim = env.rep.action_space.n
    else:
        action_dim = env.num_actions

    if config.vec_cont is True and config.model != 'contconv':
        logger.warning("Setting model to `contconv` due to the vec_cont flag")
        config.model = 'contconv'

    if config.encoder.model is not None:
        # Dataset-based modes already set model
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

        # Dataset-based VIPCGRL needs no encoder because CLIP embeddings are precomputed
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

    elif config.model == "cnnclipconv" and hasattr(config, 'decoder'):
        network = EncoderCLIPConvForward(
            config=config.encoder,
            encoder=get_cnnclip_decoder_encoder(
                config.encoder,
                decoder_config=config.decoder,
            )[0],
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

    elif config.model == "pretrained_clip":
        network = EncoderCLIPConvForward(
            config=config.encoder,
            encoder=get_pretrained_clip_encoder(config.encoder)[0] if config.encoder.model else None,
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

    elif config.model == "finetuned_clip":
        # Fine-tuned CLIP: the RL side must use the same module as the checkpoint
        # parameter tree (TrainablePretrained*Encoder) for safe subtree replacement
        # by `apply_encoder_params`.
        network = EncoderCLIPConvForward(
            config=config.encoder,
            encoder=get_finetuned_clip_encoder(config.encoder)[0] if config.encoder.model else None,
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

    # Dataset-based VIPCGRL passes the CLIP embedding dimension through nlp_input_dim
    _use_nlp_dim = config.use_nlp or (
            config.use_clip and hasattr(config, 'dataset_game') and config.dataset_game is not None
    )

    # cnnclipconv/clipconv models require both nlp_input_dim and clip_input_channel
    _needs_clip_channel = config.model in ('cnnclipconv', 'clipconv', 'pretrained_clip', 'finetuned_clip')

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
        clip_input_channel=config.clip_input_channel if (config.use_clip and (not _use_nlp_dim or _needs_clip_channel)) else -1,
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


def make_directory_recursive(path):
    if path is None:
        return

    if not os.path.exists(path):
        make_directory_recursive(os.path.dirname(path))
    else:
        return
    os.makedirs(path, exist_ok=True)
