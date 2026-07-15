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
from conf.game_utils import parse_game_str, GAME_ABBR, infer_seen_games_from_ckpt_name
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
    """dataset_game 문자열(전체명 또는 약어) → 2글자 약어로 변환."""
    from conf.game_utils import GAME_ABBR_INV
    full = GAME_ABBR[dataset_game][0] if dataset_game in GAME_ABBR else dataset_game
    return GAME_ABBR_INV.get(full, full)


def _enc_str(encoder_config) -> str:
    """encoder ckpt 이름 기반 6자리 해시 (VIPCGRL 등 비-MGPCGRL 모델 전용)."""
    ckpt = getattr(encoder_config, 'ckpt_name', None) or getattr(encoder_config, 'ckpt_path', None) or ""
    h = hashlib.md5(ckpt.encode()).hexdigest()[:6] if ckpt else "scratch"
    return f'_enc-{h}'


def _to_pstr(v: float) -> str:
    """0.05 → '0p05', 1.0 → '1' 형태로 변환."""
    return f"{v:g}".replace('.', 'p')


def _build_unseen_suffix(un_abbr, ur, sr, encgame=None) -> str:
    """un_abbr/ur/sr/encgame → '_un-XX_ur-YY_sr-ZZ_encgame-WW' 형태 suffix.

    모두 None/empty 이면 빈 문자열을 반환한다 (unseen 정보가 없으면 생략).
    sr == 1.0 이면 실험명에 포함하지 않는다.
    encgame 은 encoder 의 game 이 'all' 이 아닐 때만 붙는다
    (``_parse_encgame_from_ckpt`` 참고).
    """
    parts = []
    if un_abbr:
        parts.append(f'un-{un_abbr}')
    if ur is not None:
        parts.append(f'ur-{_to_pstr(ur)}')
    if sr is not None and sr != 1.0:
        parts.append(f'sr-{_to_pstr(sr)}')
    if encgame:
        parts.append(f'encgame-{encgame}')
    return ('_' + '_'.join(parts)) if parts else ''


def _parse_unseen_from_ckpt(ckpt_name: str):
    """encoder ckpt 이름에서 (un_abbr, ur, sr) 추출. 없으면 None."""
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


def _parse_encgame_from_ckpt(ckpt_name: str):
    """encoder ckpt 이름의 ``game-<games>`` 값을 추출. 'all' 이거나 없으면 None.

    RL 은 보통 encoder 와 무관하게 game='all' 로 돌기 때문에, encoder 쪽 game 을
    제한해 실험을 나누는 경우(source_target: game=source+target) RL exp_dir 이
    encoder 간에 충돌한다 — (un, ur, sr) 이 모두 같기 때문이다. 이를 구분하려면
    encoder 의 game 자체가 경로에 들어가야 한다.

    encoder 가 game-all 인 (기존) 실험은 None 을 반환해 경로를 그대로 유지한다.
    """
    if not ckpt_name:
        return None
    import re
    m_game = re.search(r'(?:^|[_-])game-([^_]+)', ckpt_name)
    if not m_game:
        return None

    game = m_game.group(1)
    return None if game == 'all' else game


def _unseen_abbr_from_seen_games(seen_games):
    """seen game list에서 canonical unseen game 약어 문자열을 만든다."""
    if not seen_games:
        return None

    from conf.game_utils import GAME_ABBR_INV, GAME_ABBR
    seen_game_set = {("doom" if g == "doom2" else g) for g in seen_games}
    all_games = [
        g for games in GAME_ABBR.values() for g in games
        if g not in ("doom2",)
    ]
    unseen = [g for g in all_games if g not in seen_game_set and g != "doom2"]
    if not unseen:
        return None

    abbr_parts, seen_abbrs = [], set()
    for g in unseen:
        abbr = GAME_ABBR_INV.get(g, g[:2])
        if abbr not in seen_abbrs:
            abbr_parts.append(abbr)
            seen_abbrs.add(abbr)
    return ''.join(abbr_parts)



def _unseen_suffix(config) -> str:
    """공통 unseen suffix 빌더 (VIPCGRL / MGPCGRL 공용).

    우선순위:
      1. config.train_unseen_abbr / config.train_unseen_ratio / config.train_seen_ratio (명시 파라미터, MGPCGRL only)
      2. config.encoder.ckpt_name 에서 파싱 (VIPCGRL / MGPCGRL 공통)
      3. config.reward_seen_games/seen_games 에서 자동 계산 (un_abbr only)

    unseen 정보가 전혀 없으면 빈 문자열을 반환한다 (suffix 생략).
    형식: '_un-XX_ur-YY_sr-ZZ[_encgame-WW]'
    """
    # ── 1. 명시 파라미터 (MGPCGRL config 에만 존재) ──────────────────────────
    un_abbr = getattr(config, 'train_unseen_abbr', None)   # e.g. "zd"
    ur      = getattr(config, 'train_unseen_ratio', None)  # e.g. 0.05
    # train_seen_ratio는 RL 학습 파라미터이므로 sr 경로명에는 사용하지 않음
    # encoder ckpt_name에서 파싱한 sr을 우선 사용
    sr      = None

    enc_cfg = getattr(config, 'encoder', None)
    ckpt_name = (getattr(enc_cfg, 'ckpt_name', None)
                 or getattr(enc_cfg, 'ckpt_path', None) or "")

    # ── encoder game: 'all' 이 아닌 경우만 (RL game 과 독립적인 경로 식별자) ──
    encgame = _parse_encgame_from_ckpt(ckpt_name)

    # ── 2. encoder.ckpt_name 에서 파싱 ──────────────────────────────────────
    if un_abbr is None or ur is None or sr is None:
        c_un, c_ur, c_sr = _parse_unseen_from_ckpt(ckpt_name)
        if un_abbr is None: un_abbr = c_un
        if ur is None:      ur      = c_ur
        if sr is None:      sr      = c_sr

        # Older full-shot subset encoder names do not include ``_unseen-XX``.
        # Example: ``clip-game-dgpk_exp-def_0`` means seen={dg,pk}, so the
        # downstream VIPCGRL/IPCGRL/MGPCGRL run still needs an unseen suffix to
        # avoid folder collisions across encoder choices.
        if un_abbr is None:
            un_abbr = _unseen_abbr_from_seen_games(
                infer_seen_games_from_ckpt_name(ckpt_name)
            )

    # ── train_seen_ratio fallback (encoder sr 없을 때만, 1.0 제외) ───────────
    if sr is None:
        train_sr = getattr(config, 'train_seen_ratio', None)
        if train_sr is not None and train_sr != 1.0:
            sr = train_sr

    # ── 3. seen-game metadata 에서 자동 계산 (un_abbr only) ───────────────
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

    # unseen 정보가 전혀 없으면 suffix 생략
    return _build_unseen_suffix(un_abbr, ur, sr, encgame)


def get_exp_group(config) -> str:
    """실험 그룹명 반환 (시드 미포함).

    WandB group 및 exp_dir 경로 prefix로 사용된다.
    """
    exp_name = getattr(config, 'exp_name', None) or 'def'

    # ── Random policy ──────────────────────────────────────────────────────────
    if getattr(config, 'random_agent', False):
        return f'random_exp-{exp_name}'

    # ── MultiGameDataset 기반 모드 (CPCGRL / IPCGRL / VIPCGRL / MGPCGRL) ──────
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
    # encoder ckpt 이름에서 unseen 정보 파싱해 suffix 추가 (VIPCGRL / MGPCGRL 와 동일 규칙).
    # unseen 정보가 없으면 suffix 생략.
    # MIPCGRL 은 동일한 use_nlp=True 분기를 타지만 ``is_mipcgrl`` 플래그로 prefix 를
    # 구분해 IPCGRL 체크포인트와 디스크/wandb 충돌을 방지한다.
    if getattr(config, 'use_nlp', False):
        enc = _unseen_suffix(config)
        kind = 'mipcgrl' if getattr(config, 'is_mipcgrl', False) else 'ipcgrl'
        return f'{kind}_game-{game}{re_s}{exp_s}{enc}'

    # PretrainedCLIP: model=pretrained_clip, enc suffix 없음
    if getattr(config, 'model', None) == 'pretrained_clip':
        return f'preclip_pcgrl_game-{game}{re_s}{exp_s}'

    # FinetunedCLIP: model=finetuned_clip, encoder ckpt 해시 suffix 포함
    # (동일 게임/실험명이라도 어떤 fine-tuned ckpt 를 inject 했는지에 따라
    # exp_dir 가 분리되도록 — mgpcgrl 와 동일 시맨틱)
    if getattr(config, 'model', None) == 'finetuned_clip':
        enc = _enc_str(config.encoder)
        return f'finclip_pcgrl_game-{game}{re_s}{exp_s}{enc}'

    # MGPCGRL: explicit param-based path (un-XX / ur-XX / sr-XX)
    if hasattr(config, 'decoder'):
        rdm = getattr(config, 'reward_decoder_mode', 'unseen')
        enc = _unseen_suffix(config)

        # encoder delta_weight suffix (if non-zero)
        encoder_delta_w = getattr(config, 'encoder_delta_weight', 0.0)
        delta_s = f'_dw-{_to_pstr(encoder_delta_w)}' if encoder_delta_w != 0.0 else ''

        # dataset_unseen_ratio suffix (only when not default 1.0)
        dur = getattr(config, 'dataset_unseen_ratio', 1.0)
        uro_s = f'_uro-{_to_pstr(dur)}' if dur != 1.0 else ''

        return f'mgpcgrl_game-{game}{re_s}_rdm-{rdm}{enc}{delta_s}{uro_s}{exp_s}'

    # VIPCGRL: encoder ckpt 이름에서 unseen 정보 파싱해 suffix 추가 (MGPCGRL 와 동일 규칙).
    # unseen 정보가 없으면 suffix 생략.
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

    # ── random_expname: exp_name 을 UUID 기반으로 랜덤 결정 ──
    if getattr(config, 'random_exp_name', False):
        config.exp_name = uuid.uuid4().hex[:8]
        logger.info(f"[random_exp_name] exp_name randomly set to: {config.exp_name}")

    # ── game 약어 → include_* 자동 동기화 ──
    if hasattr(config, 'game') and config.game:
        includes = parse_game_str(config.game)
        for key, val in includes.items():
            setattr(config, key, val)

        # ── dataset_game 동기화: game 파라미터가 명시적으로 전달된 경우 dataset_game을 override ──
        # dataset_game이 None이거나 기본값("all")인 경우 game 값으로 덮어씌운다.
        if hasattr(config, 'dataset_game'):
            _dg_val = getattr(config, 'dataset_game', None)
            if _dg_val is None or _dg_val == 'all':
                # 약어(dg 등)를 정식 게임명으로 변환
                _game_key = config.game.lower()
                if _game_key in GAME_ABBR:
                    config.dataset_game = GAME_ABBR[_game_key][0]
                else:
                    config.dataset_game = config.game
                logger.debug(f"[init_config] dataset_game overridden by game='{config.game}' → '{config.dataset_game}'")

    # ── MultiGameDataset 기반 CPCGRL / IPCGRL / VIPCGRL 모드 ─────────────
    if hasattr(config, 'dataset_game') and config.dataset_game is not None:
        config.raw_obs = True
        # instruct_csv는 사용하지 않음
        config.instruct_csv = None

        if config.use_clip:
            # ── VIPCGRL / PretrainedCLIP 모드: CLIP latent embedding을 입력으로 사용 ──
            config.vec_cont = False
            config.use_nlp = False
            if config.encoder.model is None:
                config.encoder.model = 'cnnclip'
            if config.nlp_input_dim <= 0:
                config.nlp_input_dim = config.encoder.output_dim  # encoder output dim (e.g. 64)
            config.vec_input_dim = config.nlp_input_dim
            # dataset 기반 VIPCGRL: cnnclipconv/clipconv 가 이미 설정된 경우 유지
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
            config.vec_input_dim = 5
            config.nlp_input_dim = 0
            logger.info(f"[CPCGRL] dataset_game={config.dataset_game}, "
                        f"dataset_reward_enum={getattr(config, 'dataset_reward_enum', None)}")

            if config.vec_cont is True and config.model != 'contconv':
                config.model = 'contconv'
                logger.info("[CPCGRL] Setting model to `contconv` due to the vec_cont flag")

        # exp_dir 등 공통 설정은 아래에서 계속 처리

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

        # ── encoder.ckpt_path 가 이미 지정된 경우 스킵 ──
        if config.encoder.ckpt_path is not None:
            logger.info(f"Encoder checkpoint path already set: [{config.encoder.ckpt_path}]")

        # ── encoder.ckpt_name 으로 pretrained_encoders/ 에서 직접 로드 ──
        elif config.encoder.ckpt_name is not None:
            _project_root = os.path.dirname(os.path.dirname(os.path.dirname(abspath(__file__))))
            _pretrained_dir = join(_project_root, config.encoder.ckpt_dir, config.encoder.ckpt_name, "ckpts")
            if not os.path.isdir(_pretrained_dir):
                logger.error(f"Pretrained encoder checkpoint not found: {_pretrained_dir}")
                exit(-1)
            config.encoder.ckpt_path = _pretrained_dir
            logger.info(f"Encoder checkpoint set from ckpt_name='{config.encoder.ckpt_name}' → [{config.encoder.ckpt_path}]")

        # encoder.ckpt 가 지정되지 않은 경우(dataset 기반 IPCGRL 등) 체크포인트 탐색 스킵
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
        # Fine-tuned CLIP: ckpt 파라미터 트리(TrainablePretrained*Encoder) 와
        # 동일한 모듈을 RL 측에도 사용해야 `apply_encoder_params` 의 subtree
        # replace 가 안전하다.
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

    # dataset 기반 VIPCGRL 은 nlp_input_dim 으로 CLIP embedding 차원을 전달
    _use_nlp_dim = config.use_nlp or (
            config.use_clip and hasattr(config, 'dataset_game') and config.dataset_game is not None
    )

    # cnnclipconv/clipconv 모델은 nlp_input_dim과 clip_input_channel 둘 다 필요
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
