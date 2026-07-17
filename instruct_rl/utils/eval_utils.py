"""
eval_utils.py
=============
each eval_*.py entry point in  text  common evaluation entry point.

Example:
    from instruct_rl.utils.eval_utils import main_eval_entry
    main_eval_entry(config, inject_obs_fn=inject_cpcgrl_obs)
"""
import os
import logging
import time
from datetime import datetime
import numpy as np
import pandas as pd
import jax
import wandb

from instruct_rl.utils.env_loader import get_wandb_key
from instruct_rl.utils.path_utils import init_config
from instruct_rl.utils.logger import get_wandb_name_eval
from instruct_rl.utils.dataset_loader import load_dataset_instruct
from envs.probs.multigame import render_multigame_maps_batch

logger = logging.getLogger(__name__)


def main_chunk(config, rng, *, inject_obs_fn=None):
    """checkpoint load  after  make_eval Usage."""
    from instruct_rl.utils.checkpointer import init_checkpointer
    from instruct_rl.eval.runner import make_eval

    if not config.random_agent:
        _, restored_ckpt, encoder_param = init_checkpointer(config)

        # ── checkpoint load text ──────────────────────────────────────────────
        if restored_ckpt is None:
            if getattr(config, 'ignore_checkpoint', False):
                logger.warning(
                    "⚠️  No checkpoint found at '%s'. "
                    "Proceeding with randomly-initialized weights (ignore_checkpoint=True).",
                    config.exp_dir,
                )
            else:
                raise FileNotFoundError(
                    f"No checkpoint found at '{config.exp_dir}'. "
                    "Ensure the model has been trained before running evaluation. "
                    "To skip this check and use random weights, set ignore_checkpoint=True."
                )
        else:
            ckpt_step = restored_ckpt.get("steps_prev_complete", "?")
            logger.info("✅  Checkpoint loaded — step=%s  (path: %s)", ckpt_step, config.exp_dir)
    else:
        restored_ckpt, encoder_param = None, None

    # train and  sametext MultiGameDataset based eval instruct load
    eval_inst = None
    eval_inst_meta = None
    gt_levels = None
    gt_images = None
    if hasattr(config, 'dataset_game') and config.dataset_game is not None:

        _, eval_inst, samples = load_dataset_instruct(config)  # test split text for
        logger.info(f"Loaded eval instruct from dataset: {eval_inst.reward_i.shape[0]} samples")

        # sample metadata DataFrame (game, instruction, reward_enum)
        # instruction_prefix(name/desc/none)  'none'   text text, CLIP embedding compute and
        # sametext seed(42) to  prefix  applytext results_tb  of  instruction text
        # text to  CLIP  in  text text  applytext also text text.
        from instruct_rl.utils.dataset_loader_helpers.preprocessing import build_effective_instructions
        _effective_instructions = build_effective_instructions(
            samples, instruction_prefix=getattr(config, 'instruction_prefix', 'none')
        )
        eval_inst_meta = pd.DataFrame({
            'game':        [s.game for s in samples],
            'instruction': _effective_instructions,
            'reward_enum': [s.meta.get('reward_enum', None) for s in samples],
        })

        # GT level: samples in  direct extract  after  n_epstext repetition
        # → pred_levels (N*n_eps, H, W)  and  batch size  text
        _n_eps = getattr(config, 'n_eps', 1)
        _gt_raw = np.stack([s.array.astype(np.int32) for s in samples])  # (M, H, W)
        gt_levels = np.repeat(_gt_raw, _n_eps, axis=0)                   # (M*n_eps, H, W)
        logger.info(f"GT levels: {_gt_raw.shape} × n_eps={_n_eps} → {gt_levels.shape}")

        # GT rendering image: text text tile batch rendering (render_multigame_maps_batch)
        _tile_size = getattr(config, 'vit_tile_size', 16)
        logger.info(f"Rendering GT images (tile_size={_tile_size}) ...")
        _gt_images_raw = render_multigame_maps_batch(
            np.stack([s.array.astype(np.int32) for s in samples]),  # (M, H, W)
            tile_size=_tile_size,
        )  # (M, H*ts, W*ts, 3)
        gt_images = np.repeat(_gt_images_raw, _n_eps, axis=0)  # (M*n_eps, H*ts, W*ts, 3)
        logger.info(f"GT images: {_gt_images_raw.shape} × n_eps={_n_eps} → {gt_images.shape}")

        # ── dry-run: max_samples  to  text inside text ─────────────────────────────
        max_samples = getattr(config, 'max_samples', None)
        if max_samples is not None and eval_inst.reward_i.shape[0] > max_samples:
            import jax.numpy as jnp
            logger.info(
                f"[dry-run] max_samples={max_samples}: "
                f"eval_inst {eval_inst.reward_i.shape[0]} → {max_samples}"
            )
            eval_inst = eval_inst.replace(
                reward_i=eval_inst.reward_i[:max_samples],
                condition=eval_inst.condition[:max_samples],
                embedding=eval_inst.embedding[:max_samples],
                condition_id=eval_inst.condition_id[:max_samples],
            )
            eval_inst_meta = eval_inst_meta.iloc[:max_samples].reset_index(drop=True)
            gt_levels = gt_levels[:max_samples * _n_eps]
            gt_images = gt_images[:max_samples * _n_eps]

    eval_fn = make_eval(
        config, restored_ckpt, encoder_param,
        inject_obs_fn=inject_obs_fn,
        eval_inst=eval_inst,
        eval_inst_meta=eval_inst_meta,
        gt_levels=gt_levels,
        gt_images=gt_images,
    )
    out = eval_fn(rng)
    jax.block_until_ready(out)
    return out


def main_eval_entry(config, *, inject_obs_fn=None):
    """Hydra @main  in  calltext  common evaluation entry point.

    Args:
        config: Hydra config (EvalConfig text  text sub class).
        inject_obs_fn: obs inject callback. None  text config based inject  to text text for .
    """
    _eval_start = time.perf_counter()

    config = init_config(config)

    if config.n_eps < 2 and config.diversity:
        raise Exception("Diversity evaluation requires n_eps > 1")

    rng = jax.random.PRNGKey(config.seed)

    exp_dir = config.exp_dir
    logger.info(f"Running experiment at {exp_dir}")

    # ── train_setting.json: load from train exp_dir → inject into config (WandB) ──
    _rdm_path = os.path.join(exp_dir, "train_setting.json")
    if os.path.exists(_rdm_path):
        try:
            import json as _json
            from conf.game_utils import compute_seen_unseen_split

            with open(_rdm_path) as _f:
                _rdm_data = _json.load(_f)
            _rdm = _rdm_data.get("reward_decoder_mode")
            if _rdm and hasattr(config, "reward_decoder_mode"):
                config.reward_decoder_mode = _rdm

            # Canonicalize seen/unseen (doom2 → doom, 5-game total) regardless
            # of what is stored in train_setting.json (old runs may have empty
            # or doom2-containing lists).
            _seen_raw = _rdm_data.get("seen_games", [])
            _seen, _ = compute_seen_unseen_split(_seen_raw)
            if hasattr(config, "seen_games"):
                config.seen_games = list(_seen)
            _unseen_raw = _rdm_data.get("unseen_games", [])
            if hasattr(config, "unseen_games"):
                config.unseen_games = sorted(set(list(config.unseen_games) + _unseen_raw))

            _enc_name = _rdm_data.get("encoder_ckpt_name")
            if _enc_name and hasattr(config, "encoder"):
                config.encoder.ckpt_name = _enc_name

            logger.info(
                "Loaded train_setting: mode=%s, encoder=%s, seen=%s, unseen=%s",
                _rdm, _enc_name, _seen, config.unseen_games,
            )
        except Exception as _e:
            logger.warning("Failed to load train_setting.json: %s", _e)

    # eval_reward_decoder_mode to  text condition text  text.
    # reward_decoder_mode  exp_dir path text for  as  as-is keeptext.
    _eval_rdm = getattr(config, 'eval_reward_decoder_mode', None)
    if _eval_rdm is not None:
        config.reward_decoder_mode = _eval_rdm

    _re = getattr(config, 'dataset_reward_enum', None)
    _re_enums = getattr(config, 'eval_dataset_reward_enums', None)

    # eval_dataset_reward_enums   text text → text text  as-is suffix to  text for
    # e.g. "01234" → "_re-01234", [0,1,2] → "_re-012"
    if _re_enums is not None:
        _re_enums_str = ''.join(str(x) for x in _re_enums) if not isinstance(_re_enums, str) else _re_enums
        _re_suffix = f"_re-{_re_enums_str}"
    elif _re is not None:
        _re_suffix = f"_re-{_re}"
    else:
        _re_suffix = ""

    # eval_games   text text abbreviation  foldertext in  text (if missing game text for )
    _eval_games = getattr(config, 'eval_games', None) or getattr(config, 'game', None)
    _game_suffix = f"_game-{_eval_games}" if _eval_games else ""

    eval_dir = os.path.join(
        exp_dir,
        f"ev{_re_suffix}{_game_suffix}",
    )
    config.eval_dir = eval_dir

    if config.reevaluate:
        if os.path.exists(eval_dir):
            logger.info(f"Removing existing evaluation directory at {eval_dir}")
            os.system(f"rm -r {eval_dir}")
        else:
            logger.info(f"No existing evaluation directory found at {eval_dir}")
    else:
        if os.path.exists(eval_dir):
            raise Exception(
                f"Evaluation directory already exists at {eval_dir}. "
                "Set reevaluate=True to overwrite."
            )

    os.makedirs(eval_dir, exist_ok=True)
    logger.info(f"Running evaluation at {eval_dir}")


    wandb_key = get_wandb_key()

    if wandb_key:
        dt = datetime.now().strftime("%Y%m%d%H%M%S")
        wandb_id = f"{get_wandb_name_eval(config)}-{dt}"
        wandb.login(key=wandb_key)
        wandb.init(
            project=config.wandb_project,
            entity=config.wandb_entity,
            name=get_wandb_name_eval(config),
            id=wandb_id,
            save_code=True,
            config=wandb.helper.parse_config(
                dict(config),
                exclude=("wandb_key", "_vid_dir", "_img_dir", "_numpy_dir", "overwrite", "initialize"),
            ),
        )
        wandb.config.update(dict(config), allow_val_change=True)

    main_chunk(config, rng, inject_obs_fn=inject_obs_fn)

    _elapsed = time.perf_counter() - _eval_start
    _h, _rem = divmod(int(_elapsed), 3600)
    _m, _s   = divmod(_rem, 60)
    _time_str = f"{_h:02d}h {_m:02d}m {_s:02d}s  ({_elapsed:.1f}s total)"
    logger.info("=" * 60)
    logger.info(f"  ✅  Evaluation finished  —  elapsed: {_time_str}")
    logger.info("=" * 60)

    if wandb.run:
        wandb.finish()
