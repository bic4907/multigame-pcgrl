"""
eval_utils.py
=============
각 eval_*.py 엔트리포인트에서 공유하는 공통 평가 엔트리포인트.

사용 예:
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
    """체크포인트 로드 후 make_eval 실행."""
    from instruct_rl.utils.checkpointer import init_checkpointer
    from instruct_rl.eval.runner import make_eval

    if not config.random_agent:
        _, restored_ckpt, encoder_param = init_checkpointer(config)

        # ── 체크포인트 로드 보장 ──────────────────────────────────────────────
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

    # train과 동일하게 MultiGameDataset 기반 eval instruct 로드
    eval_inst = None
    eval_inst_meta = None
    gt_levels = None
    gt_images = None
    if hasattr(config, 'dataset_game') and config.dataset_game is not None:

        _, eval_inst, samples = load_dataset_instruct(config)  # test split 사용
        logger.info(f"Loaded eval instruct from dataset: {eval_inst.reward_i.shape[0]} samples")

        # 샘플 메타데이터 DataFrame (game, instruction, reward_enum)
        eval_inst_meta = pd.DataFrame({
            'game':        [s.game for s in samples],
            'instruction': [getattr(s, 'instruction', None) for s in samples],
            'reward_enum': [s.meta.get('reward_enum', None) for s in samples],
        })

        # GT 레벨: samples에서 직접 추출 후 n_eps배 반복
        # → pred_levels (N*n_eps, H, W) 와 배치 크기를 맞춤
        _n_eps = getattr(config, 'n_eps', 1)
        _gt_raw = np.stack([s.array.astype(np.int32) for s in samples])  # (M, H, W)
        gt_levels = np.repeat(_gt_raw, _n_eps, axis=0)                   # (M*n_eps, H, W)
        logger.info(f"GT levels: {_gt_raw.shape} × n_eps={_n_eps} → {gt_levels.shape}")

        # GT 렌더링 이미지: 스프라이트 타일 배치 렌더링 (render_multigame_maps_batch)
        _tile_size = getattr(config, 'vit_tile_size', 16)
        logger.info(f"Rendering GT images (tile_size={_tile_size}) ...")
        _gt_images_raw = render_multigame_maps_batch(
            np.stack([s.array.astype(np.int32) for s in samples]),  # (M, H, W)
            tile_size=_tile_size,
        )  # (M, H*ts, W*ts, 3)
        gt_images = np.repeat(_gt_images_raw, _n_eps, axis=0)  # (M*n_eps, H*ts, W*ts, 3)
        logger.info(f"GT images: {_gt_images_raw.shape} × n_eps={_n_eps} → {gt_images.shape}")

        # ── dry-run: max_samples 로 잘라내기 ─────────────────────────────
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
    """Hydra @main 에서 호출하는 공통 평가 엔트리포인트.

    Args:
        config: Hydra config (EvalConfig 또는 그 하위 클래스).
        inject_obs_fn: obs 주입 콜백. None 이면 config 기반 주입 로직 사용.
    """
    _eval_start = time.perf_counter()

    config = init_config(config)

    if config.eval_aug_type is not None and config.eval_embed_type is not None and config.eval_instruct is not None:
        config.eval_instruct_csv = f'{config.eval_aug_type}/{config.eval_embed_type}/{config.eval_instruct}'

    if config.n_eps < 2 and config.diversity:
        raise Exception("Diversity evaluation requires n_eps > 1")

    rng = jax.random.PRNGKey(config.seed)

    exp_dir = config.exp_dir
    logger.info(f"Running experiment at {exp_dir}")

    _re = getattr(config, 'dataset_reward_enum', None)
    _re_suffix = f"_re-{_re}" if _re is not None else ""

    # eval_games 가 지정된 경우 약어를 폴더명에 포함 (없으면 game 사용)
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
