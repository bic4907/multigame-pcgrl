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
from datetime import datetime

import jax
import wandb

from instruct_rl.utils.path_utils import init_config
from instruct_rl.utils.logger import get_wandb_name_eval

logger = logging.getLogger(__name__)


def main_chunk(config, rng, *, inject_obs_fn=None):
    """체크포인트 로드 후 make_eval 실행."""
    from instruct_rl.utils.checkpointer import init_checkpointer
    from instruct_rl.eval.runner import make_eval

    if not config.random_agent:
        _, restored_ckpt, encoder_param = init_checkpointer(config)
    else:
        restored_ckpt, encoder_param = None, None

    # train과 동일하게 MultiGameDataset 기반 eval instruct 로드
    eval_inst = None
    if hasattr(config, 'dataset_game') and config.dataset_game is not None:
        from instruct_rl.utils.dataset_loader import load_dataset_instruct
        _, eval_inst = load_dataset_instruct(config)  # test split 사용
        logger.info(f"Loaded eval instruct from dataset: {eval_inst.reward_i.shape[0]} samples")

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

    eval_fn = make_eval(
        config, restored_ckpt, encoder_param,
        inject_obs_fn=inject_obs_fn,
        eval_inst=eval_inst,
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

    config = init_config(config)

    if config.eval_aug_type is not None and config.eval_embed_type is not None and config.eval_instruct is not None:
        config.eval_instruct_csv = f'{config.eval_aug_type}/{config.eval_embed_type}/{config.eval_instruct}'

    if config.n_eps < 2 and config.diversity:
        raise Exception("Diversity evaluation requires n_eps > 1")

    rng = jax.random.PRNGKey(config.seed)

    exp_dir = config.exp_dir
    logger.info(f"Running experiment at {exp_dir}")

    eval_dir = os.path.join(
        exp_dir,
        f"ev_{config.eval_instruct}_{config.eval_modality}"
        f"{f'_{config.eval_exp_name}' if config.eval_exp_name else ''}",
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
            config=wandb.helper.parse_config(
                dict(config),
                exclude=("wandb_key", "_vid_dir", "_img_dir", "_numpy_dir", "overwrite", "initialize"),
            ),
        )
        wandb.config.update(dict(config), allow_val_change=True)

    main_chunk(config, rng, inject_obs_fn=inject_obs_fn)

    if wandb.run:
        wandb.finish()
