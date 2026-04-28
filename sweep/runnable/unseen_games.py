"""
sweep/runnable_sweep/unseen_games.py
=====================================
Seen/Unseen 게임 분리 + **unseen_ratios Sweep** 실험 스크립트.

config의 unseen_ratios 리스트에 있는 모든 ratio에 대해 순차적으로
CLIP Decoder 모델을 학습·평가하고, 게임별 reward_accuracy를 측정한다.

단일 ratio 실험은 train_clip_decoder_unseen.py 를 사용하세요.

Usage
-----
    # 프로젝트 루트에서 실행
    python sweep/runnable_sweep/unseen_games.py game=all unseen_games=zd
    python sweep/runnable_sweep/unseen_games.py game=all unseen_games=pkzd \\
        unseen_ratios="[0.0,0.05,0.1,0.5,1.0]"
"""

import datetime
import json
import os
import shutil
import sys
import logging
from os.path import basename

# ── 프로젝트 루트를 sys.path에 추가 (sweep/ 하위에서 실행 시 필요) ──
_PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import hydra
import jax
import numpy as np
import wandb
from transformers import CLIPProcessor

from conf.config import CLIPDecoderUnseenSweepConfig
from encoder.data.clip_batch import CLIPDatasetBuilder
from encoder.utils.path import init_config
from encoder.utils.training import build_multigame_dataset
from instruct_rl.utils.logger import get_wandb_name

from train_clip_decoder import (
    parse_unseen_game_names,
    subset_clip_dataset,
    split_dataset_by_game,
    build_train_indices_for_ratio,
    train_step,  # noqa: F401
    train_and_evaluate_ratio,
)

log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logger = logging.getLogger(basename(__file__))
logger.setLevel(getattr(logging, log_level, logging.INFO))
logging.getLogger("absl").setLevel(logging.ERROR)


# ═══════════════════════════════════════════════════════════════════════════════
#  Ratio Sweep train
# ═══════════════════════════════════════════════════════════════════════════════

def make_train_unseen_sweep(config: CLIPDecoderUnseenSweepConfig):
    """unseen_ratios 리스트를 순회하면서 각 ratio 별 학습·평가를 수행한다."""

    def train(rng_key):
        rng_key, subkey = jax.random.split(rng_key)

        # ── 1. 전체 데이터셋 빌드 ──
        dataset = build_multigame_dataset(config)
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

        dataset_builder = CLIPDatasetBuilder(
            processor=processor,
            paired_data=dataset,
            rng_key=subkey,
            max_len=config.encoder.token_max_len,
            train_ratio=1.0,
            max_samples=config.max_samples,
            prepend_game_prefix=config.prepend_game_prefix,
            prepend_game_desc=config.prepend_game_desc,
        )

        full_dataset = dataset_builder.get_dataset()
        cond_norm_min, cond_norm_max = dataset_builder.get_condition_norm_stats()

        # ── 2. Seen/Unseen 게임 파싱 ──
        unseen_game_set = parse_unseen_game_names(config.unseen_games)
        all_game_names = np.array(
            [rc["game_name"] for rc in full_dataset.reward_cond]
        )
        unique_games = sorted(set(all_game_names))
        seen_games = [g for g in unique_games if g not in unseen_game_set]
        unseen_games = [g for g in unique_games if g in unseen_game_set]

        logger.info("=" * 70)
        logger.info("  Seen/Unseen Split  (SWEEP over %d ratios)", len(config.unseen_ratios))
        logger.info("  Seen games  : %s", seen_games)
        logger.info("  Unseen games: %s", unseen_games)
        logger.info("  Total samples: %d", len(full_dataset.class_ids))
        logger.info("  unseen_ratios: %s", list(config.unseen_ratios))
        logger.info("=" * 70)

        if not unseen_games:
            logger.error("No unseen games found in dataset! Check 'game' and 'unseen_games' config.")
            return

        # ── 3. 게임별 train pool / test 분할 (seed 고정) ──
        game_train_pool, game_test, _ = split_dataset_by_game(
            full_dataset,
            unseen_game_set,
            test_ratio=config.unseen_test_ratio,
            test_seed=config.unseen_test_seed,
        )

        test_indices = np.concatenate(
            [game_test[g] for g in sorted(game_test.keys())]
        )
        test_ds = subset_clip_dataset(full_dataset, test_indices)
        test_game_names = np.array(
            [rc["game_name"] for rc in test_ds.reward_cond]
        )

        for g in sorted(game_test.keys()):
            tag = "(unseen)" if g in unseen_game_set else "(seen)"
            logger.info(
                "    %-12s %s  train_pool=%d, test=%d",
                g, tag, len(game_train_pool[g]), len(game_test[g]),
            )
        logger.info("  Total test: %d", len(test_indices))

        # ── 4. 각 ratio 순회 ──
        ratios = list(config.unseen_ratios)
        all_results: dict = {}

        for ratio_idx, ratio in enumerate(ratios):
            logger.info(
                "\n[Sweep %d/%d]  unseen_ratio = %.4f",
                ratio_idx + 1, len(ratios), ratio,
            )

            train_indices = build_train_indices_for_ratio(
                game_train_pool, unseen_game_set, ratio,
                seen_ratio=config.seen_ratio,
            )

            if len(train_indices) == 0:
                logger.warning(
                    "ratio=%.2f: 0 training samples — evaluating untrained model", ratio
                )
                train_ds = subset_clip_dataset(full_dataset, np.array([0]))
            else:
                train_ds = subset_clip_dataset(full_dataset, train_indices)

            _train_games = np.array([rc["game_name"] for rc in train_ds.reward_cond])
            _game_counts = {
                g: int(np.sum(_train_games == g)) for g in sorted(set(_train_games))
            }
            logger.info("  Train set = %d samples %s", len(train_indices), _game_counts)

            rng_key, ratio_key = jax.random.split(rng_key)
            per_game_acc, per_game_reg, per_game_enum_diff = train_and_evaluate_ratio(
                config=config,
                rng_key=ratio_key,
                train_ds=train_ds,
                test_ds=test_ds,
                test_game_names=test_game_names,
                unseen_game_names=unseen_game_set,
                cond_norm_min=cond_norm_min,
                cond_norm_max=cond_norm_max,
                ratio=ratio,
                ratio_idx=ratio_idx,
                total_ratios=len(ratios),
            )

            all_results[str(ratio)] = {
                "accuracy": per_game_acc,
                "reg_loss": per_game_reg,
            }

            # W&B 로깅
            if wandb.run is not None:
                log_dict = {"sweep/ratio": ratio, "sweep/ratio_idx": ratio_idx}
                for g, acc in per_game_acc.items():
                    log_dict[f"sweep/acc_{g}"] = acc
                for g, reg in per_game_reg.items():
                    log_dict[f"sweep/reg_{g}"] = reg
                wandb.log(log_dict, step=ratio_idx)

            # 요약 출력
            logger.info("  RESULT  (ratio=%.4f)", ratio)
            for g in sorted(unique_games):
                logger.info(
                    "    %-12s  acc=%.4f  reg=%.4f",
                    g,
                    per_game_acc.get(g, float("nan")),
                    per_game_reg.get(g, float("nan")),
                )
            logger.info("  overall        acc=%.4f", per_game_acc.get("overall", float("nan")))
            logger.info("  seen_overall   acc=%.4f", per_game_acc.get("seen_overall", float("nan")))
            logger.info("  unseen_overall acc=%.4f", per_game_acc.get("unseen_overall", float("nan")))

        # ── 5. 전체 결과 저장 ──
        results_path = os.path.join(config.exp_dir, "unseen_sweep_results.json")
        with open(results_path, "w") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        logger.info("\nAll sweep results saved: %s", results_path)

        # ── 최종 sweep 요약 ──
        logger.info("\n" + "=" * 70)
        logger.info("  SWEEP SUMMARY  (%d ratios)", len(ratios))
        logger.info("  %-8s  %s", "ratio", "  ".join(f"{g:>12}" for g in sorted(unique_games)))
        logger.info("-" * 70)
        for ratio in ratios:
            accs = all_results[str(ratio)]["accuracy"]
            row = "  ".join(
                f"{accs.get(g, float('nan')):>12.4f}" for g in sorted(unique_games)
            )
            logger.info("  %-8.4f  %s", ratio, row)
        logger.info("=" * 70)

    return lambda rng_key: train(rng_key)


# ═══════════════════════════════════════════════════════════════════════════════
#  Entry Point
# ═══════════════════════════════════════════════════════════════════════════════

@hydra.main(
    version_base=None,
    config_path=os.path.join(_PROJECT_ROOT, "conf"),
    config_name="train_clip_decoder_unseen_sweep_schema",
)
def main(config: CLIPDecoderUnseenSweepConfig):
    if config.encoder.model is None:
        config.encoder.model = "cnnclip"
        logger.warning("encoder.model is None, using default value: cnnclip")

    config = init_config(config)

    rng_key = jax.random.PRNGKey(config.seed)
    np.random.seed(config.seed)

    from instruct_rl.utils.env_loader import get_wandb_key

    wandb_key = get_wandb_key()
    if wandb_key:
        dt = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        wandb_id = f"{get_wandb_name(config)}-sweep-{dt}"
        wandb.login(key=wandb_key)
        wandb.init(
            project=config.wandb_project,
            group=config.instruct,
            entity=config.wandb_entity,
            name=f"{get_wandb_name(config)}-sweep",
            id=wandb_id,
            save_code=True,
        )
        wandb.config.update(dict(config), allow_val_change=True)

    exp_dir = config.exp_dir
    logger.info(f"jax devices: {jax.devices()}")
    logger.info(f"running sweep experiment at {exp_dir}")
    logger.info(f"sweep ratios: {list(config.unseen_ratios)}")

    if config.overwrite and os.path.exists(exp_dir):
        shutil.rmtree(exp_dir)

    os.makedirs(exp_dir, exist_ok=True)

    make_train_unseen_sweep(config)(rng_key)

    if wandb.run:
        wandb.finish()


if __name__ == "__main__":
    main()

