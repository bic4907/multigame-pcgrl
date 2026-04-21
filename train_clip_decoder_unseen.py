"""
train_clip_decoder_unseen.py
============================
Seen/Unseen 게임 분리 + 단일 Few-shot Ratio 실험 스크립트.

Seen 게임의 전체 학습 데이터 + Unseen 게임의 **단일 비율(unseen_ratio)** 학습 데이터로
CLIP Decoder 모델을 학습하고, 고정된 테스트셋에서 게임별 reward_accuracy를 측정한다.

ratio sweep 을 원할 경우 sweep/runnable_sweep/unseen_games.py 를 사용하세요.

Usage
-----
    python train_clip_decoder_unseen.py game=all unseen_games=zd unseen_ratio=0.1
    python train_clip_decoder_unseen.py game=all unseen_games=pkzd unseen_ratio=0.0
"""

import datetime
import json
import os
import shutil
import logging
from os.path import basename

import hydra
import jax
import numpy as np
import wandb
from transformers import CLIPProcessor

from conf.config import CLIPDecoderUnseenConfig
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
#  Single-ratio train
# ═══════════════════════════════════════════════════════════════════════════════

def make_train_unseen(config: CLIPDecoderUnseenConfig):
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
            longtail_cut=True,
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
        logger.info("  Seen/Unseen Split")
        logger.info("  Seen games  : %s", seen_games)
        logger.info("  Unseen games: %s", unseen_games)
        logger.info("  Total samples: %d", len(full_dataset.class_ids))
        logger.info("  unseen_ratio : %.4f", config.unseen_ratio)
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

        # ── 4. 단일 ratio로 학습 인덱스 구성 ──
        ratio = config.unseen_ratio
        train_indices = build_train_indices_for_ratio(
            game_train_pool, unseen_game_set, ratio,
            seen_ratio=config.seen_ratio,
        )

        if len(train_indices) == 0:
            logger.warning("ratio=%.2f: 0 training samples — evaluating untrained model", ratio)
            train_ds = subset_clip_dataset(full_dataset, np.array([0]))
        else:
            train_ds = subset_clip_dataset(full_dataset, train_indices)

        _train_games = np.array([rc["game_name"] for rc in train_ds.reward_cond])
        _game_counts = {g: int(np.sum(_train_games == g)) for g in sorted(set(_train_games))}
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
            ratio_idx=0,
            total_ratios=1,
        )

        # ── 5. 결과 저장 ──
        save_data = {
            str(ratio): {"accuracy": per_game_acc, "reg_loss": per_game_reg}
        }
        results_path = os.path.join(config.exp_dir, "unseen_results.json")
        with open(results_path, "w") as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)
        logger.info("Results saved: %s", results_path)

        # ── 6. W&B 로깅 ──
        if wandb.run is not None:
            log_dict = {"unseen/ratio": ratio}
            for g, acc in per_game_acc.items():
                log_dict[f"unseen/acc_{g}"] = acc
            for g, reg in per_game_reg.items():
                log_dict[f"unseen/reg_{g}"] = reg
            wandb.log(log_dict)

        # ── 최종 요약 출력 ──
        logger.info("\n" + "=" * 70)
        logger.info("  RESULT  (ratio=%.4f)", ratio)
        logger.info("=" * 70)
        for g in sorted(unique_games):
            logger.info("  %-12s  acc=%.4f  reg=%.4f",
                        g, per_game_acc.get(g, float('nan')), per_game_reg.get(g, float('nan')))
        logger.info("  overall       acc=%.4f", per_game_acc.get("overall", float('nan')))
        logger.info("  seen_overall  acc=%.4f", per_game_acc.get("seen_overall", float('nan')))
        logger.info("  unseen_overall acc=%.4f", per_game_acc.get("unseen_overall", float('nan')))
        logger.info("=" * 70)

    return lambda rng_key: train(rng_key)


# ═══════════════════════════════════════════════════════════════════════════════
#  Entry Point
# ═══════════════════════════════════════════════════════════════════════════════

@hydra.main(version_base=None, config_path="./conf", config_name="train_clip_decoder_unseen")
def main(config: CLIPDecoderUnseenConfig):
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
        wandb_id = f"{get_wandb_name(config)}-{dt}"
        wandb.login(key=wandb_key)
        wandb.init(
            project=config.wandb_project,
            group=config.instruct,
            entity=config.wandb_entity,
            name=get_wandb_name(config),
            id=wandb_id,
            save_code=True,
        )
        wandb.config.update(dict(config), allow_val_change=True)

    exp_dir = config.exp_dir
    logger.info(f"jax devices: {jax.devices()}")
    logger.info(f"running experiment at {exp_dir}")

    if config.overwrite and os.path.exists(exp_dir):
        shutil.rmtree(exp_dir)

    os.makedirs(exp_dir, exist_ok=True)

    make_train_unseen(config)(rng_key)

    if wandb.run:
        wandb.finish()


if __name__ == "__main__":
    main()

