"""
train_clip_decoder_unseen.py
============================
Seen/Unseen 게임 분리 + Few-shot Ratio Sweep 실험 스크립트.

Seen 게임의 전체 학습 데이터 + Unseen 게임의 가변 비율 학습 데이터로
CLIP Decoder 모델을 학습하고, **고정된** 테스트셋에서 게임별 reward_accuracy를 측정한다.

최종 출력: few-shot ratio (x) vs. per-game reward accuracy (y) 그래프

Usage
-----
    python train_clip_decoder_unseen.py game=all unseen_games=zd
    python train_clip_decoder_unseen.py game=all unseen_games=pkzd \\
        'unseen_ratios=[0.0,0.1,0.5,1.0]' n_epochs=30
"""

import datetime
import json
import os
import shutil
import logging
from os.path import basename
from typing import Dict

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

# ── train_clip_decoder.py 에서 공통 함수 재활용 ──
from train_clip_decoder import (
    parse_unseen_game_names,
    subset_clip_dataset,
    split_dataset_by_game,
    build_train_indices_for_ratio,
    train_step,  # noqa: F401 (JIT-compiled, module-level 등록 필요)
    train_and_evaluate_ratio,
    create_fewshot_plot,
)

log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logger = logging.getLogger(basename(__file__))
logger.setLevel(getattr(logging, log_level, logging.INFO))
logging.getLogger("absl").setLevel(logging.ERROR)


# ═══════════════════════════════════════════════════════════════════════════════
#  Main Sweep
# ═══════════════════════════════════════════════════════════════════════════════

def make_train_unseen(config: CLIPDecoderUnseenConfig):
    def train(rng_key):
        rng_key, subkey = jax.random.split(rng_key)

        # ── 1. 전체 데이터셋 빌드 (한 번만) ──
        dataset = build_multigame_dataset(config)
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

        dataset_builder = CLIPDatasetBuilder(
            processor=processor,
            paired_data=dataset,
            rng_key=subkey,
            max_len=config.encoder.token_max_len,
            train_ratio=1.0,  # 자체 split 수행 → 빌더의 split 사용 안 함
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
        logger.info("  Seen/Unseen Split")
        logger.info("  Seen games  : %s", seen_games)
        logger.info("  Unseen games: %s", unseen_games)
        logger.info("  Total samples: %d", len(full_dataset.class_ids))
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

        # 고정 테스트 인덱스 (모든 게임)
        test_indices = np.concatenate(
            [game_test[g] for g in sorted(game_test.keys())]
        )
        test_ds = subset_clip_dataset(full_dataset, test_indices)
        test_game_names = np.array(
            [rc["game_name"] for rc in test_ds.reward_cond]
        )

        # 로깅: 분할 요약
        logger.info("  Test set (fixed, seed=%d):", config.unseen_test_seed)
        for g in sorted(game_test.keys()):
            tag = "(unseen)" if g in unseen_game_set else "(seen)"
            logger.info(
                "    %-12s %s  train_pool=%d, test=%d",
                g, tag, len(game_train_pool[g]), len(game_test[g]),
            )
        logger.info("  Total test: %d", len(test_indices))

        # ── 4. Few-shot ratio sweep ──
        ratios = list(config.unseen_ratios)
        results: Dict[float, Dict[str, float]] = {}
        reg_results: Dict[float, Dict[str, float]] = {}
        enum_diff_results: Dict[float, Dict[str, Dict[int, float]]] = {}

        for ratio_idx, ratio in enumerate(ratios):
            # 학습 인덱스 구성
            train_indices = build_train_indices_for_ratio(
                game_train_pool, unseen_game_set, ratio,
                seen_ratio=config.seen_ratio,
            )

            if len(train_indices) == 0:
                logger.warning(
                    "ratio=%.2f: 0 training samples — evaluating untrained model",
                    ratio,
                )
                # seen game도 없는 특이 케이스: 최소 1개 샘플로 모델 초기화만 수행
                train_ds = subset_clip_dataset(full_dataset, np.array([0]))
            else:
                train_ds = subset_clip_dataset(full_dataset, train_indices)

            # train pool 구성 로그
            _train_games = np.array(
                [rc["game_name"] for rc in train_ds.reward_cond]
            )
            _game_counts = {g: int(np.sum(_train_games == g)) for g in sorted(set(_train_games))}
            logger.info(
                "  Ratio %.2f: train set = %d samples %s",
                ratio, len(train_indices), _game_counts,
            )

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
            results[ratio] = per_game_acc
            reg_results[ratio] = per_game_reg
            enum_diff_results[ratio] = per_game_enum_diff

            # W&B 로깅 (per-ratio 최종 결과 + incremental plot)
            if wandb.run is not None:
                log_dict = {"unseen/ratio": ratio}
                for g, acc in per_game_acc.items():
                    log_dict[f"unseen/acc_{g}"] = acc
                for g, reg in per_game_reg.items():
                    log_dict[f"unseen/reg_{g}"] = reg
                wandb.log(log_dict)

        # ── 5. 결과 저장 ──
        save_data = {
            str(r): {"accuracy": results[r], "reg_loss": reg_results[r]}
            for r in results
        }
        results_path = os.path.join(config.exp_dir, "fewshot_results.json")
        with open(results_path, "w") as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)
        logger.info("Results saved: %s", results_path)

        # ── 6. 최종 그래프 생성 (결과 디렉토리에 저장) ──
        plot_path = create_fewshot_plot(
            results, reg_results, unseen_game_set, config.exp_dir
        )

        if wandb.run is not None:

            # reward_enum 전체 목록 수집
            all_enums = sorted({
                e for ratio_ed in enum_diff_results.values()
                for game_ed in ratio_ed.values()
                for e in game_ed.keys()
            })

            # 게임 이름 (overall 등 제외)
            game_names_sorted = sorted(
                {g for r in results.values() for g in r
                 if g not in ("overall", "seen_overall", "unseen_overall")}
            )

            # ratio=1.0 baseline diff 계산 (정규화 기준)
            baseline_ratio = 1.0
            baseline_seen_diff = None
            baseline_unseen_diff = None
            if baseline_ratio in enum_diff_results:
                _s, _u = [], []
                for g in game_names_sorted:
                    game_ed = enum_diff_results[baseline_ratio].get(g, {})
                    vals = [v for v in game_ed.values() if v is not None]
                    if vals:
                        avg = sum(vals) / len(vals)
                        if g in unseen_game_set:
                            _u.append(avg)
                        else:
                            _s.append(avg)
                baseline_seen_diff = sum(_s) / len(_s) if _s else None
                baseline_unseen_diff = sum(_u) / len(_u) if _u else None

            # 테이블: 각 행 = 하나의 ratio
            # norm_diff = raw_diff / baseline_diff (ratio=1.0 대비 상대 에러)
            columns = [
                "ratio", "seen_ratio", "game", "unseen_games",
                "seen_acc", "unseen_acc",
                "seen_reg", "unseen_reg",
                "seen_avg_diff", "unseen_avg_diff",
                "seen_norm_diff", "unseen_norm_diff",
            ]
            for g in game_names_sorted:
                for e in all_enums:
                    columns.append(f"{g}_enum_{e}")

            table = wandb.Table(columns=columns)
            for ratio_val in sorted(results.keys()):
                # seen/unseen 집계
                seen_acc = results[ratio_val].get("seen_overall", None)
                unseen_acc = results[ratio_val].get("unseen_overall", None)
                seen_reg = reg_results[ratio_val].get("seen_overall", None)
                unseen_reg = reg_results[ratio_val].get("unseen_overall", None)

                # 게임별 enum diff 평균 → seen/unseen raw avg diff
                seen_diffs, unseen_diffs = [], []
                for g in game_names_sorted:
                    game_ed = enum_diff_results[ratio_val].get(g, {})
                    vals = [v for v in game_ed.values() if v is not None]
                    if vals:
                        avg = sum(vals) / len(vals)
                        if g in unseen_game_set:
                            unseen_diffs.append(avg)
                        else:
                            seen_diffs.append(avg)
                seen_avg = sum(seen_diffs) / len(seen_diffs) if seen_diffs else None
                unseen_avg = sum(unseen_diffs) / len(unseen_diffs) if unseen_diffs else None

                # ratio=1.0 대비 정규화 (1.0 = baseline과 동일, >1.0 = 더 나쁨)
                seen_norm = (seen_avg / baseline_seen_diff
                             if seen_avg is not None and baseline_seen_diff else None)
                unseen_norm = (unseen_avg / baseline_unseen_diff
                               if unseen_avg is not None and baseline_unseen_diff else None)

                row = [
                    float(ratio_val), float(config.seen_ratio), config.game, config.unseen_games,
                    seen_acc, unseen_acc,
                    seen_reg, unseen_reg,
                    seen_avg, unseen_avg,
                    seen_norm, unseen_norm,
                ]
                for g in game_names_sorted:
                    game_ed = enum_diff_results[ratio_val].get(g, {})
                    for e in all_enums:
                        row.append(game_ed.get(e, None))
                table.add_data(*row)

            wandb.log({
                "table/results": table,
                "table/fewshot_plot": wandb.Image(plot_path),
            })

        # ── 최종 요약 출력 ──
        logger.info("\n" + "=" * 70)
        logger.info("  SWEEP COMPLETE — Summary")
        logger.info("=" * 70)
        header = f"  {'ratio':>6}"
        for g in sorted(unique_games):
            header += f"  {g:>10}"
        header += f"  {'overall':>10}"
        logger.info(header)
        logger.info("  " + "-" * (len(header) - 2))
        for ratio_val in sorted(results.keys()):
            row = f"  {ratio_val:>6.2f}"
            for g in sorted(unique_games):
                row += f"  {results[ratio_val].get(g, float('nan')):>10.4f}"
            row += f"  {results[ratio_val].get('overall', float('nan')):>10.4f}"
            logger.info(row)
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

