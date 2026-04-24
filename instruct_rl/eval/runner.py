"""
runner.py
=========
make_eval — 환경 루프 core.

make_eval(config, restored_ckpt, encoder_params, *, inject_obs_fn=None)
  → callable(rng) → losses (np.ndarray)
"""
import logging
import math
import time

import jax
import jax.numpy as jnp
import os
import numpy as np
import optax
import pandas as pd
import wandb
from os.path import join
from flax.training.train_state import TrainState
from tqdm import tqdm

from envs.pcgrl_env import gen_dummy_queued_state
from instruct_rl.dataclass import Instruct
from instruct_rl.eval.wrappers import DiversityWrapper, ViTScoreWrapper, TPKLWrapper
from instruct_rl.eval.wrappers.progress import ProgressWrapper
from instruct_rl.eval.agg import iqr_mean
from instruct_rl.eval.batch_save import save_batch_results
from instruct_rl.evaluate import get_loss_batch
from instruct_rl.utils.path_utils import gymnax_pcgrl_make, init_network
from instruct_rl.eval.hdf5_store import open_eval_store
from instruct_rl.eval.image_utils import sample_wandb_images
from purejaxrl.experimental.s5.wrappers import LogWrapper
from purejaxrl.structures import Transition, RunnerState

logger = logging.getLogger(__name__)


def make_eval(config, restored_ckpt, encoder_params, *, inject_obs_fn=None, eval_inst=None, eval_inst_meta=None, gt_levels=None, gt_images=None):
    """평가 함수를 생성하여 반환.

    Args:
        config         : EvalConfig (또는 하위 클래스).
        restored_ckpt  : 체크포인트 dict (None 이면 랜덤 초기화).
        encoder_params : 사전학습 encoder 파라미터 (None 이면 스킵).
        inject_obs_fn  : obs 주입 콜백. None 이면 config 기반 주입.

    Returns:
        eval_fn(rng) → losses
    """
    env, env_params = gymnax_pcgrl_make(config.env_name, config=config)
    env = LogWrapper(env)
    env.init_graphics()

    def eval_fn(rng):
        # ── 네트워크 / TrainState 초기화 ─────────────────────────────────────
        network = init_network(env, env_params, config)
        rng, _rng = jax.random.split(rng)
        init_x = env.gen_dummy_obs(env_params)

        network_params = (
            restored_ckpt['runner_state'].train_state.params
            if restored_ckpt is not None
            else network.init(_rng, init_x)
        )

        if config.ANNEAL_LR:
            def _lr_schedule(count):
                frac = (
                    1.0
                    - (count // (config.NUM_MINIBATCHES * config.update_epochs))
                    / config.NUM_UPDATES
                )
                return config.LR * frac
            tx = optax.chain(
                optax.clip_by_global_norm(config.MAX_GRAD_NORM),
                optax.adam(learning_rate=_lr_schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(config.MAX_GRAD_NORM),
                optax.adam(config.lr, eps=1e-5),
            )

        train_state = TrainState.create(apply_fn=network.apply, params=network_params, tx=tx)

        # ── 환경 초기화 ───────────────────────────────────────────────────────
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config.n_envs)
        vmap_reset_fn = jax.vmap(env.reset, in_axes=(0, None, None))
        obsv, env_state = vmap_reset_fn(reset_rng, env_params, gen_dummy_queued_state(env))

        rng, _rng = jax.random.split(rng)
        runner_state = (
            restored_ckpt["runner_state"]
            if restored_ckpt is not None
            else RunnerState(train_state, env_state, obsv, rng, update_i=0)
        )

        if encoder_params is not None:
            logger.info(f"Parameters loaded from encoder checkpoint ({config.encoder.ckpt_path})")
            runner_state.train_state.params['params']['subnet']['encoder'] = encoder_params

        # ── Instruct 준비 (dataset 모드 or CSV 모드) ──────────────────────────
        instruct = eval_inst
        n_inst = instruct.reward_i.shape[0]
        # 출력용 DataFrame 생성
        reward_i_flat = instruct.reward_i[:, 0].tolist()
        instruct_df = pd.DataFrame({
            'row_i': list(range(n_inst)),
            'reward_enum': reward_i_flat,
        })
        # 샘플 메타데이터 (game, instruction) 병합
        if eval_inst_meta is not None:
            instruct_df['game']        = eval_inst_meta['game'].values[:n_inst]
            instruct_df['instruction'] = eval_inst_meta['instruction'].values[:n_inst]
        else:
            instruct_df['game']        = 'unknown'
            instruct_df['instruction'] = None
        for c_i in range(instruct.condition.shape[1]):
            instruct_df[f'condition_{c_i}'] = instruct.condition[:, c_i].tolist()
        # 컬럼 순서 정렬: game, instruction, reward_enum, condition_*
        cond_cols = [c for c in instruct_df.columns if c.startswith('condition_')]
        ordered_cols = ['row_i', 'game', 'instruction', 'reward_enum'] + cond_cols
        instruct_df = instruct_df[[c for c in ordered_cols if c in instruct_df.columns]]
        # -1 (null 센티널) → NaN (CSV 빈값)
        instruct_df['reward_enum'] = instruct_df['reward_enum'].replace(-1, float('nan'))
        for c in cond_cols:
            instruct_df[c] = instruct_df[c].replace(-1.0, float('nan'))
        instruct_df.to_csv(join(config.eval_dir, 'input.csv'), index=False)

        # ── 배치 구성 ─────────────────────────────────────────────────────────
        n_envs = config.n_envs
        n_eps = config.n_eps
        eval_batches = jnp.array(sorted(np.tile(list(range(len(instruct_df))), n_eps)))
        repetitions = np.tile(list(range(n_eps)), len(instruct_df))
        n_rows = len(eval_batches)

        if len(eval_batches) != len(repetitions):
            raise Exception(
                f"eval_batches/repetitions length mismatch: "
                f"{len(eval_batches)} != {len(repetitions)}"
            )

        n_batches = math.ceil(n_rows / n_envs)
        losses, values, features = [], [], []
        losses_s0, features_s0 = [], []
        eval_rendered = []
        loop_start_time = time.time()

        # ── 평가 루프 시작 전 요약 로그 ───────────────────────────────────────
        logger.info(
            "[Eval Loop] total_items=%d  (samples=%d × n_eps=%d)  "
            "batch_size(n_envs)=%d  n_batches=%d",
            n_rows, len(instruct_df), n_eps, n_envs, n_batches,
        )

        # ── 평가 루프 ─────────────────────────────────────────────────────────
        with open_eval_store(config.eval_dir, mode="a") as h5_store, \
             tqdm(total=n_batches, desc="Rollout Batches") as pbar:
            for batch_i in range(n_batches):
                batch_start_time = time.time()
                start_idx = batch_i * n_envs
                end_idx = min((batch_i + 1) * n_envs, n_rows)
                idxes = eval_batches[start_idx:end_idx]
                batch_valid_size = len(idxes)

                batch_embedding = instruct.embedding[idxes]
                batch_reward_i = instruct.reward_i[idxes]
                batch_condition = instruct.condition[idxes]
                batch_repetition = repetitions[start_idx:end_idx]

                # 마지막 배치 패딩
                if len(batch_embedding) < n_envs:
                    pad = n_envs - len(batch_embedding)
                    batch_embedding = jnp.pad(batch_embedding, ((0, pad), (0, 0)))
                    batch_condition = jnp.pad(batch_condition, ((0, pad), (0, 0)))
                    batch_reward_i = jnp.pad(batch_reward_i, ((0, pad),))
                    batch_repetition = jnp.pad(batch_repetition, ((0, pad),))

                batch_instruct = Instruct(
                    reward_i=batch_reward_i,
                    condition=batch_condition,
                    embedding=batch_embedding,
                    condition_id=None,
                )

                reset_rng = jnp.stack([jax.random.PRNGKey(s) for s in batch_repetition])
                init_obs, init_state = vmap_reset_fn(
                    reset_rng, env_params, gen_dummy_queued_state(env)
                )
                done = jnp.zeros((n_envs,), dtype=bool)

                # ── 단일 스텝 ─────────────────────────────────────────────────
                def _env_step(carry, _):
                    rng, last_obs, state, done = carry

                    if inject_obs_fn is not None:
                        last_obs = inject_obs_fn(last_obs, state, batch_instruct, config, env)
                    else:
                        if config.use_nlp or config.use_clip:
                            last_obs = last_obs.replace(nlp_obs=batch_instruct.embedding)
                        if config.vec_cont:
                            vmap_cont = jax.vmap(env.prob.get_cont_obs, in_axes=(0, 0, None))
                            cont_obs = vmap_cont(
                                state.env_state.env_map,
                                batch_instruct.condition,
                                config.raw_obs,
                            )
                            last_obs = last_obs.replace(nlp_obs=cont_obs)

                    rng, _rng = jax.random.split(rng)
                    pi, value, _, _, _ = network.apply(
                        runner_state.train_state.params, last_obs,
                        return_text_embed=False,
                        return_state_embed=False,
                    )
                    action = pi.sample(seed=_rng)
                    log_prob = pi.log_prob(action)

                    rng, _rng = jax.random.split(rng)
                    rng_step = jax.random.split(_rng, config.n_envs)
                    obsv, next_state, reward_env, done, info = jax.vmap(
                        env.step, in_axes=(0, 0, 0, None)
                    )(rng_step, state, action, env_params)

                    transition = Transition(
                        done, action, value, reward_env, log_prob, obsv, info,
                        next_state.env_state, None,
                    )
                    return (rng, obsv, next_state, done), (transition, state)

                # ── scan 래퍼 ─────────────────────────────────────────────────
                @jax.jit
                def run_eval_step(rng, init_obs, init_state, done):
                    env_step_len = int(
                        (config.map_width ** 2)
                        * config.max_board_scans
                        * (2 if config.representation == "turtle" else 1)
                    )
                    _, (_, states) = jax.lax.scan(
                        _env_step, (rng, init_obs, init_state, done), None,
                        length=env_step_len,
                    )
                    states = jax.tree.map(
                        lambda x, y: jnp.concatenate([x[None], y], axis=0),
                        init_state, states,
                    )
                    last_states = jax.tree.map(lambda x: x[[-1], ...], states)

                    rendered = jax.vmap(
                        lambda s: jax.vmap(env.render)(s.env_state)
                    )(last_states).transpose(1, 0, 2, 3, 4)

                    rendered_raw = jax.vmap(
                        lambda s: jax.vmap(env.render_env_map)(s.env_state.env_map)
                    )(last_states)
                    _n_row, _n_eps, _h, _w, _c = rendered_raw.shape
                    rendered_raw = rendered_raw.reshape(-1, _h, _w, _c)

                    result = get_loss_batch(
                        reward_i=batch_instruct.reward_i,
                        condition=batch_instruct.condition,
                        env_maps=states.env_state.env_map[-1, :, :, :],
                    )
                    result_s0 = get_loss_batch(
                        reward_i=batch_instruct.reward_i,
                        condition=batch_instruct.condition,
                        env_maps=init_state.env_state.env_map,
                    )
                    return result, result_s0, rendered, rendered_raw, last_states

                rng_eval = jax.random.PRNGKey(30)
                result, result_s0, rendered, raw_rendered, last_states = run_eval_step(
                    rng_eval, init_obs, init_state, done
                )

                result = jax.device_get(result)
                result_s0 = jax.device_get(result_s0)
                losses.append(result.loss)
                values.append(result.value)
                features.append(result.feature)
                losses_s0.append(result_s0.loss)
                features_s0.append(result_s0.feature)
                eval_rendered.append(jax.device_get(raw_rendered))
                rendered = jax.device_get(rendered)

                # ── 이미지/상태 저장 ─────────────────────────────────────────
                save_batch_results(
                    idxes, batch_valid_size,
                    batch_reward_i, batch_repetition,
                    result, rendered, jax.device_get(raw_rendered), last_states,
                    instruct_df=instruct_df,
                    h5=h5_store,
                )
                batch_elapsed = time.time() - batch_start_time
                logger.debug(
                    f"[Batch {batch_i+1}/{n_batches}] elapsed: {batch_elapsed:.1f}s  "
                    f"(cumulative: {time.time() - loop_start_time:.1f}s)"
                )
                pbar.update(1)


        # ── 결과 DataFrame 구성 ───────────────────────────────────────────────
        total_elapsed = time.time() - loop_start_time
        logger.info(
            f"[Eval] Done: {n_batches} batches / {n_rows} samples  "
            f"total: {total_elapsed:.1f}s  "
            f"(avg per batch: {total_elapsed/n_batches:.1f}s)"
        )

        df_ctrl_sim = instruct_df.iloc[eval_batches].copy()
        df_ctrl_sim['seed'] = repetitions
        df_ctrl_sim = df_ctrl_sim.reset_index(drop=True)

        ##################################
        # Append the controllability loss#
        ##################################
        losses_arr = np.stack(losses, axis=0).reshape(-1)[:n_rows]
        losses_s0_arr = np.stack(losses_s0, axis=0).reshape(-1)[:n_rows]

        # features: list of (n_envs, feat_dim) → (n_rows, feat_dim)
        features_stacked = np.stack(features, axis=0)          # (n_batches, n_envs, feat_dim)
        actual_feat_dim = features_stacked.shape[-1]
        features_arr = features_stacked.reshape(-1, actual_feat_dim)[:n_rows]  # (n_rows, feat_dim)

        features_s0_stacked = np.stack(features_s0, axis=0)
        features_s0_arr = features_s0_stacked.reshape(-1, actual_feat_dim)[:n_rows]

        df_ctrl_sim['loss'] = losses_arr
        df_ctrl_sim['loss_s0'] = losses_s0_arr

        # condition_* 컬럼 수만큼 feat 컬럼 생성 (e.g. feat_region, feat_plength, ...)
        cond_cols = [c for c in df_ctrl_sim.columns if c.startswith('condition_')]
        n_cond = len(cond_cols)

        # reward_enum에 해당하는 feat 컬럼에만 측정값을 채우고 나머지는 NaN
        reward_enum_arr = df_ctrl_sim['reward_enum'].values  # (n_rows,)
        feat_df = pd.DataFrame(
            np.full((n_rows, n_cond), np.nan),
            columns=[f"feat_{i}" for i in range(n_cond)],
        )
        feat_s0_df = pd.DataFrame(
            np.full((n_rows, n_cond), np.nan),
            columns=[f"feat_{i}_s0" for i in range(n_cond)],
        )
        for col_i in range(min(n_cond, actual_feat_dim)):
            mask = (reward_enum_arr == col_i)   # reward_enum은 0-based
            feat_df.loc[mask, f"feat_{col_i}"] = features_arr[mask, col_i]
            feat_s0_df.loc[mask, f"feat_{col_i}_s0"] = features_s0_arr[mask, col_i]

        df_ctrl_sim = pd.concat([df_ctrl_sim.reset_index(drop=True), feat_df, feat_s0_df], axis=1)

        # progress 측정: condition_* (cont_value), feat_*, feat_*_s0 → progress_*
        df_ctrl_sim = ProgressWrapper(n_cond=n_cond).run(df_ctrl_sim)

        ##################################
        # Similarity scores              #
        ##################################

        if config.vit_score:
            vit_scores = ViTScoreWrapper(config).run(
                instruct_df=instruct_df, n_eps=n_eps, gt_images=gt_images
            )
            df_ctrl_sim['vit_score'] = vit_scores

        if config.tpkldiv:
            tpkl_scores = TPKLWrapper(config).run(
                instruct_df=instruct_df, n_eps=n_eps, gt_levels=gt_levels
            )
            df_ctrl_sim['tpkldiv'] = tpkl_scores

        # ── wandb / CSV 출력 ──────────────────────────────────────────────────
        ctrl_sim_path = join(config.eval_dir, "ctrl_sim.csv")
        df_ctrl_sim.to_csv(ctrl_sim_path, index=False)
        logger.info("[Eval] Saved ctrl_sim → %s", ctrl_sim_path)

        ##################################
        # Diversity scores               #
        ##################################

        # ── 후처리 메트릭 ─────────────────────────────────────────────────────
        if config.diversity:
            diversity_df = DiversityWrapper(config).run(instruct_df=instruct_df, n_eps=n_eps)
            diversity_path = join(config.eval_dir, "diversity.csv")
            diversity_df.to_csv(diversity_path, index=False)
            logger.info("[Eval] Saved diversity → %s", diversity_path)
        else:
            diversity_df = None

        # ── row_i 단위 mean DataFrame 생성 ───────────────────────────────────
        mean_cols = (['progress'] if 'progress' in df_ctrl_sim.columns else [])
        if 'vit_score' in df_ctrl_sim.columns:
            mean_cols.append('vit_score')
        if 'tpkldiv' in df_ctrl_sim.columns:
            mean_cols.append('tpkldiv')

        # row_i 기준 IQR-trimmed mean (seed 축 집계)
        meta_cols = ['row_i', 'game', 'instruction', 'reward_enum']
        df_results = df_ctrl_sim.groupby('row_i', sort=True)[mean_cols].agg(iqr_mean).reset_index()
        # 메타데이터 병합 (row_i당 첫 번째 행 사용)
        meta_df = df_ctrl_sim[meta_cols].drop_duplicates(subset='row_i').reset_index(drop=True)
        df_results = meta_df.merge(df_results, on='row_i')

        # diversity 있으면 옆에 붙이기
        if diversity_df is not None:
            df_results = df_results.merge(
                diversity_df[['row_i', 'diversity']],
                on='row_i',
                how='left',
            )

        results_path = join(config.eval_dir, "results.csv")
        df_results.to_csv(results_path, index=False)
        logger.info("[Eval] Saved results → %s", results_path)

        # ── 전체 mean 요약 CSV 생성 및 wandb 업로드 ─────────────────────────────
        summary_metric_cols = [c for c in ['progress', 'vit_score', 'tpkldiv', 'diversity'] if c in df_results.columns]
        if summary_metric_cols:
            summary_series = df_results[summary_metric_cols].mean()
            df_summary = summary_series.reset_index()
            df_summary.columns = ['metric', 'mean']
            summary_path = join(config.eval_dir, "summary.csv")
            df_summary.to_csv(summary_path, index=False)
            logger.info("[Eval] Saved summary → %s", summary_path)

        if wandb.run:
            log_dict = {
                'ctrl_sim_tb': wandb.Table(dataframe=df_ctrl_sim),
                'results_tb': wandb.Table(dataframe=df_results),
            }
            if diversity_df is not None:
                log_dict['diversity_tb'] = wandb.Table(dataframe=diversity_df)
            if summary_metric_cols:
                log_dict.update({row['metric']: row['mean'] for _, row in df_summary.iterrows()})
            wandb.log(log_dict)

            # ── 샘플 이미지 wandb 업로드 (N개, 조건 다양성 확보) ────────────────
            wandb_images = sample_wandb_images(
                df_ctrl_sim, eval_rendered, n_rows,
                n_samples=getattr(config, 'n_sample_images', 10),
            )
            if wandb_images:
                wandb.log({f'images/{i}': img for i, img in enumerate(wandb_images)})
                logger.info("[Eval] Uploaded %d sample images → wandb", len(wandb_images))

            # ── CSV artifact 업로드 ───────────────────────────────────────────
            csv_artifact = wandb.Artifact(name="eval_csv", type="dataset")
            csv_artifact.add_file(ctrl_sim_path, name="ctrl_sim.csv")
            csv_artifact.add_file(results_path, name="results.csv")
            if diversity_df is not None:
                csv_artifact.add_file(diversity_path, name="diversity.csv")
            if summary_metric_cols:
                csv_artifact.add_file(summary_path, name="summary.csv")
            wandb.log_artifact(csv_artifact)
            logger.info("[Eval] Uploaded CSV files → wandb artifact (eval_csv)")

            h5_path = join(config.eval_dir, "eval.h5")
            if os.path.exists(h5_path):
                artifact = wandb.Artifact(name="eval_h5", type="dataset")
                artifact.add_file(h5_path, name="eval.h5")
                wandb.log_artifact(artifact)
                logger.info("[Eval] Uploaded eval.h5 → wandb artifact")

        return losses_arr

    return eval_fn

