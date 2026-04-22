"""
runner.py
=========
make_eval — 환경 루프 core.

make_eval(config, restored_ckpt, encoder_params, *, inject_obs_fn=None)
  → callable(rng) → losses (np.ndarray)
"""
import logging
import math
import os
import time

import cv2
import imageio
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pandas as pd
import wandb
from os.path import join
from flax.training.train_state import TrainState
from tqdm import tqdm

from envs.pcgrl_env import gen_dummy_queued_state
from evaluator import get_reward_batch
from instruct_rl import NUM_FEATURES, FEATURE_NAMES
from instruct_rl.dataclass import Instruct
from instruct_rl.eval.embedding import prepare_instruct
from instruct_rl.eval.metrics import run_post_eval
from instruct_rl.evaluate import get_loss_batch
from instruct_rl.utils.path_utils import gymnax_pcgrl_make, init_network
from purejaxrl.experimental.s5.wrappers import LogWrapper
from instruct_rl.utils.checkpointer import init_checkpointer
from purejaxrl.structures import Transition, RunnerState

logger = logging.getLogger(__name__)


def make_eval(config, restored_ckpt, encoder_params, *, inject_obs_fn=None, eval_inst=None, eval_inst_meta=None):
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

        train_state = TrainState.create(
            apply_fn=network.apply, params=network_params, tx=tx
        )

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
        if eval_inst is not None:
            # train과 동일한 MultiGameDataset test split 사용
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
            logger.info(f"[Dataset mode] eval instruct: {n_inst} samples")
        else:
            # CSV 모드 (NLP/CLIP 기반 모델)
            from os.path import abspath, dirname
            csv_path = abspath(join(dirname(__file__), "..", "..", "instruct",
                                    f"{config.eval_instruct_csv}.csv"))
            instruct_df = pd.read_csv(csv_path).reset_index().rename(columns={'index': 'row_i'})
            instruct_df.to_csv(join(config.eval_dir, 'input.csv'), index=False)
            instruct = prepare_instruct(config, network, runner_state, instruct_df, init_x)
            logger.info(f"[CSV mode] eval instruct: {len(instruct_df)} samples")

        # ── 배치 구성 ─────────────────────────────────────────────────────────
        n_envs = config.n_envs
        n_eps = config.n_eps
        eval_batches = jnp.array(
            sorted(np.tile(list(range(len(instruct_df))), n_eps))
        )
        repetitions = np.tile(list(range(1, n_eps + 1)), len(instruct_df))
        n_rows = len(eval_batches)

        if len(eval_batches) != len(repetitions):
            raise Exception(
                f"eval_batches/repetitions length mismatch: "
                f"{len(eval_batches)} != {len(repetitions)}"
            )

        n_batches = math.ceil(n_rows / n_envs)
        losses, values, features = [], [], []
        eval_rendered = []
        loop_start_time = time.time()

        # ── 평가 루프 시작 전 요약 로그 ───────────────────────────────────────
        logger.info(
            "[Eval Loop] total_items=%d  (samples=%d × n_eps=%d)  "
            "batch_size(n_envs)=%d  n_batches=%d",
            n_rows, len(instruct_df), n_eps, n_envs, n_batches,
        )

        # ── 평가 루프 ─────────────────────────────────────────────────────────
        from instruct_rl.eval.hdf5_store import open_eval_store
        with open_eval_store(config.eval_dir, mode="a") as h5_store, \
             tqdm(total=n_batches, desc="Evaluating Batches") as pbar:
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

                    cond_reward = get_reward_batch(
                        batch_instruct.reward_i,
                        batch_instruct.condition,
                        state.env_state.env_map,
                        next_state.env_state.env_map,
                    )
                    reward = jnp.where(done, reward_env, cond_reward)

                    next_state = next_state.replace(
                        returned_episode_returns=(
                            next_state.returned_episode_returns - reward_env + reward
                        )
                    )
                    info["returned_episode_returns"] = next_state.returned_episode_returns

                    transition = Transition(
                        done, action, value, reward, log_prob, obsv, info,
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
                    return result, rendered, rendered_raw, last_states

                rng_eval = jax.random.PRNGKey(30)
                result, rendered, raw_rendered, last_states = run_eval_step(
                    rng_eval, init_obs, init_state, done
                )

                result = jax.device_get(result)
                losses.append(result.loss)
                values.append(result.value)
                features.append(result.feature)
                eval_rendered.append(jax.device_get(raw_rendered))
                rendered = jax.device_get(rendered)

                # ── 이미지/상태 저장 ─────────────────────────────────────────
                _save_batch_results(
                    config, idxes, batch_valid_size,
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
        losses_arr = np.stack(losses, axis=0).reshape(-1)[:n_rows]

        # feature 차원을 실제 결과에서 동적으로 결정 (get_loss_batch는 5개 반환)
        features_stacked = np.stack(features, axis=0)          # (n_batches, n_envs, feat_dim)
        actual_feat_dim = features_stacked.shape[-1]
        features_arr = features_stacked.reshape(-1, actual_feat_dim)[:n_rows]

        # FEATURE_NAMES 길이가 실제 feat_dim보다 클 수 있으므로 맞춰서 자름
        feat_col_names = FEATURE_NAMES[:actual_feat_dim]
        if len(feat_col_names) < actual_feat_dim:
            feat_col_names = feat_col_names + [
                f"feat_{i}" for i in range(len(feat_col_names), actual_feat_dim)
            ]

        df_output = instruct_df.iloc[eval_batches].copy()
        df_output = df_output.loc[:, ~df_output.columns.str.startswith("embed")]
        df_output['seed'] = repetitions
        df_output['loss'] = losses_arr
        df_output = df_output.reset_index()

        feat_df = pd.DataFrame(
            features_arr, columns=[f"feat_{n}" for n in feat_col_names]
        ).reset_index()
        df_output = pd.concat([df_output, feat_df], axis=1)

        # ── 후처리 메트릭 ─────────────────────────────────────────────────────
        df_output = run_post_eval(
            config, instruct_df, df_output, eval_rendered, n_rows, n_eps
        )

        # ── wandb / CSV 출력 ──────────────────────────────────────────────────
        df_output.to_csv(f"{config.eval_dir}/loss.csv", index=False)

        if wandb.run:
            mean_loss = (
                df_output.groupby('reward_enum')['loss']
                .mean()
                .reset_index()
            )
            wandb.log({
                f"Loss/{int(row['reward_enum'])}": row['loss']
                for _, row in mean_loss.iterrows()
            })
            wandb.log({'raw': wandb.Table(dataframe=df_output)})

        if wandb.run and config.flush:
            for row_i, row in instruct_df.iterrows():
                game      = row.get('game', 'unknown')
                re_val    = int(row.get('reward_enum', row_i))
                folder_name = f"{game}_re{re_val}_{int(row_i):04d}"
                os.system(f"rm -r {config.eval_dir}/{folder_name}")

        return losses_arr

    return eval_fn


# ── 이미지 저장 헬퍼 ──────────────────────────────────────────────────────────

def _save_batch_results(
    config, idxes, batch_valid_size,
    batch_reward_i, batch_repetition,
    result, rendered, raw_rendered, last_states,
    instruct_df=None,
    h5=None,
):
    from instruct_rl.eval.hdf5_store import write_sample

    for idx, (row_i, reward_i, repeat_i, feature, state) in enumerate(zip(
        idxes,
        batch_reward_i[:batch_valid_size],
        batch_repetition[:batch_valid_size],
        result.feature[:batch_valid_size],
        last_states.env_state.env_map[0, :][:batch_valid_size],
    )):
        # 폴더명: {game}_re{re}_{row_i:04d}  (메타 없으면 기존 reward_{row_i} 유지)
        if instruct_df is not None and row_i < len(instruct_df):
            meta = instruct_df.iloc[int(row_i)]
            game   = str(meta.get('game', 'unknown'))
            re_val = int(meta.get('reward_enum', int(reward_i[0]) if hasattr(reward_i, '__len__') else int(reward_i)))
            folder_name = f"{game}_re{re_val}_{int(row_i):04d}"
        else:
            folder_name = f"reward_{row_i}"

        # ── 프레임 배열 조합 (RGBA→RGB, 텍스트 오버레이) ──────────────────
        frames_rgb = []
        for frame in rendered[idx]:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
            task_text = _build_task_text(reward_i, feature)
            frame = cv2.putText(
                frame, task_text, (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA,
            )
            frames_rgb.append(frame)
        frames_rgb = np.array(frames_rgb, dtype=np.uint8)  # (n_frames, H, W, 3)

        # ── HDF5 저장 ─────────────────────────────────────────────────────
        if h5 is not None:
            write_sample(h5, folder_name, int(repeat_i), frames_rgb, np.array(state))

        # ── wandb 로깅 (메모리에서 직접) ──────────────────────────────────
        if wandb.run:
            wandb.log({f"RawImage/{folder_name}/seed_{repeat_i}": wandb.Image(raw_rendered[idx])})
            for frame in frames_rgb:
                wandb.log({f"Image/{folder_name}/seed_{repeat_i}": wandb.Image(frame)})


def _build_task_text(reward_i, feature) -> str:
    labels = {1: f"RG: {int(feature[0])} | ",
              2: f"PL: {int(feature[1])} | ",
              3: f"WC: {int(feature[2])} | ",
              4: f"BC: {int(feature[3])} | ",
              5: "BD | "}
    return "".join(v for k, v in labels.items() if k in reward_i)


