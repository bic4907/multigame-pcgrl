"""
instruct_rl/utils/callbacks.py
==============================
학습/평가 로그 콜백 함수 모음.
train_cpcgrl.py 에서 분리.
"""
import logging
from os.path import basename
from timeit import default_timer as timer

import jax.numpy as jnp
import pandas as pd

from utils import render_callback

logger = logging.getLogger(basename(__file__))


def log_callback(metric, loss_mean, return_info, steps_prev_complete, config, writer, train_start_time):
    timesteps = metric["timestep"][metric["returned_episode"]] * config.n_envs
    return_values = metric["returned_episode_returns"][metric["returned_episode"]]

    if len(timesteps) > 0:
        t = timesteps[-1].item()
        ep_return_mean = return_values.mean()
        ep_return_max = return_values.max()
        ep_return_min = return_values.min()

        ep_length = metric["returned_episode_lengths"][
            metric["returned_episode"]
        ].mean()
        fps = (t - steps_prev_complete) / (timer() - train_start_time)

        prefix = (
            f"Iteration_{config.current_iteration}/train/"
            if config.current_iteration > 0
            else ""
        )

        metric = {
            f"Train/{prefix}ep_return": ep_return_mean,
            f"Train/{prefix}ep_return_max": ep_return_max,
            f"Train/{prefix}ep_return_min": ep_return_min,
            f"Train/{prefix}ep_length": ep_length,
            f"Train/{prefix}fps": fps,
            f"Train/{prefix}total_loss": loss_mean.total_loss,
            f"Train/{prefix}value_loss": loss_mean.value_loss,
            f"Train/{prefix}actor_loss": loss_mean.actor_loss,
            f"Train/{prefix}entropy": loss_mean.entropy,
            f"Train/{prefix}cond_return": jnp.mean(return_info.cond_return),
            f"Train/{prefix}sim_return": jnp.mean(return_info.sim_return),
            f"Train/{prefix}coef_sim_return": jnp.mean(return_info.coef_sim_return),
            f"Train/{prefix}total_return": jnp.mean(return_info.total_return),
            f"Train/Step": t,
        }

        # log metrics
        writer.log(metric, t)

        print(
            f"[train] global step={t}; episodic return mean: {ep_return_mean:.02f}, "
            + f"max: {ep_return_max:.02f}, min: {ep_return_min:.02f}, fps: {fps:.02f}, "
            + f"loss: {loss_mean.total_loss:.02f}, "
            + f"actor_loss: {loss_mean.actor_loss:.02f}, "
            + f"value_loss: {loss_mean.value_loss:.02f}, "
            + f"entropy: {loss_mean.entropy:.02f}"
        )


def eval_callback(
    eval_metric,
    train_metric,
    states,
    frames,
    steps_prev_complete,
    config,
    writer,
    train_start_time,
):
    timesteps = (
        train_metric["timestep"][train_metric["returned_episode"]] * config.n_envs
    )
    return_values = eval_metric["returned_episode_returns"][
        eval_metric["returned_episode"]
    ]

    if len(timesteps) > 0:
        t = timesteps[-1].item()
        ep_return_mean = return_values.mean()
        ep_return_max = return_values.max()
        ep_return_min = return_values.min()

        ep_length = eval_metric["returned_episode_lengths"][
            eval_metric["returned_episode"]
        ].mean()
        prefix = (
            f"Iteration_{config.current_iteration}/train/"
            if config.current_iteration > 0
            else ""
        )

        metric = {
            f"Eval/{prefix}ep_return": ep_return_mean,
            f"Eval/{prefix}ep_return_max": ep_return_max,
            f"Eval/{prefix}ep_return_min": ep_return_min,
            f"Eval/{prefix}ep_length": ep_length,
            "Train/Step": t,
        }

        # log metrics
        writer.log(metric, t)
        render_callback(
            frames=frames,
            states=states,
            video_dir=config._vid_dir,
            image_dir=config._img_dir,
            numpy_dir=config._numpy_dir,
            traj_dir=config._traj_dir,
            logger=logger,
            config=config,
            t=t,
        )

        print(
            f"[eval] global step={t}; episodic return mean: {ep_return_mean} "
            + f"max: {ep_return_max}, min: {ep_return_min}"
        )


def loss_callback(metric, loss, config, writer):
    timesteps = metric["timestep"][metric["returned_episode"]] * config.n_envs

    if len(timesteps) > 0:
        t = timesteps[-1].item()

        result_df = pd.DataFrame({"reward_enum": loss.reward_enum, "loss": loss.loss})

        mean_loss = result_df.groupby("reward_enum").agg({"loss": ["mean"]})
        mean_loss.columns = mean_loss.columns.droplevel(0)
        mean_loss = mean_loss.reset_index()

        dict_loss = dict()
        for _, row in mean_loss.iterrows():
            reward_enum, mean = row
            dict_loss[f"Loss/{str(int(reward_enum))}"] = mean

        writer.log(dict_loss, t)
        dict_str = ", ".join([f"{k}: {v}" for k, v in dict_loss.items()])
        print(f"[eval] global step={t}; loss: {dict_str}")

