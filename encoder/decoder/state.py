from __future__ import annotations

import logging

import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState

from encoder.clip_model import get_cnnclip_decoder_encoder
from encoder.schedular import create_learning_rate_fn


def get_train_state(config, rng_key, cond_norm_min=None, cond_norm_max=None):
    lr_schedular = create_learning_rate_fn(config, config.lr, config.steps_per_epoch)

    def create_train_state(module, rng_key, pretrained_params):
        def replace_params(params, key, replacement):
            for k in params.keys():
                if k == key:
                    params[k] = replacement
                    logging.info(f"replaced {key} in params")
                    return
                if isinstance(params[k], type(params)):
                    replace_params(params[k], key, replacement)

        rng_key, init_rng = jax.random.split(rng_key)
        input_ids = jnp.ones((1, config.encoder.token_max_len), dtype=jnp.int32)
        attention_mask = jnp.ones((1, config.encoder.token_max_len), dtype=jnp.int32)

        if config.encoder.model == "cnnclip":
            pixel_values = jnp.ones(
                (1, 16, 16, config.clip_input_channel), dtype=jnp.float32
            )
        elif config.encoder.model == "clip":
            pixel_values = jnp.ones(
                (1, 224, 224, config.clip_input_channel), dtype=jnp.float32
            )
        else:
            raise NotImplementedError(f"Model not implemented: {config.encoder.model}")

        variables = module.init(
            init_rng,
            input_ids,
            attention_mask,
            pixel_values,
            reward_enum=jnp.zeros((1,), dtype=jnp.int32),
            mode=config.encoder.mode,
            training=False,
        )

        if pretrained_params is not None:
            for key in pretrained_params:
                replace_params(variables, key, pretrained_params[key])

        def _create_mask(variables):
            import jax.tree_util as jtu

            flat = jtu.tree_map(lambda _: True, variables.get("params", {}))
            frozen = jtu.tree_map(lambda _: False, variables.get("norm_stats", {}))
            return {"params": flat, "norm_stats": frozen}

        mask = _create_mask(variables)
        tx = optax.masked(
            optax.adamw(learning_rate=lr_schedular, weight_decay=config.weight_decay),
            mask,
        )
        return TrainState.create(apply_fn=module.apply, params=variables, tx=tx)

    module, pretrained_params = get_cnnclip_decoder_encoder(
        config.encoder,
        decoder_config=config.decoder,
        cond_norm_min=cond_norm_min,
        cond_norm_max=cond_norm_max,
        RL_training=False,
    )

    state = create_train_state(module, rng_key=rng_key, pretrained_params=pretrained_params)
    return state, lr_schedular
