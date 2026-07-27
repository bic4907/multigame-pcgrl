"""
encoder/finetuned_clip_model.py
================================
Trainable variant for fine-tuning HuggingFace pretrained CLIP on user-provided
(image, text) data. Its parameter-tree names and structure exactly match
`encoder.pretrained_clip_model`, so fine-tuned checkpoints can be injected
directly into the existing RL pipeline through `apply_encoder_params`.

Differences
------
- Removes `jax.lax.stop_gradient`, allowing gradients through all CLIP parameters
- All other module names and shapes match `pretrained_clip_model`
"""
from typing import Dict

import jax
import jax.numpy as jnp
from flax import linen as nn
from transformers import FlaxCLIPModel

from conf.config import EncoderConfig
from encoder.pretrained_clip_model import ContrastiveModule  # as-is reuse


class TrainablePretrainedTextEncoder(nn.Module):
    """Same structure as `PretrainedTextEncoder`, without stop_gradient."""
    pretrained_text_encoder: nn.Module

    @nn.compact
    def __call__(self, input_ids, attention_mask, position_ids):
        x = self.pretrained_text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
        ).pooler_output
        x = nn.Dense(512, name="pretrained_text_projection", use_bias=False)(x)
        return x


class TrainablePretrainedImageEncoder(nn.Module):
    """Same structure as `PretrainedImageEncoder`, without stop_gradient."""
    pretrained_state_encoder: nn.Module

    @nn.compact
    def __call__(self, pixel_values):
        x = self.pretrained_state_encoder(pixel_values).pooler_output
        x = nn.Dense(512, name="pretrained_image_projection", use_bias=False)(x)
        return x


def get_finetuned_clip_encoder(config: EncoderConfig):
    """training for  ContrastiveModule + HF pretrained initial parameter dict  return.

    The returned format and structure match `get_pretrained_clip_encoder`, so
    `train_clip.py:get_train_state` can reuse its `replace_params` logic.
    """
    pretrained_params = {}

    clip = FlaxCLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_module, clip_variables = clip.module, {"params": clip.params}

    text_model, text_model_vars = clip_module.bind(clip_variables).text_model.unbind()
    pretrained_params["pretrained_text_encoder"] = text_model_vars["params"]
    pretrained_params["pretrained_text_projection"] = clip_variables["params"]["text_projection"]

    vision_model, vision_model_vars = clip_module.bind(clip_variables).vision_model.unbind()
    pretrained_params["pretrained_state_encoder"] = vision_model_vars["params"]
    pretrained_params["pretrained_image_projection"] = clip_variables["params"]["visual_projection"]

    text_encoder_def = TrainablePretrainedTextEncoder(text_model)
    state_encoder_def = TrainablePretrainedImageEncoder(vision_model)

    if config.state:
        encoder_dict = dict(state=state_encoder_def, text=text_encoder_def)
    else:
        encoder_dict = dict(text=text_encoder_def)

    encoder = ContrastiveModule(encoders=encoder_dict)
    return encoder, pretrained_params
