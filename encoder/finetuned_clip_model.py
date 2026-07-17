"""
encoder/finetuned_clip_model.py
================================
HuggingFace pretrained CLIP  text for text of  (image, text) data to  text abovetext
trainable text. `encoder.pretrained_clip_model`  and  parameter text(name/structure)
textwalltext sametext also text text, fine-tune result checkpoint  as-is existing RL training
pipeline(`apply_encoder_params`) as  inject text text text.

text text
------
- `jax.lax.stop_gradient` remove → text CLIP parameter in  text text
- text text text name·shape text `pretrained_clip_model`  and  same
"""
from typing import Dict

import jax
import jax.numpy as jnp
from flax import linen as nn
from transformers import FlaxCLIPModel

from conf.config import EncoderConfig
from encoder.pretrained_clip_model import ContrastiveModule  # as-is reuse


class TrainablePretrainedTextEncoder(nn.Module):
    """`PretrainedTextEncoder`  and  same structure text stop_gradient none."""
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
    """`PretrainedImageEncoder`  and  same structure text stop_gradient none."""
    pretrained_state_encoder: nn.Module

    @nn.compact
    def __call__(self, pixel_values):
        x = self.pretrained_state_encoder(pixel_values).pooler_output
        x = nn.Dense(512, name="pretrained_image_projection", use_bias=False)(x)
        return x


def get_finetuned_clip_encoder(config: EncoderConfig):
    """training for  ContrastiveModule + HF pretrained initial parameter dict  return.

    return text (·structure)  `get_pretrained_clip_encoder`  and  sametext to
    `train_clip.py:get_train_state`  of  `replace_params`  to text  as-is text for text text text.
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

