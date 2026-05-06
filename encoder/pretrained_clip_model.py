from transformers import FlaxCLIPModel, CLIPConfig
from transformers.models.clip.modeling_flax_clip import FlaxCLIPTextTransformer, FlaxCLIPVisionTransformer
from typing import Dict

import jax
import jax.numpy as jnp
from flax import linen as nn

from conf.config import EncoderConfig


class PretrainedTextEncoder(nn.Module):
    pretrained_text_encoder: nn.Module

    @nn.compact
    def __call__(self, input_ids, attention_mask, position_ids):
        x = self.pretrained_text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
        ).pooler_output
        x = nn.Dense(512, name="pretrained_text_projection", use_bias=False)(x)
        x = jax.lax.stop_gradient(x)
        return x


class PretrainedImageEncoder(nn.Module):
    pretrained_state_encoder: nn.Module

    @nn.compact
    def __call__(self, pixel_values):
        x = self.pretrained_state_encoder(pixel_values).pooler_output
        x = nn.Dense(512, name="pretrained_image_projection", use_bias=False)(x)
        x = jax.lax.stop_gradient(x)
        return x


class ContrastiveModule(nn.Module):
    encoders: Dict[str, nn.Module]

    def setup(self):
        self.text_state_temperature = self.param(
            "text_state_temperature", nn.initializers.constant(jnp.log(0.07)), ()
        )

    def encode_text(
        self, input_ids: jnp.ndarray, attention_mask: jnp.ndarray, position_ids: jnp.ndarray
    ) -> jnp.ndarray:
        x = self.encoders["text"](input_ids, attention_mask, position_ids)
        x = x / (jnp.linalg.norm(x, axis=-1, keepdims=True) + 1e-6)
        return x

    def encode_state(self, pixel_values: jnp.ndarray) -> jnp.ndarray:
        x = self.encoders["state"](pixel_values)
        x = x / (jnp.linalg.norm(x, axis=-1, keepdims=True) + 1e-6)
        return x

    @nn.compact
    def __call__(
        self,
        input_ids: jnp.ndarray = None,
        attention_mask: jnp.ndarray = None,
        pixel_values: jnp.ndarray = None,
        mode: str = "text_state",
        **kwargs,
    ):
        output_dict = dict()
        modes = mode.split("_")

        if "state" in modes:
            state_embed = self.encode_state(pixel_values)
            output_dict["state_embed"] = state_embed
            output_dict["text_state_temperature"] = self.text_state_temperature

        if "text" in modes:
            batch_size, seq_len = input_ids.shape
            position_ids = jnp.arange(seq_len)[None, :].repeat(batch_size, axis=0)
            text_embed = self.encode_text(input_ids, attention_mask, position_ids)
            output_dict["text_embed"] = text_embed
            output_dict["text_state_temperature"] = self.text_state_temperature

        output_dict["text_state_temperature"] = self.text_state_temperature
        return output_dict


def get_pretrained_clip_encoder(config: EncoderConfig):
    """
    Pretrained CLIP encoder (inference only, no gradient).
    """
    pretrained_params = {}

    clip = FlaxCLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clip, clip_variables = clip.module, {"params": clip.params}

    text_model, text_model_vars = clip.bind(clip_variables).text_model.unbind()
    pretrained_params["pretrained_text_encoder"] = text_model_vars["params"]
    pretrained_params["pretrained_text_projection"] = clip_variables["params"]["text_projection"]

    vision_model, vision_model_vars = clip.bind(clip_variables).vision_model.unbind()

    text_encoder_def = PretrainedTextEncoder(text_model)
    state_encoder_def = PretrainedImageEncoder(vision_model)

    mode = "text"

    if config.state:
        encoder_dict = dict(
            state=state_encoder_def,
            text=text_encoder_def,
        )
        mode += "_state"
    else:
        encoder_dict = dict(
            text=text_encoder_def,
        )

    encoder = ContrastiveModule(encoders=encoder_dict)

    return encoder, pretrained_params

