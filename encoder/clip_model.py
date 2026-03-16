
from transformers import FlaxCLIPModel, CLIPConfig
from transformers.models.clip.modeling_clip import CLIPEncoder
from transformers.models.clip.modeling_flax_clip import FlaxCLIPTextTransformer, FlaxCLIPVisionTransformer
from typing import Dict

import jax
import jax.numpy as jnp
from flax import linen as nn

from encoder.data import CLIPContrastiveBatch
from conf.config import CLIPTrainConfig, EncoderConfig

class PretrainedTextEncoder(nn.Module):
    pretrained_text_encoder: nn.Module
    freeze_encoder: bool = False
    projection_dim: int = None

    @nn.compact
    def __call__(self, input_ids, attention_mask, position_ids):
        x = self.pretrained_text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids
            ).pooler_output
        x = nn.Dense(
                512, 
                name="pretrained_text_projection", 
                use_bias=False
                )(x)
        if self.freeze_encoder:
            x = jax.lax.stop_gradient(x)
        if self.projection_dim is not None:
            x = nn.Dense(
                self.projection_dim, 
                name="final_text_projection", 
                kernel_init=jax.nn.initializers.normal(0.02),
                use_bias=False
                )(x)
        return x



class PretrainedImageEncoder(nn.Module):
    pretrained_state_encoder: nn.Module
    freeze_encoder: bool = False
    projection_dim: int = None

    @nn.compact
    def __call__(self, pixel_values):
        x = self.pretrained_state_encoder(pixel_values).pooler_output
        x = nn.Dense(
                512, 
                name="pretrained_image_projection", 
                use_bias=False
                )(x)
        if self.freeze_encoder:
            x = jax.lax.stop_gradient(x)
        if self.projection_dim is not None:
            x = nn.Dense(
                self.projection_dim, 
                name="final_image_projection", 
                kernel_init=jax.nn.initializers.normal(0.02),
                use_bias=False)(x)
        return x


class SqueezeExcite(nn.Module):
    reduction: int = 4

    @nn.compact
    def __call__(self, x):
        c = x.shape[-1]
        s = jnp.mean(x, axis=(1, 2), keepdims=True)
        s = nn.Dense(c // self.reduction, use_bias=False)(s)
        s = nn.gelu(s)
        s = nn.Dense(c, use_bias=False)(s)
        s = nn.sigmoid(s)
        return x * s


class ResBlock(nn.Module):
    out_ch: int
    drop_rate: float = 0.0
    use_se: bool = False

    @nn.compact
    def __call__(self, x, training: bool):
        residual = x
        ch = x.shape[-1]

        x = nn.Conv(ch,(3,3), padding='SAME', feature_group_count=ch)(x)
        x = nn.gelu(x)
        x = nn.LayerNorm()(x)

        x = nn.Conv(self.out_ch, (1,1), use_bias=False)(x)

        if self.use_se:
            x = SqueezeExcite()(x)
        if ch != self.out_ch:
            residual = nn.Conv(self.out_ch, (1,1), use_bias=False)(residual)

        if self.drop_rate > 0.0 and training:
            x = nn.Dropout(self.drop_rate)(x, deterministic=not training)

        x = x + residual
        return x


class CNNResMapEncoder(nn.Module):
    projection_dim: int = None
    drop_rate: float = 0.0

    @nn.compact
    def __call__(self, pixel_values, training:bool):
        x = nn.Conv(64, (3, 3), padding="SAME")(pixel_values)
        x = nn.gelu(x)
        x = nn.LayerNorm()(x)

        x = ResBlock(128, drop_rate=self.drop_rate, use_se=True)(x, training)

        x = nn.Conv(128, (3,3), strides=(2,2), padding='SAME')(x)
        x = nn.gelu(x)
        x = nn.LayerNorm()(x)

        x = ResBlock(256, drop_rate=self.drop_rate, use_se=True)(x, training)

        x = jnp.mean(x, axis=(1,2))

        x = nn.Dense(256)(x)
        x = nn.gelu(x)
        x = nn.LayerNorm()(x)
        x = nn.Dropout(self.drop_rate)(x, deterministic=not training)
        x = nn.Dense(self.projection_dim, use_bias=False)(x)
        return x


class CNNResSketchEncoder(nn.Module):
    projection_dim: int = None
    drop_rate: float = 0.0

    @nn.compact
    def __call__(self, pixel_values, training: bool):
        # Input: (224, 224, 3)
        x = nn.Conv(32, (7, 7), strides=(2, 2), padding="SAME")(pixel_values)  # (224 → 112)
        x = nn.gelu(x)
        x = nn.LayerNorm()(x)

        x = ResBlock(32, drop_rate=self.drop_rate, use_se=True)(x, training)

        x = nn.Conv(64, (3, 3), strides=(2, 2), padding='SAME')(x)  # (112 → 56)
        x = nn.gelu(x)
        x = nn.LayerNorm()(x)
        x = ResBlock(64, drop_rate=self.drop_rate, use_se=True)(x, training)

        x = nn.Conv(128, (3, 3), strides=(2, 2), padding='SAME')(x)  # (56 → 28)
        x = nn.gelu(x)
        x = nn.LayerNorm()(x)
        x = ResBlock(128, drop_rate=self.drop_rate, use_se=True)(x, training)

        x = nn.Conv(256, (3, 3), strides=(2, 2), padding='SAME')(x)  # (28 → 14)
        x = nn.gelu(x)
        x = nn.LayerNorm()(x)
        x = ResBlock(256, drop_rate=self.drop_rate, use_se=True)(x, training)

        # Global Average Pooling: (B, 14, 14, 256) → (B, 256)
        x = jnp.mean(x, axis=(1, 2))

        # Projection Head
        x = nn.Dense(256)(x)
        x = nn.gelu(x)
        x = nn.LayerNorm()(x)
        x = nn.Dropout(self.drop_rate)(x, deterministic=not training)
        x = nn.Dense(self.projection_dim, use_bias=False)(x)
        return x


class ContrastiveModule(nn.Module):
    encoders: Dict[str, nn.Module]
    dropout_rate: float = 0.0

    def setup(self):
        self.text_state_temperature = self.param(
            "text_state_temperature", nn.initializers.constant(jnp.log(0.07)), ()
        )
        self.text_sketch_temperature = self.param(
            "text_sketch_temperature", nn.initializers.constant(jnp.log(0.07)), ()
        )
        self.state_sketch_temperature = self.param(
            "state_sketch_temperature", nn.initializers.constant(jnp.log(0.07)), ()
        )

    def encode_text(
            self, input_ids: jnp.ndarray, attention_mask: jnp.ndarray, position_ids: jnp.ndarray, training: bool
    ) -> jnp.ndarray:
        x = self.encoders["text"](input_ids, attention_mask, position_ids)
        # x = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(x)
        x = x / (jnp.linalg.norm(x, axis=-1, keepdims=True) + 1e-6)
        return x

    def encode_state(
            self, pixel_values: jnp.ndarray, training: bool
    ) -> jnp.ndarray:
        x = self.encoders["state"](pixel_values, training)
        x = x / (jnp.linalg.norm(x, axis=-1, keepdims=True) + 1e-6)
        return x

    def encode_sketch(self, sketch_img: jnp.ndarray, training: bool) -> jnp.ndarray:
        x = self.encoders["sketch"](sketch_img, training)
        x = x / (jnp.linalg.norm(x, axis=-1, keepdims=True) + 1e-6)
        return x

    @nn.compact
    def __call__(
            self,
            input_ids: jnp.ndarray = None,
            attention_mask: jnp.ndarray = None,
            pixel_values: jnp.ndarray = None,
            sketch_values: jnp.ndarray = None,
            mode: str = "text_state_sketch",
            training: bool = False,
    ):

        output_dict = dict()
        modes = mode.split("_")

        if "state" in modes:
            state_embed = self.encode_state(pixel_values, training)
            output_dict["state_embed"] = state_embed
            output_dict["text_state_temperature"] = self.text_state_temperature

        if "text" in modes:
            batch_size, seq_len = input_ids.shape
            position_ids = jnp.arange(seq_len)[None, :].repeat(batch_size, axis=0)

            text_embed = self.encode_text(input_ids, attention_mask, position_ids, training)

            output_dict["text_embed"] = text_embed
            output_dict["text_state_temperature"] = self.text_state_temperature

        if "sketch" in modes:
            sketch_embed = self.encode_sketch(sketch_values, training)
            output_dict["sketch_embed"] = sketch_embed

        output_dict['text_state_temperature'] = self.text_state_temperature
        output_dict['text_sketch_temperature'] = self.text_sketch_temperature
        output_dict['state_sketch_temperature'] = self.state_sketch_temperature

        return output_dict


def get_clip_encoder(config: EncoderConfig, RL_training: bool=True):
    """
    Pretrained CLIP encoder with text and image encoders.
    """
    pretrained_params = None
    if RL_training:
        clip_conf = CLIPConfig.from_pretrained("openai/clip-vit-base-patch32")
        text_model = FlaxCLIPTextTransformer(clip_conf.text_config)
        vision_model = FlaxCLIPVisionTransformer(clip_conf.vision_config)
        sketch_model = FlaxCLIPVisionTransformer(clip_conf.vision_config)
    else:
        pretrained_params = {}

        clip = FlaxCLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        clip, clip_variables = clip.module, {"params": clip.params}

        # Get text model
        text_model, text_model_vars = clip.bind(clip_variables).text_model.unbind()
        pretrained_params["pretrained_text_encoder"] = text_model_vars["params"]
        pretrained_params["pretrained_text_projection"] = clip_variables["params"]["text_projection"]

        # Get vision model
        vision_model, vision_model_vars = clip.bind(clip_variables).vision_model.unbind()

        # Get sketch model
        sketch_model, sketch_model_vars = clip.bind(clip_variables).vision_model.unbind()

    text_encoder_def = PretrainedTextEncoder(text_model, projection_dim=config.output_dim,
                                             freeze_encoder=config.freeze_text_enc)
    state_encoder_def = PretrainedImageEncoder(vision_model, projection_dim=config.output_dim,
                                               freeze_encoder=config.freeze_state_enc)
    sketch_encoder_def = PretrainedImageEncoder(sketch_model, projection_dim=config.output_dim,
                                                 freeze_encoder=config.freeze_sketch_enc)

    mode = "text"

    if config.sketch and config.state:
        encoder_dict = dict(
            state=state_encoder_def,
            text=text_encoder_def,
            sketch=sketch_encoder_def,
        )
        mode += "_state_sketch"
    elif config.sketch:
        encoder_dict = dict(
            text=text_encoder_def,
            sketch=sketch_encoder_def,
        )
        mode += "_sketch"
    elif config.state:
        encoder_dict = dict(
            state=state_encoder_def,
            text=text_encoder_def,
        )
        mode += "_state"
    else:
        encoder_dict = dict(
            text=text_encoder_def,
        )

    encoder = ContrastiveModule(
        encoders=encoder_dict,
        dropout_rate=config.dropout_rate,
        mode=mode
    )


    return encoder, pretrained_params


def get_cnnclip_encoder(config: EncoderConfig, RL_training: bool = True):
    """
    CNN-based CLIP encoder with text and state encoders.
    """
    state_encoder_def = CNNResMapEncoder(projection_dim=config.output_dim, drop_rate=config.dropout_rate)

    sketch_encoder_def = CNNResSketchEncoder(projection_dim=config.output_dim, drop_rate=config.dropout_rate)

    pretrained_params = None

    if RL_training:
        clip_conf = CLIPConfig.from_pretrained("openai/clip-vit-base-patch32")
        text_model = FlaxCLIPTextTransformer(clip_conf.text_config)

    else:
        pretrained_params = {}

        clip = FlaxCLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        clip, clip_variables = clip.module, {"params": clip.params}

        # Get text model
        text_model, text_model_vars = clip.bind(clip_variables).text_model.unbind()
        pretrained_params["pretrained_text_encoder"] = text_model_vars["params"]
        pretrained_params["pretrained_text_projection"] = clip_variables["params"]["text_projection"]

    text_encoder_def = PretrainedTextEncoder(text_model, projection_dim=config.output_dim,
                                             freeze_encoder=config.freeze_text_enc)

    if config.sketch and config.state:
        encoder_dict = dict(
                state=state_encoder_def,
                text=text_encoder_def,
                sketch=sketch_encoder_def,
            )
    elif config.sketch:
        encoder_dict = dict(
            text=text_encoder_def,
            sketch=sketch_encoder_def,
        )
    elif config.state:
        encoder_dict = dict(
            state=state_encoder_def,
            text=text_encoder_def,
        )
    else:
        encoder_dict = dict(
            text=text_encoder_def,
        )

    encoder = ContrastiveModule(
        encoders=encoder_dict,
        dropout_rate=config.dropout_rate,
    )

    return encoder, pretrained_params


if __name__ == "__main__":
    # Test
    batch_size = 2
    seq_len = 32
    image_shape = (224, 224, 6)
    config = CLIPTrainConfig()

    dummy_data = CLIPContrastiveBatch(
            class_ids=jnp.ones((1,), dtype=jnp.int32),
            input_ids=jnp.ones((1, config.encoder.token_max_len), dtype=jnp.int32),
            attention_mask=jnp.ones((1, config.encoder.token_max_len), dtype=jnp.int32),
            pixel_values=jnp.ones((1, 224, 224, 6), dtype=jnp.float32),
            duplicate_matrix=jnp.ones((1, 1), dtype=jnp.float32),
        )
    encoders = get_clip_encoder(config.encoder)
    encoders, pretrained_params = get_clip_encoder(config.encoder)

    model_def = ContrastiveModule(
            encoders=encoders,
            dropout_rate=config.dropout_rate
        )
    # Initialize and run model
    variables = model_def.init(jax.random.PRNGKey(0), dummy_data, mode="text_state_sketch")
    outputs = model_def.apply(variables, **dummy_data)

    print("Output shapes:")
    for k, v in outputs.items():
        print(f"{k}: {v.shape}")
