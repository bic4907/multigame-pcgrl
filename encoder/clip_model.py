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


class ContrastiveModule(nn.Module):
    encoders: Dict[str, nn.Module]
    dropout_rate: float = 0.0

    def setup(self):
        self.text_state_temperature = self.param(
            "text_state_temperature", nn.initializers.constant(jnp.log(0.07)), ()
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

    @nn.compact
    def __call__(
            self,
            input_ids: jnp.ndarray = None,
            attention_mask: jnp.ndarray = None,
            pixel_values: jnp.ndarray = None,
            mode: str = "text_state",
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


        output_dict['text_state_temperature'] = self.text_state_temperature

        return output_dict


class _TransformerBlock(nn.Module):
    """Pre-norm Transformer encoder block (MHSA + MLP)."""
    dim: int
    num_heads: int = 4
    mlp_ratio: float = 4.0
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = False) -> jnp.ndarray:
        # ── Multi-head self-attention ──
        h = nn.LayerNorm()(x)
        h = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            dropout_rate=self.dropout_rate,
        )(h, h, deterministic=not training)
        x = x + h
        # ── Feed-forward ──
        h = nn.LayerNorm()(x)
        h = nn.Dense(int(self.dim * self.mlp_ratio))(h)
        h = nn.gelu(h)
        h = nn.Dropout(self.dropout_rate)(h, deterministic=not training)
        h = nn.Dense(self.dim)(h)
        h = nn.Dropout(self.dropout_rate)(h, deterministic=not training)
        x = x + h
        return x


class RewardDecoder(nn.Module):
    """Universal transition reward model (spatial, attention-based).

    Instead of consuming pre-computed state *embeddings*, this module ingests the
    spatial one-hot tile maps of the previous/next state plus an explicit change
    (diff) feature map.  A small conv stem extracts local features which are then
    processed by a ViT-style Transformer whose tokens are conditioned on the
    instruction (text) embedding.  The scalar transition reward is read out from
    a dedicated CLS token.

    Input feature maps (per cell):
      - prev one-hot map        : C channels (what tile was there)
      - curr one-hot map        : C channels (what tile is there now)
      - signed diff (curr-prev) : C channels (which categories were removed/added)
      - changed mask            : 1 channel  (where *any* change happened)
    """
    num_reward_classes: int = 6   # kept for construction compatibility
    hidden_dim: int = 128
    num_layers: int = 2
    dropout_rate: float = 0.1
    num_heads: int = 4

    @nn.compact
    def __call__(
        self,
        text_embed: jnp.ndarray,
        prev_map: jnp.ndarray,
        curr_map: jnp.ndarray,
        training: bool = False,
    ):
        """
        Args:
            text_embed: (B, D_text)         — instruction embedding (L2-normalized)
            prev_map:   (B, H, W, C)        — s_t  one-hot (+ coord) feature map
            curr_map:   (B, H, W, C)        — s_t+1 one-hot (+ coord) feature map
        Returns:
            reward: (B,) scalar transition reward
        """
        diff = curr_map - prev_map                                   # (B,H,W,C)
        changed = jnp.max(jnp.abs(diff), axis=-1, keepdims=True)     # (B,H,W,1)
        feat = jnp.concatenate([prev_map, curr_map, diff, changed], axis=-1)

        B, H, W, _ = feat.shape

        # ── Conv stem: extract local transition features ──
        x = nn.Conv(self.hidden_dim, (3, 3), padding="SAME", name="stem_conv1")(feat)
        x = nn.gelu(x)
        x = nn.LayerNorm()(x)
        x = nn.Conv(self.hidden_dim, (3, 3), padding="SAME", name="stem_conv2")(x)
        x = nn.gelu(x)

        # ── Tokenize spatial grid ──
        tokens = x.reshape(B, H * W, self.hidden_dim)
        pos_embed = self.param(
            "pos_embed", nn.initializers.normal(0.02), (1, H * W, self.hidden_dim)
        )
        tokens = tokens + pos_embed

        # ── Instruction token (text conditioning) ──
        text_token = nn.Dense(self.hidden_dim, name="text_proj")(text_embed)
        text_token = text_token[:, None, :]                          # (B,1,D)

        # ── Learnable CLS readout token ──
        cls = self.param("cls_token", nn.initializers.normal(0.02), (1, 1, self.hidden_dim))
        cls = jnp.broadcast_to(cls, (B, 1, self.hidden_dim))

        x = jnp.concatenate([cls, text_token, tokens], axis=1)       # (B, 2+H*W, D)
        x = nn.Dropout(self.dropout_rate)(x, deterministic=not training)

        # ── Transformer encoder ──
        for i in range(self.num_layers):
            x = _TransformerBlock(
                dim=self.hidden_dim,
                num_heads=self.num_heads,
                dropout_rate=self.dropout_rate,
                name=f"block_{i}",
            )(x, training=training)
        x = nn.LayerNorm()(x)

        # ── Reward read-out from CLS token ──
        pooled = x[:, 0]                                             # (B, D)
        h = nn.Dense(self.hidden_dim // 2, name="head_hidden")(pooled)
        h = nn.gelu(h)
        reward = nn.Dense(1, name="reward_head")(h)
        return jnp.squeeze(reward, axis=-1)


class ContrastiveDecoderModule(nn.Module):
    """ContrastiveModule + RewardDecoder.

    기존 contrastive 학습에 디코더 브랜치를 추가하여
    embedding 으로부터 reward_enum과 condition을 예측한다.

    reward_enum_onehot_dim > 0 이면, pixel_values에 reward_enum의
    one-hot 인코딩을 공간 차원으로 broadcast하여 채널 concat한다.
    → CNN이 해당 레벨이 어떤 reward_enum인지 알 수 있다.
    """
    encoders: Dict[str, nn.Module]
    decoder: RewardDecoder
    dropout_rate: float = 0.0
    reward_enum_onehot_dim: int = 0  # num_reward_classes for one-hot; 0 = disabled

    def setup(self):
        self.text_state_temperature = self.param(
            "text_state_temperature", nn.initializers.constant(jnp.log(0.07)), ()
        )

    def encode_text(
            self, input_ids: jnp.ndarray, attention_mask: jnp.ndarray,
            position_ids: jnp.ndarray, training: bool
    ) -> jnp.ndarray:
        x = self.encoders["text"](input_ids, attention_mask, position_ids)
        x = x / (jnp.linalg.norm(x, axis=-1, keepdims=True) + 1e-6)
        return x

    def encode_state(
            self, pixel_values: jnp.ndarray, training: bool,
            reward_enum: jnp.ndarray = None,
    ) -> jnp.ndarray:
        # ── reward_enum one-hot concat ──
        if self.reward_enum_onehot_dim > 0:
            B, H, W, _ = pixel_values.shape
            if reward_enum is not None:
                # (B,) → (B, num_classes) → (B, 1, 1, num_classes) → (B, H, W, num_classes)
                onehot = jax.nn.one_hot(reward_enum, self.reward_enum_onehot_dim)
                onehot = jnp.broadcast_to(
                    onehot[:, None, None, :], (B, H, W, self.reward_enum_onehot_dim)
                )
            else:
                # reward_enum 미제공 시 zeros (정보 없음)
                onehot = jnp.zeros((B, H, W, self.reward_enum_onehot_dim))
            pixel_values = jnp.concatenate([pixel_values, onehot], axis=-1)

        x = self.encoders["state"](pixel_values, training)
        x = x / (jnp.linalg.norm(x, axis=-1, keepdims=True) + 1e-6)
        return x

    @nn.compact
    def __call__(
            self,
            input_ids: jnp.ndarray = None,
            attention_mask: jnp.ndarray = None,
            pixel_values: jnp.ndarray = None,
            prev_pixel_values: jnp.ndarray = None,
            curr_pixel_values: jnp.ndarray = None,
            reward_enum: jnp.ndarray = None,
            mode: str = "text_state",
            training: bool = False,
    ):
        output_dict = dict()
        modes = mode.split("_")

        if "state" in modes and pixel_values is not None:
            state_embed = self.encode_state(pixel_values, training, reward_enum=reward_enum)
            output_dict["state_embed"] = state_embed
            output_dict["text_state_temperature"] = self.text_state_temperature

        if "text" in modes:
            batch_size, seq_len = input_ids.shape
            position_ids = jnp.arange(seq_len)[None, :].repeat(batch_size, axis=0)
            text_embed = self.encode_text(input_ids, attention_mask, position_ids, training)
            output_dict["text_embed"] = text_embed
            output_dict["text_state_temperature"] = self.text_state_temperature

        output_dict['text_state_temperature'] = self.text_state_temperature

        if (
            "text_embed" in output_dict
            and prev_pixel_values is not None
            and curr_pixel_values is not None
        ):
            # Spatial reward model: feed prev/curr feature maps directly so the
            # decoder can reason about *where* and *what* changed via its own
            # conv + attention feature extractor (no pre-pooled state embedding).
            output_dict["reward_pred"] = self.decoder(
                output_dict["text_embed"],
                prev_pixel_values,
                curr_pixel_values,
                training=training,
            )

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

    text_encoder_def = PretrainedTextEncoder(text_model, projection_dim=None,
                                             freeze_encoder=config.freeze_text_enc)
    state_encoder_def = PretrainedImageEncoder(vision_model, projection_dim=None,
                                               freeze_encoder=config.freeze_state_enc)

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

    if config.state:
        encoder_dict = dict(
            text=text_encoder_def,
            state=state_encoder_def
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


def get_cnnclip_decoder_encoder(config: EncoderConfig, decoder_config=None,
                                cond_norm_min=None, cond_norm_max=None,
                                RL_training: bool = False):
    """
    CNN-based CLIP encoder + RewardDecoder.
    ContrastiveDecoderModule을 반환한다.

    Parameters
    ----------
    cond_norm_min : jnp.ndarray | None
        (num_reward_classes,) — reward_enum별 condition min. 역변환용.
    cond_norm_max : jnp.ndarray | None
        (num_reward_classes,) — reward_enum별 condition max. 역변환용.
    """
    from conf.config import DecoderConfig as _DC
    if decoder_config is None:
        decoder_config = _DC()

    state_encoder_def = CNNResMapEncoder(projection_dim=config.output_dim, drop_rate=config.dropout_rate)

    pretrained_params = None

    if RL_training:
        clip_conf = CLIPConfig.from_pretrained("openai/clip-vit-base-patch32")
        text_model = FlaxCLIPTextTransformer(clip_conf.text_config)
    else:
        pretrained_params = {}
        clip = FlaxCLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        clip, clip_variables = clip.module, {"params": clip.params}
        text_model, text_model_vars = clip.bind(clip_variables).text_model.unbind()
        pretrained_params["pretrained_text_encoder"] = text_model_vars["params"]
        pretrained_params["pretrained_text_projection"] = clip_variables["params"]["text_projection"]

    text_encoder_def = PretrainedTextEncoder(text_model, projection_dim=config.output_dim,
                                             freeze_encoder=config.freeze_text_enc)

    if config.state:
        encoder_dict = dict(text=text_encoder_def, state=state_encoder_def)
    else:
        encoder_dict = dict(text=text_encoder_def)

    decoder = RewardDecoder(
        num_reward_classes=decoder_config.num_reward_classes,
        hidden_dim=decoder_config.hidden_dim,
        num_layers=decoder_config.num_layers,
        dropout_rate=config.dropout_rate,
        num_heads=getattr(decoder_config, "num_heads", 4),
    )

    # reward_enum one-hot 채널 추가 여부 결정
    _onehot_dim = decoder_config.num_reward_classes if getattr(decoder_config, 'cnn_reward_enum_onehot', False) else 0

    module = ContrastiveDecoderModule(
        encoders=encoder_dict,
        decoder=decoder,
        dropout_rate=config.dropout_rate,
        reward_enum_onehot_dim=_onehot_dim,
    )

    return module, pretrained_params


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
    variables = model_def.init(jax.random.PRNGKey(0), dummy_data, mode="text_state")
    outputs = model_def.apply(variables, **dummy_data)

    print("Output shapes:")
    for k, v in outputs.items():
        print(f"{k}: {v.shape}")
