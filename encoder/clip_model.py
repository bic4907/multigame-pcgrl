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
    def __call__(self, pixel_values, training: bool = False):
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


class RewardDecoder(nn.Module):
    """Embedding → (reward_enum classification, condition regression) 디코더.

    Architecture
    ------------
    Shared trunk (MLP) → 분기
      ├─ Classification head (hidden → num_reward_classes)  : reward_enum 분류
      └─ Regression head    (hidden → sigmoid → [0,1])      : 정규화된 condition 예측
          → denorm 시 cond_min/cond_max 로 원래 스케일 복원

    cond_norm_min / cond_norm_max 는 Flax state variable ("norm_stats" collection)
    로 저장되어, 체크포인트에 함께 포함된다.  학습되지 않는 상수이다.

    Parameters
    ----------
    num_reward_classes : int
        reward_enum 종류 수 (예: 6).
    hidden_dim : int
        MLP hidden dimension.
    num_layers : int
        shared trunk hidden layer 수 (≥1).
    dropout_rate : float
        Dropout rate.
    cond_norm_min_init : jnp.ndarray | None
        (num_reward_classes,) — 초기화 시 전달하는 reward_enum별 condition min 값.
    cond_norm_max_init : jnp.ndarray | None
        (num_reward_classes,) — 초기화 시 전달하는 reward_enum별 condition max 값.
    """
    num_reward_classes: int = 6
    hidden_dim: int = 128
    num_layers: int = 2
    dropout_rate: float = 0.1
    cond_norm_min_init: jnp.ndarray = None
    cond_norm_max_init: jnp.ndarray = None

    @nn.compact
    def __call__(self, embed: jnp.ndarray, training: bool = False):
        """
        Args:
            embed: (B, D) — 인코더 임베딩 (L2-normalized).
        Returns:
            reward_logits:      (B, num_reward_classes) — reward_enum 분류 logits
            condition_pred:     (B, num_reward_classes) — 정규화된 [0,1] condition 예측 (loss용)
            condition_pred_raw: (B, num_reward_classes) — 원래 스케일 condition 예측 (추론용)
        """
        # ── Norm stats를 state variable로 등록 (학습 불가, 체크포인트에 저장) ──
        _default_min = jnp.zeros(self.num_reward_classes) if self.cond_norm_min_init is None else self.cond_norm_min_init
        _default_max = jnp.ones(self.num_reward_classes)  if self.cond_norm_max_init is None else self.cond_norm_max_init

        cond_min = self.variable(
            "norm_stats", "cond_norm_min",
            lambda: _default_min,
        ).value
        cond_max = self.variable(
            "norm_stats", "cond_norm_max",
            lambda: _default_max,
        ).value

        # stop_gradient: 역전파에서 제외
        cond_min = jax.lax.stop_gradient(cond_min)
        cond_max = jax.lax.stop_gradient(cond_max)

        # ── Shared trunk ──
        x = embed
        for _ in range(self.num_layers):
            x = nn.Dense(self.hidden_dim)(x)
            x = nn.gelu(x)
            x = nn.LayerNorm()(x)
            x = nn.Dropout(self.dropout_rate)(x, deterministic=not training)

        # ── Classification head (reward_enum) ──
        cls_h = nn.Dense(self.hidden_dim // 2, name="cls_hidden")(x)
        cls_h = nn.gelu(cls_h)
        reward_logits = nn.Dense(self.num_reward_classes, name="reward_cls_head")(cls_h)

        # ── Regression head (condition value per reward_enum) ──
        reg_h = nn.Dense(self.hidden_dim // 2, name="reg_hidden")(x)
        reg_h = nn.gelu(reg_h)
        reg_logits = nn.Dense(self.num_reward_classes, name="condition_reg_head")(reg_h)

        # sigmoid → [0, 1] 정규화 공간 (loss는 이 값으로 계산)
        condition_pred = jax.nn.sigmoid(reg_logits)

        # 역변환 → 원래 스케일 (추론 시 사용)
        # log1p 공간에서 정규화되었으므로: denorm → expm1
        scale = cond_max - cond_min                              # (num_classes,)
        condition_pred_log = condition_pred * scale + cond_min          # log1p 공간
        condition_pred_raw = jnp.expm1(jnp.maximum(condition_pred_log, 0.0))  # 원래 스케일

        return reward_logits, condition_pred, condition_pred_raw


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
            reward_enum: jnp.ndarray = None,
            mode: str = "text_state",
            training: bool = False,
    ):
        output_dict = dict()
        modes = mode.split("_")

        if "state" in modes:
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

        # ── 디코더: state embedding 으로부터 reward_enum & condition 예측 ──
        if "state" in modes:
            reward_logits, condition_pred, condition_pred_raw = self.decoder(
                output_dict["state_embed"], training=training
            )
            output_dict["reward_logits"] = reward_logits
            output_dict["condition_pred"] = condition_pred              # [0,1] 정규화 (loss용)
            output_dict["condition_pred_raw"] = condition_pred_raw      # 원래 스케일 (추론용)

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

    text_encoder_def = PretrainedTextEncoder(text_model, projection_dim=config.output_dim,
                                             freeze_encoder=config.freeze_text_enc)
    state_encoder_def = PretrainedImageEncoder(vision_model, projection_dim=config.output_dim,
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
    )


    return encoder, pretrained_params


def get_clip_hf_pretrained_params(config: EncoderConfig):
    """HuggingFace CLIP pretrained weights를 ContrastiveModule param tree 형식으로 반환."""
    clip = FlaxCLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_module, clip_variables = clip.module, {"params": clip.params}

    text_model, text_model_vars = clip_module.bind(clip_variables).text_model.unbind()
    vision_model, vision_model_vars = clip_module.bind(clip_variables).vision_model.unbind()

    # Flax은 Dict[str, nn.Module] 필드를 "{field}_{key}" 형태로 flat하게 저장함
    # encoders: Dict[str, nn.Module] → encoders_text, encoders_state
    # text_projection / visual_projection은 HF에서 이미 {"kernel": array} 형태로 저장됨
    pretrained_params = {
        "encoders_text": {
            "pretrained_text_encoder": text_model_vars["params"],
            "pretrained_text_projection": clip_variables["params"]["text_projection"],
        },
    }

    if config.state:
        # clipconv uses 224×224×3 RGB input, fully compatible with HF patch_embedding (3ch)
        vision_params = vision_model_vars["params"]
        pretrained_params["encoders_state"] = {
            "pretrained_state_encoder": vision_params,
            "pretrained_image_projection": clip_variables["params"]["visual_projection"],
        }

    return pretrained_params


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
        cond_norm_min_init=cond_norm_min,
        cond_norm_max_init=cond_norm_max,
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
