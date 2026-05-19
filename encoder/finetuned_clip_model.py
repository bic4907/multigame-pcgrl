"""
encoder/finetuned_clip_model.py
================================
HuggingFace pretrained CLIP을 사용자의 (image, text) 데이터로 파인튜닝하기 위한
trainable 변종. `encoder.pretrained_clip_model` 와 파라미터 트리(이름/구조)가
완벽히 동일하도록 작성되어, fine-tune 결과 체크포인트를 그대로 기존 RL 학습
파이프라인(`apply_encoder_params`)으로 inject 할 수 있다.

차이점
------
- `jax.lax.stop_gradient` 제거 → 모든 CLIP 파라미터에 그래디언트 흐름
- 그 외 모듈 이름·shape 모두 `pretrained_clip_model` 과 동일
"""
from typing import Dict

import jax
import jax.numpy as jnp
from flax import linen as nn
from transformers import FlaxCLIPModel

from conf.config import EncoderConfig
from encoder.pretrained_clip_model import ContrastiveModule  # 그대로 재사용


class TrainablePretrainedTextEncoder(nn.Module):
    """`PretrainedTextEncoder` 와 동일 구조이되 stop_gradient 없음."""
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
    """`PretrainedImageEncoder` 와 동일 구조이되 stop_gradient 없음."""
    pretrained_state_encoder: nn.Module

    @nn.compact
    def __call__(self, pixel_values):
        x = self.pretrained_state_encoder(pixel_values).pooler_output
        x = nn.Dense(512, name="pretrained_image_projection", use_bias=False)(x)
        return x


def get_finetuned_clip_encoder(config: EncoderConfig):
    """학습용 ContrastiveModule + HF pretrained 초기 파라미터 dict를 반환.

    반환 형식 (·구조)은 `get_pretrained_clip_encoder` 와 동일하므로
    `train_clip.py:get_train_state` 의 `replace_params` 로직을 그대로 사용할 수 있다.
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

