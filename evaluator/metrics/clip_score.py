"""
evaluator/metrics/clip_score.py
================================
CLIPScore 지표 — HuggingFace CLIP / SigLIP 텍스트–이미지 유사도.

입력: LevelBundle.text (instruction) + LevelBundle.image (rendered RGB)
유사도: text_i ↔ image_j 코사인 유사도 (대칭화: (ti + tj) / 2)
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
from PIL import Image

from .base import BaseMetricEvaluator, LevelBundle

_FLAX_PREFIXES  = ("openai/clip-vit",)
_SIGLIP_PREFIXES = ("google/siglip",)


def _infer_backend(model_name: str, backend: Optional[str]) -> str:
    if backend is not None:
        return backend
    for p in _FLAX_PREFIXES:
        if model_name.startswith(p):
            return "flax"
    return "torch"


class CLIPScoreMetric(BaseMetricEvaluator):
    """
    CLIPScore 지표.

    text–image 코사인 유사도를 대칭화하여 유사도 행렬로 반환한다.

    Parameters
    ----------
    model_name : str
        HuggingFace 모델 ID.
        Flax 백엔드: "openai/clip-vit-large-patch14" (기본)
                     "openai/clip-vit-large-patch14-336"
        Torch 백엔드: "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
                      "google/siglip-so400m-patch14-384"
    backend : {"flax", "torch"} | None
        None 이면 model_name 으로 자동 추론.
    """

    _DEFAULT_MODEL = "openai/clip-vit-large-patch14-336"

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL,
        backend: Optional[str] = None,
    ) -> None:
        self._model_name = model_name
        self._backend    = _infer_backend(model_name, backend)
        self._is_siglip  = any(model_name.startswith(p) for p in _SIGLIP_PREFIXES)
        self._load_model()

    # ── 모델 로드 ─────────────────────────────────────────────────────────────

    def _load_model(self) -> None:
        if self._backend == "flax":
            from transformers import FlaxCLIPModel, CLIPProcessor
            try:
                self._model = FlaxCLIPModel.from_pretrained(self._model_name)
            except OSError:
                self._model = FlaxCLIPModel.from_pretrained(self._model_name, from_pt=True)
            self._processor = CLIPProcessor.from_pretrained(self._model_name)
        else:
            import torch
            from transformers import AutoProcessor, AutoModel
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            self._model  = AutoModel.from_pretrained(self._model_name).to(self._device).eval()
            self._processor = AutoProcessor.from_pretrained(self._model_name)

    # ── BaseMetricEvaluator 구현 ──────────────────────────────────────────────

    @property
    def name(self) -> str:
        return f"CLIPScore[{self._model_name.split('/')[-1]}]"

    def similarity_matrix(self, bundles: List[LevelBundle]) -> np.ndarray:
        """
        (N, N) 대칭 CLIPScore 행렬.
        sim[i, j] = (cos(text_i, img_j) + cos(text_j, img_i)) / 2
        """
        texts  = [b.text for b in bundles]
        images = [Image.fromarray(b.image).convert("RGB") for b in bundles]

        text_embs  = self._encode_texts(texts)
        image_embs = self._encode_images(images)

        # (N, N) 코사인 유사도
        ti_mat = np.matmul(text_embs, image_embs.T)
        return (ti_mat + ti_mat.T) / 2

    # ── 인코딩 ────────────────────────────────────────────────────────────────

    def _encode_texts(self, texts: List[str]) -> np.ndarray:
        if self._backend == "flax":
            import jax.numpy as jnp
            inputs = self._processor(text=texts, return_tensors="np",
                                     padding=True, truncation=True, max_length=77)
            feats = self._model.get_text_features(
                input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
            feats = np.array(feats)
        else:
            import torch
            inputs = self._processor(text=texts, return_tensors="pt",
                                     padding=True, truncation=True, max_length=77)
            inputs = {k: v.to(self._device) for k, v in inputs.items()}
            with torch.no_grad():
                feats = self._model.get_text_features(**inputs).cpu().numpy()
        return feats / (np.linalg.norm(feats, axis=-1, keepdims=True) + 1e-8)

    def _encode_images(self, images: List[Image.Image]) -> np.ndarray:
        if self._backend == "flax":
            inputs = self._processor(images=images, return_tensors="np")
            feats  = np.array(self._model.get_image_features(
                pixel_values=inputs["pixel_values"]))
        else:
            import torch
            inputs = self._processor(images=images, return_tensors="pt")
            inputs = {k: v.to(self._device) for k, v in inputs.items()}
            with torch.no_grad():
                feats = self._model.get_image_features(**inputs).cpu().numpy()
        return feats / (np.linalg.norm(feats, axis=-1, keepdims=True) + 1e-8)
