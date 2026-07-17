"""
evaluator/metrics/clip_score.py
================================
CLIPScore texttable — HuggingFace CLIP / SigLIP text–image text also .

text: LevelBundle.text (instruction) + LevelBundle.image (rendered RGB)
text also : text_i ↔ image_j text text also  (text: (ti + tj) / 2)
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
    CLIPScore texttable.

    text–image text text also   text text also  rowtext to  returntext.

    Parameters
    ----------
    model_name : str
        HuggingFace text ID.
        Flax text: "openai/clip-vit-large-patch14" (default)
                     "openai/clip-vit-large-patch14-336"
        Torch text: "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
                      "google/siglip-so400m-patch14-384"
    backend : {"flax", "torch"} | None
        None  text model_name  as  automatic text.
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

    # ── text load ─────────────────────────────────────────────────────────────

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

    # ── BaseMetricEvaluator text ──────────────────────────────────────────────

    @property
    def name(self) -> str:
        return f"CLIPScore[{self._model_name.split('/')[-1]}]"

    def similarity_matrix(self, bundles: List[LevelBundle]) -> np.ndarray:
        """
        (N, N) text CLIPScore rowtext.
        sim[i, j] = (cos(text_i, img_j) + cos(text_j, img_i)) / 2
        """
        texts  = [b.text for b in bundles]
        images = [Image.fromarray(b.image).convert("RGB") for b in bundles]

        text_embs  = self._encode_texts(texts)
        image_embs = self._encode_images(images)

        # (N, N) text text also
        ti_mat = np.matmul(text_embs, image_embs.T)
        return (ti_mat + ti_mat.T) / 2

    # ── text ────────────────────────────────────────────────────────────────

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
