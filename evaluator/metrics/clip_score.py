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
        from evaluator.clipscore.clip_score_evaluator import CLIPScoreEvaluator
        self._ev = CLIPScoreEvaluator(model_name, backend)
        self._model_name = model_name

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

        text_embs  = self._ev.encode_texts(texts)
        image_embs = self._ev.encode_images(images)
        ti_mat = np.array(self._ev.similarity_matrix(text_embs, image_embs))

        # 대칭화: text_i→img_j 와 text_j→img_i 의 평균
        return (ti_mat + ti_mat.T) / 2

