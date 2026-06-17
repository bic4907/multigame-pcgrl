"""Processor/tokenizer helpers for CLIP-style datasets."""

from __future__ import annotations

import logging
import os
from os.path import basename

from transformers import CLIPProcessor, CLIPTokenizerFast


log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logger = logging.getLogger(basename(__file__))
logger.setLevel(getattr(logging, log_level, logging.INFO))


def load_clip_text_processor(model_name: str = "openai/clip-vit-base-patch32"):
    """Load the text component needed by CLIPDatasetBuilder.

    CLIPDatasetBuilder only tokenizes text. Loading CLIPProcessor also loads the
    image processor, which can fail in constrained CI/cache environments even
    though the image processor is unused for CNN-CLIP level tensors.
    """
    try:
        return CLIPTokenizerFast.from_pretrained(model_name)
    except Exception as tokenizer_exc:
        logger.warning(
            "Failed to load CLIPTokenizerFast for %s; falling back to CLIPProcessor: %s",
            model_name,
            tokenizer_exc,
        )
        return CLIPProcessor.from_pretrained(model_name)
