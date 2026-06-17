"""
encoder/data/mlp_batch.py
=========================
Annotation 형식 MultiGameDataset → MLP 인코더 학습용 Dataset.

CLIPDatasetBuilder 와 동일한 전처리 파이프라인(instruction 필터, prefix,
longtail cut, log1p 정규화, stratified split, pixel_values 포함)을 그대로
위임하고, BERT CLS 임베딩만 추가로 계산한다.

인코더 타입(CLIP vs MLP)만 바꾸면서 나머지 변인을 통제해야 하므로
생성자 시그니처를 CLIPDatasetBuilder 와 최대한 동일하게 맞춘다.
BERT 임베딩은 instruct_rl.utils.dataset_loader._compute_bert_embeddings 재사용.
"""

from __future__ import annotations

import logging
import os
from os.path import basename
from typing import Iterator, Optional, Set, Tuple

import jax
import numpy as np
from chex import dataclass
from transformers import CLIPProcessor

log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logger = logging.getLogger(basename(__file__))
logger.setLevel(getattr(logging, log_level, logging.INFO))
logging.getLogger("absl").setLevel(logging.ERROR)


# ── Dataclass ─────────────────────────────────────────────────────────────────

@dataclass
class MLPDataset:
    """MLP 인코더 학습용 Dataset.

    CLIPDataset 과 대응 관계:
        bert_embeddings  ← (CLIPDataset 에는 없음)  BERT CLS 임베딩
        pixel_values     ← CLIPDataset.pixel_values  레벨 맵 (H, W, C)
        condition_targets           ← 동일
        reward_enum_targets         ← 동일
        game_names                  ← reward_cond 에서 추출
        instructions                ← 원본 instruction 텍스트
        is_train                    ← 동일 (unseen 게임은 False)
    """
    bert_embeddings: np.ndarray      # (N, nlp_input_dim)
    pixel_values: np.ndarray         # (N, H, W, C)  레벨 맵 (CLIPDataset 과 동일)
    condition_targets: np.ndarray    # (N,)  log1p + per-enum min-max → [0, 1]
    reward_enum_targets: np.ndarray  # (N,)  0-indexed
    game_names: np.ndarray           # (N,)  str
    instructions: np.ndarray         # (N,)  str
    is_train: np.ndarray             # (N,)  bool


# ── Dataset Builder ───────────────────────────────────────────────────────────

class MLPDatasetBuilder:
    """CLIPDatasetBuilder + BERT 임베딩으로 MLPDataset 을 생성한다.

    생성자 시그니처는 CLIPDatasetBuilder 와 동일하며,
    MLP 전용 파라미터(exclude_games, nlp_input_dim)만 추가한다.

    Parameters  (CLIPDatasetBuilder 와 동일한 순서·이름)
    ----------
    processor : CLIPProcessor
        CLIPDatasetBuilder 내부 tokenizer. BERT 임베딩 계산과는 무관.
    paired_data : MultiGameDataset
    rng_key : jax.random.PRNGKey
    train_ratio : float
    max_len : int
    max_samples : int | None
    prepend 옵션 (단일 instruction_prefix)
    -------------------------------------
    instruction_prefix : str | None
        "name" / "desc" / "none" (또는 None) — CLIPDatasetBuilder 와 동일.
    longtail_cut : bool

    Parameters  (MLP 전용)
    ----------------------
    exclude_games : set[str] | None
        unseen 게임 이름 집합. ``unseen_ratio`` 만큼만 학습에 포함하고
        나머지는 is_train=False (val) 로 돌려 zero/few-shot 평가에 사용한다.
    nlp_input_dim : int
        BERT 임베딩 차원 (기본 768).
    unseen_ratio : float
        unseen 게임(train pool) 중 학습에 사용할 비율 (few-shot ratio).
        0.0 = zero-shot (unseen 학습 데이터 0%), 1.0 = unseen train pool 전부.
    seen_ratio : float
        seen 게임(train pool) 중 학습에 사용할 비율. 1.0 = 전부 사용.
    """

    def __init__(
        self,
        processor: CLIPProcessor,
        paired_data,
        rng_key: jax.random.PRNGKey,
        train_ratio: float = 0.8,
        max_len: int = 77,
        max_samples: Optional[int] = None,
        instruction_prefix: Optional[str] = "name",
        longtail_cut: bool = True,
        tile_offset: int = 0,
        # MLP 전용
        exclude_games: Optional[Set[str]] = None,
        nlp_input_dim: int = 768,
        unseen_ratio: float = 0.0,
        seen_ratio: float = 1.0,
    ) -> None:
        from encoder.data.clip_batch import CLIPDatasetBuilder

        self.exclude_games: Set[str] = exclude_games or set()
        self.nlp_input_dim = nlp_input_dim
        self.unseen_ratio = float(unseen_ratio)
        self.seen_ratio = float(seen_ratio)

        # 1. CLIPDatasetBuilder 로 전처리 (필터·prefix·정규화·split·pixel_values)
        self._clip_builder = CLIPDatasetBuilder(
            processor=processor,
            paired_data=paired_data,
            rng_key=rng_key,
            train_ratio=train_ratio,
            max_len=max_len,
            max_samples=max_samples,
            instruction_prefix=instruction_prefix,
            longtail_cut=longtail_cut,
            tile_offset=tile_offset,
        )
        clip_ds = self._clip_builder.get_dataset()
        d = self._clip_builder.preprocessed_dataset_dict

        # 2. CLIPDatasetBuilder 가 전처리한 instruction(prefix 포함)으로 BERT 임베딩 계산
        instructions: list[str] = d["language_inst"]
        bert_embeds = self._compute_bert(instructions)

        # 3. game_names 추출 (CLIPDataset 에서는 reward_cond 안에 있음)
        game_names = np.array(d["game_type"])

        # 4. is_train: CLIPDatasetBuilder split + few-shot ratio 적용
        #    - seen  게임: train pool 중 seen_ratio prefix 만 학습에 사용
        #    - unseen 게임: train pool 중 unseen_ratio prefix 만 학습에 사용
        #      (나머지는 is_train=False 로 돌려 zero/few-shot 평가에 사용)
        is_train = self._apply_fewshot_split(game_names, clip_ds.is_train.copy())

        # 5. 요약 로그
        self._log_split(game_names, is_train)

        self._dataset = MLPDataset(
            bert_embeddings=bert_embeds,
            pixel_values=np.array(d["pixel_values"]),
            condition_targets=clip_ds.condition_targets,
            reward_enum_targets=clip_ds.reward_enum_targets,
            game_names=game_names,
            instructions=np.array(instructions),
            is_train=is_train,
        )

    # ── 공개 API ──────────────────────────────────────────────────────────────

    def get_dataset(self) -> MLPDataset:
        return self._dataset

    def get_condition_norm_stats(self) -> tuple[dict, dict]:
        """CLIPDatasetBuilder 의 reward_enum별 condition 정규화 파라미터 반환."""
        return self._clip_builder.get_condition_norm_stats()

    # ── 내부 메서드 ───────────────────────────────────────────────────────────

    def _compute_bert(self, instructions: list[str]) -> np.ndarray:
        """CLIPDatasetBuilder 가 전처리한 instruction 문자열 → BERT CLS 임베딩."""
        from instruct_rl.utils.dataset_loader import _compute_bert_embeddings

        class _FakeSample:
            __slots__ = ("instruction",)
            def __init__(self, inst: str):
                self.instruction = inst

        fake_samples = [_FakeSample(inst) for inst in instructions]
        return np.array(_compute_bert_embeddings(fake_samples, self.nlp_input_dim))

    def _apply_fewshot_split(
        self, game_names: np.ndarray, is_train: np.ndarray
    ) -> np.ndarray:
        """게임별 train pool 에 few-shot ratio 를 적용한다.

        CLIPDatasetBuilder 의 자연 train/val split 결과(``is_train``)를 받아
        각 게임의 train pool(is_train=True) 중 ratio prefix 만 학습에 남긴다.

        - seen  게임 (game ∉ exclude_games): seen_ratio prefix 사용
        - unseen 게임 (game ∈ exclude_games): unseen_ratio prefix 사용
          (unseen_ratio=0.0 → 전부 제외 = 기존 exclude_games 동작 = zero-shot)

        prefix 선택은 자연 인덱스 순서 기준으로 결정적(deterministic)이며,
        CLIP few-shot(``build_train_indices_for_ratio`` 의 pool[:n_use])과
        동일한 시맨틱을 갖는다.
        """
        new_is_train = is_train.copy()
        for game in sorted(set(game_names.tolist())):
            is_unseen = game in self.exclude_games
            ratio = self.unseen_ratio if is_unseen else self.seen_ratio
            if ratio >= 1.0:
                continue  # 전부 사용 → 변경 없음
            game_train_idx = np.where((game_names == game) & is_train)[0]
            n_use = int(len(game_train_idx) * ratio)
            drop_idx = game_train_idx[n_use:]  # ratio 초과분은 학습 제외
            new_is_train[drop_idx] = False
        return new_is_train

    def _log_split(self, game_names: np.ndarray, is_train: np.ndarray) -> None:
        unique_games = sorted(set(game_names))
        logger.info("=" * 60)
        logger.info(
            "  MLPDataset split  (unseen=%s, unseen_ratio=%.4f, seen_ratio=%.4f)",
            self.exclude_games or "none", self.unseen_ratio, self.seen_ratio,
        )
        for g in unique_games:
            mask = game_names == g
            tag = "(unseen)" if g in self.exclude_games else "(seen)"
            logger.info(
                "  %-12s %s  total=%d, train=%d, val=%d",
                g, tag, mask.sum(), (mask & is_train).sum(), (mask & ~is_train).sum(),
            )
        logger.info(
            "  Total: %d (train=%d, val=%d)",
            len(game_names), is_train.sum(), (~is_train).sum(),
        )
        logger.info("=" * 60)


# ── Batch Generator ───────────────────────────────────────────────────────────

def create_mlp_batches(
    dataset: MLPDataset,
    batch_size: int,
    train: bool,
    rng: jax.random.PRNGKey,
) -> Iterator[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """MLPDataset 에서 미니배치를 생성한다.

    Parameters
    ----------
    dataset : MLPDataset
    batch_size : int
    train : bool
        True → is_train=True 샘플, False → val 샘플.
    rng : jax.random.PRNGKey

    Yields
    ------
    (bert_embeds, pixel_values, cond_targets, game_names, reward_enums)
    """
    mask = dataset.is_train if train else ~dataset.is_train
    indices = np.where(mask)[0]

    if len(indices) == 0:
        return

    perm = np.array(jax.random.permutation(rng, len(indices)))
    indices = indices[perm]

    for start in range(0, len(indices), batch_size):
        end = min(start + batch_size, len(indices))
        batch_idx = indices[start:end]
        if len(batch_idx) < batch_size:
            extra = np.random.choice(len(indices), batch_size - len(batch_idx), replace=True)
            batch_idx = np.concatenate([batch_idx, indices[extra]])

        yield (
            dataset.bert_embeddings[batch_idx],       # (B, nlp_input_dim)
            dataset.pixel_values[batch_idx],           # (B, H, W, C)
            dataset.condition_targets[batch_idx],      # (B,)
            dataset.game_names[batch_idx],             # (B,) str
            dataset.reward_enum_targets[batch_idx],    # (B,)
        )
