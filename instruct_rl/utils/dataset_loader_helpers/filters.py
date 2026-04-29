from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional

from .constants import REWARD_ENUM_NAMES


def _parse_dataset_reward_enum_filter(raw_value, *, field_name: str = "dataset_reward_enum"):
    """dataset_reward_enum 설정을 정규화한다.

    Returns
    -------
    list[int] | None
        None 이면 필터 비활성화(=전체 reward_enum 허용).
        예: "01" -> [0, 1], "0,1" -> [0, 1], 2 -> [2]
    """
    if raw_value is None:
        return None

    parsed = None
    if isinstance(raw_value, str):
        v = raw_value.strip().lower()
        if v in ("", "none", "all"):
            return None
        try:
            if "," in v:
                parsed = [int(x.strip()) for x in v.split(",") if x.strip()]
            else:
                parsed = [int(c) for c in v]
        except ValueError as e:
            raise ValueError(
                f"Invalid {field_name}='{raw_value}'. "
                f"Use digits like '01', comma list like '0,1', or 'all'."
            ) from e
    elif isinstance(raw_value, int):
        parsed = [int(c) for c in str(raw_value)]
    else:
        try:
            parsed = [int(x) for x in raw_value]
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"Invalid {field_name}={raw_value!r}. "
                f"Use int/list, digits-string, comma-list, or 'all'."
            ) from e

    valid = set(REWARD_ENUM_NAMES.keys())
    normalized = []
    seen = set()
    for re_val in parsed:
        if re_val not in valid:
            raise ValueError(
                f"Invalid {field_name} value: {re_val}. "
                f"Valid enums are {sorted(valid)}."
            )
        if re_val not in seen:
            normalized.append(re_val)
            seen.add(re_val)
    return normalized if normalized else None


def _parse_reward_enum_list(raw_value, *, field_name: str = "eval_dataset_reward_enums"):
    """복수 reward_enum 설정을 int 리스트로 정규화한다.

    None/'none'/'' -> None
    'all'          -> [0,1,2,3,4]
    '012'          -> [0,1,2]   (기존 동작 유지)
    '0,2,4'        -> [0,2,4]
    """
    if raw_value is None:
        return None

    if isinstance(raw_value, str):
        v = raw_value.strip().lower()
        if v in ("", "none"):
            return None
        if v == "all":
            return sorted(REWARD_ENUM_NAMES.keys())
        try:
            if "," in v:
                return [int(x.strip()) for x in v.split(",") if x.strip()]
            return [int(c) for c in v]
        except ValueError as e:
            raise ValueError(
                f"Invalid {field_name}='{raw_value}'. "
                f"Use digits like '012', comma list like '0,1,2', or 'all'."
            ) from e

    if isinstance(raw_value, int):
        return [int(c) for c in str(raw_value)]

    try:
        return [int(x) for x in raw_value]
    except (TypeError, ValueError) as e:
        raise ValueError(
            f"Invalid {field_name}={raw_value!r}. "
            f"Use iterable of ints, digits-string, comma-list, or 'all'."
        ) from e


@dataclass
class _ConditionFilter:
    """단일 condition 필터 조건."""

    enum_idx: int
    min_val: Optional[float] = None
    max_val: Optional[float] = None


def _parse_condition_filters(filter_str: str) -> List[_ConditionFilter]:
    """필터 문자열을 파싱한다.

    포맷 (쉼표로 여러 개 구분):
        enum_{i}_min_{lo}_max_{hi}   — lo ≤ condition[i] ≤ hi
        enum_{i}_min_{lo}            — lo ≤ condition[i]
        enum_{i}_max_{hi}            — condition[i] ≤ hi
    """
    pattern = re.compile(
        r"enum_(\d+)"
        r"(?:_min_([\d.]+))?"
        r"(?:_max_([\d.]+))?"
    )
    filters = []
    for token in filter_str.split(","):
        token = token.strip()
        if not token:
            continue
        match = pattern.fullmatch(token)
        if match is None:
            raise ValueError(
                f"Invalid condition filter: '{token}'. "
                f"Expected format: enum_{{i}}_min_{{lo}}_max_{{hi}} "
                f"(min/max are each optional but at least one required)"
            )
        idx = int(match.group(1))
        min_val = float(match.group(2)) if match.group(2) is not None else None
        max_val = float(match.group(3)) if match.group(3) is not None else None
        if min_val is None and max_val is None:
            raise ValueError(
                f"Condition filter 'enum_{idx}' has neither min nor max — "
                f"at least one bound is required."
            )
        filters.append(_ConditionFilter(enum_idx=idx, min_val=min_val, max_val=max_val))
    return filters


def _apply_condition_filters(samples, filters: List[_ConditionFilter]):
    """필터 리스트를 AND 조합으로 적용하여 샘플을 필터링한다."""
    for filt in filters:
        def _keep(sample, _f=filt):
            conds = sample.meta.get("conditions", {})
            val = conds.get(_f.enum_idx, conds.get(str(_f.enum_idx), None))
            if val is None:
                return False
            val = float(val)
            if _f.min_val is not None and val < _f.min_val:
                return False
            if _f.max_val is not None and val > _f.max_val:
                return False
            return True

        samples = [sample for sample in samples if _keep(sample)]
    return samples

