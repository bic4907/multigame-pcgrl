"""
results/utils
=============
results/ 스크립트들이 공유하는 유틸리티 패키지.

내부 구조:
    core/       핵심 헬퍼 — io, stats, normalization, run_output
    pipeline/   파이프라인 step runner
    doc/        문서 변환 도구 (embed_markdown_images_base64, render_markdown_pdf)
    dev/        개발·테스트 도구 (_make_test_data)

빠른 임포트 예시:
    from utils import load_cfg, make_run_dir, setup_logger
    from utils import parse_run_tokens, sort_key_reward_enum
    from utils import safe_std, to_float
    from utils import run_experiment_processing
"""

from .core.run_output import load_cfg, make_run_dir, setup_logger
from .core.io import (
    normalize_reward_enum,
    parse_run_tokens,
    iter_summary_paths,
    iter_results_paths,
    read_summary,
    sort_key_reward_enum,
    load_run_config,
    get_game_split,
)
from .core.stats import safe_std, to_float, iqr_mean
from .core.normalization import (
    compute_normalization_scale,
    apply_normalization,
    save_normalization_scale,
    load_normalization_scale,
)
from .pipeline import (
    run_experiment_processing,
    run_processing_step,
    get_experiment_names,
    get_experiment_projects,
    PROCESSING_STEPS,
)

__all__ = [
    # core.run_output
    "load_cfg", "make_run_dir", "setup_logger",
    # core.io
    "normalize_reward_enum", "parse_run_tokens",
    "iter_summary_paths", "iter_results_paths", "read_summary",
    "sort_key_reward_enum", "load_run_config", "get_game_split",
    # core.stats
    "safe_std", "to_float", "iqr_mean",
    # core.normalization
    "compute_normalization_scale", "apply_normalization",
    "save_normalization_scale", "load_normalization_scale",
    # pipeline
    "run_experiment_processing", "run_processing_step",
    "get_experiment_names", "get_experiment_projects", "PROCESSING_STEPS",
]
