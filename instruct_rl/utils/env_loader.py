"""
instruct_rl/utils/env_loader.py
================================
.env 파일을 읽어 os.environ 에 등록하는 유틸리티.
dotenv 패키지 없이 직접 파싱한다.
"""

from __future__ import annotations

import os
from pathlib import Path


def load_dotenv(path: str | Path | None = None) -> dict[str, str]:
    """.env 파일을 한 줄씩 읽어 ``os.environ`` 에 등록한다.

    Parameters
    ----------
    path : str | Path | None
        .env 파일 경로. None 이면 프로젝트 루트(이 파일 기준 3단계 상위)의
        .env 를 사용한다.

    Returns
    -------
    dict[str, str]
        파싱·등록된 key-value 쌍.

    Notes
    -----
    - 빈 줄, ``#`` 으로 시작하는 주석 줄은 무시한다.
    - 값 양쪽의 작은따옴표 / 큰따옴표를 자동으로 벗긴다.
    - 이미 os.environ 에 존재하는 키는 덮어쓰지 않는다(시스템 환경변수 우선).
    """
    if path is None:
        # instruct_rl/utils/env_loader.py → 프로젝트 루트
        root = Path(__file__).resolve().parent.parent.parent
        path = root / ".env"
    else:
        path = Path(path)

    loaded: dict[str, str] = {}

    if not path.is_file():
        return loaded

    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()

            # 빈 줄·주석 무시
            if not line or line.startswith("#"):
                continue

            # export KEY=VALUE 형태 지원
            if line.startswith("export "):
                line = line[len("export "):].strip()

            if "=" not in line:
                continue

            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip()

            if not key:
                continue

            # 따옴표 제거
            if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
                value = value[1:-1]

            loaded[key] = value

            # 시스템 환경변수가 이미 있으면 덮어쓰지 않음
            if key not in os.environ:
                os.environ[key] = value

    return loaded


def get_wandb_key() -> str | None:
    """WANDB_API_KEY 를 환경변수에서 가져온다.

    .env 가 아직 로드되지 않았을 수 있으므로 한 번 더 load_dotenv() 를 호출한다.
    """
    load_dotenv()
    return os.environ.get("WANDB_API_KEY")

