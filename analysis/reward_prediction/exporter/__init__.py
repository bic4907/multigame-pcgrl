"""
exporter
========
MGPCGRL 체크포인트에서 reward_enum / condition 예측 결과를 CSV로 내보냅니다.

Public API
----------
run_export(args) → None
    argparse.Namespace(또는 동등한 객체)를 받아 전체 export 파이프라인 실행.

ExportConfig(dataclass)
    pipeline.py 에서 설정을 넘길 때 사용하는 간단한 설정 컨테이너.
"""
from .checkpoint_csv import ExportConfig, run_export

__all__ = ["ExportConfig", "run_export"]

