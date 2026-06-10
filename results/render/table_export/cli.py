from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

from .h5_utils import analyze_h5_file
from .markdown import export_markdown_table
from .models import RunResult
from .utils import load_config
from .wandb_artifacts import process_config_items


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _fallback_single_h5(config, script_dir: Path):
    if isinstance(config, dict) and "h5_path" in config:
        h5_path = Path(config["h5_path"])
    else:
        h5_path = script_dir.parent.parent / "eval.h5"

    if not h5_path.exists():
        raise FileNotFoundError(f"h5 파일을 찾을 수 없습니다: {h5_path}")

    h5_stats = analyze_h5_file(h5_path)
    return [
        RunResult(
            method="Local",
            run_url="N/A",
            run_name=h5_path.name,
            h5_path=h5_path,
            h5_stats=h5_stats,
        )
    ]


def main():
    print("=" * 80)
    print("W&B Run 평가 H5 다운로드 → 분석 → Markdown Export")
    print("=" * 80)

    script_dir = Path(__file__).resolve().parent.parent
    config_path = script_dir / "config.json"
    print(f"\n📄 Config 로드: {config_path}")

    config = load_config(config_path)
    print(f"   ✅ {len(config) if isinstance(config, list) else 'dict'} items loaded")
    reward_enums = (
        config.get("reward_enums", [0, 1, 2, 3, 4])
        if isinstance(config, dict)
        else [0, 1, 2, 3, 4]
    )

    if isinstance(config, list):
        run_items = config
    elif isinstance(config, dict) and isinstance(config.get("runs"), list):
        run_items = config["runs"]
    else:
        run_items = None

    if run_items is not None:
        run_results = process_config_items(run_items, script_dir, reward_enums=reward_enums)
    else:
        run_results = _fallback_single_h5(config, script_dir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = script_dir / "outputs" / timestamp
    output_path = output_dir / "table.md"
    content = export_markdown_table(config, run_results, output_path)

    print(f"\n📝 Markdown 생성: {output_path}")
    print(f"   ✅ {len(run_results)}개 항목 처리 완료")
    print(f"   파일 크기: {len(content)} bytes")

    print("\n" + "=" * 80)
    print("생성된 Markdown 미리보기:")
    print("=" * 80)
    print(content)
    print("=" * 80)
    print(f"\n✨ 완료! open {output_path}")

