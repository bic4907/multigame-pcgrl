# results/ — 실험 결과 처리 파이프라인

> 설정값(프로젝트 이름, 경로 등) → **[CONFIG.md](CONFIG.md)**

---

## 디렉토리 구조

```
results/
│   # ── Entry point (유일한 직접 실행 스크립트) ─────────────
├── run_pipeline.py                     # ★ 파이프라인 통합 진입점
│
│   # ── Config ──────────────────────────────────────────────
├── config.json                         # 실험/프로젝트 설정
├── run_config_exclude_keys.json        # W&B run_config 저장 제외 키 목록
│
│   # ── Shared utilities ────────────────────────────────────
└── utils/
    ├── __init__.py                     # 전체 public 심볼 re-export
    │
    ├── core/                           # 핵심 공유 헬퍼
    │   ├── io.py                       # 파일 탐색, 토큰 파싱, CSV 읽기
    │   ├── stats.py                    # safe_std, to_float, iqr_mean
    │   ├── normalization.py            # min-max 정규화 (계산·적용·저장·로드)
    │   └── run_output.py              # 출력 디렉토리 생성, 로거, config 로드
    │
    ├── pipeline/                       # 파이프라인 step runner
    │   └── __init__.py                # run_experiment_processing, PROCESSING_STEPS
    │
    ├── experiment/                     # 실험 처리 스크립트 (pipeline이 호출)
    │   ├── benchmark.py               # 벤치마크 테이블 + re.png / re_overall.png
    │   ├── condition_progress_report.py # condition vs metric 플롯 + 리포트
    │   ├── seen_unseen_report.py      # seen/unseen 분리 분석 + unseen.png
    │   ├── make_eval_summary.py       # ctrl_sim.csv → results/summary.csv
    │   ├── reward_enum_visualizer.py  # reward_enum 타일맵 시각화
    │   ├── process_allseen.py         # allseen 실험 래퍼
    │   └── process_unseen.py
    │
    ├── wandb/                          # W&B 연동 도구
    │   └── eval_downloader.py         # eval 아티팩트 다운로드
    │
    ├── doc/                            # 문서 변환 도구
    │   ├── embed_markdown_images_base64.py
    │   └── render_markdown_pdf.py
    │
    └── dev/                            # 개발·테스트 도구
        └── _make_test_data.py
```

---

## 파이프라인 흐름

```
W&B
 │
 ▼ [Step 1] eval_downloader.py
    wandb_projects/<project>/<run>/<eval>/
        ctrl_sim.csv   diversity.csv   eval.h5   run_config.json
 │
 ▼ [Step 2] make_eval_summary.py
    wandb_projects/<project>/<run>/<eval>/
        results.csv    summary.csv
 │
 ▼ [Step 3] benchmark.py           (allseen 실험)
    outputs/<run_dir>/
        benchmark_table.md/.csv
        re.png          re_overall.png    re_game.png
 │
 ▼ [Step 4] condition_progress_report.py
    outputs/<run_dir>/
        plots/          report.md
 │
 ▼ [Step 5] seen_unseen_report.py  (unseen 실험)
    outputs/<run_dir>/
        unseen.png
        seen_table.md/.csv    unseen_table.md/.csv
        seen_unseen_table.md/.csv
```

---

## 빠른 시작

### 전체 파이프라인 한 번에 실행

```bash
# 모든 experiment 전체 단계 실행
python results/run_pipeline.py

# 특정 experiment만
python results/run_pipeline.py --experiment allseen
python results/run_pipeline.py --experiment unseen

# 특정 step만
python results/run_pipeline.py --steps 3
python results/run_pipeline.py --steps 4 5

# 실행 계획만 확인 (실제 실행 없음)
python results/run_pipeline.py --dry-run

# step 목록 보기
python results/run_pipeline.py --list
```

### 각 step 개별 실행

```bash
# Step 1: W&B 다운로드
python results/eval_downloader.py
python results/eval_downloader.py --experiment allseen
python results/eval_downloader.py --h5              # eval.h5 포함
python results/eval_downloader.py --finished-only --workers 4

# Step 2: ctrl_sim.csv → results/summary.csv
python results/make_eval_summary.py
python results/make_eval_summary.py --experiment allseen --input wandb_projects

# Step 3: 벤치마크 테이블 + 플롯
python results/benchmark.py
python results/benchmark.py --experiment allseen
python results/benchmark.py --group-by folder_game_reward_enum
python results/benchmark.py --no-plot

# Step 4: condition vs metric 리포트
python results/condition_progress_report.py
python results/condition_progress_report.py --experiment allseen

# Step 5: seen/unseen 분리 분석
python results/seen_unseen_report.py --experiment unseen
python results/seen_unseen_report.py --experiment unseen --no-plot
```

---

## 출력 파일

| 파일 | 설명 |
|------|------|
| `benchmark_table.md` / `.csv` | `--group-by` 기준 집계 테이블 |
| `re.png` | reward_enum 별 모델 비교 바 플롯 |
| `re_overall.png` | reward_enum 통합 단일 바 플롯 |
| `re_game.png` | game × reward_enum 서브플롯 |
| `unseen.png` | unseen 게임 성능 바 플롯 + seen 기준선 |
| `seen_table.md/.csv` | seen 게임만 집계 |
| `unseen_table.md/.csv` | unseen 게임만 집계 |
| `seen_unseen_table.md/.csv` | seen vs unseen 비교 |
| `report.md` | condition vs metric 플롯 리포트 |
| `normalization_scale.json` | 파이프라인 공유 정규화 스케일 |

---

## utils/ 패키지

```python
# utils/__init__.py 가 모든 심볼을 re-export하므로 아래처럼 바로 임포트 가능
from utils import load_cfg, make_run_dir, setup_logger
from utils import parse_run_tokens, sort_key_reward_enum
from utils import safe_std, to_float, iqr_mean
from utils import compute_normalization_scale, apply_normalization
from utils import run_experiment_processing
```

| 모듈 | 주요 제공 기능 |
|------|--------------|
| `utils.core.run_output` | `load_cfg`, `make_run_dir`, `setup_logger` |
| `utils.core.io` | `parse_run_tokens`, `iter_results_paths`, `read_summary`, `get_game_split` |
| `utils.core.stats` | `safe_std`, `to_float`, `iqr_mean` |
| `utils.core.normalization` | `compute_normalization_scale`, `apply_normalization`, 저장/로드 |
| `utils.pipeline` | `run_experiment_processing`, `run_processing_step`, `PROCESSING_STEPS` |
| `utils.doc.embed_markdown_images_base64` | MD 이미지 base64 임베드 |
| `utils.doc.render_markdown_pdf` | Markdown → PDF 변환 |

---

## 입력 파일 형식

**`summary.csv`** — `<input>/<project>/<run>/[<eval>/]summary.csv`
```
metric,mean
progress,0.832
vit_score,0.741
tpkldiv,1.23
diversity,0.95
```

**`results.csv`** — `<input>/<project>/<run>/[<eval>/]results.csv`
```
game,reward_enum,progress,vit_score,tpkldiv,diversity
dungeon,0,0.84,0.73,1.21,0.96
```

**`run_config.json`** — seen/unseen 게임 분류 정보
```json
{
  "seen_games": ["dungeon", "pokemon", "sokoban"],
  "unseen_games": ["doom", "zelda"]
}
```

run/eval 폴더명은 `game-all_re-0_s-42` 형식 토큰으로 자동 파싱됩니다.

---

## 의존성

```bash
pip install matplotlib seaborn pandas h5py tqdm
```

---

## 리팩토링 이력

| 변경 | 내용 |
|------|------|
| `process_shared.py` → `utils/pipeline.py` | 공유 파이프라인 유틸을 utils 패키지로 이전 |
| `utils/__init__.py` | 모든 서브모듈 심볼 re-export 추가 |
| `condition_progress_report.py` | 중복 `_load_cfg()` 제거, `utils.run_output.load_cfg` 사용 |
| `benchmark.py`, `seen_unseen_report.py` | `make_run_dir`/`setup_logger` 모듈 레벨 임포트로 정리 |
