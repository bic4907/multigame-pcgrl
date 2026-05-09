# 설정 레퍼런스

모든 설정값은 **`results/config.json`** 한 파일에서 관리합니다.  
스크립트 실행 시 자동으로 읽으며, 값이 없으면 각 스크립트의 하드코딩 기본값이 사용됩니다.

> **출력 경로는 config.json에 고정하지 않습니다.**  
> 스크립트 실행마다 `<outputs_root>/<script명>_<YYYYMMDD_HHMMSS>/` 폴더가 자동 생성되고,  
> 실행에 사용된 설정이 `used_config.json`으로 함께 저장됩니다.

---

## config.json 전체 구조

```json
{
  "outputs_root": "results/outputs",
  "wandb": {
    "target_projects": ["aaai27_eval_cpcgrl", "aaai27_eval_cpcgrl_all"],
    "num_workers": 4
  },
  "paths": {
    "eval_output":            "results/eval",
    "gamewise_target_dir":    "results/results/eval/aaai27_eval_cpcgrl",
    "condition_report_input": "results/wandb_download",
    "reward_viz_root":        "results/wandb_download/aaai27_eval_cpcgrl"
  },
  "games": {
    "code_to_name": { "dg": "dungeon", "pk": "pokemon", "sk": "sokoban", "dm": "doom", "zd": "zelda" },
    "colors":       { "doom": "#1f77b4", "dungeon": "#d62728", "pokemon": "#2ca02c", "sokoban": "#ff7f0e", "zelda": "#9467bd" }
  },
  "metrics": {
    "default_order":  ["progress", "vit_score", "tpkldiv", "diversity"],
    "display_names":  { "progress": "Progress", "vit_score": "ViTScore", "tpkldiv": "TPKL-Div", "diversity": "Diversity" }
  },
  "plots": {
    "preferred_folder_order": ["aaai27_eval_cpcgrl", "aaai27_eval_cpcgrl_gamegroup", "aaai27_eval_cpcgrl_all"]
  },
  "reward_enums": {
    "labels":          { "0": "Region", "1": "Path Length", "2": "Interactable", "3": "Hazard", "4": "Collectable" },
    "num_slots":       4,
    "run_dir_pattern": "cpcgrl_game-all_re-{reward_enum}_exp-def_s-0"
  }
}
```

---

## 키별 설명

### `outputs_root`

| 키 | 기본값 | 설명 |
|----|--------|------|
| `outputs_root` | `"results/outputs"` | 타임스탬프 출력 폴더의 루트 경로 |

실행 시 `<outputs_root>/build_benchmark_table_20260507_143022/` 형태로 자동 생성됩니다.  
각 폴더 안에는 결과물과 함께 `used_config.json`이 저장됩니다.

---

### `wandb`

| 키 | 설명 | 사용 스크립트 |
|----|------|--------------|
| `target_projects` | 다운로드할 W&B 프로젝트 목록 | `eval_downloader.py` |
| `num_workers` | 병렬 다운로드 스레드 수 | `eval_downloader.py` |

> W&B 엔티티(`DEFAULT_ENTITY`)는 `sweep/wandb_utils/config.py` 에서 별도 관리합니다.

---

### `paths` (input paths only)

| Key | Description | Used by |
|----|------|--------------|
| `eval_output` | Download root for `eval_downloader` | `eval_downloader.py` |
| `gamewise_target_dir` | Target folder for per-game summary rebuild | `make_gamewise_summary.py` |
| `condition_report_input` | Root for searching `ctrl_sim.csv` | `condition_progress_report.py` |
| `reward_viz_root` | Root for `eval.h5` / `ctrl_sim.csv` | `reward_enum_visualizer.py` |

```json
"paths": {
  "eval_output":            "wandb_projects",
  "gamewise_target_dir":    "wandb_projects/aaai27_eval_cpcgrl",
  "condition_report_input": "wandb_projects",
  "reward_viz_root":        "wandb_projects/aaai27_eval_cpcgrl"
}
```

---

### `games`

| 키 | 설명 | 사용 스크립트 |
|----|------|--------------|
| `code_to_name` | 폴더명 게임 코드 → 게임 이름 매핑 | `make_gamewise_summary.py` |
| `colors` | 게임별 플롯 색상 (hex) | `condition_progress_report.py` |

---

### `metrics`

| 키 | 설명 | 사용 스크립트 |
|----|------|--------------|
| `default_order` | 테이블/플롯의 메트릭 표시 순서 | `build_benchmark_table.py` |
| `display_names` | 메트릭 레이블 (플롯 축 이름) | `build_benchmark_table.py` |

---

### `plots`

| 키 | 설명 | 사용 스크립트 |
|----|------|--------------|
| `preferred_folder_order` | 비교 플롯에서 폴더(실험) 표시 순서 | `build_benchmark_table.py` |

---

### `reward_enums`

| 키 | 설명 | 사용 스크립트 |
|----|------|--------------|
| `labels` | reward_enum 번호 → 표시 이름 | `condition_progress_report.py`, `reward_enum_visualizer.py` |
| `num_slots` | 대표 샘플 구간 수 | `reward_enum_visualizer.py` |
| `run_dir_pattern` | reward_enum별 run 폴더명 패턴 (`{reward_enum}` 치환) | `reward_enum_visualizer.py` |

