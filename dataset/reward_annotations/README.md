# Reward Annotation Pipeline

게임 레벨 샘플에 대해 reward measure를 계산하고, OpenAI Batch API로 자연어 instruction을 생성하는 2단계 파이프라인.

```
캐시 (MultiGameDataset)
    ↓  annotate.py
{key}.ann.json  (5개 measure 값, instruction 비어있음)
    ↓  generate_instructions.py --run
{key}.ann.json  (instruction_raw / instruction_uni 채워진 최종본)
```

---

## 전체 실행 (--run)

```bash
# Step 1: measure annotation
python dataset/reward_annotations/annotate.py

# Step 2: instruction 생성 (제출 → 완료 대기 → 자동 저장)
python dataset/reward_annotations/generate_instructions.py --run
```

`--run`은 게임별로 배치를 제출하고, 완료될 때까지 폴링하며, 완료 즉시 ann.json에 결과를 반영한다.

### 부분 실행 (특정 게임 / enum)

```bash
python dataset/reward_annotations/annotate.py --games doom zelda

python dataset/reward_annotations/generate_instructions.py --run \
    --games doom zelda --enums 0 1
```

---

## Step 1 — annotate.py

캐시(`dataset/multigame/cache/artifacts/`)에서 맵 배열을 읽어 5가지 measure를 계산하고 `{key}.ann.json`에 저장한다.

### Reward Enum 정의

| enum | feature_name | 내용 |
|------|------|------|
| 0 | `region` | 연결된 passable 영역 수 |
| 1 | `path_length` | 가장 긴 경로 길이 |
| 2 | `interactable_count` | Interactive 타일 수 |
| 3 | `hazard_count` | Hazard 타일 수 |
| 4 | `collectable_count` | Collectable 타일 수 |

**passable 기준**: unified EMPTY(1) + HAZARD(4) + COLLECTABLE(5) — 모든 게임 공통

### 게임별 sub_condition (count 계산 기준 타일)

| game | interactable | hazard | collectable |
|------|------|------|------|
| doom | spawn + door + danger | enemy | item |
| zelda | door + block + start | mob | object |
| sokoban | box | — | — |
| pokemon | spawn + water | enemy | object |
| dungeon | — | enemy | treasure |

### 주요 옵션

| 옵션 | 기본값 | 설명 |
|------|------|------|
| `--games` | 전체 5개 | 처리할 게임 목록 |
| `--cache-dir` | `dataset/multigame/cache/artifacts` | 캐시 루트 디렉토리 |
| `--force` | False | 기존 ann.json 덮어쓰기 |

### 출력

`dataset/multigame/cache/artifacts/{hash}/{game}/{key}.ann.json`

```json
{
  "game": "doom",
  "n_samples": 1000,
  "has_instructions": false,
  "annotations": [
    {
      "key": "dm000000",
      "source_id": "Doom1_map01_000",
      "reward_enum": 0,
      "feature_name": "region",
      "condition_0": 3.0,
      "instruction_raw": null,
      "instruction_uni": null
    }
  ]
}
```

---

## Step 2 — generate_instructions.py

ann.json을 읽어 각 샘플에 대해 GPT에게 `instruction_raw` / `instruction_uni` 생성을 요청하고 결과를 ann.json에 저장한다.

### --run 동작 흐름

```
1. threshold=None 인 (game, feature) 조합 → instruction을 "None"으로 직접 채움
2. 미처리 행에 대해 게임별 JSONL 파일 생성 (batches/{timestamp}.jsonl)
3. OpenAI Batch API에 게임별로 배치 제출
4. 폴링 루프 (기본 10초 간격):
     - 완료된 배치 → 결과 파싱 → ann.json 업데이트
     - 실패/만료 → 로그 출력 후 skip
5. 모든 배치 완료 시 종료
```

### instruction_raw vs instruction_uni

| 필드 | 기준 |
|------|------|
| `instruction_raw` | 게임 원본 타일 이름 사용 (ENEMY, DOOR, SPAWN 등) |
| `instruction_uni` | unified 카테고리 사용 (empty / wall / interactive / hazard / collectable) |

### 주요 옵션

| 옵션 | 기본값 | 설명 |
|------|------|------|
| `--games` | 전체 5개 | 처리할 게임 목록 |
| `--enums` | `0 1 2 3 4` | 처리할 reward_enum |
| `--cache-dir` | `dataset/multigame/cache/artifacts` | 캐시 루트 |
| `--force` | False | 이미 채워진 instruction도 재생성 |
| `--poll-interval` | 10 | 폴링 간격 (초) |
| `--limit` | None | 처리할 최대 행 수 (테스트용) |

### 단계별 개별 실행

```bash
# JSONL 생성 + 배치 제출만 (완료 대기 없음)
python dataset/reward_annotations/generate_instructions.py --submit

# 완료된 배치 결과 조회 + ann.json 업데이트
python dataset/reward_annotations/generate_instructions.py --retrieve BATCH_ID

# 배치 상태 확인
python dataset/reward_annotations/generate_instructions.py --status BATCH_ID

# 배치 이력 조회
python dataset/reward_annotations/generate_instructions.py --log
```

배치 제출 이력은 `batches/batch_log.csv`에 기록된다.

---

## 프롬프트 수정

### system_prompt.txt

GPT에게 전달되는 시스템 지시문. 출력 형식, 문체, 제약 조건을 정의한다.

```
You are a game level description writer for PCGRL.
Write one sentence (≤10 words) describing the level's intensity.
Output JSON: {"instruction_raw": "...", "instruction_uni": "..."}
```

**주요 수정 포인트:**
- 문장 길이 제한 (`STRICT LIMIT: 10 words or fewer`)
- 문체 방향 (`brief, factual description` / `NOT a design command`)
- 금지 표현 (숫자·수치 언급 금지)

### instruction_config.py

유저 프롬프트 생성에 사용되는 모든 설정 상수. 아래 항목을 수정해 프롬프트 내용을 제어한다.

#### CUSTOM_THRESHOLDS

feature 값을 4개 intensity level로 분할하는 경계값 (3개 경계 → 4 구간).
`None`이면 GPT 호출 없이 `"None"` 문자열로 채워진다.

```python
CUSTOM_THRESHOLDS = {
    "dungeon_region":       [1.5, 4.5, 14.5],  # very few / somewhat few / somewhat many / very many
    "sokoban_hazard_count": None,               # 소코반에 hazard 없음 → 스킵
    ...
}
```

> 경계값 변경 후 instruction을 재생성하려면 `--force` 옵션 사용.

#### FEATURE_ZONE_LABELS

intensity level 0~3에 대응하는 레이블 문자열. 유저 프롬프트의 `Intensity level` 항목에 표시된다.

```python
FEATURE_ZONE_LABELS = {
    "region": ["very few regions", "somewhat few regions",
               "somewhat many regions", "very many regions"],
    ...
}
```

#### VOCAB_SETS

각 intensity level별로 GPT에게 제안하는 어휘 목록. 프롬프트에 `Suggested vocabulary` 힌트로 삽입되며, 매 요청마다 목록에서 랜덤하게 1개 선택된다.

```python
VOCAB_SETS = {
    "region": [
        ["few", "sparse", "marginal"],           # level 0 — very few
        ["some", "moderate", "slight"],           # level 1 — somewhat few
        ["several", "multiple", "partitioned"],   # level 2 — somewhat many
        ["fragmented", "numerous", "many"],       # level 3 — very many
    ],
    ...
}
```

#### GAME_DESCRIPTIONS / FEATURE_DESCRIPTIONS

유저 프롬프트 상단에 삽입되는 게임 및 feature 설명. GPT가 문맥을 이해하는 데 사용된다.

```python
GAME_DESCRIPTIONS = {
    "doom": "Doom (top-down view of a first-person shooter dungeon map)",
    ...
}

FEATURE_DESCRIPTIONS = {
    "region": "number of disconnected passable-area clusters — count of separate walkable zones",
    ...
}
```

---

## 모델 설정

`generate_instructions.py` 상단에서 직접 수정:

```python
MODEL       = "gpt-5.4-mini"   # 사용할 OpenAI 모델
MAX_TOKENS  = 300              # 최대 출력 토큰
TEMPERATURE = 2.0              # 다양성 (높을수록 다양한 표현)
```

---

## 환경 변수

`.env` 파일에 설정:

```
OPENAI_API_KEY=sk-...
```

---

## 파일 구조

```
dataset/reward_annotations/
├── annotate.py              # Step 1: measure 계산 → ann.json 생성
├── generate_instructions.py # Step 2: OpenAI Batch API → instruction 채움
├── instruction_config.py    # 프롬프트 설정 상수 (threshold, vocab, 설명)
├── system_prompt.txt        # GPT 시스템 프롬프트
└── batches/
    ├── batch_log.csv        # 배치 제출/완료 이력
    └── batch_{timestamp}.jsonl  # 게임별 배치 요청 파일
```
