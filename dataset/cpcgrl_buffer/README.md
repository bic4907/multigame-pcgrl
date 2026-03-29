# CPCGRL Pair Dataset

## 개요

CPCGRL (Conditional PCGRL) 에이전트의 학습 중 수집된 trajectory 버퍼로부터
**연속 2-step env_map 쌍** `(before, after)`을 추출하여 전처리한 데이터셋입니다.

5개 reward_enum (1=region, 2=path_length, 3=block, 4=bat_amount, 5=bat_direction) 별로
학습된 에이전트의 버퍼에서 골고루 추출한 뒤, **쌍 단위 중복을 완전히 제거**하여
단일 `.npz` 파일로 저장합니다.

## 파일

```
dataset/cpcgrl_buffer/
├── __init__.py                 # CPCGRLBufferDataset, MapTransitionPair export
├── build_pair_dataset.py       # 전처리 스크립트
├── cpcgrl_pair_dataset.npz     # 최종 데이터셋 (단일 파일)
├── dataset.py                  # 데이터셋 클래스
├── metadata.json               # 빌드 메타데이터
└── README.md
```

## 데이터 Shape

`cpcgrl_pair_dataset.npz` 내부 키:

| 키 | Shape | dtype | 설명 |
|---|---|---|---|
| `env_map_pairs` | `(6792, 2, 16, 16)` | int32 | (before, after) env_map 쌍 |
| `reward_enums` | `(6792,)` | int32 | 각 쌍의 reward_enum 라벨 (1~5) |
| `timesteps` | `(6792,)` | int64 | 각 쌍의 시작 timestep (total_timesteps 기준) |

- `env_map_pairs[:, 0]` → **before** (t 시점의 맵)
- `env_map_pairs[:, 1]` → **after** (t+1 시점의 맵)
- 맵 크기: 16×16, 타일 값: dungeon3 기준 정수 (0~7)

## reward_enum 별 분포

| reward_enum | feature | 쌍 수 |
|:-----------:|---------|------:|
| 1 | region | 3,469 |
| 2 | path_length | 1,616 |
| 3 | block | 884 |
| 4 | bat_amount | 534 |
| 5 | bat_direction | 289 |
| **합계** | | **6,792** |

> 중복 제거 전 20,000 쌍 → 제거 후 **6,792** 쌍.
> reward_enum 별 중복률이 다른 이유는 일부 에이전트가 유사한 맵을 반복 생성하기 때문입니다.

## metadata.json

빌드 시 자동 생성되는 메타데이터 파일:

| 키 | 설명 | 예시 |
|---|---|---|
| `created_at` | 생성 시각 | `"2026-03-29 16:37:24"` |
| `hostname` | 생성 PC 이름 | `"MacBookPro.local"` |
| `platform` | OS/아키텍처 | `"macOS-15.7.4-arm64-arm-64bit"` |
| `total_pairs` | 중복 제거 후 최종 쌍 수 | `6554` |
| `total_before_dedup` | 중복 제거 전 쌍 수 | `16000` |
| `tile_min` / `tile_max` | env_map 타일 값 범위 | `1` / `7` |
| `env_map_shape` | 데이터 shape | `[6554, 2, 16, 16]` |
| `reward_enum_distribution` | re별 쌍 수 | `{"1": 483, "3": 2745, ...}` |
| `seed` | 랜덤 시드 | `42` |

## 생성 방법

```bash
# saves/ 에 학습 버퍼가 있는 상태에서 실행
python dataset/cpcgrl_buffer/build_pair_dataset.py \
    --saves_dir saves \
    --pairs_per_re 4000 \
    --seed 42
```

### 전처리 파이프라인

1. `saves/` 에서 `_re-{N}_` 패턴으로 reward_enum 별 버퍼 디렉토리 자동 탐색
2. 각 버퍼의 env_map 을 전부 로드하여 **연속 2-step 쌍** `(env_map[t], env_map[t+1])` 생성
   - `done=True` 경계 건너뜀 (에피소드 끊긴 곳)
   - timestep 비정상 점프 건너뜀
3. reward_enum 당 4,000개씩 중복 없이 랜덤 샘플링
4. 전체 병합 후 **쌍 단위 중복 제거** (env_map 2장이 완전히 동일한 쌍 제거)
5. 셔플 후 단일 `.npz` 로 저장

### 원본 버퍼 출처

학습 50%~100% 구간에서 `BufferCollector`가 수집한 trajectory:

```
saves/model-contconv_exp-def_game-dungeon_re-{1..5}_vec_ro_s-0/buffer/
    buffer_000000_ts460800.npz
    buffer_000001_ts537600.npz
    ...
```

## 사용법

### 기본 사용

```python
from dataset.cpcgrl_buffer import CPCGRLBufferDataset

ds = CPCGRLBufferDataset()
print(ds)
# CPCGRLBufferDataset(n=12655, reward_enums=[1, 3, 4, 5])

print(len(ds))       # 12655
print(ds.map_shape)  # (16, 16)
```

### MapTransitionPair — before/after 쌍 접근

```python
pair = ds[0]
print(pair)
# MapTransitionPair(re=3, ts=537604, map=16x16, changes=1)

pair.before       # (16, 16) int32 — t 시점의 맵
pair.after        # (16, 16) int32 — t+1 시점의 맵
pair.pair         # (2, 16, 16)    — 스택된 형태
pair.reward_enum  # 3
pair.timestep     # 537604

# 변화 분석
pair.diff          # (16, 16) int16 — after - before
pair.changed_mask  # (16, 16) bool  — 변경된 위치
pair.n_changes     # 1              — 변경된 타일 수
```

### reward_enum 필터링

```python
# region(re-1) 쌍만 가져오기
region_ds = ds.by_reward_enum(1)
print(region_ds)
# CPCGRLBufferDataset(n=736, reward_enums=[1])

# 여러 reward_enum 동시 필터
sub_ds = ds.by_reward_enum(1, 3)
print(sub_ds)
# CPCGRLBufferDataset(n=6205, reward_enums=[1, 3])
```

### 배치 접근 (numpy 배열)

```python
ds.pairs             # (N, 2, 16, 16) — 전체
ds.before_maps       # (N, 16, 16)    — 모든 before
ds.after_maps        # (N, 16, 16)    — 모든 after
ds.reward_enums_array  # (N,) int32
ds.timesteps_array     # (N,) int64
```

### 랜덤 샘플링

```python
pair = ds.sample(seed=42)          # 1개
pairs = ds.sample(n=100, seed=42)  # 100개 리스트
```

### 슬라이싱

```python
first_10 = ds[:10]      # CPCGRLBufferDataset(n=10, ...)
```

### 통계 / 메타데이터

```python
ds.summary()
# {'total_pairs': 12655, 'map_shape': (16, 16),
#  'tile_min': 1, 'tile_max': 7,
#  'reward_enum_distribution': {1: 736, 3: 5469, 4: 3347, 5: 3103}}

ds.metadata
# {'created_at': '2026-03-29 16:40:29', 'hostname': 'MacBookPro.local', ...}
```

