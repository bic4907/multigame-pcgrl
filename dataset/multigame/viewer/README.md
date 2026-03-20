# Dataset Viewer

Lightweight browser viewer for local dataset inspection.

## Features

- Shows sample counts per game (including `dungeon`, `pokemon`, `boxoban`, `doom`)
- Select game in browser and browse by index
- Keyboard navigation with left/right arrows (`←` / `→`)
- **Three rendering modes:**
  - **🎨 Raw** – Original game-specific tile colors (per-game palette)
    - ✅ Dungeon, Boxoban, DOOM 지원
    - ⚠️ POKEMON: 팔레트 미정의 → 기본값 사용 (흐릿함)
  - **🗂 Unified** – 7-category unified palette (empty/wall/floor/enemy/object/spawn/hazard)
    - ✅ 모든 게임 완벽 지원 (권장 ✅)
    - 게임 간 비교에 최적
  - **🔤 Symbol** – Tile name text overlay on unified colors
    - ✅ 모든 게임 지원
    - 타일 이름 확인에 유용
- **Live legend** – Shows only tiles present in the current level
- **Tile mapping panel** - Shows `raw tile -> unified category` loaded from `dataset/multigame/tile_mapping.json`
- **Album view** - Shows multiple samples at once (6/8/12 per page), click a card to open single view

## Run

### ⚠️ 중요: 프로젝트 루트에서 실행하세요!

```bash
# 반드시 프로젝트 루트로 이동
cd /home/cilab/Projects/Py/multigame-pcgrl
```

### Method 1: Direct Module Execution (권장 ✅)

```bash
# 프로젝트 루트에서:
python -m dataset.multigame.viewer.server --host 127.0.0.1 --port 8765
```

**❌ 주의: 이것은 작동하지 않습니다!**
```bash
cd dataset/multigame/viewer
python -m dataset.multigame.viewer.server  # ← ModuleNotFoundError!
```

### Method 2: Using __main__.py

```bash
# 프로젝트 루트에서:
python -m dataset.multigame.viewer  # __init__.py가 server를 시작함
```

### Custom Dataset Paths

```bash
# 프로젝트 루트에서:
python -m dataset.multigame.viewer.server \
  --host 127.0.0.1 \
  --port 8765 \
  --dungeon-root /path/to/dungeon_level_dataset \
  --pokemon-root /path/to/five-dollar-model \
  --boxoban-root /path/to/boxoban_levels \
  --doom-root /path/to/doom_levels
```

### PYTHONPATH 설정 (대안)

원하면 현재 디렉토리에서도 실행 가능:

```bash
cd dataset/multigame/viewer
PYTHONPATH=/home/cilab/Projects/Py/multigame-pcgrl python -m dataset.multigame.viewer.server
```

## Usage

1. **Select game** from dropdown (e.g., `dungeon`, `pokemon`, `boxoban`, `doom`)
2. **Switch rendering mode** by clicking tabs:
   - `Raw` – See original palette colors
     - **Note:** POKEMON은 팔레트가 미정의되어 모든 타일이 동일 색상으로 표시됨
     - **해결책:** `Unified` 모드 사용 권장
   - `Unified` – See 7-category abstraction (useful for cross-game comparison)
     - ✅ 모든 게임 완벽 지원 (권장)
   - `Symbol` – See tile names overlaid (e.g., "WAL", "FLO", "ENE")
3. **Navigate samples:**
   - `Prev` / `Next` buttons
   - Arrow keys: `←` / `→`
   - Jump to specific index with `Index` input + `Go`
4. **Album mode:**
   - Set `보기 = Album`
   - Choose `앨범크기` (6 / 8 / 12)
   - Click a thumbnail card to return to single detail view at that index

### 렌더링 모드 선택 가이드

| 게임 | Raw | Unified | Symbol |
|------|-----|---------|--------|
| Dungeon | ✅ | ✅ | ✅ |
| Sokoban | ✅ | ✅ | ✅ |
| POKEMON | ⚠️ (권장 아님) | ✅ **권장** | ✅ |
| DOOM | ✅ | ✅ | ✅ |
| DOOM 2 | ✅ | ✅ | ✅ |

## Notes

- Legend panel updates dynamically to show only tiles used in the current level
- Symbol mode is most readable when tile size ≥ 12px (automatically scaled)
- All rendering happens client-side after initial JSON fetch (fast mode switching)
- Viewer automatically detects available datasets (dungeon, pokemon, boxoban, doom)
- Missing datasets are simply skipped without error

## 문제 해결

### ❌ POKEMON이 핑크색으로 표시됨

**원인**: POKEMON 게임의 팔레트가 tile_mapping.json에 정의되지 않음

**해결책**:
1. **Unified 모드 사용 (권장)** ✅
   - `Unified` 탭을 클릭하면 7-category 색상으로 표시됨
   - 모든 게임에 완벽히 지원됨

2. **tile_mapping.json 업데이트** (장기 해결책)
   - `dataset/multigame/tile_mapping.json`의 pokemon 섹션에 `_tile_colors` 추가

### ⚠️ DOOM 렌더링 경고 (RuntimeWarning)

**원인**: DOOM 맵이 16x16이 아닌 크기를 가짐 (예: 133x96)
- DOOM 맵들이 원본 크기를 유지하고 있음
- 시스템이 자동으로 top-left 16x16으로 정규화함

**현상**: 경고는 표시되지만 렌더링은 정상 작동
```
RuntimeWarning: [doom] ... has shape (133, 96); normalizing to (16, 16)
```

**해결책**:
1. **경고 무시** (현재 권장) - 기능상 문제 없음
2. **DOOM 슬라이싱 활성화** (미래 개선)
   - DoomHandler의 슬라이싱 기능을 활용하여 맵을 작은 섹션으로 분할

### ⚠️ Boxoban 매핑 경고

**원인**: 게임 태그가 `sokoban`이지만 `boxoban` 태그로 요청됨

**현상**:
```
RuntimeWarning: [tile_utils] No mapping found for game 'boxoban'.
```

**해결책**: 자동으로 처리됨 - Unified 모드에서 정상 렌더링

## Mapping Source

Viewer mapping is loaded from:

- `dataset/multigame/tile_mapping.json`

API endpoint:

- `/api/mapping?game=<game_tag>`

The browser caches mapping per game and reuses it while navigating indices.
