"""
dataset/scripts/render_samples.py
---------------------------------
간단한 스크립트: 프로젝트 내 `dataset` 폴더에 있는 멀티게임 데이터셋에서
샘플을 렌더링하여 PNG 이미지로 저장합니다.

사용법 (프로젝트 루트에서):
    python dataset/scripts/render_samples.py

생성물:
    dataset/samples/zelda_sample.png
    dataset/samples/dungeon_sample.png
    dataset/samples/grid_samples.png

참고: 이 스크립트는 로컬 `dataset` 디렉터리에 있는 `multigame` 패키지를 직접 임포트합니다.
"""
from __future__ import annotations

from pathlib import Path

HERE = Path(__file__).resolve().parent.parent

from dataset.multigame.dataset import MultiGameDataset
from dataset.multigame.render import save_rendered, save_grid

OUT_DIR = HERE / "samples"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("Loading datasets (this may take a moment)...")
    ds = MultiGameDataset(vglc_games=["zelda", "mario", "lode_runner"], include_dungeon=True)
    print(f"Loaded MultiGameDataset with total samples: {len(ds)}")

    # Zelda 샘플 (첫 번째)
    zelda_pool = ds.by_game("zelda")
    if zelda_pool:
        z = zelda_pool[0]
        out = OUT_DIR / "zelda_sample.png"
        save_rendered(z, out, tile_size=8)
        print("Saved:", out)
    else:
        print("No Zelda samples found")

    # Dungeon 샘플 (첫 번째)
    dungeon_pool = ds.by_game("dungeon")
    if dungeon_pool:
        d = dungeon_pool[0]
        out = OUT_DIR / "dungeon_sample.png"
        save_rendered(d, out, tile_size=8)
        print("Saved:", out)
    else:
        print("No Dungeon samples found")

    # 그리드: Zelda 3개 + Dungeon 3개 (가능한 경우)
    grid_samples = []
    grid_samples.extend(zelda_pool[:3])
    grid_samples.extend(dungeon_pool[:3])
    if grid_samples:
        out = OUT_DIR / "grid_samples.png"
        save_grid(grid_samples, out, cols=3, tile_size=6)
        print("Saved grid:", out)
    else:
        print("Not enough samples for grid")


if __name__ == "__main__":
    main()

