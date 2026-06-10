"""
dataset/multigame/render.py
===========================
타일 그리드 렌더링 유틸리티.

- 팔레트 기반 렌더링 (array_to_rgb, render_sample 등)
- 타일 이미지 기반 렌더링 (GameLevelRenderer)

외부 의존: numpy, Pillow (PIL)
Pillow가 없을 경우 numpy 배열만 반환.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from .base import GameSample
from .handlers.vglc_games import PALETTES
from .handlers.dungeon_handler import DUNGEON_PALETTE

# ── 전체 팔레트 통합 ─────────────────────────────────────────────────────────────
_ALL_PALETTES: Dict[str, Dict[int, Tuple[int, int, int]]] = {
    **PALETTES,
    "dungeon": DUNGEON_PALETTE,
}

_DEFAULT_UNKNOWN_COLOR = (255, 0, 255)   # 매핑 없을 때 마젠타
_DEFAULT_TILE_SIZE     = 16              # 픽셀 단위 타일 크기


def get_palette(game: str) -> Dict[int, Tuple[int, int, int]]:
    """게임 태그로 팔레트 dict 반환."""
    return _ALL_PALETTES.get(game, {})


def array_to_rgb(
    array: np.ndarray,
    palette: Dict[int, Tuple[int, int, int]],
    unknown_color: Tuple[int, int, int] = _DEFAULT_UNKNOWN_COLOR,
) -> np.ndarray:
    """
    (H, W) int 배열 → (H, W, 3) uint8 RGB 배열.

    Parameters
    ----------
    array         : (H, W) int32/int64 타일 ID 배열
    palette       : tile_id → (R, G, B) 매핑
    unknown_color : 팔레트에 없는 타일의 색
    """
    h, w = array.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for tile_id, color in palette.items():
        mask = array == tile_id
        rgb[mask] = color
    # 팔레트 미등록 타일
    registered = set(palette.keys())
    for r in range(h):
        for c in range(w):
            if int(array[r, c]) not in registered:
                rgb[r, c] = unknown_color
    return rgb


def render_sample(
    sample: GameSample,
    tile_size: int = _DEFAULT_TILE_SIZE,
    unknown_color: Tuple[int, int, int] = _DEFAULT_UNKNOWN_COLOR,
) -> np.ndarray:
    """
    GameSample → (H*tile_size, W*tile_size, 3) uint8 RGB 배열.

    Returns
    -------
    numpy ndarray (Pillow 없어도 동작)
    """
    palette = get_palette(sample.game)
    small   = array_to_rgb(sample.array, palette, unknown_color)
    if tile_size == 1:
        return small
    # 업스케일
    return np.repeat(np.repeat(small, tile_size, axis=0), tile_size, axis=1)


def render_sample_pil(
    sample: GameSample,
    tile_size: int = _DEFAULT_TILE_SIZE,
    unknown_color: Tuple[int, int, int] = _DEFAULT_UNKNOWN_COLOR,
):
    """
    GameSample → PIL Image.
    Pillow가 설치되지 않은 경우 ImportError 발생.
    """
    from PIL import Image
    rgb = render_sample(sample, tile_size=tile_size, unknown_color=unknown_color)
    return Image.fromarray(rgb, mode="RGB")


def save_rendered(
    sample: GameSample,
    save_path: Path | str,
    tile_size: int = _DEFAULT_TILE_SIZE,
    unknown_color: Tuple[int, int, int] = _DEFAULT_UNKNOWN_COLOR,
) -> Path:
    """
    GameSample을 PNG로 저장.

    Returns
    -------
    저장된 파일 경로
    """
    from PIL import Image
    img = render_sample_pil(sample, tile_size=tile_size, unknown_color=unknown_color)
    out = Path(save_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(out))
    return out


def render_grid(
    samples: List[GameSample],
    cols: int = 4,
    tile_size: int = _DEFAULT_TILE_SIZE,
    gap: int = 2,
    bg_color: Tuple[int, int, int] = (30, 30, 30),
) -> np.ndarray:
    """
    여러 GameSample을 격자 형태로 배치한 이미지 반환.

    Parameters
    ----------
    samples   : GameSample 리스트
    cols      : 열 수
    tile_size : 타일 픽셀 크기
    gap       : 셀 간 간격 (픽셀)
    bg_color  : 배경 색 (R, G, B)

    Returns
    -------
    (total_H, total_W, 3) uint8 RGB 배열
    """
    if not samples:
        return np.zeros((1, 1, 3), dtype=np.uint8)

    rendered = [render_sample(s, tile_size=tile_size) for s in samples]
    max_h = max(r.shape[0] for r in rendered)
    max_w = max(r.shape[1] for r in rendered)

    rows = (len(samples) + cols - 1) // cols
    total_h = rows * max_h + (rows + 1) * gap
    total_w = cols * max_w + (cols + 1) * gap

    canvas = np.full((total_h, total_w, 3), bg_color, dtype=np.uint8)
    for idx, img in enumerate(rendered):
        row_i = idx // cols
        col_i = idx % cols
        y = gap + row_i * (max_h + gap)
        x = gap + col_i * (max_w + gap)
        h, w = img.shape[:2]
        canvas[y:y + h, x:x + w] = img

    return canvas


def save_grid(
    samples: List[GameSample],
    save_path: Path | str,
    cols: int = 4,
    tile_size: int = _DEFAULT_TILE_SIZE,
    gap: int = 2,
    bg_color: Tuple[int, int, int] = (30, 30, 30),
) -> Path:
    """render_grid 결과를 PNG로 저장."""
    from PIL import Image
    canvas = render_grid(samples, cols=cols, tile_size=tile_size,
                         gap=gap, bg_color=bg_color)
    out = Path(save_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(canvas, mode="RGB").save(str(out))
    return out


# ── 타일 이미지 기반 렌더링 ───────────────────────────────────────────────────────

class GameLevelRenderer:
    """
    타일 이미지 기반 게임 레벨 렌더러.

    tile_mapping.json과 tile_ims/ 디렉토리를 사용하여
    각 타일을 실제 타일 이미지로 렌더링합니다.
    """

    def __init__(self, tile_ims_dir: Optional[Union[str, Path]] = None):
        """
        Parameters
        ----------
        tile_ims_dir : 타일 이미지 디렉토리 경로 (기본값: 현재 디렉토리의 tile_ims)
        """
        if tile_ims_dir is None:
            current_dir = Path(__file__).parent
            tile_ims_dir = current_dir / "tile_ims"

        self.tile_ims_dir = Path(tile_ims_dir)
        self.mapping_file = self.tile_ims_dir.parent / "tile_mapping.json"

        # tile_mapping.json 로드
        with open(self.mapping_file, 'r', encoding='utf-8') as f:
            self.tile_mapping = json.load(f)

        # 타일 이미지 캐시
        self._tile_cache = {}

    def _load_tile_image(self, game: str, tile_id: int, tile_size: int = 16) -> np.ndarray:
        """
        특정 게임의 타일 ID에 해당하는 이미지를 로드합니다.

        Parameters
        ----------
        game : 게임 이름 (dungeon, doom, pokemon, sokoban, zelda)
        tile_id : 타일 ID
        tile_size : 타일 크기 (픽셀)

        Returns
        -------
        np.ndarray : (tile_size, tile_size, 3) RGB 이미지
        """
        cache_key = (game, tile_id, tile_size)
        if cache_key in self._tile_cache:
            return self._tile_cache[cache_key]

        # tile_mapping.json에서 타일 이미지 경로 찾기
        game_config = self.tile_mapping.get(game, {})
        tile_images = game_config.get("_tile_images", {})

        # 타일 ID를 문자열로 변환
        tile_id_str = str(tile_id)

        # 해당 타일 이미지 파일명 찾기
        if tile_id_str in tile_images:
            tile_filename = tile_images[tile_id_str]
        else:
            tile_filename = "empty.png"

        # 파일 경로 구성
        if tile_filename.startswith("pokemon/"):
            tile_path = self.tile_ims_dir / tile_filename
        else:
            tile_path = self.tile_ims_dir / game / tile_filename

        # 이미지 로드
        if not tile_path.exists():
            # 플레이스홀더 생성 (회색)
            tile_img = np.full((tile_size, tile_size, 3), 128, dtype=np.uint8)
        else:
            from PIL import Image
            img = Image.open(tile_path).convert('RGB')
            if img.size != (tile_size, tile_size):
                img = img.resize((tile_size, tile_size), Image.NEAREST)
            tile_img = np.array(img)

        # 캐시에 저장
        self._tile_cache[cache_key] = tile_img
        return tile_img

    def render(
        self,
        game: str,
        level: np.ndarray,
        tile_size: int = 16,
        save_path: Optional[Union[str, Path]] = None,
        show_tile_numbers: bool = False
    ):
        """
        게임 레벨을 타일 이미지로 렌더링합니다.

        Parameters
        ----------
        game : 게임 이름 (dungeon, doom, pokemon, sokoban, zelda)
        level : 게임 레벨 배열 (2D numpy array, 각 셀은 타일 ID)
        tile_size : 각 타일의 픽셀 크기
        save_path : 저장 경로 (None이면 저장하지 않음)
        show_tile_numbers : 타일 번호를 이미지 위에 표시할지 여부

        Returns
        -------
        PIL.Image.Image : 렌더링된 이미지
        """
        from PIL import Image, ImageDraw, ImageFont

        if not isinstance(level, np.ndarray):
            level = np.array(level)

        if level.ndim != 2:
            raise ValueError(f"Level must be 2D array, got shape {level.shape}")

        height, width = level.shape

        # 렌더링 캔버스 생성
        canvas = np.zeros((height * tile_size, width * tile_size, 3), dtype=np.uint8)

        # 각 타일 렌더링
        for y in range(height):
            for x in range(width):
                tile_id = int(level[y, x])
                tile_img = self._load_tile_image(game, tile_id, tile_size)

                # 캔버스에 타일 배치
                y_start = y * tile_size
                y_end = y_start + tile_size
                x_start = x * tile_size
                x_end = x_start + tile_size
                canvas[y_start:y_end, x_start:x_end] = tile_img

        # PIL 이미지로 변환
        img = Image.fromarray(canvas, mode='RGB')

        # 타일 번호 표시
        if show_tile_numbers:
            draw = ImageDraw.Draw(img)

            # 폰트 크기 설정
            font_size = max(8, int(tile_size * 0.45))
            font = None

            try:
                font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", font_size)
            except Exception:
                try:
                    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
                except:
                    try:
                        font = ImageFont.load_default()
                    except:
                        return img

            # 각 타일 위에 번호 표시
            for y in range(height):
                for x in range(width):
                    tile_id = int(level[y, x])
                    text = str(tile_id)

                    x_pos = x * tile_size + tile_size // 2
                    y_pos = y * tile_size + tile_size // 2

                    try:
                        if hasattr(font, 'getbbox'):
                            bbox = font.getbbox(text)
                            text_width = bbox[2] - bbox[0]
                            text_height = bbox[3] - bbox[1]
                        else:
                            text_width = len(text) * font_size // 2
                            text_height = font_size
                    except:
                        text_width = len(text) * font_size // 2
                        text_height = font_size

                    text_x = x_pos - text_width // 2
                    text_y = y_pos - text_height // 2

                    # 배경 사각형
                    padding = 1
                    rect_bbox = [
                        text_x - padding,
                        text_y - padding,
                        text_x + text_width + padding,
                        text_y + text_height + padding
                    ]
                    draw.rectangle(rect_bbox, fill=(0, 0, 0, 180))

                    # 텍스트 그리기
                    try:
                        draw.text((text_x, text_y), text, fill=(255, 255, 255), font=font)
                    except Exception:
                        pass

        # 저장
        if save_path is not None:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            img.save(save_path)

        return img


def render_game_level(
    game: str,
    level: np.ndarray,
    tile_size: int = 16,
    save_path: Optional[Union[str, Path]] = None,
    show_tile_numbers: bool = False,
    tile_ims_dir: Optional[Union[str, Path]] = None
):
    """
    게임 레벨을 타일 이미지로 렌더링하는 편의 함수.

    Parameters
    ----------
    game : 게임 이름
    level : 2D numpy array
    tile_size : 타일 크기 (픽셀)
    save_path : 저장 경로
    show_tile_numbers : 타일 번호 표시 여부
    tile_ims_dir : 타일 이미지 디렉토리 (선택)

    Returns
    -------
    PIL.Image.Image : 렌더링된 이미지
    """
    renderer = GameLevelRenderer(tile_ims_dir=tile_ims_dir)
    return renderer.render(
        game,
        level,
        tile_size=tile_size,
        save_path=save_path,
        show_tile_numbers=show_tile_numbers
    )

