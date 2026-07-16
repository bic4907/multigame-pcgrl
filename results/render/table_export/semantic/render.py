from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from dataset.multigame.render import GameLevelRenderer
from .constants import COUNT_TILE_ID_BY_REWARD_ENUM, PASSABLE_TILE_IDS, PROJECT_ROOT, _fmt_num
from .metrics import _path_metric_and_coords


def _load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        str(PROJECT_ROOT / "debug" / "Pretendard-Regular.ttf"),
    ]
    for path in candidates:
        if path and Path(path).exists():
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                pass
    return ImageFont.load_default()

def _draw_level_image(game: str, state: np.ndarray, out_path: Path, tile_size: int, renderer: GameLevelRenderer) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    renderer.render(game=game, level=state, tile_size=tile_size, show_tile_numbers=False).save(str(out_path))
    return out_path

def _draw_region_boundaries(draw: ImageDraw.ImageDraw, state: np.ndarray, tile_size: int) -> None:
    passable = np.isin(state, list(PASSABLE_TILE_IDS))
    height, width = state.shape
    outer_width = max(3, tile_size // 4)
    inner_width = max(2, tile_size // 6)
    for y, x in zip(*np.where(passable)):
        x0 = int(x) * tile_size
        y0 = int(y) * tile_size
        x1 = x0 + tile_size
        y1 = y0 + tile_size
        edges = []
        if y == 0 or not passable[y - 1, x]:
            edges.append((x0, y0, x1, y0))
        if y + 1 >= height or not passable[y + 1, x]:
            edges.append((x0, y1, x1, y1))
        if x == 0 or not passable[y, x - 1]:
            edges.append((x0, y0, x0, y1))
        if x + 1 >= width or not passable[y, x + 1]:
            edges.append((x1, y0, x1, y1))
        for edge in edges:
            draw.line(edge, fill=(45, 35, 0, 210), width=outer_width)
            draw.line(edge, fill=(255, 214, 10, 255), width=inner_width)

def _overlay_metric(
    image_path: Path,
    state: np.ndarray,
    reward_enum: int,
    metric_value: float,
    target_value: float | None,
    out_path: Path,
    tile_size: int,
) -> Path:
    img = Image.open(image_path).convert("RGBA")
    draw = ImageDraw.Draw(img, "RGBA")

    if reward_enum == 0:
        _draw_region_boundaries(draw, state, tile_size)
    elif reward_enum == 1:
        _, coords = _path_metric_and_coords(state)
        if len(coords) >= 2:
            points = [(x * tile_size + tile_size // 2, y * tile_size + tile_size // 2) for y, x in coords]
            outer_width = max(5, tile_size // 2)
            inner_width = max(3, tile_size // 3)
            draw.line(points, fill=(45, 35, 0, 190), width=outer_width, joint="curve")
            draw.line(points, fill=(255, 214, 10, 255), width=inner_width, joint="curve")
            radius = max(4, tile_size // 3)
            for px, py in (points[0], points[-1]):
                draw.ellipse(
                    (px - radius, py - radius, px + radius, py + radius),
                    fill=(255, 214, 10, 255),
                    outline=(45, 35, 0, 230),
                    width=2,
                )
    else:
        tile_id = COUNT_TILE_ID_BY_REWARD_ENUM.get(reward_enum)
        if tile_id is not None:
            ys, xs = np.where(state == tile_id)
            box_pad = max(2, tile_size // 5)
            box_width = max(3, tile_size // 4)
            for y, x in zip(ys, xs):
                x0 = int(x) * tile_size
                y0 = int(y) * tile_size
                x1 = x0 + tile_size
                y1 = y0 + tile_size
                draw.rectangle(
                    (
                        max(0, x0 - box_pad),
                        max(0, y0 - box_pad),
                        min(img.width - 1, x1 + box_pad),
                        min(img.height - 1, y1 + box_pad),
                    ),
                    outline=(255, 214, 10, 255),
                    width=box_width,
                )

    label = f"Target={_fmt_num(target_value)} | Result={_fmt_num(metric_value)}"
    pad = max(8, tile_size // 4)
    font_size = max(42, int(tile_size * 1.40))
    font = _load_font(font_size, bold=True)
    bbox = draw.textbbox((0, 0), label, font=font)
    while bbox[2] - bbox[0] + 2 * pad > img.width and font_size > 30:
        font_size -= 2
        font = _load_font(font_size, bold=True)
        bbox = draw.textbbox((0, 0), label, font=font)
    box = (0, 0, bbox[2] - bbox[0] + 2 * pad, bbox[3] - bbox[1] + 2 * pad)
    draw.rectangle(box, fill=(255, 255, 255, 225))
    draw.text((pad, pad), label, fill=(20, 24, 30, 255), font=font)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.convert("RGB").save(str(out_path))
    return out_path

def _combine_triplet(
    low_img: Path,
    mid_img: Path,
    high_img: Path,
    out_path: Path,
    label_low: str,
    label_mid: str,
    label_high: str,
) -> Path:
    low = Image.open(low_img).convert("RGBA")
    mid = Image.open(mid_img).convert("RGBA")
    high = Image.open(high_img).convert("RGBA")
    gap = 22
    header_h = 24
    width = low.width + mid.width + high.width + 2 * gap
    height = max(low.height, mid.height, high.height) + header_h
    canvas = Image.new("RGBA", (width, height), (255, 255, 255, 0))
    draw = ImageDraw.Draw(canvas)
    font = _load_font(14, bold=True)
    canvas.alpha_composite(low, (0, header_h))
    canvas.alpha_composite(mid, (low.width + gap, header_h))
    canvas.alpha_composite(high, (low.width + mid.width + 2 * gap, header_h))
    draw.text((4, 2), label_low, fill=(20, 24, 30, 255), font=font)
    draw.text((low.width + gap + 4, 2), label_mid, fill=(20, 24, 30, 255), font=font)
    draw.text((low.width + mid.width + 2 * gap + 4, 2), label_high, fill=(20, 24, 30, 255), font=font)
    y = header_h + max(low.height, mid.height, high.height) // 2
    arrow_segments = [
        (low.width + 4, low.width + gap - 4),
        (low.width + gap + mid.width + 4, low.width + mid.width + 2 * gap - 4),
    ]
    for x0, x1 in arrow_segments:
        draw.line((x0, y, x1, y), fill=(70, 78, 90, 255), width=3)
        draw.polygon([(x1, y), (x1 - 8, y - 5), (x1 - 8, y + 5)], fill=(70, 78, 90, 255))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.convert("RGB").save(str(out_path))
    return out_path

