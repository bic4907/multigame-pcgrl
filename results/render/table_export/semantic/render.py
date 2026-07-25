from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from dataset.multigame.render import GameLevelRenderer
from .constants import COUNT_TILE_ID_BY_REWARD_ENUM, PASSABLE_TILE_IDS, PROJECT_ROOT, _fmt_num
from .metrics import _eval_path_metric_and_coords


def _load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        str(PROJECT_ROOT / "debug" / "Pretendard-Regular.ttf"),
        "/System/Library/Fonts/Supplemental/Pretendard-Regular.otf",
        "/System/Library/Fonts/Supplemental/Pretendard.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
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
    inset = max(2, tile_size // 8)
    ignored_holes = _interior_nonpassable_holes(passable)
    for y, x in zip(*np.where(passable)):
        x0 = int(x) * tile_size
        y0 = int(y) * tile_size
        x1 = x0 + tile_size
        y1 = y0 + tile_size
        edges = []
        if _is_region_outer_edge(passable, ignored_holes, y - 1, x):
            edges.append((x0 + inset, y0 + inset, x1 - inset, y0 + inset))
        if _is_region_outer_edge(passable, ignored_holes, y + 1, x):
            edges.append((x0 + inset, y1 - inset, x1 - inset, y1 - inset))
        if _is_region_outer_edge(passable, ignored_holes, y, x - 1):
            edges.append((x0 + inset, y0 + inset, x0 + inset, y1 - inset))
        if _is_region_outer_edge(passable, ignored_holes, y, x + 1):
            edges.append((x1 - inset, y0 + inset, x1 - inset, y1 - inset))
        for edge in edges:
            draw.line(edge, fill=(45, 35, 0, 210), width=outer_width)
            draw.line(edge, fill=(255, 214, 10, 255), width=inner_width)


def _is_region_outer_edge(passable: np.ndarray, ignored_holes: set[tuple[int, int]], y: int, x: int) -> bool:
    height, width = passable.shape
    if y < 0 or y >= height or x < 0 or x >= width:
        return True
    return not passable[y, x] and (y, x) not in ignored_holes


def _interior_nonpassable_holes(passable: np.ndarray) -> set[tuple[int, int]]:
    height, width = passable.shape
    seen = np.zeros_like(passable, dtype=bool)
    holes: set[tuple[int, int]] = set()
    for start_y, start_x in zip(*np.where(~passable)):
        if seen[start_y, start_x]:
            continue
        queue: deque[tuple[int, int]] = deque([(int(start_y), int(start_x))])
        seen[start_y, start_x] = True
        component: list[tuple[int, int]] = []
        touches_border = False
        adjacent_passable: set[tuple[int, int]] = set()
        while queue:
            y, x = queue.popleft()
            component.append((y, x))
            touches_border = touches_border or y == 0 or x == 0 or y + 1 == height or x + 1 == width
            for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                ny, nx = y + dy, x + dx
                if ny < 0 or ny >= height or nx < 0 or nx >= width:
                    continue
                if passable[ny, nx]:
                    adjacent_passable.add((ny, nx))
                elif not seen[ny, nx]:
                    seen[ny, nx] = True
                    queue.append((ny, nx))
        if not touches_border and _adjacent_to_single_region(passable, adjacent_passable):
            holes.update(component)
    return holes


def _adjacent_to_single_region(passable: np.ndarray, starts: set[tuple[int, int]]) -> bool:
    if not starts:
        return False
    target = next(iter(starts))
    queue: deque[tuple[int, int]] = deque([target])
    seen = {target}
    while queue:
        y, x = queue.popleft()
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ny, nx = y + dy, x + dx
            if ny < 0 or ny >= passable.shape[0] or nx < 0 or nx >= passable.shape[1]:
                continue
            if passable[ny, nx] and (ny, nx) not in seen:
                seen.add((ny, nx))
                queue.append((ny, nx))
    return starts.issubset(seen)

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
        _, coords = _eval_path_metric_and_coords(state)
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

    label_lines = [f"Target={_fmt_num(target_value)}", f"Result={_fmt_num(metric_value)}"]
    pad_x = max(16, int(tile_size * 0.50))
    pad_y = max(12, int(tile_size * 0.38))
    inset_x = max(10, int(tile_size * 0.36))
    inset_y = max(10, int(tile_size * 0.36))
    radius = max(8, int(tile_size * 0.35))
    line_gap = max(4, int(tile_size * 0.16))
    font_size = max(54, int(tile_size * 1.78))
    font = _load_font(font_size, bold=True)

    def line_boxes() -> list[tuple[int, int, int, int]]:
        return [draw.textbbox((0, 0), line, font=font) for line in label_lines]

    bboxes = line_boxes()
    text_w = max(bbox[2] - bbox[0] for bbox in bboxes)
    text_h = sum(bbox[3] - bbox[1] for bbox in bboxes) + line_gap * (len(label_lines) - 1)
    while text_w + 2 * pad_x + inset_x > img.width and font_size > 36:
        font_size -= 2
        font = _load_font(font_size, bold=True)
        bboxes = line_boxes()
        text_w = max(bbox[2] - bbox[0] for bbox in bboxes)
        text_h = sum(bbox[3] - bbox[1] for bbox in bboxes) + line_gap * (len(label_lines) - 1)
    box = (
        inset_x,
        inset_y,
        inset_x + text_w + 2 * pad_x,
        inset_y + text_h + 2 * pad_y,
    )
    draw.rounded_rectangle(box, radius=radius, fill=(255, 255, 255, 230), outline=(20, 24, 30, 210), width=2)
    y = inset_y + pad_y
    for line, bbox in zip(label_lines, bboxes):
        text_xy = (inset_x + pad_x, y - bbox[1])
        draw.text(text_xy, line, fill=(20, 24, 30, 255), font=font)
        draw.text((text_xy[0] + 1, text_xy[1]), line, fill=(20, 24, 30, 255), font=font)
        y += bbox[3] - bbox[1] + line_gap

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
