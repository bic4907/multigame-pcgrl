from __future__ import annotations

import csv
import json
import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from ..models import CandidateRow, RenderCell
from .constants import (
    GAME_LABEL,
    GAME_PREVIEW_LABEL,
    METHOD_ORDER,
    TRANSITION_LABELS,
    _feature_name_for_game,
    _latex_escape,
)
from .render import _load_font


def _relative_for_tex(path: Path, base: Path) -> str:
    try:
        return path.relative_to(base).as_posix()
    except ValueError:
        return path.as_posix()

def _cell_latex(cell: RenderCell, output_dir: Path) -> str:
    def panel(image: Path, candidate: CandidateRow, seed: int) -> str:
        img = _relative_for_tex(image, output_dir)
        return (
            r"\begin{minipage}[t]{.32\linewidth}\centering "
            rf"\includegraphics[width=\linewidth]{{{img}}}"
            r"\end{minipage}"
        )

    image_row = (
        panel(cell.low_overlay, cell.low, cell.low_seed)
        + r"\hfill"
        + panel(cell.mid_overlay, cell.mid, cell.mid_seed)
        + r"\hfill"
        + panel(cell.high_overlay, cell.high, cell.high_seed)
    )
    return (
        r"\parbox[t]{\linewidth}{\centering "
        rf"\makebox[\linewidth][s]{{{image_row}}}"
        r"}"
    )

def _build_latex(
    cells: dict[tuple[str, str, str], RenderCell],
    features: list[str],
    games: list[str],
    output_dir: Path,
    methods: list[str] | None = None,
) -> str:
    methods = methods or METHOD_ORDER
    game_headers = [GAME_LABEL.get(game, _latex_escape(game.title())) for game in games]
    game_col_width = max(0.15, min(0.295, 0.885 / max(1, len(games))))
    colspec = (
        r"@{}>{\centering\arraybackslash}m{.055\textwidth}"
        + "".join([rf"p{{{game_col_width:.3f}\textwidth}}" for _ in games])
        + r"@{}"
    )
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{\textbf{Domain-specific grounding of shared semantic transitions.} "
        r"The same normalized semantic transition corresponds to different raw condition values across game domains. "
        r"The rows show fixed method outputs under each game-specific condition scale. "
        r"Each visualization is rendered from the fixed row and seed listed in the render configuration.}",
        r"\label{tab:domain_specific_semantic_transition}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2pt}",
        r"\renewcommand{\arraystretch}{1.14}",
        rf"\begin{{tabular}}{{{colspec}}}",
        r"\toprule",
        "Method & " + " & ".join(game_headers) + r" \\",
        r"\midrule",
    ]

    for feature_i, feature in enumerate(features):
        transition = TRANSITION_LABELS.get(feature, _latex_escape(feature))
        lines.append(rf"\multicolumn{{{len(games) + 1}}}{{@{{}}l}}{{\textbf{{{transition}}}}} \\[-0.15em]")
        for method_i, method in enumerate(methods):
            game_cells = []
            for game in games:
                cell = cells.get((method, game, feature))
                game_cells.append(_cell_latex(cell, output_dir) if cell else "-")
            lines.append(f"{method} & " + " & ".join(game_cells) + r" \\")
        if feature_i + 1 < len(features):
            lines.append(r"\midrule")

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table*}",
            "",
        ]
    )
    return "\n".join(lines)

def _draw_text(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, font: ImageFont.ImageFont, fill=(20, 24, 30)) -> int:
    x, y = xy
    for line in text.split("\n"):
        draw.text((x, y), line, font=font, fill=fill)
        y += draw.textbbox((x, y), line, font=font)[3] - draw.textbbox((x, y), line, font=font)[1] + 4
    return y

def _make_preview_png_pdf(
    cells: dict[tuple[str, str, str], RenderCell],
    features: list[str],
    games: list[str],
    output_dir: Path,
    methods: list[str] | None = None,
) -> tuple[Path, Path]:
    methods = methods or METHOD_ORDER
    cell_w = 660 if len(games) > 2 else 760
    left_w = 230
    method_w = 110
    header_h = 70
    row_h = 280
    image_h = 220
    width = left_w + method_w + len(games) * cell_w + 36
    height = header_h + len(features) * len(methods) * row_h + 40
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    font_h = _load_font(30, bold=True)
    font = _load_font(22)
    font_b = _load_font(22, bold=True)

    x_transition = 18
    x_method = x_transition + left_w
    x_game0 = x_method + method_w
    for game_i, game in enumerate(games):
        x0 = x_game0 + game_i * cell_w
        label = GAME_PREVIEW_LABEL.get(game, game.title())
        bbox = draw.textbbox((0, 0), label, font=font_h)
        label_w = bbox[2] - bbox[0]
        draw.text((x0 + (cell_w - label_w) // 2, 18), label, font=font_h, fill=(20, 24, 30))
    draw.line((18, header_h - 8, width - 18, header_h - 8), fill=(40, 44, 52), width=2)

    y = header_h
    for feature in features:
        transition = TRANSITION_LABELS.get(feature, feature).replace("$", "").replace("\\rightarrow", "->")
        transition = transition.replace("\\", "")
        block_top = y
        block_bottom = y + len(methods) * row_h
        draw.rectangle((18, block_top, width - 18, block_bottom - 1), outline=(220, 224, 230), width=1)
        transition_bbox = draw.multiline_textbbox((0, 0), transition, font=font_b, spacing=4)
        transition_h = transition_bbox[3] - transition_bbox[1]
        _draw_text(draw, (x_transition, block_top + max(18, (block_bottom - block_top - transition_h) // 2)), transition, font_b)
        for method_i, method in enumerate(methods):
            row_top = y + method_i * row_h
            draw.line((x_method - 10, row_top, width - 18, row_top), fill=(232, 235, 240), width=1)
            draw.text((x_method, row_top + 118), method, font=font_b, fill=(20, 24, 30))
            for game_i, game in enumerate(games):
                x0 = x_game0 + game_i * cell_w
                cell = cells.get((method, game, feature))
                if cell is None:
                    draw.text((x0 + 20, row_top + 122), "-", font=font, fill=(20, 24, 30))
                    continue
                img = Image.open(cell.triplet_overlay).convert("RGB")
                scale = min((cell_w - 48) / img.width, image_h / img.height)
                img = img.resize((int(img.width * scale), int(img.height * scale)), Image.Resampling.LANCZOS)
                canvas.paste(img, (x0 + (cell_w - img.width) // 2, row_top + 30))
        for method_i in range(1, len(methods)):
            row_y = block_top + method_i * row_h
            draw.line((18, row_y, width - 18, row_y), fill=(232, 235, 240), width=1)
        y += len(methods) * row_h

    png_path = output_dir / "table2_semantic_transition_crop.png"
    pdf_path = output_dir / "table2_semantic_transition_crop.pdf"
    canvas.save(png_path)
    canvas.save(pdf_path, "PDF", resolution=300.0)
    return png_path, pdf_path

def _save_image_pdf(src: Path, dst: Path) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    Image.open(src).convert("RGB").save(dst, "PDF", resolution=300.0)
    return dst

def _make_overleaf_bundle(cells: dict[tuple[str, str, str], RenderCell], output_dir: Path) -> Path:
    bundle_dir = output_dir / "experiment"
    figure_dir = bundle_dir / "tbl_qualitative_figs"
    figure_dir.mkdir(parents=True, exist_ok=True)

    tex = (output_dir / "tbl_qualitative.tex").read_text(encoding="utf-8")
    include_paths = sorted(set(re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^{}]+)\}", tex)))
    for include_path in include_paths:
        src = output_dir / include_path
        if not src.exists():
            continue
        dst = figure_dir / src.with_suffix(".pdf").name
        _save_image_pdf(src, dst)
        tex = tex.replace(include_path, f"experiment/tbl_qualitative_figs/{dst.name}")

    tex = "\n".join(
        [
            r"% Requires: \usepackage{graphicx,booktabs,array}",
            tex,
        ]
    )
    (bundle_dir / "tbl_qualitative.tex").write_text(tex, encoding="utf-8")
    return bundle_dir

def _write_manifest(cells: dict[tuple[str, str, str], RenderCell], output_dir: Path) -> Path:
    rows = []
    for key in sorted(cells):
        cell = cells[key]
        for side, candidate, seed, image, overlay in [
            ("low", cell.low, cell.low_seed, cell.low_image, cell.low_overlay),
            ("mid", cell.mid, cell.mid_seed, cell.mid_image, cell.mid_overlay),
            ("high", cell.high, cell.high_seed, cell.high_image, cell.high_overlay),
        ]:
            mean, std = candidate.mean_std
            rows.append(
                {
                    "method": cell.method,
                    "game": cell.game,
                    "task": cell.feature,
                    "grounded_task": _feature_name_for_game(cell.feature, cell.game),
                    "reward_enum": candidate.reward_enum,
                    "side": side,
                    "row_i": candidate.row_i,
                    "seed": seed,
                    "condition_value": candidate.target,
                    "realized_seed_stat": candidate.seed_metrics.get(seed),
                    "realized_mean": mean,
                    "realized_std": std,
                    "instruction": candidate.instruction,
                    "image": image.relative_to(output_dir).as_posix(),
                    "overlay": overlay.relative_to(output_dir).as_posix(),
                    "triplet_overlay": cell.triplet_overlay.relative_to(output_dir).as_posix(),
                }
            )

    csv_path = output_dir / "rendered_samples.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)
    (output_dir / "rendered_samples.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    return csv_path
