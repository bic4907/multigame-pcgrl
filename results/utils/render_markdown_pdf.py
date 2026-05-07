"""
Render a markdown file to PDF (basic markdown subset).

Supported elements:
- headings (#, ##, ###)
- bullets (- ...)
- plain text paragraphs
- markdown images: ![alt](path)
- two-image table row style:
  | ![a](img_a.png) | ![b](img_b.png) |
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


IMG_PATTERN = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render markdown to PDF.")
    parser.add_argument("input_md", help="Input markdown file path.")
    parser.add_argument("output_pdf", help="Output PDF file path.")
    parser.add_argument(
        "--single-page",
        action="store_true",
        help="Render all image pairs into a single PDF page.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Output DPI for PDF raster content.",
    )
    return parser.parse_args()


class PdfCanvas:
    def __init__(self, pdf: PdfPages, page_size=(8.27, 11.69)):
        self.pdf = pdf
        self.page_size = page_size
        self.fig = None
        self.y = None
        self._new_page()

    def _new_page(self):
        if self.fig is not None:
            self.pdf.savefig(self.fig, bbox_inches="tight")
            plt.close(self.fig)
        self.fig = plt.figure(figsize=self.page_size)
        self.fig.patch.set_facecolor("white")
        self.y = 0.96

    def _ensure_space(self, needed: float):
        if self.y - needed < 0.04:
            self._new_page()

    def add_text(self, text: str, size: float = 11.0, weight: str = "normal", gap: float = 0.012):
        line_h = 0.013 * (size / 11.0) + gap
        self._ensure_space(line_h)
        self.fig.text(0.06, self.y, text, fontsize=size, fontweight=weight, va="top", ha="left")
        self.y -= line_h

    def add_blank(self, h: float = 0.010):
        self._ensure_space(h)
        self.y -= h

    def add_image_single(self, path: Path):
        img = mpimg.imread(path)
        hpx, wpx = img.shape[0], img.shape[1]
        x = 0.07
        w = 0.86
        h = w * (hpx / max(1, wpx))
        self._ensure_space(h + 0.015)
        ax = self.fig.add_axes([x, self.y - h, w, h])
        ax.imshow(img)
        ax.axis("off")
        self.y -= h + 0.015

    def add_image_pair(self, left: Path, right: Path):
        img_l = mpimg.imread(left)
        img_r = mpimg.imread(right)
        h_l, w_l = img_l.shape[0], img_l.shape[1]
        h_r, w_r = img_r.shape[0], img_r.shape[1]

        x_l = 0.06
        x_r = 0.52
        w = 0.42
        h = max(w * (h_l / max(1, w_l)), w * (h_r / max(1, w_r)))
        self._ensure_space(h + 0.015)

        ax_l = self.fig.add_axes([x_l, self.y - h, w, h])
        ax_l.imshow(img_l)
        ax_l.axis("off")

        ax_r = self.fig.add_axes([x_r, self.y - h, w, h])
        ax_r.imshow(img_r)
        ax_r.axis("off")

        self.y -= h + 0.015

    def close(self):
        if self.fig is not None:
            self.pdf.savefig(self.fig, bbox_inches="tight")
            plt.close(self.fig)
            self.fig = None


def resolve_image_paths(md_line: str, base_dir: Path) -> list[Path]:
    rels = IMG_PATTERN.findall(md_line)
    paths = []
    for r in rels:
        p = (base_dir / r).resolve()
        paths.append(p)
    return paths


def extract_image_pairs_from_report(input_md: Path) -> list[tuple[str, Path, Path]]:
    lines = input_md.read_text(encoding="utf-8").splitlines()
    base_dir = input_md.parent
    current_label: str | None = None
    pairs: list[tuple[str, Path, Path]] = []

    for raw in lines:
        line = raw.strip()
        if line.startswith("## "):
            heading = line[3:].strip()
            if heading and heading.lower() != "overview":
                current_label = heading
            continue
        if line.startswith("|") and "![" in line:
            imgs = resolve_image_paths(line, base_dir)
            if len(imgs) >= 2 and imgs[0].exists() and imgs[1].exists():
                label = current_label or f"Section {len(pairs) + 1}"
                pairs.append((label, imgs[0], imgs[1]))
    return pairs


def render_single_page_pairs(input_md: Path, output_pdf: Path, dpi: int = 300) -> None:
    pairs = extract_image_pairs_from_report(input_md)
    if not pairs:
        render_markdown_to_pdf(input_md=input_md, output_pdf=output_pdf)
        return

    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    n_rows = len(pairs)
    # A3 portrait gives enough room for 5x2 image grid on one page.
    fig = plt.figure(figsize=(11.69, 16.54))
    gs = fig.add_gridspec(
        nrows=n_rows,
        ncols=2,
        left=0.02,
        right=0.98,
        top=0.99,
        bottom=0.01,
        wspace=0.03,
        hspace=0.03,
    )

    for r, (label, left_img_path, right_img_path) in enumerate(pairs):
        left_img = mpimg.imread(left_img_path)
        right_img = mpimg.imread(right_img_path)

        ax_l = fig.add_subplot(gs[r, 0])
        ax_r = fig.add_subplot(gs[r, 1])

        ax_l.imshow(left_img, interpolation="nearest")
        ax_r.imshow(right_img, interpolation="nearest")

        for ax in (ax_l, ax_r):
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)

        # Overlay section label once per row, no extra title space.
        ax_l.text(
            0.01,
            0.98,
            label,
            transform=ax_l.transAxes,
            ha="left",
            va="top",
            fontsize=10,
            color="#111111",
            bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
        )

    fig.savefig(output_pdf, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def render_markdown_to_pdf(input_md: Path, output_pdf: Path) -> None:
    base_dir = input_md.parent
    lines = input_md.read_text(encoding="utf-8").splitlines()
    output_pdf.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(output_pdf) as pdf:
        canvas = PdfCanvas(pdf)
        for raw in lines:
            line = raw.rstrip()
            if not line.strip():
                canvas.add_blank(0.010)
                continue

            if line.startswith("### "):
                canvas.add_text(line[4:].strip(), size=12.0, weight="bold", gap=0.012)
                continue
            if line.startswith("## "):
                canvas.add_text(line[3:].strip(), size=14.0, weight="bold", gap=0.014)
                continue
            if line.startswith("# "):
                canvas.add_text(line[2:].strip(), size=16.0, weight="bold", gap=0.016)
                continue
            if line.startswith("- "):
                canvas.add_text(f"• {line[2:].strip()}", size=10.8, weight="normal", gap=0.010)
                continue

            # Skip markdown table separators/headers
            if line.startswith("| ---") or line.strip() == "| --- | --- |":
                continue
            if line.startswith("| Condition vs Progress"):
                continue

            # image row inside table
            if line.startswith("|") and "![" in line:
                imgs = resolve_image_paths(line, base_dir)
                if len(imgs) >= 2 and imgs[0].exists() and imgs[1].exists():
                    canvas.add_image_pair(imgs[0], imgs[1])
                    continue
                if len(imgs) >= 1 and imgs[0].exists():
                    canvas.add_image_single(imgs[0])
                    continue

            # normal image line
            if line.startswith("!["):
                imgs = resolve_image_paths(line, base_dir)
                if imgs and imgs[0].exists():
                    canvas.add_image_single(imgs[0])
                continue

            # fallback plain text
            canvas.add_text(line, size=10.6, weight="normal", gap=0.010)

        canvas.close()


def main():
    args = parse_args()
    input_md = Path(args.input_md).resolve()
    output_pdf = Path(args.output_pdf).resolve()
    if args.single_page:
        render_single_page_pairs(input_md=input_md, output_pdf=output_pdf, dpi=args.dpi)
    else:
        render_markdown_to_pdf(input_md=input_md, output_pdf=output_pdf)
    print(f"[OK] input_md   : {input_md}")
    print(f"[OK] output_pdf : {output_pdf}")


if __name__ == "__main__":
    main()
