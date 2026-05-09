"""
Embed local markdown image files as base64 data URIs.

Usage:
    python results/embed_markdown_images_base64.py input.md output_embedded.md
"""

from __future__ import annotations

import argparse
import base64
import mimetypes
import re
from pathlib import Path


IMG_MD_PATTERN = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Embed local markdown images as base64 data URIs.")
    parser.add_argument("input_md", help="Input markdown path.")
    parser.add_argument("output_md", help="Output markdown path.")
    return parser.parse_args()


def _guess_mime(path: Path) -> str:
    mime, _ = mimetypes.guess_type(str(path))
    if mime:
        return mime
    suffix = path.suffix.lower()
    if suffix == ".png":
        return "image/png"
    if suffix in (".jpg", ".jpeg"):
        return "image/jpeg"
    if suffix == ".gif":
        return "image/gif"
    if suffix == ".webp":
        return "image/webp"
    return "application/octet-stream"


def _normalize_md_target(target: str) -> str:
    t = target.strip()
    if t.startswith("<") and t.endswith(">"):
        t = t[1:-1].strip()
    return t


def embed_markdown_images(input_md: Path, output_md: Path) -> tuple[int, int]:
    text = input_md.read_text(encoding="utf-8")
    base_dir = input_md.parent
    converted = 0
    missing = 0

    def repl(match: re.Match[str]) -> str:
        nonlocal converted, missing
        alt = match.group(1)
        raw_target = _normalize_md_target(match.group(2))

        lowered = raw_target.lower()
        if lowered.startswith("data:") or lowered.startswith("http://") or lowered.startswith("https://"):
            return match.group(0)

        img_path = Path(raw_target)
        if not img_path.is_absolute():
            img_path = (base_dir / img_path).resolve()

        if not img_path.exists() or not img_path.is_file():
            missing += 1
            return match.group(0)

        mime = _guess_mime(img_path)
        b64 = base64.b64encode(img_path.read_bytes()).decode("ascii")
        converted += 1
        return f"![{alt}](data:{mime};base64,{b64})"

    out_text = IMG_MD_PATTERN.sub(repl, text)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(out_text, encoding="utf-8")
    return converted, missing


def main() -> None:
    args = parse_args()
    input_md = Path(args.input_md).resolve()
    output_md = Path(args.output_md).resolve()

    converted, missing = embed_markdown_images(input_md, output_md)
    print(f"[OK] input_md   : {input_md}")
    print(f"[OK] output_md  : {output_md}")
    print(f"[OK] converted  : {converted}")
    print(f"[OK] missing    : {missing}")


if __name__ == "__main__":
    main()
