"""
ann.json file of  instruction_uni  text tabletext  text result  instruction_raw in  text.
instruction_uni  text text text.

term_dst.json of  dst text  text text Usagetext text.

Usage:
    python -m dataset.multigame.scripts.replace_instruction_terms
    python -m dataset.multigame.scripts.replace_instruction_terms --dry-run
    python -m dataset.multigame.scripts.replace_instruction_terms --enums 2 3
    python -m dataset.multigame.scripts.replace_instruction_terms --games dungeon zelda
    python -m dataset.multigame.scripts.replace_instruction_terms --dst-file path/to/term_dst.json
"""

import argparse
import json
import re
import shutil
from pathlib import Path

# ── text  tabletext (src) fixed text of  ────────────────────────────────────────────────────
# each enumtext longer match(tile text)  text, shorter(text)  text during  in  text.
# text(s) also  text text.
# text text  sametext dst to  text.

TERM_PATTERNS: dict[int, list[str]] = {
    2: [r"interactive tiles?", r"interactives?"],   # longer first
    3: [r"hazard tiles?",      r"hazards?"],
    4: [r"collectable tiles?", r"collectables?"],
}

# ── dst file default path ────────────────────────────────────────────────────────────
DEFAULT_DST_FILE = Path(__file__).parent / "term_dst.json"

ANN_DIR = Path("dataset/multigame/cache/artifacts")


def load_dst(dst_file: Path) -> dict[str, dict[int, str]]:
    """term_dst.json load. { game: { enum(int): dst(str) } }"""
    with dst_file.open(encoding="utf-8") as f:
        raw = json.load(f)
    return {
        game: {int(enum): dst for enum, dst in enum_map.items()}
        for game, enum_map in raw.items()
    }


def _pluralize(word: str) -> str:
    """text text rule apply."""
    w = word.lower()
    if w.endswith(("s", "ss", "sh", "ch", "x", "z")):
        return word + "es"
    if w.endswith("y") and len(w) >= 2 and w[-2] not in "aeiou":
        return word[:-1] + "ies"
    if w.endswith("fe"):
        return word[:-2] + "ves"
    if w.endswith("f") and not w.endswith("ff"):
        return word[:-1] + "ves"
    return word + "s"


def _replace_pattern(text: str, pattern: str, dst: str) -> str:
    """text text  text pattern → dst text.
    - text text  s to  text dst  text as  convert (collectables → items)
    - text text text textcharacter preserve
    """
    compiled = re.compile(r"\b" + pattern + r"\b", re.IGNORECASE)

    def replacer(m: re.Match) -> str:
        matched = m.group()
        result = _pluralize(dst) if matched[-1].lower() == "s" else dst
        if matched[0].isupper():
            return result[0].upper() + result[1:] if result else ""
        return result

    return compiled.sub(replacer, text)


def replace_in_text(
    text: str,
    game: str,
    enum: int,
    dst_map: dict[str, dict[int, str]],
) -> str:
    if not text or text.lower() == "none":
        return text
    dst = dst_map.get(game, {}).get(enum, "")
    if not dst:
        return text
    for pattern in TERM_PATTERNS.get(enum, []):
        text = _replace_pattern(text, pattern, dst)
    return text


COPY_ENUMS = {0, 1}   # instruction_uni  instruction_raw in  as-is copy


def process_file(
    ann_path: Path,
    enums: set[int],
    dst_map: dict[str, dict[int, str]],
    dry_run: bool,
) -> dict:
    game = ann_path.parent.name
    original_text = ann_path.read_text(encoding="utf-8")
    data = json.loads(original_text)

    result_text = original_text
    changed = 0
    pos = 0  # ordertext to  text abovetext text start abovetext

    for ann in data["annotations"]:
        enum = int(ann["reward_enum"])
        old_raw = ann.get("instruction_raw") or ""

        if enum in COPY_ENUMS:
            new_raw = ann.get("instruction_uni") or ""
        elif enum in enums:
            uni = ann.get("instruction_uni") or ""
            new_raw = replace_in_text(uni, game, enum, dst_map)
            if new_raw == uni:
                # text none — instruction_raw abovetext pos  before text
                old_field = '"instruction_raw":' + json.dumps(old_raw, ensure_ascii=False)
                idx = result_text.find(old_field, pos)
                if idx >= 0:
                    pos = idx + len(old_field)
                continue
        else:
            continue

        old_field = '"instruction_raw":' + json.dumps(old_raw, ensure_ascii=False)
        new_field = '"instruction_raw":' + json.dumps(new_raw, ensure_ascii=False)
        idx = result_text.find(old_field, pos)
        if idx >= 0:
            result_text = result_text[:idx] + new_field + result_text[idx + len(old_field):]
            pos = idx + len(new_field)
        changed += 1

    if changed and not dry_run:
        backup = ann_path.parent / (ann_path.name + ".bak")
        if not backup.exists():
            shutil.copy2(ann_path, backup)
        ann_path.write_text(result_text, encoding="utf-8")

    return {"path": ann_path, "changed": changed, "total": len(data["annotations"])}


def main() -> None:
    parser = argparse.ArgumentParser(description="instruction_uni tabletext text")
    parser.add_argument("--ann-dir", type=Path, default=ANN_DIR)
    parser.add_argument("--dst-file", type=Path, default=DEFAULT_DST_FILE,
                        help=f"dst text of  JSON file (default: {DEFAULT_DST_FILE})")
    parser.add_argument("--enums", nargs="*", type=int, default=[2, 3, 4],
                        help="text target enum (default: 2 3 4). enum 0,1  always uni→raw copy)")
    parser.add_argument("--games", nargs="*", default=None)
    parser.add_argument("--dry-run", action="store_true",
                        help="text file text text  text  inside text text")
    args = parser.parse_args()

    enums = set(args.enums)
    dst_map = load_dst(args.dst_file)
    print(f"dst file: {args.dst_file.resolve()}")

    # dst  textwith text warning
    empty_dst = [
        f"{g} / enum {e}"
        for g, enum_map in dst_map.items()
        if args.games is None or g in args.games
        for e, dst in enum_map.items()
        if e in enums and not dst
    ]
    if empty_dst:
        print("[WARN] dst  text text:")
        for item in empty_dst:
            print(f"  {item}")

    ann_files = sorted(args.ann_dir.rglob("*.ann.json"))
    if args.games:
        ann_files = [p for p in ann_files if p.parent.name in args.games]

    print(f"\n{'[DRY RUN] ' if args.dry_run else ''}process target: {len(ann_files)}text file  enums={sorted(enums)}\n")

    total_changed = 0
    for ann_path in ann_files:
        result = process_file(ann_path, enums, dst_map, dry_run=args.dry_run)
        game = ann_path.parent.name
        status = "CHANGED" if result["changed"] else "no change"
        print(f"  [{game}]  {status}  ({result['changed']:,} / {result['total']:,} annotations)")
        total_changed += result["changed"]

    print(f"\ntotal text: {total_changed:,}text annotation")
    if args.dry_run:
        print("(dry-run: file  text text)")


if __name__ == "__main__":
    main()
