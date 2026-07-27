"""
Read instruction_uni from ann.json and write replaced expressions to instruction_raw.
Do not modify instruction_uni.

Fill in dst values in term_dst.json before running.

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

# ── Fixed definitions of source expressions ───────────────────────────────────
# For each enum, replace longer matches containing "tile" before shorter ones.
# Match plural forms ending in s as well. Every pattern maps to the same dst.

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
    """Apply English pluralization rules."""
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
    """Replace pattern with dst while respecting word boundaries.
    - Pluralize dst when the match ends in s (collectables -> items).
    - Preserve the case of the first matched letter.
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
    pos = 0  # Search start for ordered replacements

    for ann in data["annotations"]:
        enum = int(ann["reward_enum"])
        old_raw = ann.get("instruction_raw") or ""

        if enum in COPY_ENUMS:
            new_raw = ann.get("instruction_uni") or ""
        elif enum in enums:
            uni = ann.get("instruction_uni") or ""
            new_raw = replace_in_text(uni, game, enum, dst_map)
            if new_raw == uni:
                # No replacement; advance only within instruction_raw
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
    parser = argparse.ArgumentParser(description="Replace instruction_uni expressions")
    parser.add_argument("--ann-dir", type=Path, default=ANN_DIR)
    parser.add_argument("--dst-file", type=Path, default=DEFAULT_DST_FILE,
                        help=f"JSON file defining dst values (default: {DEFAULT_DST_FILE})")
    parser.add_argument("--enums", nargs="*", type=int, default=[2, 3, 4],
                        help="Enums to replace (default: 2 3 4); enums 0 and 1 always copy uni to raw")
    parser.add_argument("--games", nargs="*", default=None)
    parser.add_argument("--dry-run", action="store_true",
                        help="Print changes without modifying files")
    args = parser.parse_args()

    enums = set(args.enums)
    dst_map = load_dst(args.dst_file)
    print(f"dst file: {args.dst_file.resolve()}")

    # Warn about entries with empty dst
    empty_dst = [
        f"{g} / enum {e}"
        for g, enum_map in dst_map.items()
        if args.games is None or g in args.games
        for e, dst in enum_map.items()
        if e in enums and not dst
    ]
    if empty_dst:
        print("[WARN] Skipping entries with empty dst:")
        for item in empty_dst:
            print(f"  {item}")

    ann_files = sorted(args.ann_dir.rglob("*.ann.json"))
    if args.games:
        ann_files = [p for p in ann_files if p.parent.name in args.games]

    print(f"\n{'[DRY RUN] ' if args.dry_run else ''}Targets: {len(ann_files)} files  enums={sorted(enums)}\n")

    total_changed = 0
    for ann_path in ann_files:
        result = process_file(ann_path, enums, dst_map, dry_run=args.dry_run)
        game = ann_path.parent.name
        status = "CHANGED" if result["changed"] else "no change"
        print(f"  [{game}]  {status}  ({result['changed']:,} / {result['total']:,} annotations)")
        total_changed += result["changed"]

    print(f"\nTotal changes: {total_changed:,} annotations")
    if args.dry_run:
        print("(dry run: no files were modified)")


if __name__ == "__main__":
    main()
