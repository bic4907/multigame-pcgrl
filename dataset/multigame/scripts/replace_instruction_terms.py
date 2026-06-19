"""
ann.json 파일의 instruction_uni를 읽어 표현을 치환한 결과를 instruction_raw에 쓴다.
instruction_uni는 변경하지 않는다.

term_dst.json의 dst 값을 채운 뒤 실행하면 된다.

사용법:
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

# ── 찾을 표현 (src) 고정 정의 ────────────────────────────────────────────────────
# 각 enum마다 longer match(tile 포함)를 먼저, shorter(단독)를 나중에 치환한다.
# 복수형(s)도 함께 매칭한다.
# 모든 패턴이 동일한 dst로 치환된다.

TERM_PATTERNS: dict[int, list[str]] = {
    2: [r"interactive tiles?", r"interactives?"],   # longer first
    3: [r"hazard tiles?",      r"hazards?"],
    4: [r"collectable tiles?", r"collectables?"],
}

# ── dst 파일 기본 경로 ────────────────────────────────────────────────────────────
DEFAULT_DST_FILE = Path(__file__).parent / "term_dst.json"

ANN_DIR = Path("dataset/multigame/cache/artifacts")


def load_dst(dst_file: Path) -> dict[str, dict[int, str]]:
    """term_dst.json 로드. { game: { enum(int): dst(str) } }"""
    with dst_file.open(encoding="utf-8") as f:
        raw = json.load(f)
    return {
        game: {int(enum): dst for enum, dst in enum_map.items()}
        for game, enum_map in raw.items()
    }


def _pluralize(word: str) -> str:
    """영어 복수형 규칙 적용."""
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
    """단어 경계를 지키면서 pattern → dst 치환.
    - 매칭된 단어가 s로 끝나면 dst를 복수형으로 변환 (collectables → items)
    - 매칭 첫 글자 대소문자 보존
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


COPY_ENUMS = {0, 1}   # instruction_uni를 instruction_raw에 그대로 복사


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
    pos = 0  # 순서대로 교체하기 위한 검색 시작 위치

    for ann in data["annotations"]:
        enum = int(ann["reward_enum"])
        old_raw = ann.get("instruction_raw") or ""

        if enum in COPY_ENUMS:
            new_raw = ann.get("instruction_uni") or ""
        elif enum in enums:
            uni = ann.get("instruction_uni") or ""
            new_raw = replace_in_text(uni, game, enum, dst_map)
            if new_raw == uni:
                # 치환 없음 — instruction_raw 위치만 pos 전진
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
    parser = argparse.ArgumentParser(description="instruction_uni 표현 치환")
    parser.add_argument("--ann-dir", type=Path, default=ANN_DIR)
    parser.add_argument("--dst-file", type=Path, default=DEFAULT_DST_FILE,
                        help=f"dst 정의 JSON 파일 (기본: {DEFAULT_DST_FILE})")
    parser.add_argument("--enums", nargs="*", type=int, default=[2, 3, 4],
                        help="치환 대상 enum (기본: 2 3 4). enum 0,1은 항상 uni→raw 복사)")
    parser.add_argument("--games", nargs="*", default=None)
    parser.add_argument("--dry-run", action="store_true",
                        help="실제 파일 수정 없이 변경 내역만 출력")
    args = parser.parse_args()

    enums = set(args.enums)
    dst_map = load_dst(args.dst_file)
    print(f"dst 파일: {args.dst_file.resolve()}")

    # dst가 비어있는 항목 경고
    empty_dst = [
        f"{g} / enum {e}"
        for g, enum_map in dst_map.items()
        if args.games is None or g in args.games
        for e, dst in enum_map.items()
        if e in enums and not dst
    ]
    if empty_dst:
        print("[WARN] dst가 비어있어 건너뜀:")
        for item in empty_dst:
            print(f"  {item}")

    ann_files = sorted(args.ann_dir.rglob("*.ann.json"))
    if args.games:
        ann_files = [p for p in ann_files if p.parent.name in args.games]

    print(f"\n{'[DRY RUN] ' if args.dry_run else ''}처리 대상: {len(ann_files)}개 파일  enums={sorted(enums)}\n")

    total_changed = 0
    for ann_path in ann_files:
        result = process_file(ann_path, enums, dst_map, dry_run=args.dry_run)
        game = ann_path.parent.name
        status = "CHANGED" if result["changed"] else "no change"
        print(f"  [{game}]  {status}  ({result['changed']:,} / {result['total']:,} annotations)")
        total_changed += result["changed"]

    print(f"\n총 변경: {total_changed:,}개 annotation")
    if args.dry_run:
        print("(dry-run: 파일은 수정되지 않았습니다)")


if __name__ == "__main__":
    main()
