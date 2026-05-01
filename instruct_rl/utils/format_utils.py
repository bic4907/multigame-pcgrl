"""instruct_rl/utils/format_utils.py

공통 문자열 포매팅 유틸리티.
"""


def simple_table(rows, headers):
    """psql 스타일 ASCII 테이블 문자열을 반환한다.

    Parameters
    ----------
    rows    : list of tuple — 각 행의 셀 값
    headers : list of str  — 헤더 이름

    Example output
    --------------
    +---------+--------+-----------+
    | game    | acc    | reg_loss  |
    +---------+--------+-----------+
    | dungeon | 0.9230 | 0.0312    |
    +---------+--------+-----------+
    """
    col_w = [
        max(len(str(h)), max((len(str(r[i])) for r in rows), default=0))
        for i, h in enumerate(headers)
    ]
    sep = "+-" + "-+-".join("-" * w for w in col_w) + "-+"
    fmt = "| " + " | ".join(f"{{:<{w}}}" for w in col_w) + " |"
    lines = [sep, fmt.format(*headers), sep] + [fmt.format(*r) for r in rows] + [sep]
    return "\n".join(lines)

