"""instruct_rl/utils/format_utils.py

common string text utility.
"""


def simple_table(rows, headers):
    """psql texttile ASCII text text string  returntext.

    Parameters
    ----------
    rows    : list of tuple — each row of  cell text
    headers : list of str  — text name

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

