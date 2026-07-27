"""instruct_rl/utils/format_utils.py

Common string-formatting utilities.
"""


def simple_table(rows, headers):
    """Return a psql-style ASCII table string.

    Parameters
    ----------
    rows    : list of tuples containing cell values for each row
    headers : list of header names

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
