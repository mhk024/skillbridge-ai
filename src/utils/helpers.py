"""Small shared helpers."""


def normalize_output(text: str) -> str:
    """Normalize program output for comparison.

    Unifies line endings, strips trailing spaces on each line and removes
    trailing blank lines. Leading/inner content is left untouched.
    """
    lines = [ln.rstrip() for ln in (text or "").replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    while lines and lines[-1] == "":
        lines.pop()
    return "\n".join(lines)