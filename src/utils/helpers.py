"""Small shared helpers."""

from datetime import datetime, timezone


def normalize_output(text: str) -> str:
    """Normalize program output for comparison.

    Unifies line endings, strips trailing spaces on each line and removes
    trailing blank lines. Leading/inner content is left untouched.
    """
    lines = [ln.rstrip() for ln in (text or "").replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    while lines and lines[-1] == "":
        lines.pop()
    return "\n".join(lines)


def utc_to_local_str(value: datetime | None, fmt: str = "%d %b %Y, %H:%M") -> str:
    """Show a naive-UTC database timestamp in the server's local time zone."""
    if value is None:
        return ""
    return value.replace(tzinfo=timezone.utc).astimezone().strftime(fmt)