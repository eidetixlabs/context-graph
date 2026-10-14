from __future__ import annotations

import re

_EXCESS_BLANK_LINES = re.compile(r"\n{3,}")


def trim_content(text: str) -> str:
    text = text.strip()
    return _EXCESS_BLANK_LINES.sub("\n\n", text)
