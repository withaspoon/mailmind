from __future__ import annotations

import re
from html import unescape
from typing import Iterable, List


_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


def html_to_text(html: str) -> str:
    """Very simple HTML→text fallback (no external deps)."""
    # Remove script/style contents
    html = re.sub(r"<script[\s\S]*?</script>", " ", html, flags=re.IGNORECASE)
    html = re.sub(r"<style[\s\S]*?</style>", " ", html, flags=re.IGNORECASE)
    # Replace breaks with newlines
    html = re.sub(r"<br\s*/?>", "\n", html, flags=re.IGNORECASE)
    html = re.sub(r"</p>", "\n\n", html, flags=re.IGNORECASE)
    # Strip tags
    text = _TAG_RE.sub(" ", html)
    text = unescape(text)
    # Collapse whitespace
    text = _WS_RE.sub(" ", text)
    # Normalize newlines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_text(text: str, target_size: int = 800) -> List[str]:
    """Chunk text into ~target_size characters, respecting paragraph breaks when possible."""
    if not text:
        return []
    paras = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    chunks: List[str] = []
    buf: List[str] = []
    size = 0
    for p in paras:
        if (size + len(p)) > target_size and buf:
            chunks.append("\n\n".join(buf))
            buf = []
            size = 0
        buf.append(p)
        size += len(p) + 2
    if buf:
        chunks.append("\n\n".join(buf))
    # If no paras, fallback to hard split
    if not chunks:
        s = text
        for i in range(0, len(s), target_size):
            chunks.append(s[i : i + target_size])
    return chunks

