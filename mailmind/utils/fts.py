from __future__ import annotations

import re
from typing import List


_TOKEN_RE = re.compile(r"[A-Za-z0-9_@.-]+", re.UNICODE)
_STOPWORDS = {
    # English minimal
    "a",
    "an",
    "the",
    "and",
    "or",
    "is",
    "are",
    "to",
    "of",
    "in",
    "on",
    "for",
    "with",
    "it",
    "this",
    "that",
    "there",
    "s",
    # Common variants
    "pdf",
}


def nl_to_fts_query(q: str, min_token_len: int = 2) -> str:
    """Turn a natural language string into a safe FTS5 query.

    - Extract alnum tokens (and _ @ . -), lowercase
    - Drop very short tokens and simple stopwords
    - Join with AND to encourage intersection
    """
    if not q:
        return ""
    toks = [t.lower() for t in _TOKEN_RE.findall(q)]
    toks = [t for t in toks if len(t) >= min_token_len and t not in _STOPWORDS]
    if not toks:
        return ""
    # Join with AND; caller can also OR by repeating calls if desired
    return " AND ".join(toks)

