from __future__ import annotations

import re
from typing import List


_TOKEN_RE = re.compile(r"[\w@.\-]+", re.UNICODE)
_STOPWORDS = set()  # Avoid language-specific stopwords; rely on embeddings + planner


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
    # Join with AND; keeps FTS safe and language-agnostic
    return " AND ".join(toks)
