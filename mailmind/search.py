from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Literal, Optional


ResultType = Literal["message", "chunk"]


@dataclass
class SearchResult:
    type: ResultType
    score: float
    message_id: str
    subject: str
    from_email: str
    to_emails: str
    date_ts: int
    folder: str
    account: str
    snippet: str
    chunk_id: Optional[int] = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


def _open(db_path: Path) -> sqlite3.Connection:
    con = sqlite3.connect(str(db_path))
    con.row_factory = sqlite3.Row
    return con


def search(db_path: Path, query: str, limit: int = 20) -> List[SearchResult]:
    con = _open(db_path)
    out: List[SearchResult] = []
    try:
        # Messages FTS over body primarily; get metadata via join on message_id
        q1 = (
            """
            SELECT m.account, m.folder, m.message_id AS mid, m.subject, m.from_email, m.to_emails, m.date_ts,
                   snippet(messages_fts, 5, '[', ']', '…', 8) AS snip,
                   bm25(messages_fts) AS score
            FROM messages_fts
            JOIN messages m ON m.message_id = messages_fts.message_id
            WHERE messages_fts MATCH ?
            ORDER BY score LIMIT ?
            """
        )
        for row in con.execute(q1, (query, limit)):
            out.append(
                SearchResult(
                    type="message",
                    score=float(row["score"]),
                    message_id=row["mid"],
                    subject=row["subject"] or "",
                    from_email=row["from_email"] or "",
                    to_emails=row["to_emails"] or "",
                    date_ts=int(row["date_ts"]) if row["date_ts"] is not None else 0,
                    folder=row["folder"] or "",
                    account=row["account"] or "",
                    snippet=row["snip"] or "",
                )
            )

        # Chunk FTS; join via chunk_id field
        q2 = (
            """
            SELECT m.account, m.folder, m.message_id AS mid, m.subject, m.from_email, m.to_emails, m.date_ts,
                   c.id AS chunk_id,
                   snippet(chunks_fts, 1, '[', ']', '…', 8) AS snip,
                   bm25(chunks_fts) AS score
            FROM chunks_fts
            JOIN chunks c ON c.id = CAST(chunks_fts.chunk_id AS INTEGER)
            JOIN messages m ON m.id = c.message_id
            WHERE chunks_fts MATCH ?
            ORDER BY score LIMIT ?
            """
        )
        for row in con.execute(q2, (query, limit)):
            out.append(
                SearchResult(
                    type="chunk",
                    score=float(row["score"]),
                    message_id=row["mid"],
                    subject=row["subject"] or "",
                    from_email=row["from_email"] or "",
                    to_emails=row["to_emails"] or "",
                    date_ts=int(row["date_ts"]) if row["date_ts"] is not None else 0,
                    folder=row["folder"] or "",
                    account=row["account"] or "",
                    snippet=row["snip"] or "",
                    chunk_id=int(row["chunk_id"]) if row["chunk_id"] is not None else None,
                )
            )
    finally:
        con.close()

    # Sort combined results by score ascending (bm25 lower is better), then stable
    out.sort(key=lambda r: r.score)
    return out[:limit]

