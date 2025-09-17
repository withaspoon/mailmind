from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import List

from .hybrid import HybridConfig, hybrid_search
from .llm import LLMConfig, load_llm_from_env


@dataclass
class ContextDoc:
    message_id: str
    subject: str
    from_email: str
    folder: str
    account: str
    snippet: str
    text: str


def _fetch_full_text_for_message(db_path: Path, message_id: str, char_limit: int = 2000) -> str:
    con = sqlite3.connect(str(db_path))
    con.row_factory = sqlite3.Row
    try:
        row = con.execute(
            "SELECT body FROM messages_fts WHERE message_id=?", (message_id,)
        ).fetchone()
        if row and row["body"]:
            body = row["body"]
            if len(body) > char_limit:
                return body[:char_limit] + "…"
            return body
        # Fallback to concatenation of chunk texts
        rows = con.execute(
            """
            SELECT c.text FROM chunks c
            JOIN messages m ON m.id = c.message_id
            WHERE m.message_id=?
            ORDER BY c.id LIMIT 5
            """,
            (message_id,),
        ).fetchall()
        parts = [r["text"] for r in rows if r and r["text"]]
        s = "\n\n".join(parts)
        if len(s) > char_limit:
            return s[:char_limit] + "…"
        return s
    finally:
        con.close()


def build_prompt(query: str, docs: List[ContextDoc]) -> str:
    lines = [
        "You are a local email research assistant. Answer concisely with citations.",
        "Question:",
        query,
        "",
        "Context (each item shows subject, from, folder/account, id, then excerpt):",
    ]
    for i, d in enumerate(docs, 1):
        lines.append(f"[{i}] {d.subject} — {d.from_email} ({d.account}/{d.folder}) id={d.message_id}")
        lines.append(d.text)
        lines.append("")
    lines.append("Instructions: Provide a short answer. When referencing, cite like [1], [2].")
    return "\n".join(lines)


def summarize_query(db_path: Path, root: Path, query: str, top_k: int = 8) -> str:
    cfg = HybridConfig(vectors_path=root / "vectors" / "mailmind_hnsw.bin")
    results = hybrid_search(db_path, query, cfg)
    # Build contexts
    docs: List[ContextDoc] = []
    for r in results[:top_k]:
        text = _fetch_full_text_for_message(db_path, r.message_id, char_limit=1800)
        if not text:
            text = r.snippet
        docs.append(
            ContextDoc(
                message_id=r.message_id,
                subject=r.subject,
                from_email=r.from_email,
                folder=r.folder,
                account=r.account,
                snippet=r.snippet,
                text=text,
            )
        )
    if not docs:
        return "No results found."

    prompt = build_prompt(query, docs)
    llm = load_llm_from_env()
    return llm.generate(prompt)

