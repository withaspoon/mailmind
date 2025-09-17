from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import List

from .hybrid import HybridConfig, hybrid_search_nl
from .llm import LLMConfig, load_llm_from_env


@dataclass
class ContextDoc:
    message_id: str
    subject: str
    from_email: str
    folder: str
    account: str
    date_ts: int
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


def _fmt_date(ts: int) -> str:
    if not ts:
        return ""
    try:
        from datetime import datetime

        return datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d")
    except Exception:
        return ""


def build_prompt(query: str, docs: List[ContextDoc]) -> str:
    lines = [
        "You are a local email research assistant. Answer concisely with citations.",
        "Question:",
        query,
        "",
        "Context (each item shows subject, from, folder/account, id, then excerpt):",
    ]
    for i, d in enumerate(docs, 1):
        date_s = _fmt_date(d.date_ts)
        date_part = f" date={date_s}" if date_s else ""
        lines.append(f"[{i}] {d.subject} — {d.from_email} ({d.account}/{d.folder}){date_part} id={d.message_id}")
        lines.append(d.text)
        lines.append("")
    lines.append("Instructions: Provide a short answer. When referencing, cite like [1 2024-03-12], [2 2024-05-01] including date if shown.")
    return "\n".join(lines)


def summarize_query(db_path: Path, root: Path, query: str, top_k: int = 8) -> tuple[object, str]:
    cfg = HybridConfig(vectors_path=root / "vectors" / "mailmind_hnsw.bin")
    plan, results = hybrid_search_nl(db_path, query, cfg)
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
                date_ts=r.date_ts,
                snippet=r.snippet,
                text=text,
            )
        )
    if not docs:
        return plan, "No results found."

    prompt = build_prompt(query, docs)
    llm = load_llm_from_env()
    summary = llm.generate(prompt)
    # Append sources with dates for determinism
    src_lines = ["", "Sources:"]
    for i, d in enumerate(docs, 1):
        date_s = _fmt_date(d.date_ts)
        date_part = f" {date_s}" if date_s else ""
        src_lines.append(f"- [{i}{date_part}] {d.subject} — {d.from_email} ({d.account}/{d.folder}) id={d.message_id}")
    return plan, summary + "\n" + "\n".join(src_lines)
