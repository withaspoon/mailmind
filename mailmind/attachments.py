from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from .parsing.docs import OCRConfig, extract_attachment_text
from .utils.text import chunk_text
from .progress import ProgressReporter, ProgressConfig, resolve_progress_mode


def process_attachments(
    db_path: Path,
    attachments_root: Path,
    langs: str = "eng",
    min_chars_per_page: int = 120,
    limit: int = 0,
    reprocess: bool = False,
    progress_mode: str | None = None,
    progress_root: Path | None = None,
) -> Tuple[int, int, int]:
    """Extract text (with OCR if necessary) for attachments and index into FTS.

    Returns (total_attachments_considered, processed, chunk_rows_added).
    """
    cfg = OCRConfig(langs=langs, min_chars_per_page=min_chars_per_page)
    con = sqlite3.connect(str(db_path))
    con.execute("PRAGMA journal_mode=WAL;")
    con.execute("PRAGMA foreign_keys=ON;")
    cur = con.cursor()

    # Fetch attachment rows; group by sha256 to avoid re-OCR duplicates.
    q = "SELECT id, message_id, filename, mime, sha256, path_original, path_text FROM attachments ORDER BY id"
    rows = list(cur.execute(q))
    if limit and limit > 0:
        rows = rows[:limit]

    # Build a map sha -> cached text path and content
    text_cache: Dict[str, Tuple[Path | None, str]] = {}
    processed = 0
    chunks_added = 0

    reporter = ProgressReporter(
        ProgressConfig(
            mode=resolve_progress_mode(progress_mode),
            root=progress_root,
            key="process-attachments",
            total=len(rows),
            desc="Attachments",
        )
    )
    for aid, mid, fname, mime, sha, p_orig, p_text in rows:
        # Skip if already has text and not reprocessing
        if not reprocess and p_text:
            # Still ensure chunks exist
            existing = cur.execute(
                "SELECT COUNT(*) FROM chunks WHERE attachment_id=? AND kind='attachment'", (aid,)
            ).fetchone()[0]
            if existing:
                continue
        try:
            original_path = Path(p_orig)
        except Exception:
            continue
        cache_dir = attachments_root / (sha or "")

        if sha in text_cache:
            path_text, text = text_cache[sha]
        else:
            path_text, text = extract_attachment_text(original_path, mime or "", cache_dir, cfg)
            text_cache[sha] = (path_text, text)

        if path_text and text:
            processed += 1
            # Update DB path_text (idempotent)
            cur.execute("UPDATE attachments SET path_text=? WHERE id=?", (str(path_text), aid))
            # Insert chunks if not present
            ex = cur.execute(
                "SELECT COUNT(*) FROM chunks WHERE attachment_id=? AND kind='attachment'", (aid,)
            ).fetchone()[0]
            if ex == 0:
                for ch in chunk_text(text, target_size=800):
                    cur.execute(
                        "INSERT INTO chunks (message_id, attachment_id, kind, text, token_count) VALUES (?, ?, 'attachment', ?, NULL)",
                        (mid, aid, ch),
                    )
                    cid = cur.lastrowid
                    cur.execute("INSERT INTO chunks_fts (chunk_id, text) VALUES (?, ?)", (cid, ch))
                    chunks_added += 1
        # Commit periodically
        if (processed + chunks_added) % 100 == 0:
            con.commit()
        reporter.update(1)

    con.commit()
    con.close()
    reporter.close()
    return (len(rows), processed, chunks_added)
