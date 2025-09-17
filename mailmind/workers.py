from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from typing import List, Tuple

from .parsing.docs import OCRConfig, extract_attachment_text
from .utils.text import chunk_text
from .embedder import Embedder
from .ann import HnswIndex
from .progress import ProgressReporter, ProgressConfig, resolve_progress_mode


def _now() -> int:
    return int(time.time())


def worker_attachments_once(
    db_path: Path,
    attachments_root: Path,
    langs: str = "eng",
    min_chars_per_page: int = 120,
    max_jobs: int = 50,
    progress_mode: str | None = None,
    progress_root: Path | None = None,
) -> Tuple[int, int, int]:
    """Process up to max_jobs pending attachment OCR tasks.

    Returns (taken, succeeded, chunks_added).
    """
    cfg = OCRConfig(langs=langs, min_chars_per_page=min_chars_per_page)
    con = sqlite3.connect(str(db_path))
    con.execute("PRAGMA journal_mode=WAL;")
    con.execute("PRAGMA foreign_keys=ON;")
    cur = con.cursor()

    # Pick pending jobs
    rows = list(
        cur.execute(
            """
            SELECT j.id, a.id as aid, a.message_id, a.mime, a.sha256, a.path_original
            FROM jobs_attachments j
            JOIN attachments a ON a.id = j.attachment_id
            WHERE j.status = 'pending'
            ORDER BY j.id LIMIT ?
            """,
            (max_jobs,),
        )
    )
    taken = len(rows)
    success = 0
    chunks_added = 0
    reporter = ProgressReporter(
        ProgressConfig(
            mode=resolve_progress_mode(progress_mode),
            root=progress_root,
            key="worker-attachments",
            total=taken,
            desc="OCR jobs",
        )
    )

    for jid, aid, mid, mime, sha, p_orig in rows:
        cur.execute(
            "UPDATE jobs_attachments SET status='processing', tries=tries+1, updated_ts=? WHERE id=?",
            (_now(), jid),
        )
        con.commit()
        try:
            path = Path(p_orig)
            cache_dir = attachments_root / (sha or "")
            path_text, text = extract_attachment_text(path, mime or "", cache_dir, cfg)
            if path_text and text:
                cur.execute("UPDATE attachments SET path_text=? WHERE id=?", (str(path_text), aid))
                # Insert chunks if not present
                ex = cur.execute(
                    "SELECT COUNT(*) FROM chunks WHERE attachment_id=? AND kind='attachment'",
                    (aid,),
                ).fetchone()[0]
                if ex == 0:
                    for ch in chunk_text(text, target_size=800):
                        cur.execute(
                            "INSERT INTO chunks (message_id, attachment_id, kind, text, token_count) VALUES (?, ?, 'attachment', ?, NULL)",
                            (mid, aid, ch),
                        )
                        cid = cur.lastrowid
                        cur.execute("INSERT INTO chunks_fts (chunk_id, text) VALUES (?, ?)", (cid, ch))
                        # Enqueue embedding job for new chunks
                        cur.execute(
                            "INSERT OR IGNORE INTO jobs_embeddings(chunk_id, status, tries, updated_ts) VALUES (?, 'pending', 0, ?)",
                            (cid, _now()),
                        )
                        chunks_added += 1
                cur.execute(
                    "UPDATE jobs_attachments SET status='done', last_error=NULL, updated_ts=? WHERE id=?",
                    (_now(), jid),
                )
                success += 1
            else:
                cur.execute(
                    "UPDATE jobs_attachments SET status='error', last_error=?, updated_ts=? WHERE id=?",
                    ("no text extracted", _now(), jid),
                )
        except Exception as e:
            cur.execute(
                "UPDATE jobs_attachments SET status='error', last_error=?, updated_ts=? WHERE id=?",
                (str(e)[:500], _now(), jid),
            )
        con.commit()
        reporter.update(1)

    con.close()
    reporter.close()
    return taken, success, chunks_added


def worker_embeddings_once(
    db_path: Path,
    vectors_path: Path,
    model: str,
    dim: int,
    backend: str = "auto",
    batch_size: int = 128,
    max_jobs: int = 200,
    progress_mode: str | None = None,
    progress_root: Path | None = None,
) -> Tuple[int, int]:
    """Process up to max_jobs pending embedding tasks.

    Returns (taken, embedded).
    """
    con = sqlite3.connect(str(db_path))
    con.row_factory = sqlite3.Row
    cur = con.cursor()

    rows = list(
        cur.execute(
            """
            SELECT j.id, c.id AS chunk_id, c.text
            FROM jobs_embeddings j
            JOIN chunks c ON c.id = j.chunk_id
            WHERE j.status = 'pending'
            ORDER BY j.id LIMIT ?
            """,
            (max_jobs,),
        )
    )
    taken = len(rows)
    if not rows:
        con.close()
        return 0, 0

    # Mark as processing
    cur.executemany(
        "UPDATE jobs_embeddings SET status='processing', tries=tries+1, updated_ts=? WHERE id=?",
        [(_now(), int(r["id"])) for r in rows],
    )
    con.commit()

    chunk_ids = [int(r["chunk_id"]) for r in rows]
    texts = [r["text"] or "" for r in rows]
    reporter = ProgressReporter(
        ProgressConfig(
            mode=resolve_progress_mode(progress_mode),
            root=progress_root,
            key="worker-embeddings",
            total=taken,
            desc="Embed jobs",
        )
    )

    emb = Embedder(model=model, dim=dim, backend=backend)
    vecs = emb.encode(texts, batch_size=batch_size)

    ann = HnswIndex(dim=dim, path=vectors_path)
    ann_ok = False
    if ann.available():
        try:
            ann.load()
            ann_ok = True
        except Exception:
            # init new index
            ann.init(capacity=max(len(vecs) * 2, 10000))
            ann_ok = True

    embedded = 0
    try:
        if ann_ok:
            ann.add(vecs, chunk_ids)
            ann.save()
        # Record mapping and mark jobs done
        cur.executemany(
            "INSERT OR IGNORE INTO chunk_vectors (chunk_id, model, dim) VALUES (?, ?, ?)",
            [(cid, model, int(dim)) for cid in chunk_ids],
        )
        cur.executemany(
            "UPDATE jobs_embeddings SET status='done', last_error=NULL, updated_ts=? WHERE chunk_id=?",
            [(_now(), cid) for cid in chunk_ids],
        )
        embedded = len(chunk_ids)
        con.commit()
    except Exception as e:
        # Mark errors
        cur.executemany(
            "UPDATE jobs_embeddings SET status='error', last_error=?, updated_ts=? WHERE chunk_id=?",
            [(str(e)[:500], _now(), cid) for cid in chunk_ids],
        )
        con.commit()
    finally:
        con.close()
        # Consider all processed in this batch
        reporter.update(taken)
        reporter.close()

    return taken, embedded
