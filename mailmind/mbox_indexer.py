from __future__ import annotations

import email
import json
import mailbox
import sqlite3
from pathlib import Path
from typing import List, Tuple

from .indexer import (
    ParsedMessage,
    _extract_body_and_attachments,
    _normalize_addrs,
    _parse_date_to_ts,
)
from .utils.text import chunk_text
from .progress import ProgressReporter, ProgressConfig, resolve_progress_mode
import time


def _parsed_from_message(msg: email.message.EmailMessage, account: str, folder: str) -> ParsedMessage:
    message_id = (msg.get("Message-ID") or "").strip()
    subject = (msg.get("Subject") or "").strip()
    from_email = _normalize_addrs(msg.get("From"))
    to_emails = _normalize_addrs(msg.get("To"))
    cc_emails = _normalize_addrs(msg.get("Cc"))
    date_ts = _parse_date_to_ts(msg.get("Date"))
    in_reply_to = (msg.get("In-Reply-To") or "").strip()

    headers = {k: v for (k, v) in msg.items() if k}
    # body + attachments
    body, atts, has_atts = _extract_body_and_attachments(msg)

    if not message_id:
        # Create synthetic ID if missing
        import hashlib

        base = (subject + "|" + from_email + "|" + to_emails + "|" + str(date_ts)).encode()
        body_hash = hashlib.sha256((body or "").encode()).hexdigest()
        message_id = "synthetic-" + hashlib.sha256(base + body_hash.encode()).hexdigest()

    return ParsedMessage(
        path=Path(""),
        account=account,
        folder=folder,
        message_id=message_id,
        subject=subject,
        from_email=from_email,
        to_emails=to_emails,
        cc_emails=cc_emails,
        date_ts=date_ts,
        in_reply_to=in_reply_to,
        headers_json=json.dumps(headers, ensure_ascii=False),
        body=body,
        has_attachments=has_atts,
        attachments=atts,
    )


def index_mbox(
    mbox_path: Path,
    db_path: Path,
    attachments_root: Path,
    account: str = "default",
    folder: str | None = None,
    batch_size: int = 200,
    progress_mode: str | None = None,
    progress_root: Path | None = None,
) -> None:
    mbox_path = mbox_path.expanduser().resolve()
    if not mbox_path.exists():
        raise FileNotFoundError(str(mbox_path))
    folder = folder or f"mbox/{mbox_path.stem}"

    con = sqlite3.connect(str(db_path))
    con.execute("PRAGMA journal_mode=WAL;")
    con.execute("PRAGMA foreign_keys=ON;")
    cur = con.cursor()

    # Local import to avoid extra deps
    import hashlib
    import os

    def _ensure_attachment_saved(filename: str, content: bytes) -> tuple[str, Path]:
        sha = hashlib.sha256(content).hexdigest()
        d = attachments_root / sha
        d.mkdir(parents=True, exist_ok=True)
        safe_name = os.path.basename(filename or "attachment.bin")
        out = d / f"original_{safe_name}"
        if not out.exists():
            out.write_bytes(content)
        return sha, out

    count = 0
    inserted = 0
    skipped = 0

    box = mailbox.mbox(str(mbox_path))
    try:
        total = len(box)
    except Exception:
        total = None
    reporter = ProgressReporter(
        ProgressConfig(
            mode=resolve_progress_mode(progress_mode),
            root=progress_root,
            key=f"index-mbox:{mbox_path.name}",
            total=total,
            desc=f"Index {mbox_path.name}",
        )
    )
    try:
        for msg in box:
            count += 1
            if not isinstance(msg, email.message.EmailMessage):
                # Convert legacy Message to EmailMessage if needed
                try:
                    msg = email.message_from_bytes(msg.as_bytes())  # type: ignore
                except Exception:
                    continue
            parsed = _parsed_from_message(msg, account=account, folder=folder)

            try:
                cur.execute(
                    """
                    INSERT OR IGNORE INTO messages
                    (account, folder, message_id, subject, from_email, to_emails, cc_emails, date_ts, in_reply_to, thread_id, has_attachments, headers_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, '', ?, ?)
                    """,
                    (
                        parsed.account,
                        parsed.folder,
                        parsed.message_id,
                        parsed.subject,
                        parsed.from_email,
                        parsed.to_emails,
                        parsed.cc_emails,
                        parsed.date_ts,
                        parsed.in_reply_to,
                        1 if parsed.has_attachments else 0,
                        parsed.headers_json,
                    ),
                )

                if cur.rowcount == 0:
                    skipped += 1
                    if count % batch_size == 0:
                        con.commit()
                    continue

                inserted += 1
                cur.execute("SELECT id FROM messages WHERE message_id=?", (parsed.message_id,))
                row = cur.fetchone()
                if not row:
                    continue
                mid = int(row[0])

                cur.execute(
                    "INSERT INTO messages_fts (message_id, subject, from_email, to_emails, cc_emails, body) VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        parsed.message_id,
                        parsed.subject,
                        parsed.from_email,
                        parsed.to_emails,
                        parsed.cc_emails,
                        parsed.body,
                    ),
                )

                attach_names = []
                for (fname, mime, content) in parsed.attachments:
                    if not content:
                        continue
                    sha, out = _ensure_attachment_saved(fname, content)
                    cur.execute(
                        "INSERT INTO attachments (message_id, filename, mime, sha256, bytes, path_original) VALUES (?, ?, ?, ?, ?, ?)",
                        (mid, fname, mime, sha, len(content), str(out)),
                    )
                    cur.execute(
                        "INSERT OR IGNORE INTO jobs_attachments(attachment_id, status, tries, updated_ts) VALUES ((SELECT last_insert_rowid()), 'pending', 0, ?)",
                        (int(time.time()),),
                    )
                    attach_names.append(fname or "")

                if parsed.body:
                    for ch in chunk_text(parsed.body, target_size=800):
                        cur.execute(
                            "INSERT INTO chunks (message_id, attachment_id, kind, text, token_count) VALUES (?, NULL, 'body', ?, NULL)",
                            (mid, ch),
                        )
                        cid = cur.lastrowid
                        cur.execute("INSERT INTO chunks_fts (chunk_id, text) VALUES (?, ?)", (cid, ch))
                        cur.execute(
                            "INSERT OR IGNORE INTO jobs_embeddings(chunk_id, status, tries, updated_ts) VALUES (?, 'pending', 0, ?)",
                            (cid, int(time.time())),
                        )

                # Meta chunk per message
                meta_parts = [
                    f"SUBJECT: {parsed.subject}",
                    f"FROM: {parsed.from_email}",
                    f"TO: {parsed.to_emails}",
                    f"CC: {parsed.cc_emails}",
                    f"FOLDER: {folder}",
                    f"ATTACHMENTS: {', '.join(x for x in attach_names if x)}",
                ]
                if parsed.body:
                    meta_parts.append("BODY: " + parsed.body[:500])
                meta_text = "\n".join(meta_parts)
                if meta_text.strip():
                    cur.execute(
                        "INSERT INTO chunks (message_id, attachment_id, kind, text, token_count) VALUES (?, NULL, 'meta', ?, NULL)",
                        (mid, meta_text),
                    )
                    meta_cid = cur.lastrowid
                    cur.execute("INSERT INTO chunks_fts (chunk_id, text) VALUES (?, ?)", (meta_cid, meta_text))
                    cur.execute(
                        "INSERT OR IGNORE INTO jobs_embeddings(chunk_id, status, tries, updated_ts) VALUES (?, 'pending', 0, ?)",
                        (meta_cid, int(time.time())),
                    )

                if count % batch_size == 0:
                    con.commit()

            except sqlite3.Error:
                con.rollback()
                continue
            finally:
                reporter.update(1)
    finally:
        box.close()

    con.commit()
    con.close()
    print(f"MBOX indexing done. msgs={count} inserted={inserted} skipped_existing={skipped}")
    reporter.close()
