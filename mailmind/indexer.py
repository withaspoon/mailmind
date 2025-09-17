from __future__ import annotations

import email
import email.policy
import email.utils
import hashlib
import json
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Tuple

from .utils.text import html_to_text, chunk_text
from .progress import ProgressReporter, ProgressConfig, resolve_progress_mode
import time


@dataclass
class ParsedMessage:
    path: Path
    account: str
    folder: str
    message_id: str
    subject: str
    from_email: str
    to_emails: str
    cc_emails: str
    date_ts: int
    in_reply_to: str
    headers_json: str
    body: str
    has_attachments: bool
    attachments: List[Tuple[str, str, bytes]]  # (filename, mime, content)


def _normalize_addrs(value: Optional[str]) -> str:
    if not value:
        return ""
    pairs = email.utils.getaddresses([value])
    return ", ".join(addr for _name, addr in pairs if addr)


def _parse_date_to_ts(value: Optional[str]) -> int:
    if not value:
        return 0
    try:
        dt = email.utils.parsedate_to_datetime(value)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp())
    except Exception:
        return 0


def _extract_body_and_attachments(msg: email.message.EmailMessage) -> Tuple[str, List[Tuple[str, str, bytes]], bool]:
    plain_parts: List[str] = []
    html_parts: List[str] = []
    atts: List[Tuple[str, str, bytes]] = []
    has_atts = False

    if msg.is_multipart():
        for part in msg.walk():
            ctype = part.get_content_type()
            disp = (part.get_content_disposition() or "").lower()
            if ctype == "text/plain" and disp != "attachment":
                try:
                    plain_parts.append(part.get_content().strip())
                except Exception:
                    payload = part.get_payload(decode=True) or b""
                    try:
                        plain_parts.append(payload.decode(errors="ignore"))
                    except Exception:
                        pass
            elif ctype == "text/html" and disp != "attachment":
                try:
                    html_parts.append(part.get_content())
                except Exception:
                    payload = part.get_payload(decode=True) or b""
                    try:
                        html_parts.append(payload.decode(errors="ignore"))
                    except Exception:
                        pass
            else:
                filename = part.get_filename() or ""
                if disp in {"attachment", "inline"} or (filename and disp != "attachment") or not ctype.startswith("text/"):
                    try:
                        content = part.get_payload(decode=True) or b""
                    except Exception:
                        content = b""
                    if filename or content:
                        has_atts = True
                        atts.append((filename or f"part.{ctype.replace('/', '.')}", ctype, content))
    else:
        ctype = msg.get_content_type()
        if ctype == "text/plain":
            try:
                plain_parts.append(msg.get_content())
            except Exception:
                payload = msg.get_payload(decode=True) or b""
                try:
                    plain_parts.append(payload.decode(errors="ignore"))
                except Exception:
                    pass
        elif ctype == "text/html":
            try:
                html_parts.append(msg.get_content())
            except Exception:
                payload = msg.get_payload(decode=True) or b""
                try:
                    html_parts.append(payload.decode(errors="ignore"))
                except Exception:
                    pass

    body = "\n".join(plain_parts).strip()
    if not body and html_parts:
        # Fallback to HTML→text
        body = html_to_text("\n".join(html_parts))
    return body, atts, has_atts or bool(atts)


def parse_eml(path: Path, account: str, folder: str) -> ParsedMessage | None:
    try:
        raw = path.read_bytes()
    except Exception:
        return None
    try:
        msg = email.message_from_bytes(raw, policy=email.policy.default)
    except Exception:
        return None

    message_id = (msg.get("Message-ID") or "").strip()
    subject = (msg.get("Subject") or "").strip()
    from_email = _normalize_addrs(msg.get("From"))
    to_emails = _normalize_addrs(msg.get("To"))
    cc_emails = _normalize_addrs(msg.get("Cc"))
    date_ts = _parse_date_to_ts(msg.get("Date"))
    in_reply_to = (msg.get("In-Reply-To") or "").strip()

    # Serialize a small subset of headers to JSON for provenance
    headers = {k: v for (k, v) in msg.items() if k.lower() in {
        "message-id", "subject", "from", "to", "cc", "date", "in-reply-to", "references"
    }}
    headers_json = json.dumps(headers, ensure_ascii=False)

    body, atts, has_atts = _extract_body_and_attachments(msg)

    # Fallback synthetic Message-ID for idempotency when header is missing
    if not message_id:
        base = (subject + "|" + from_email + "|" + to_emails + "|" + str(date_ts)).encode()
        body_hash = hashlib.sha256((body or "").encode()).hexdigest()
        message_id = "synthetic-" + hashlib.sha256(base + body_hash.encode()).hexdigest()

    return ParsedMessage(
        path=path,
        account=account,
        folder=folder,
        message_id=message_id,
        subject=subject,
        from_email=from_email,
        to_emails=to_emails,
        cc_emails=cc_emails,
        date_ts=date_ts,
        in_reply_to=in_reply_to,
        headers_json=headers_json,
        body=body,
        has_attachments=has_atts,
        attachments=atts,
    )


def _walk_maildir(maildir: Path) -> Iterator[Tuple[str, str, Path]]:
    """Yield (folder_name, kind, file_path) for each message file in any cur/ or new/ under maildir.

    Works with nested Maildir structures. Folder name is the directory name that contains cur/new.
    """
    maildir = maildir.expanduser().resolve()
    if not maildir.exists():
        return

    for kind in ("cur", "new"):
        for kpath in maildir.rglob(kind):
            if not kpath.is_dir():
                continue
            folder_dir = kpath.parent
            folder_name = str(folder_dir.relative_to(maildir))
            for fp in kpath.iterdir():
                if fp.is_file():
                    yield (folder_name, kind, fp)


def _ensure_attachment_saved(root: Path, filename: str, content: bytes) -> Tuple[str, Path]:
    sha = hashlib.sha256(content).hexdigest()
    d = root / sha
    d.mkdir(parents=True, exist_ok=True)
    safe_name = filename or "attachment.bin"
    # Avoid path traversal in filename
    safe_name = os.path.basename(safe_name)
    out = d / f"original_{safe_name}"
    if not out.exists():
        out.write_bytes(content)
    return sha, out


def index_maildir(
    maildir: Path,
    db_path: Path,
    attachments_root: Path,
    account: str = "default",
    batch_size: int = 200,
    progress_mode: str | None = None,
    progress_root: Path | None = None,
) -> None:
    con = sqlite3.connect(str(db_path))
    con.execute("PRAGMA journal_mode=WAL;")
    con.execute("PRAGMA foreign_keys=ON;")
    cur = con.cursor()

    count = 0
    inserted = 0
    skipped = 0

    reporter = ProgressReporter(
        ProgressConfig(
            mode=resolve_progress_mode(progress_mode),
            root=progress_root,
            key=f"index-maildir:{account}",
            total=None,
            desc=f"Index {account}",
        )
    )
    for folder, _kind, fp in _walk_maildir(maildir):
        count += 1
        parsed = parse_eml(fp, account=account, folder=folder)
        if not parsed:
            continue

        try:
            # Insert message (idempotent on message_id)
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
                # Still consider adding chunks to FTS if new body? Skip for simplicity.
                if count % batch_size == 0:
                    con.commit()
                continue

            inserted += 1
            # Fetch message id
            cur.execute("SELECT id FROM messages WHERE message_id=?", (parsed.message_id,))
            row = cur.fetchone()
            if row is None:
                continue
            mid = int(row[0])

            # Insert into messages_fts
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

            # Save attachments and insert rows
            attach_names = []
            for (fname, mime, content) in parsed.attachments:
                if not content:
                    continue
                sha, out = _ensure_attachment_saved(attachments_root, fname, content)
                cur.execute(
                    """
                    INSERT INTO attachments (message_id, filename, mime, sha256, bytes, path_original)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (mid, fname, mime, sha, len(content), str(out)),
                )
                attach_names.append(fname or "")
                # Enqueue attachment job (Stage B)
                cur.execute(
                    "INSERT OR IGNORE INTO jobs_attachments(attachment_id, status, tries, updated_ts) VALUES ((SELECT last_insert_rowid()), 'pending', 0, ?)",
                    (int(time.time()),),
                )

            # Chunk body and insert into chunks + chunks_fts
            if parsed.body:
                for ch in chunk_text(parsed.body, target_size=800):
                    cur.execute(
                        "INSERT INTO chunks (message_id, attachment_id, kind, text, token_count) VALUES (?, NULL, 'body', ?, NULL)",
                        (mid, ch),
                    )
                    cid = cur.lastrowid
                    cur.execute(
                        "INSERT INTO chunks_fts (chunk_id, text) VALUES (?, ?)",
                        (cid, ch),
                    )
                    # Enqueue embedding job (Stage C)
                    cur.execute(
                        "INSERT OR IGNORE INTO jobs_embeddings(chunk_id, status, tries, updated_ts) VALUES (?, 'pending', 0, ?)",
                        (cid, int(time.time())),
                    )

            # Insert a meta-chunk per message (field-aware, language-agnostic)
            meta_parts = [
                f"SUBJECT: {parsed.subject}",
                f"FROM: {parsed.from_email}",
                f"TO: {parsed.to_emails}",
                f"CC: {parsed.cc_emails}",
                f"FOLDER: {parsed.folder}",
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
            # Keep indexing despite individual failures
            con.rollback()
            continue
        finally:
            reporter.update(1)

    con.commit()
    con.close()

    print(f"Indexing done. files={count} inserted_messages={inserted} skipped_existing={skipped}")
    reporter.close()
