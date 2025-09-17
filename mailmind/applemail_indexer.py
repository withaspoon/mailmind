from __future__ import annotations

import email
import email.policy
import email.utils
import hashlib
import os
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional, Tuple
import mimetypes

from .progress import ProgressConfig, ProgressReporter, resolve_progress_mode
from .utils.text import chunk_text, html_to_text


@dataclass
class EmlxParsed:
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
            from datetime import timezone

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
        body = html_to_text("\n".join(html_parts))
    return body, atts, has_atts or bool(atts)


def _read_emlx_bytes(path: Path) -> bytes:
    """Read an .emlx file and return the raw RFC822 bytes.

    .emlx begins with an ASCII decimal length and a newline, followed by that many bytes of message.
    Some files may not include the length header; in that case, return the full contents.
    """
    with path.open("rb") as f:
        first = f.readline()
        try:
            n = int(first.strip() or b"0")
        except Exception:
            # Not a length header; include first line
            return first + f.read()
        # Length header present; read exactly n bytes
        data = f.read(n)
        return data


def parse_emlx(path: Path, account: str, folder: str) -> Optional[EmlxParsed]:
    try:
        raw = _read_emlx_bytes(path)
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

    # Compact provenance headers
    headers = {k: v for (k, v) in msg.items() if k.lower() in {"message-id", "subject", "from", "to", "cc", "date", "in-reply-to", "references"}}
    import json

    headers_json = json.dumps(headers, ensure_ascii=False)

    body, atts, has_atts = _extract_body_and_attachments(msg)
    # Append sidecar attachments if present
    side = _collect_sidecar_attachments(path)
    if side:
        # De-duplicate by sha256 of content
        seen_sha: set[str] = set()
        for (_fn, _mt, c) in atts:
            try:
                seen_sha.add(hashlib.sha256(c or b"").hexdigest())
            except Exception:
                pass
        for fn, mt, c in side:
            try:
                sha = hashlib.sha256(c or b"").hexdigest()
            except Exception:
                sha = ""
            if sha and sha in seen_sha:
                continue
            atts.append((fn, mt, c))
        has_atts = has_atts or bool(side)

    # Fallback synthetic Message-ID for idempotency
    if not message_id:
        base = (subject + "|" + from_email + "|" + to_emails + "|" + str(date_ts)).encode()
        body_hash = hashlib.sha256((body or "").encode()).hexdigest()
        message_id = "synthetic-" + hashlib.sha256(base + body_hash.encode()).hexdigest()

    return EmlxParsed(
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


def _collect_sidecar_attachments(emlx_path: Path) -> List[Tuple[str, str, bytes]]:
    """Locate Apple Mail sidecar attachments for a given .emlx.

    Common layouts (vary by macOS version):
      - <Mailbox>.mbox/Messages/<N>.emlx
        <Mailbox>.mbox/Attachments/<N>/* (possibly nested 1/, 2/ per part)
      - <Mailbox>.mbox/Messages/<N>.emlx
        <Mailbox>.mbox/Messages/Attachments/<N>/*
    We scan a few likely candidates and aggregate files.
    """
    attachments: List[Tuple[str, str, bytes]] = []
    try:
        stem = emlx_path.stem  # message numeric id
        mbox_dir = emlx_path.parent
        if mbox_dir.name.lower() == "messages":
            mbox_dir = mbox_dir.parent  # go up to *.mbox
        candidates = [
            mbox_dir / "Attachments" / stem,
            emlx_path.parent / "Attachments" / stem,
        ]
        for base in candidates:
            if not base.exists():
                continue
            for f in base.rglob("*"):
                if f.is_file():
                    try:
                        data = f.read_bytes()
                    except Exception:
                        continue
                    mt = mimetypes.guess_type(f.name)[0] or "application/octet-stream"
                    attachments.append((f.name, mt, data))
    except Exception:
        pass
    return attachments


def _ensure_attachment_saved(root: Path, filename: str, content: bytes) -> Tuple[str, Path]:
    sha = hashlib.sha256(content).hexdigest()
    d = root / sha
    d.mkdir(parents=True, exist_ok=True)
    safe_name = os.path.basename(filename or "attachment.bin")
    out = d / f"original_{safe_name}"
    if not out.exists():
        out.write_bytes(content)
    return sha, out


def _folder_from_emlx(mail_root: Path, fp: Path) -> str:
    # Prefer path relative to mail_root, strip trailing 'Messages'
    try:
        rel = fp.relative_to(mail_root)
    except Exception:
        rel = fp
    parent = rel.parent
    if parent.name.lower() == "messages":
        parent = parent.parent
    return str(parent)


def _walk_emlx(mail_root: Path) -> Iterator[Path]:
    mail_root = mail_root.expanduser().resolve()
    if not mail_root.exists():
        return
    for fp in mail_root.rglob("*.emlx"):
        if fp.is_file():
            yield fp


def _count_emlx(mail_root: Path, folder_filter: Optional[str] = None) -> int:
    try:
        if not folder_filter:
            return sum(1 for _ in mail_root.rglob("*.emlx"))
        filt = folder_filter.lower()
        n = 0
        for fp in mail_root.rglob("*.emlx"):
            if not fp.is_file():
                continue
            folder = _folder_from_emlx(mail_root, fp).lower()
            if filt in folder:
                n += 1
        return n
    except Exception:
        return 0


def index_apple_mail(
    mail_root: Path,
    db_path: Path,
    attachments_root: Path,
    account: str = "applemail",
    batch_size: int = 200,
    progress_mode: str | None = None,
    progress_root: Path | None = None,
    compute_total: bool = True,
    folder_filter: Optional[str] = None,
    rescan_attachments: bool = False,
) -> None:
    con = sqlite3.connect(str(db_path))
    con.execute("PRAGMA journal_mode=WAL;")
    con.execute("PRAGMA foreign_keys=ON;")
    cur = con.cursor()

    processed = 0
    inserted = 0
    skipped = 0
    unchanged = 0
    now = int(time.time())

    total_count = _count_emlx(mail_root, folder_filter=folder_filter) if compute_total else None
    reporter = ProgressReporter(
        ProgressConfig(
            mode=resolve_progress_mode(progress_mode),
            root=progress_root,
            key=f"index-applemail:{account}",
            total=total_count,
            desc=f"Index Apple Mail ({account})",
        )
    )

    for fp in _walk_emlx(mail_root):
        processed += 1
        try:
            st = fp.stat()
        except Exception:
            continue
        inode = int(getattr(st, "st_ino", 0) or 0)
        size = int(getattr(st, "st_size", 0) or 0)
        mtime = int(getattr(st, "st_mtime", 0) or 0)

        # Skip unchanged files using seen_files
        cur.execute("SELECT inode, size, mtime, message_id FROM seen_files WHERE path=?", (str(fp),))
        row = cur.fetchone()
        if row is not None:
            prev_inode, prev_size, prev_mtime, prev_mid = row
            if int(prev_size or 0) == size and int(prev_mtime or 0) == mtime:
                # Optionally rescan sidecar attachments for unchanged messages
                if rescan_attachments and prev_mid:
                    sid = con.execute("SELECT id FROM messages WHERE message_id=?", (prev_mid,)).fetchone()
                    if sid is not None:
                        mid = int(sid[0])
                        # Collect sidecar attachments and insert if new
                        side = _collect_sidecar_attachments(fp)
                        for (fname, mime, content) in side:
                            if not content:
                                continue
                            sha = hashlib.sha256(content).hexdigest()
                            # Skip if attachment with same sha already stored for this message
                            ex = con.execute(
                                "SELECT 1 FROM attachments WHERE message_id=? AND sha256=?",
                                (mid, sha),
                            ).fetchone()
                            if ex:
                                continue
                            sha_calc, out_path = _ensure_attachment_saved(attachments_root, fname, content)
                            cur.execute(
                                "INSERT INTO attachments (message_id, filename, mime, sha256, bytes, path_original) VALUES (?, ?, ?, ?, ?, ?)",
                                (mid, fname, mime, sha_calc, len(content), str(out_path)),
                            )
                            cur.execute(
                                "INSERT OR IGNORE INTO jobs_attachments(attachment_id, status, tries, updated_ts) VALUES ((SELECT last_insert_rowid()), 'pending', 0, ?)",
                                (int(time.time()),),
                            )
                        con.commit()
                unchanged += 1
                if processed % batch_size == 0:
                    con.commit()
                reporter.update(1)
                continue

        folder = _folder_from_emlx(mail_root, fp)
        if folder_filter and folder_filter.lower() not in folder.lower():
            # Skip files outside the requested folder filter
            reporter.update(1)
            continue
        parsed = parse_emlx(fp, account=account, folder=folder)
        if not parsed:
            reporter.update(1)
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
            else:
                inserted += 1
                # Fetch message id
                cur.execute("SELECT id FROM messages WHERE message_id=?", (parsed.message_id,))
                row = cur.fetchone()
                if row is None:
                    reporter.update(1)
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

                # Save attachments from MIME
                attach_names: List[str] = []
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

                # Chunk body as chunks + enqueue embeddings (Stage C)
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

                # Meta-chunk per message
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

            # Upsert seen_files
            cur.execute(
                """
                INSERT INTO seen_files(path, inode, size, mtime, message_id, last_seen_ts)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(path) DO UPDATE SET inode=excluded.inode, size=excluded.size, mtime=excluded.mtime, message_id=excluded.message_id, last_seen_ts=excluded.last_seen_ts
                """,
                (str(fp), inode, size, mtime, parsed.message_id, now),
            )

            if processed % batch_size == 0:
                con.commit()
        except sqlite3.Error:
            con.rollback()
        finally:
            reporter.update(1)

    con.commit()
    con.close()
    print(
        f"Apple Mail indexing done. files={processed} inserted_messages={inserted} skipped_existing={skipped} unchanged={unchanged}"
    )
    reporter.close()
