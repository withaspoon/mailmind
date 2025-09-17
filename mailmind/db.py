from __future__ import annotations

import sqlite3
from pathlib import Path


SCHEMA = r"""
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS messages (
  id INTEGER PRIMARY KEY,
  account TEXT,
  folder TEXT,
  message_id TEXT UNIQUE,
  subject TEXT,
  from_email TEXT,
  to_emails TEXT,
  cc_emails TEXT,
  date_ts INTEGER,
  in_reply_to TEXT,
  thread_id TEXT,
  has_attachments INTEGER DEFAULT 0,
  headers_json TEXT
);

CREATE TABLE IF NOT EXISTS attachments (
  id INTEGER PRIMARY KEY,
  message_id INTEGER REFERENCES messages(id) ON DELETE CASCADE,
  filename TEXT,
  mime TEXT,
  sha256 TEXT,
  bytes INTEGER,
  path_original TEXT,
  path_text TEXT
);

CREATE TABLE IF NOT EXISTS chunks (
  id INTEGER PRIMARY KEY,
  message_id INTEGER REFERENCES messages(id) ON DELETE CASCADE,
  attachment_id INTEGER REFERENCES attachments(id) ON DELETE CASCADE,
  kind TEXT, -- body|attachment|header
  text TEXT,
  token_count INTEGER
);

-- FTS5 virtual tables (contentless; we insert directly)
CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
  message_id, subject, from_email, to_emails, cc_emails, body,
  tokenize = 'unicode61 remove_diacritics 2 tokenchars "._-@"'
);

CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
  chunk_id, text,
  tokenize = 'unicode61 remove_diacritics 2 tokenchars "._-@"'
);
"""


def init_db(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(db_path))
    try:
        con.executescript(SCHEMA)
        con.commit()
    finally:
        con.close()

