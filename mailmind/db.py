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

-- Mapping of embedded chunks to model/dim; vector data stored externally (e.g., hnswlib file)
CREATE TABLE IF NOT EXISTS chunk_vectors (
  chunk_id INTEGER PRIMARY KEY REFERENCES chunks(id) ON DELETE CASCADE,
  model TEXT,
  dim INTEGER
);

-- Incremental indexing support
CREATE TABLE IF NOT EXISTS seen_files (
  path TEXT PRIMARY KEY,
  inode INTEGER,
  size INTEGER,
  mtime INTEGER,
  message_id TEXT,
  last_seen_ts INTEGER
);

CREATE TABLE IF NOT EXISTS jobs_attachments (
  id INTEGER PRIMARY KEY,
  attachment_id INTEGER UNIQUE REFERENCES attachments(id) ON DELETE CASCADE,
  status TEXT NOT NULL DEFAULT 'pending', -- pending|processing|done|error
  last_error TEXT,
  tries INTEGER NOT NULL DEFAULT 0,
  updated_ts INTEGER
);
CREATE INDEX IF NOT EXISTS jobs_attachments_status_idx ON jobs_attachments(status);

CREATE TABLE IF NOT EXISTS jobs_embeddings (
  id INTEGER PRIMARY KEY,
  chunk_id INTEGER UNIQUE REFERENCES chunks(id) ON DELETE CASCADE,
  status TEXT NOT NULL DEFAULT 'pending', -- pending|processing|done|error
  last_error TEXT,
  tries INTEGER NOT NULL DEFAULT 0,
  updated_ts INTEGER
);
CREATE INDEX IF NOT EXISTS jobs_embeddings_status_idx ON jobs_embeddings(status);

-- FTS5 virtual tables (contentless; we insert directly)
CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
  message_id, subject, from_email, to_emails, cc_emails, body,
  tokenize = "unicode61 remove_diacritics 2 tokenchars '._-@'"
);

CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
  chunk_id, text,
  tokenize = "unicode61 remove_diacritics 2 tokenchars '._-@'"
);
"""


def init_db(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(db_path))
    try:
        try:
            con.executescript(SCHEMA)
        except sqlite3.OperationalError as e:
            msg = str(e).lower()
            if "tokenize" in msg and "parse error" in msg:
                # Fallback: create FTS without custom tokenizer options
                fallback = SCHEMA.replace(
                    "tokenize = \"unicode61 remove_diacritics 2 tokenchars '._-@'\"",
                    "tokenize = 'unicode61'",
                )
                con.executescript(fallback)
            else:
                raise
        con.commit()
    finally:
        con.close()
