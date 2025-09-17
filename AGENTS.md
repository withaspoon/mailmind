Mailmind — Agents & Contributors Guide

Scope: This file applies to the entire repo. Follow these rules for any code, docs, or scripts you add or modify.

Project goal (one-liner)
- Fully local, multi‑account email search and summarization tool that ingests Gmail/Apple Mail/Proton/etc. into a single local index, supports natural‑language queries, extraction, and summarization, and runs offline on macOS (M‑series) and Linux.

Non‑goals
- Sending or drafting email. No cloud calls. No server dependency beyond optional local services (Ollama, Proton Bridge).

System constraints & invariants
- Offline by default. Do not add network calls in code paths used by indexing/search. If optional downloads are supported (models), they must be clearly optional, and cache to `models/`.
- Deterministic, idempotent indexing. Re‑runs should not duplicate data. Use SHA‑256 content hashes and Message‑ID de‑dupe.
- Single local source of truth. A SQLite database (metadata + FTS) plus a content‑addressed attachments store.
- Hybrid retrieval: lexical (SQLite FTS5) + semantic (local embeddings) with optional reranking.
- Security: Never log entire email bodies or PII by default. Logs should be concise and redact addresses.

Target platform & performance
- Primary: macOS 14+ on Apple Silicon (M3/M4). Secondary: Linux x86_64.
- Must run fast on a laptop. Prefer CPU‑friendly models and quantization. Use batch processing and caching for embeddings and OCR.

Recommended implementation stack
- Language: Python 3.11+ for rapid iteration. Keep interfaces clean to allow later migration of hot paths to Rust if needed.
- Database: SQLite 3.44+ with FTS5. Vector store options (in this order):
  1) hnswlib (.bin index next to DB; lowest friction, portable)
  2) FAISS (optional; heavier dep) or sqlite-vss (optional; if available in environment)
- Embeddings (default): google/embeddinggemma-300m at 768d; serve with MRL truncation to 256d by default for space/speed. Alternatives: intfloat/e5-small, BAAI/bge-small.
- Reranker (optional): BAAI/bge-reranker-base (int8/onnx where possible) for precision on top‑K.
- Generator LLM (local): Llama‑3.1‑8B‑Instruct (Q4_K_M) via Ollama or llama.cpp for planning and summarization.
- OCR & parsing: ocrmypdf+tesseract, pdfplumber/camelot, BeautifulSoup/readability-lxml.

Repo structure (expected)
- mailmind/
  - cli/ (Typer/Click commands)
  - config/ (schema, loaders)
  - ingest/ (maildir/imap/mbox/apple mail adapters)
  - parsing/ (body, HTML clean, attachment text, OCR)
  - db/ (schema, migrations, FTS setup)
  - vector/ (embeddings interface, hnswlib/faiss/sqlite-vss adapters)
  - search/ (hybrid retrieval, filters, rerank)
  - llm/ (planner, summarizer, model backends: ollama/llama.cpp)
  - tools/ (table extraction, NER, aggregation)
  - utils/ (hashing, threading, chunking, time, logging)
- data/
  - db.sqlite3 (default path; configurable)
  - vectors/ (ANN indexes)
  - attachments/ (content addressed by sha256)
- models/ (local model weights/cache; not checked in)
- scripts/ (mbsync templates, maintenance)

Coding conventions
- Python: type‑annotated, pass mypy at “basic” level. Keep functions small and pure where possible.
- Style: black, ruff (when tooling is added). Avoid one‑letter variables. Do not over‑abstract early.
- Errors: Fail fast on corruption; be tolerant on individual message parse errors (log and continue). Always include provenance in exceptions when possible (account/folder/filepath).
- Logging: Use structured logging. Default level INFO. Avoid logging full message content.

Data model (high‑level)
- messages(id, account, folder, message_id, subject, from_email, to_emails, cc_emails, date_ts, in_reply_to, thread_id, has_attachments, headers_json)
- attachments(id, message_id, filename, mime, sha256, bytes, path_original, path_text)
- chunks(id, message_id, attachment_id, kind{body|attachment|header}, text, token_count)
- embeddings(id, chunk_id, model, dim, vec)  [ANN index stored separately when hnswlib/faiss]
- tags(message_id, tag)
- entities(id, chunk_id, type, value, canonical, score)
- FTS5 virtual tables for messages/chunks text search

Planner & DSL (must‑have behaviors)
- LLM produces a JSON plan with: filters (date ranges, participants, has:pdf, tags), search terms (lexical), semantic query, ops (collect_attachments, extract_tables, aggregate, summarize), and top_k. Execution is deterministic and auditable with citations back to message/attachment ids.

Performance & robustness rules
- Chunk bodies/attachments into ~512–1024 token spans. Cache embeddings by sha256(text).
- Batch embedding where possible. Normalize embeddings to unit vectors.
- Keep default embedding dim at 256 via MRL truncation to balance size and quality.
- Use incremental indexing keyed by file mtime and content hash.

Security & privacy
- Run entirely local. Never ship data off‑device. If a future UI is added, bind to localhost only.
- Provide config for encrypting DB and attachments at rest (use OS disk encryption by default; defer app‑level encryption unless required).

Testing philosophy
- Red–green testing as the default workflow: write a failing test first (red), implement minimal code to pass (green), then refactor safely.
- Aim for high coverage (target ≥85% lines, ≥80% branches) on core modules: parsing, indexing, chunking, DB I/O, and search.
- Unit tests for parsers, chunking, hashing, and DB I/O. Golden‑file tests for email parsers with synthetic fixtures (no real mail). Use `sample.mbox` as a baseline fixture for integration tests.
- Integration tests should index a small Maildir and the provided `sample.mbox`, then run representative searches end‑to‑end, asserting deterministic results and citations.
- No network during tests. All tests must be fully offline and reproducible.

Dependencies policy
- Keep core deps minimal and portable. Heavy/optional deps (FAISS, spaCy large pipelines) must be behind optional extras.
- Provide fast‑path installation on macOS with Homebrew + pip wheels only.

Agent workflow (for AI/code assistants)
- Make small, focused patches; keep changes localized. Update docs when behavior changes.
- Respect offline constraint. Do not add telemetry.
- Prefer hnswlib vector store unless explicitly asked otherwise.
- If adding new commands/flags, update CLI help and README examples.
- When adding new files, check for AGENTS.md in subdirs; deeper scopes override this file.

Operational checklists
- Indexer must handle: missing headers, weird encodings, multipart/alternative, nested attachments, zero‑byte parts, HTML‑only bodies.
- De‑dup messages by normalized Message‑ID and by (From, Date, Subject, sha256(body)) fallback.
- OCR: run only when PDF text extraction yields little/no text; cache results.

Licensing notes
- Gemma models are under Google’s Gemma license; local, private use is OK. Do not redistribute weights via this repo.
