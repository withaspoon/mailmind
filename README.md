Mailmind — Local Multi‑Account Email Search & Summarization

Overview
- Mailmind ingests all your email accounts (Gmail, Apple Mail, Proton via Bridge, IMAP, mbox exports) into a single local index. It supports natural‑language queries, hybrid search (lexical + semantic), attachment extraction (including PDFs with OCR), and summarization/aggregation — all offline on your Mac (M‑series) or Linux laptop.
- No sending or drafting email. Mailmind is a private, on‑device research assistant for your mail.

Key features
- Multi‑account ingest into one local Maildir/DB
- Hybrid retrieval: SQLite FTS5 + local embeddings (EmbeddingGemma or e5/bge)
- Optional reranking for higher precision
- Attachment processing: PDFs/Office → text; OCR for scanned PDFs/images
- Natural‑language planner for complex tasks (extract, aggregate, summarize)
- Provenance & citations to original messages and attachments
- Runs fully offline; configurable, portable, minimal deps

Architecture
- Ingest → Normalize → Index → Store → Query → Summarize
  - Ingest: mbsync/isync or offlineimap into Maildir; Proton Bridge exposes IMAP locally; Apple Mail via its local store or one‑time export
  - Normalize: parse RFC‑822, build threads, extract attachments into content‑addressed store
  - Index: FTS5 for keyword; embeddings + ANN for semantic
  - Store: SQLite DB (metadata/FTS) + ANN index files (hnswlib by default)
  - Query: NL → plan (JSON) → hybrid search → rerank → tools (tables/NER/aggregation)
  - Summarize: local LLM produces answers with citations and optional exports

What’s included (docs)
- AGENTS.md: repo rules/spec for contributors and agents
- README.md: this guide
- TODO.md: milestone roadmap with acceptance checks

System requirements
- macOS 14+ on Apple Silicon (M3/M4 recommended) or Linux x86_64
- Python 3.11+ (primary) — later we may add Rust modules for hot paths
- Disk space: depends on mailbox size; embeddings at 256‑dim ≈ 1.0 GB per 1M chunks

Recommended local tooling (macOS)
- Homebrew: `brew install isync ocrmypdf tesseract poppler ghostscript pkg-config`
- Proton Bridge (for Proton accounts): install official app
- Ollama (for LLMs): `brew install ollama` then `ollama run llama3.1:8b-instruct-q4_K_M`

Models (local)
- Embeddings (default): google/embeddinggemma-300m (768d); serve with MRL truncation to 256d by default to cut storage.
  - Place weights in `models/embeddinggemma/` or ensure they are available offline via HF cache.
  - Alternatives: `intfloat/e5-small`, `BAAI/bge-small-en-v1.5`.
- Reranker (optional): `BAAI/bge-reranker-base` (quantized or ONNX). Skip if you want minimal deps.
- Generator LLM: for planning/summarization. Suggested: Llama‑3.1‑8B‑Instruct (Q4_K_M) via Ollama, which uses Metal on M‑series.

Data layout (default; configurable)
- `data/db.sqlite3` — SQLite metadata DB + FTS virtual tables
- `data/vectors/` — ANN index files (e.g., `mailmind_hnsw.bin`)
- `data/attachments/<sha256>/original.ext` — content‑addressed attachments
- `models/` — local model weights/cache (not checked in)

Install (developer preview)
1) Create and activate a Python 3.11 virtualenv
   - macOS: `python3 -m venv .venv && source .venv/bin/activate`
2) Install core Python deps (to be defined once code is added)
   - Minimal set will include: click/typer, beautifulsoup4, lxml, pdfplumber, ocrmypdf, tesseract bindings, sentence-transformers, hnswlib, sqlite3 (builtin), watchdog
3) Install system tools (OCR)
   - `brew install ocrmypdf tesseract poppler`
4) (Optional) Install Ollama for the generator LLM
   - `brew install ollama && ollama pull llama3.1:8b-instruct-q4_K_M`

Configure
Create a config file `mailmind.yaml` in the repo root or `~/.config/mailmind/config.yaml`:

  root: ./data
  db_path: ./data/db.sqlite3
  vectors:
    backend: hnswlib   # options: hnswlib | faiss | sqlite-vss
    dim: 256           # 256 by default via MRL truncation
    path: ./data/vectors/mailmind_hnsw.bin
  models:
    embedding:
      name: google/embeddinggemma-300m
      truncate_dim: 256
      local_path: ./models/embeddinggemma
    generator:
      backend: ollama
      name: llama3.1:8b-instruct-q4_K_M
      params: { temperature: 0.2, top_p: 0.95 }
    reranker:
      name: BAAI/bge-reranker-base
      enabled: false
  ingest:
    accounts:
      - name: personal-gmail
        type: imap
        method: mbsync
        maildir: ~/Mail/personal-gmail
        mbsync_config: ~/.mbsyncrc
      - name: proton
        type: imap
        method: proton-bridge
        maildir: ~/Mail/proton
      - name: apple-mail
        type: apple_mail_dir
        path: ~/Library/Mail
  ocr:
    languages: [eng, swe, spa]
    run_when_text_ratio_below: 0.6
  index:
    chunk_tokens: 800
    batch_size: 128
    normalize_embeddings: true

IMAP sync (Maildir)
- Use mbsync/isync to mirror each account locally. Example `~/.mbsyncrc` snippet (edit per account):

  IMAPAccount personal-gmail
  Host imap.gmail.com
  SSLType IMAPS
  User your.name@gmail.com
  PassCmd "security find-internet-password -a your.name@gmail.com -s imap.gmail.com -g 2>&1 | awk -F\"\" '/password/ { print $2 }'"

  IMAPStore personal-remote
  Account personal-gmail

  MaildirStore personal-local
  Path ~/Mail/personal-gmail/
  Inbox ~/Mail/personal-gmail/INBOX

  Channel personal
  Master :personal-remote:
  Slave :personal-local:
  Patterns *
  Create Slave
  SyncState *

Proton Bridge
- Install Proton Bridge and configure the Proton account; then sync with mbsync pointing at the bridge’s local IMAP service.

Apple Mail
- Option A: Index directly from `~/Library/Mail` (read‑only). Option B: Export mailboxes to a Maildir/mbox and index the exported copy. Option A is simplest and keeps things incremental.

CLI (planned commands)
- `mailmind init --root ./data` — create folders, DB, FTS tables
- `mailmind sync` — run mbsync for configured accounts (optional helper)
- `mailmind index --incremental` — parse new messages, attachments, OCR if needed, embed, build ANN
- `mailmind search "board meeting pdf totals since 2000"` — hybrid query with rerank and citations
- `mailmind summarize --query "board meeting totals since 2000" --export summary.md` — plan + execute tools + summarization
- `mailmind extract --attachments --type pdf --query "board meeting" --out ./exports/` — copy attachment originals and text to out
- `mailmind whois --query "medical company contacted in Spain"` — NER + filters over Sent mail

Planner DSL (example)
{
  "filters": {
    "date_from": "2000-01-01",
    "tags_any": ["board_meeting"],
    "has_attachment": true,
    "attachment_types": ["pdf"]
  },
  "search": {
    "lexical": ["minutes", "financials", "P&L", "balance"],
    "semantic_query": "board meeting financial summaries totals since 2000"
  },
  "ops": [
    {"op": "collect_attachments"},
    {"op": "extract_tables"},
    {"op": "aggregate", "group_by": ["year"], "metrics": [{"sum": "TotalRevenue"}, {"sum": "EBIT"}]},
    {"op": "summarize"}
  ],
  "top_k": 200
}

Performance tips (M‑series Mac)
- Keep embedding dim at 256 via MRL (EmbeddingGemma). This cuts vector size by ~3×.
- Batch embed with size 128–256, normalize embeddings.
- Use OCR only when extracted text is sparse; cache OCR results.
- Prefer hnswlib vector store for portability; persist the ANN index to `data/vectors`.
- Keep reranker disabled unless precision really needs it.

Privacy & security
- All processing is local. No data leaves your machine. Prefer full‑disk encryption.
- Redact email addresses/domains in logs. Do not include bodies in logs.

Troubleshooting
- Slow indexing: check OCR settings; reduce OCR languages; verify batch sizes.
- Large DB: reduce embedding dim to 128 (MRL) or raise chunk size.
- Missing PDFs text: ensure `ocrmypdf` and `tesseract` are installed; confirm language packs.
- Planner quality: try a larger generator LLM or enable reranker for more precise top‑K.

Roadmap snapshot (see TODO.md)
- M0–M3: ingest, attachments, embeddings, hybrid search
- M4–M6: planner + tools, reranking, quality eval
- M7+: finetune embeddings on your mail, TUI

Licensing notes
- Gemma models are under the Gemma license; local private use is fine. You are responsible for complying with model licenses.

