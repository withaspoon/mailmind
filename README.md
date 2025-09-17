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

Incremental Indexing (20 GB+ mailboxes)
- Strategy: detect deltas and process in staged pipelines to stay responsive with hundreds of new emails/day.
  - Message dedupe keys: `Message-ID` if present; fallback to synthetic ID (From|To|Date|Subject|sha256(body)).
  - Attachment dedupe key: `sha256(content)` across accounts/folders.
- Stages:
  - Stage A (foreground, fast): parse headers/body, insert into `messages` + `messages_fts`, register `attachments` metadata. Runs on every sync.
  - Stage B (background): extract attachment text (PDF → text; OCR fallback); write `attachments.path_text`; chunk into `chunks/chunks_fts`.
  - Stage C (background): embed new chunks and update ANN (hnswlib) incrementally.
- Scheduling:
  - Run Stage A on every mbsync or fs event.
  - Stage B/C are background workers with small job queues and backpressure; pause during quiet hours or on battery.
- Idempotency & robustness:
  - All inserts are safe with `INSERT OR IGNORE` on unique keys. OCR is cached per `sha256` in `attachments/<sha>/text.txt`.
  - Transactions commit in small batches; partial failures are retried via job tables.

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
- Embeddings (default): `intfloat/multilingual-e5-small` (384‑dim). Good CPU speed and multilingual.
  - Alternatives: `intfloat/e5-small-v2` (English), `BAAI/bge-small-en-v1.5`.
  - If you want to experiment with EmbeddingGemma, pass `--model` explicitly, but note that model names and availability may vary on HF.
- Reranker (optional): `BAAI/bge-reranker-base` (quantized or ONNX). Skip if you want minimal deps.
- Generator LLM: for planning/summarization. Suggested: Llama‑3.1‑8B‑Instruct (Q4_K_M) via Ollama, which uses Metal on M‑series.

Environment variables
- `MAILMIND_LLM_BACKEND` (default: `ollama`) — set to `ollama` or leave unset. If Ollama is not installed, the summarizer falls back to a simple extractive mode.
- `MAILMIND_LLM_MODEL` (default: `llama3.1:8b-instruct-q4_K_M`) — model tag for Ollama.

Data layout (default; configurable)
- `data/db.sqlite3` — SQLite metadata DB + FTS virtual tables
- `data/vectors/` — ANN index files (e.g., `mailmind_hnsw.bin`)
- `data/attachments/<sha256>/original.ext` — content‑addressed attachments
- `models/` — local model weights/cache (not checked in)

Configuration (incremental + workers)
- In `mailmind.yaml` (example):

  root: ./data
  db_path: ./data/db.sqlite3
  vectors:
    backend: hnswlib
    dim: 256
    path: ./data/vectors/mailmind_hnsw.bin
  ocr:
    languages: [eng, swe, spa]
    min_chars_per_page: 120
  workers:
    attachments:
      concurrency: 2         # max concurrent OCR jobs
      quiet_hours: ["22:00", "07:00"]
      on_battery: pause      # pause|run
    embeddings:
      batch_size: 256
      concurrency: 1
  ingest:
    accounts:
      - name: personal-gmail
        maildir: ~/Mail/personal-gmail
      - name: proton
        maildir: ~/Mail/proton

Install (developer preview)
1) Create and activate a Python 3.11 virtualenv
   - Any shell (no activation):
     - `python3 -m venv .venv`
     - `.venv/bin/python -m pip install -e .`
     - Run commands via the venv binaries, e.g. `.venv/bin/mailmind --help`
   - If you prefer activation:
     - bash/zsh: `python3 -m venv .venv && source .venv/bin/activate`
     - fish: `python3 -m venv .venv && source .venv/bin/activate.fish`
     - tcsh/csh: `python3 -m venv .venv && source .venv/bin/activate.csh`
2) Install core Python deps (to be defined once code is added)
   - Minimal set will include: click/typer, beautifulsoup4, lxml, pdfplumber, ocrmypdf, tesseract bindings, sentence-transformers, hnswlib, sqlite3 (builtin), watchdog
3) Install system tools (OCR)
   - `brew install ocrmypdf tesseract poppler`
4) (Optional) Install Ollama for the generator LLM
   - `brew install ollama && ollama pull llama3.1:8b-instruct-q4_K_M`

Progress display and GUI plumbing
- All long-running commands support `--progress` with modes: `auto` (default), `bar`, `json`, `none`.
- JSON progress events are written to stderr as single-line objects when `--progress json` or when not running in a TTY.
- A status file is maintained at `data/state/progress.json` with the last known progress for each task key (e.g., `index-mbox:<name>`, `embed`, `worker-attachments`). This is intended for a future menu bar UI.
- You can also set `MAILMIND_PROGRESS` env var to override the mode globally (auto|bar|json|none).

Optional: Enable ANN and embeddings (recommended)
- Ensure Apple developer tools: `xcode-select --install`
- Upgrade build tools in your venv:
  - `.venv/bin/python -m pip install -U pip setuptools wheel`
- Install ANN + embedding packages (optional but recommended):
  - `.venv/bin/python -m pip install hnswlib sentence-transformers`
- For Transformers backend (e.g., EmbeddingGemma via HF):
  - `.venv/bin/python -m pip install transformers torch --upgrade`
  - On Apple Silicon, PyTorch will use Metal (MPS) automatically when available.
- Build or update the vector index:
  - `mailmind embed --dim 256`
  - You should see `data/vectors/mailmind_hnsw.bin` (and a `.labels.txt`).
- Use hybrid search:
  - `mailmind search-hybrid "invoice OR meeting" --limit 10`

Notes:
- If `hnswlib` is not installed, `embed` still computes embeddings (using a deterministic fallback if `sentence-transformers` isn’t installed) but will not build an ANN index. Hybrid search will gracefully fall back to FTS results.
- Installing `sentence-transformers` will pull PyTorch and download model weights on first use (cached locally for offline use thereafter).

Re-embed with a different model
- To switch embedding models, rebuild the mapping and index:
  - `mailmind embed --rebuild --model intfloat/multilingual-e5-small --dim 384`
  - Then run hybrid search again.

Use EmbeddingGemma directly in Python (Transformers backend)
- If you have the model available in your local HF cache or on disk:
  - `mailmind embed --rebuild --backend transformers --model <hf_id_or_local_path> --dim 256`
- Examples (model IDs may vary; ensure you have access and have downloaded them):
  - `mailmind embed --rebuild --backend transformers --model google/embeddinggemma-300m --dim 256`
- Notes:
  - The Transformers backend will auto-detect GPU/Metal (MPS) and run there when available.
  - If the model isn’t found under that name, point `--model` to a local folder where the model is stored.

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
- `mailmind index-mbox --mbox sample.mbox` — index an mbox file (added for development/testing)
- `mailmind search "board meeting pdf totals since 2000"` — lexical FTS search
- `mailmind search-hybrid "board meeting totals since 2000"` — hybrid (FTS ∪ ANN) if vectors available
- `mailmind summarize --query "board meeting totals since 2000"` — uses local LLM (Ollama) or a deterministic fallback
- `mailmind process-attachments --langs eng+spa+swe` — extract text from PDFs/images (OCR fallback) and index
- `mailmind embed --rebuild --model intfloat/multilingual-e5-small --dim 384` — recompute embeddings and rebuild ANN
- `mailmind worker attachments --max 50` — process up to 50 pending attachment OCR jobs
- `mailmind worker embeddings --max 200` — process up to 200 pending embedding jobs
- `mailmind extract-tables --query "board meeting" --since 2000 --out out/rows.csv --metrics revenue,ebit` — extract numeric cells from tables in PDFs matching the query; aggregations saved to `rows.agg.csv`
  - Add `--progress bar` to show a live progress bar, or `--progress json` to emit progress events (for GUI integration).
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

Throughput & sizing expectations
- Stage A (parse + FTS bodies): 3k–10k msgs/min with batch commits (200–1,000), WAL on.
- Stage B (PDF text): 20–80 pages/sec with pdfplumber/pdftotext; OCR: 0.5–3 pages/sec/core (tesseract/ocrmypdf). Cache per `sha256`.
- Stage C (embeddings): e5‑small CPU 2k–10k chunks/min; MPS (Metal) improves further. 256‑dim float32 ≈ 1.0 GB per 1M chunks.
- Vector index (hnswlib): on‑disk size ≈1.2–1.5× vector payload; incremental adds are fast.

Privacy & security
- All processing is local. No data leaves your machine. Prefer full‑disk encryption.
- Redact email addresses/domains in logs. Do not include bodies in logs.

Troubleshooting
- Slow indexing: check OCR settings; reduce OCR languages; verify batch sizes.
- Large DB: reduce embedding dim to 128 (MRL) or raise chunk size.
- Missing PDFs text: ensure `ocrmypdf` and `tesseract` are installed; confirm language packs.
- Planner quality: try a larger generator LLM or enable reranker for more precise top‑K.
- `summarize` requires a local LLM. By default it tries Ollama (`ollama run llama3.1:8b-instruct-q4_K_M`). If Ollama is not available, it falls back to a simple extractive summary.
- ANN not built: install `hnswlib` in your venv and re-run `mailmind embed --dim 256`.
- Hybrid model mismatch: `search-hybrid` auto-detects the model/dim used during `embed` from the database. You can override with `--model`/`--dim` if needed, but for best results keep the same model for indexing and querying.
- OCR not running: install `ocrmypdf`, `tesseract`, and `poppler` (`pdftotext`) via Homebrew; the pipeline first tries native PDF text, then runs OCR only when needed.
- Table extraction empty: ensure `pdfplumber` is installed; results depend on table layout. You can also install `camelot` if needed (optional, not used by default).

Operations & scheduling
- Sync frequency: run `mbsync` every 5–10 minutes, or use FS watchers to trigger Stage A.
- Background workers: run `process-attachments` and `embed` periodically (launchd/systemd timers) or as long‑running services that drain queues.
- DB maintenance: `ANALYZE` monthly; `VACUUM` occasionally when idle and disk is tight.
- Backups: snapshot `data/db.sqlite3`, `data/vectors/*.bin`, and `data/attachments/`.

Roadmap snapshot (see TODO.md)
- M0–M3: ingest, attachments, embeddings, hybrid search
- M4–M6: planner + tools, reranking, quality eval
- M7+: finetune embeddings on your mail, TUI

Licensing notes
- Gemma models are under the Gemma license; local private use is fine. You are responsible for complying with model licenses.
