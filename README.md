# Mailmind — Local Multi‑Account Email Search & Summarization

Private, offline search and summarization across all your email accounts. Mailmind ingests Gmail/Apple Mail/Proton/IMAP/mbox into a single local index, supports natural‑language queries, hybrid retrieval (FTS5 + embeddings), PDF/OCR processing, and optional summarization — all on your laptop.

## Contents
- Features
- How It Works
- Requirements
- Install
- Quickstart
- Progress & Status
- Configuration
- Data Layout
- CLI Overview
- Performance
- Troubleshooting
- Roadmap
- License

## Features
- Multi‑account ingest into one local database (Maildir/IMAP/mbox)
- Hybrid retrieval: SQLite FTS5 + local embeddings (hnswlib index)
- Attachment processing: PDFs/Office → text, OCR fallback for scans
- Natural‑language planner tools (extract tables, aggregate, summarize)
- Deterministic, idempotent indexing with content hashes and job queues
- Runs fully offline; no cloud calls or telemetry

## How It Works
- Stage A — Parse & FTS: parse messages, index headers/body into FTS; register attachments
- Stage B — Attachments: extract text from PDFs/images (OCR on demand), chunk and index
- Stage C — Embeddings: embed new chunks and update ANN incrementally
- Hybrid search — Fuse FTS and ANN results; deterministic, with citations back to message/attachment IDs

Field‑aware meta‑chunks: Each message also gets a compact “meta” chunk that includes subject, participants (from/to/cc), folder, attachment filenames, and a short body preview. This enables fuzzy matching across any field through vectors without hard‑coded vocab or language lists.

Incremental indexing is idempotent by design: re‑runs do not duplicate rows. OCR is cached per attachment `sha256`, embedding vectors are cached by chunk content and model/dim.

## Requirements
- macOS 14+ on Apple Silicon (M‑series) or Linux x86_64
- Python 3.11+
- Disk space: varies with mailbox size; 256‑dim embeddings ≈ ~1.0 GB per 1M chunks

## Install
1) Create a virtual environment

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip wheel setuptools
```

2) Install Mailmind (editable for development)

```bash
pip install -e .
```

3) Optional: install recommended extras
- Vector index: `pip install hnswlib`
- Embeddings (Sentence‑Transformers backend): `pip install sentence-transformers`
- Embeddings (Transformers backend): `pip install transformers torch --upgrade`
- PDF tables: `pip install pdfplumber`
- OCR pipeline: install system deps first, then `pip install ocrmypdf`
- Natural‑language dates (multilingual): `pip install dateparser`
 - Fast constraint extractor (LLM via Ollama): install Ollama and pull a small model, e.g. `ollama pull gemma3:4b`. Configure model with `MAILMIND_FAST_LLM_MODEL` (default: `gemma3:4b`).

macOS (Homebrew)

```bash
brew install isync ocrmypdf tesseract poppler ghostscript pkg-config
```

Linux (Debian/Ubuntu)

```bash
sudo apt-get update
sudo apt-get install -y isync ocrmypdf tesseract-ocr poppler-utils ghostscript pkg-config
```

## Quickstart
Initialize a workspace:

```bash
mailmind init --root ./data
```

Index a sample mbox (included in this repo) for a demo account:

```bash
mailmind index-mbox --mbox sample.mbox --account demo --progress bar
```

Run background workers once to process attachments and embeddings:

```bash
mailmind worker attachments --max 100 --langs eng+spa+swe --progress bar
mailmind worker embeddings --max 500 --progress bar
```

Search (lexical only or hybrid):

```bash
mailmind search "board meeting totals since 2000"
mailmind search-hybrid "board meeting totals since 2000"
```

Extract numeric cells from PDF tables that match a query (CSV + optional yearly aggregation):

```bash
mailmind extract-tables --query "board meeting" --since 2000 --out out/rows.csv --metrics revenue,ebit
```

Tip: A simple end‑to‑end smoke run is provided in `./test.sh`.

## Progress & Status
- All long‑running commands accept `--progress` with modes: `auto` (default), `bar`, `json`, `none`
- JSON mode writes single‑line progress events to stderr (for GUI integration)
- A status file is maintained at `data/state/progress.json` with the last known progress per task key (e.g., `index-mbox:<name>`, `embed`, `worker-attachments`)
- You can also set `MAILMIND_PROGRESS=bar|json|none` to override globally

## Configuration
Create `mailmind.yaml` in the repo root or `~/.config/mailmind/config.yaml`.

Minimal example:

```yaml
root: ./data
db_path: ./data/db.sqlite3
vectors:
  backend: hnswlib
  dim: 256
  path: ./data/vectors/mailmind_hnsw.bin
ocr:
  languages: [eng, swe]
  min_chars_per_page: 120
workers:
  attachments: { concurrency: 2 }
  embeddings: { batch_size: 256 }
ingest:
  accounts:
    - name: personal-gmail
      maildir: ~/Mail/personal-gmail
    - name: proton
      maildir: ~/Mail/proton
```

IMAP (mbsync/isync) example (`~/.mbsyncrc` — edit per account):

```text
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
```

Proton Bridge: install Proton Bridge and point mbsync at the bridge’s local IMAP service.

Apple Mail: either index directly from `~/Library/Mail` (read‑only) or export to Maildir/mbox and index the exported copy.

## Data Layout
- `data/db.sqlite3` — SQLite metadata DB + FTS virtual tables
- `data/vectors/` — ANN index files (e.g., `mailmind_hnsw.bin`)
- `data/attachments/<sha256>/original.ext` — content‑addressed originals + cached text
- `models/` — local model weights/cache (not checked in)

## CLI Overview
- `mailmind init --root ./data` — create folders, DB, FTS tables
- `mailmind index-mbox --mbox sample.mbox --account demo` — index an mbox file
- `mailmind index-maildir --root ~/Mail/personal-gmail --account personal-gmail` — index a Maildir
- `mailmind search "query"` — lexical FTS search
- `mailmind search-hybrid "query"` — hybrid (FTS ∪ ANN)
- `mailmind search-hybrid "natural language query"` — NL planner is enabled by default; add `--no-nl` to disable. A fast constraints pre‑pass (tiny LLM) infers soft date ranges and hints; hybrid retrieval applies soft time boosts and structural filters.
- `mailmind process-attachments --langs eng+spa` — extract text and index
- `mailmind worker attachments|embeddings [--max N]` — drain job queues once
- `mailmind embed --model <id> --dim 256` — embed queued chunks and update ANN
- `mailmind extract-tables --query "board" --out out/rows.csv --metrics revenue,ebit`
- `mailmind summarize --query "..."` — uses a local LLM (Ollama) when available

## Performance
- Keep embedding dim at 256 (MRL truncation) to reduce size/speed costs
- Batch embed (128–256) and normalize vectors
- Use OCR only when native PDF text is sparse; OCR is cached per sha256
- Prefer hnswlib for portability; persist ANN index to `data/vectors`
- Throughput guidance:
  - Stage A (parse + FTS bodies): 3k–10k msgs/min (WAL on, batched commits)
  - Stage B (PDF text): 20–80 pages/sec (pdfplumber/pdftotext); OCR: 0.5–3 pages/sec/core
  - Stage C (embeddings): 2k–10k chunks/min on CPU e5‑small; faster with MPS (Metal)

## Troubleshooting
- Slow indexing: tune OCR languages; verify batch sizes; keep WAL on
- Large DB: lower embedding dim (e.g., 128) or increase chunk size
- Missing PDF text: install `ocrmypdf`, `tesseract`, `poppler`; confirm language packs
- Planner quality: try a stronger local LLM or add a reranker
- Summarize requires a local LLM (e.g., Ollama). Falls back to extractive mode if unavailable
- ANN missing: `pip install hnswlib` and re‑run `mailmind worker embeddings --max ...`
- Hybrid model mismatch: `search-hybrid` auto‑detects model/dim from the DB; override with `--model`/`--dim` if needed

## Roadmap
- M0–M3: ingest, attachments, embeddings, hybrid search
- M4–M6: planner + tools, reranking, quality eval
- M7+: finetune embeddings on your mail, TUI

## License
- Code: see `pyproject.toml` (Proprietary)
- Models: comply with their respective licenses (e.g., Gemma license)
