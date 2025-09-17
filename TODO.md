Mailmind — TODO / Roadmap

Legend
- [ ] pending  [x] done  [~] in progress

Milestone 0 — Bootstrap & DB (1–2 days)
- [ ] Create Python package skeleton `mailmind/` with cli, config, ingest, parsing, db, vector, search, llm, tools, utils
- [ ] Implement config loader (YAML) with defaults and env overrides
- [ ] Define SQLite schema and migrations (messages, attachments, chunks, embeddings meta, tags, entities) + FTS5 tables
- [ ] Implement content‑addressed attachments store (sha256)
- [ ] Implement chunker (body/attachments) and hashing utilities
- Acceptance: `mailmind init` creates `data/` structure and DB with FTS tables

Milestone 1 — Ingest & Parse (3–5 days)
- [ ] Maildir walker and RFC‑822 parser (headers, bodies, attachments)
- [ ] Threading via Message‑ID / In‑Reply‑To / References
- [ ] HTML → text (readability) fallback to plaintext parts
- [ ] Attachment extraction & mimetype detection
- [ ] Apple Mail adapter (read‑only) and mbox import
- [ ] Idempotent reindex with mtime + sha256 de‑dup
- Acceptance: `mailmind index` ingests a small Maildir into DB; attachments extracted to `data/attachments`

Milestone 2 — OCR & Documents (2–4 days)
- [ ] PDF text extraction (pdfplumber) with layout heuristics
- [ ] OCR pipeline (ocrmypdf + tesseract) when text ratio < threshold
- [ ] Image OCR (tesseract) for common formats
- [ ] Cache OCR text next to original; persist to attachments table
- Acceptance: scanned PDFs produce searchable text and attach provenance

Milestone 3 — Embeddings & Vector Store (2–4 days)
- [ ] Embedding backend abstraction
- [ ] EmbeddingGemma integration via Sentence‑Transformers (serve at 256‑dim by default with MRL)
- [ ] hnswlib ANN index (persist .bin, load on demand)
- [ ] Hybrid search (FTS5 ∪ ANN) with basic scoring fusion
- Acceptance: `mailmind search "query"` returns relevant chunks with message/attachment citations

Milestone 4 — Planner & Tools (4–6 days)
- [ ] LLM backend abstraction (Ollama/llama.cpp)
- [ ] Prompted planner → JSON DSL (filters, search, ops, top_k)
- [ ] Execution engine for ops: collect_attachments, extract_tables (camelot/pdfplumber), aggregate (pandas), summarize
- [ ] Entity extraction (spaCy small) for ORG/GPE/DATE/MONEY
- Acceptance: End‑to‑end example: “board meeting totals since 2000” produces a summary and CSV with citations

Milestone 5 — Reranking & Quality (2–3 days)
- [ ] Optional cross‑encoder reranker (bge‑reranker)
- [ ] Small eval harness: Recall@K / MRR on saved queries
- [ ] Config toggles and performance tuning
- Acceptance: measurable precision gain on test queries with reranker enabled

Milestone 6 — Finetune Embeddings (advanced, optional)
- [ ] Build positives/negatives from threads/subjects and heuristics
- [ ] Sentence‑Transformers training loop (MultipleNegativesRankingLoss)
- [ ] Export tuned model; serve at 256‑dim via MRL
- Acceptance: improved Recall@K vs base model on historical queries

Milestone 7 — Packaging & UX (ongoing)
- [ ] CLI polish with Typer (help, examples, JSON output)
- [ ] Optional TUI (fzf‑like), open attachments/messages from results
- [ ] Packaging: uv/PEX/pyinstaller for single‑file distribution (if feasible)
- [ ] Docs: enrich README with concrete examples and troubleshooting

Operational tasks
- [ ] Add log redaction of emails/domains
- [ ] Add `--dry-run` and `--since` / `--until` flags to indexer
- [ ] Add `mailmind doctor` to check OCR, models, vector store availability
- [ ] Add watchdog for incremental reindex on file changes

Nice‑to‑haves
- [ ] DuckDB integration for heavy aggregations
- [ ] notmuch tag sync
- [ ] Calendar/contacts ingestion for better entity linking
- [ ] Simple HTTP API (localhost) for integrations

Acceptance checklist (MVP)
- [ ] Indexes at least one IMAP account mirrored to Maildir
- [ ] Searches via hybrid retrieval with reasonable latency (<250 ms for top‑50 on M‑class Mac)
- [ ] Extracts PDFs (including OCR) and makes them searchable
- [ ] Answers planner‑style queries with citations (paths + IDs)

