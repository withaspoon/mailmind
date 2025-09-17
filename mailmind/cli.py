from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import load_config
from .db import init_db
from .indexer import index_maildir
from .mbox_indexer import index_mbox
from .search import search as fts_search
from .embedder import Embedder
from .ann import HnswIndex
from .hybrid import HybridConfig, hybrid_search
from .summarizer import summarize_query
from .attachments import process_attachments
from .workers import worker_attachments_once, worker_embeddings_once
from .tables import extract_tables_by_query


def _add_init(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("init", help="Initialize data directories and SQLite DB")
    p.add_argument("--root", type=Path, default=Path("data"), help="Root data directory (default: ./data)")
    p.add_argument("--db", type=Path, default=None, help="SQLite DB path (default: <root>/db.sqlite3)")
    p.set_defaults(cmd="init")


def _add_index(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("index", help="Index a Maildir tree into the local DB")
    p.add_argument("--maildir", type=Path, required=True, help="Path to Maildir root (contains folders with cur/new)")
    p.add_argument("--root", type=Path, default=Path("data"), help="Root data directory (default: ./data)")
    p.add_argument("--db", type=Path, default=None, help="SQLite DB path (default: <root>/db.sqlite3)")
    p.add_argument("--account", type=str, default="default", help="Logical account name for these messages")
    p.add_argument("--batch", type=int, default=200, help="DB commit batch size (default: 200)")
    p.set_defaults(cmd="index")


def _add_index_mbox(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("index-mbox", help="Index an mbox file into the local DB")
    p.add_argument("--mbox", type=Path, required=True, help="Path to .mbox file")
    p.add_argument("--root", type=Path, default=Path("data"), help="Root data directory (default: ./data)")
    p.add_argument("--db", type=Path, default=None, help="SQLite DB path (default: <root>/db.sqlite3)")
    p.add_argument("--account", type=str, default="default", help="Logical account name for these messages")
    p.add_argument("--folder", type=str, default=None, help="Folder label (default: mbox/<name>)")
    p.add_argument("--batch", type=int, default=200, help="DB commit batch size (default: 200)")
    p.set_defaults(cmd="index-mbox")


def _add_search(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("search", help="Search indexed mail using FTS (lexical)")
    p.add_argument("query", type=str, help="FTS query string")
    p.add_argument("--root", type=Path, default=Path("data"), help="Root data directory (default: ./data)")
    p.add_argument("--db", type=Path, default=None, help="SQLite DB path (default: <root>/db.sqlite3)")
    p.add_argument("--limit", type=int, default=20, help="Max results (default: 20)")
    p.add_argument("--json", action="store_true", help="Output JSON lines")
    p.set_defaults(cmd="search")


def _add_embed(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("embed", help="Embed chunks and build ANN index (optional, local)")
    p.add_argument("--root", type=Path, default=Path("data"), help="Root data directory (default: ./data)")
    p.add_argument("--db", type=Path, default=None, help="SQLite DB path (default: <root>/db.sqlite3)")
    p.add_argument("--model", type=str, default="intfloat/multilingual-e5-small", help="Sentence-Transformers model name")
    p.add_argument("--dim", type=int, default=256, help="Embedding dimension to serve (slice if needed)")
    p.add_argument("--batch", type=int, default=128, help="Embedding batch size")
    p.add_argument("--vectors", type=Path, default=None, help="Path for ANN index file (default: <root>/vectors/mailmind_hnsw.bin)")
    p.add_argument("--limit", type=int, default=0, help="Only embed first N chunks (0 = all)")
    p.add_argument("--rebuild", action="store_true", help="Rebuild embeddings and ANN from scratch (clears mapping and index file)")
    p.add_argument("--backend", type=str, default="auto", choices=["auto", "st", "transformers"], help="Embedding backend")
    p.set_defaults(cmd="embed")


def _add_search_hybrid(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("search-hybrid", help="Hybrid search (FTS ∪ ANN) if vectors available")
    p.add_argument("query", type=str)
    p.add_argument("--root", type=Path, default=Path("data"))
    p.add_argument("--db", type=Path, default=None)
    p.add_argument("--model", type=str, default=None, help="Override embedding model used for ANN (auto-detected from DB if omitted)")
    p.add_argument("--dim", type=int, default=0, help="Override embedding dim (auto-detected from DB if 0)")
    p.add_argument("--fts-k", type=int, default=50)
    p.add_argument("--ann-k", type=int, default=100)
    p.add_argument("--limit", type=int, default=20)
    p.add_argument("--nl", action="store_true", help="Use the NL planner for fuzzy, multilingual constraints and expansions")
    p.set_defaults(cmd="search-hybrid")


def _add_summarize(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("summarize", help="Summarize answers for a query using local LLM (Ollama) with citations")
    p.add_argument("query", type=str)
    p.add_argument("--root", type=Path, default=Path("data"))
    p.add_argument("--db", type=Path, default=None)
    p.add_argument("--top", type=int, default=8, help="Top documents to include")
    p.set_defaults(cmd="summarize")


def _add_process_attachments(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "process-attachments",
        help="Extract text from attachments (PDF/image) with OCR fallback and index into FTS",
    )
    p.add_argument("--root", type=Path, default=Path("data"))
    p.add_argument("--db", type=Path, default=None)
    p.add_argument("--langs", type=str, default="eng", help="OCR languages (e.g., eng,eng+spa,swe)")
    p.add_argument(
        "--min-chars-per-page",
        type=int,
        default=120,
        help="OCR PDFs when extracted text has fewer chars per page than this threshold",
    )
    p.add_argument("--limit", type=int, default=0, help="Process at most N attachments (0 = all)")
    p.add_argument("--reprocess", action="store_true", help="Force re-extraction even if path_text exists")
    p.set_defaults(cmd="process-attachments")


def _add_worker(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("worker", help="Run background workers once (attachments/embeddings)")
    sub2 = p.add_subparsers(dest="worker_cmd")

    pa = sub2.add_parser("attachments", help="Process pending attachment OCR jobs once")
    pa.add_argument("--root", type=Path, default=Path("data"))
    pa.add_argument("--db", type=Path, default=None)
    pa.add_argument("--langs", type=str, default="eng")
    pa.add_argument("--min-chars-per-page", type=int, default=120)
    pa.add_argument("--max", type=int, default=50, help="Max jobs to process in this run")
    pa.set_defaults(cmd="worker", worker_cmd="attachments")

    pe = sub2.add_parser("embeddings", help="Process pending embedding jobs once")
    pe.add_argument("--root", type=Path, default=Path("data"))
    pe.add_argument("--db", type=Path, default=None)
    pe.add_argument("--model", type=str, default=None, help="Embedding model (defaults to last used in DB)")
    pe.add_argument("--dim", type=int, default=0, help="Embedding dim (defaults from DB)")
    pe.add_argument("--backend", type=str, default="auto", choices=["auto", "st", "transformers"])
    pe.add_argument("--batch", type=int, default=128)
    pe.add_argument("--max", type=int, default=200)
    pe.set_defaults(cmd="worker", worker_cmd="embeddings")


def _add_extract_tables(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("extract-tables", help="Extract tables from PDF attachments matching a query; optional yearly aggregation")
    p.add_argument("query", type=str)
    p.add_argument("--root", type=Path, default=Path("data"))
    p.add_argument("--db", type=Path, default=None)
    p.add_argument("--since", type=int, default=None, help="Only include rows from this year and onwards")
    p.add_argument("--out", type=Path, required=True, help="Output CSV path for raw numeric rows")
    p.add_argument(
        "--metrics",
        type=str,
        default=None,
        help="Comma-separated metric name fragments to aggregate (e.g., 'revenue,ebit,total')",
    )
    p.set_defaults(cmd="extract-tables")


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser("mailmind", description="Local email indexer and searcher (offline)")
    ap.add_argument(
        "--progress",
        choices=["auto", "bar", "json", "none"],
        default="auto",
        help="Progress reporting mode (default: auto)",
    )
    sub = ap.add_subparsers(dest="cmd")
    _add_init(sub)
    _add_index(sub)
    _add_index_mbox(sub)
    _add_search(sub)
    _add_embed(sub)
    _add_search_hybrid(sub)
    _add_summarize(sub)
    _add_process_attachments(sub)
    _add_worker(sub)
    _add_extract_tables(sub)
    ns = ap.parse_args(argv)

    if ns.cmd is None:
        ap.print_help()
        return 2

    # Load config file if present (optional)
    cfg = load_config()

    root: Path = (ns.root or Path(cfg.get("root", "data"))).expanduser().resolve()
    db_path: Path = (ns.db or Path(cfg.get("db_path", root / "db.sqlite3"))).expanduser()
    if db_path.is_dir():
        db_path = db_path / "db.sqlite3"

    if ns.cmd == "init":
        root.mkdir(parents=True, exist_ok=True)
        (root / "attachments").mkdir(parents=True, exist_ok=True)
        (root / "vectors").mkdir(parents=True, exist_ok=True)
        init_db(db_path)
        print(f"Initialized DB at {db_path} and data root {root}")
        return 0

    if ns.cmd == "index":
        root.mkdir(parents=True, exist_ok=True)
        (root / "attachments").mkdir(parents=True, exist_ok=True)
        init_db(db_path)
        index_maildir(
            maildir=ns.maildir,
            db_path=db_path,
            attachments_root=root / "attachments",
            account=ns.account,
            batch_size=ns.batch,
            progress_mode=ns.progress,
            progress_root=root,
        )
        return 0

    if ns.cmd == "index-mbox":
        root.mkdir(parents=True, exist_ok=True)
        (root / "attachments").mkdir(parents=True, exist_ok=True)
        init_db(db_path)
        index_mbox(
            mbox_path=ns.mbox,
            db_path=db_path,
            attachments_root=root / "attachments",
            account=ns.account,
            folder=ns.folder,
            batch_size=ns.batch,
            progress_mode=ns.progress,
            progress_root=root,
        )
        return 0

    if ns.cmd == "search":
        results = fts_search(db_path=db_path, query=ns.query, limit=ns.limit)
        if ns.json:
            for r in results:
                print(r.to_json())
        else:
            for i, r in enumerate(results, 1):
                dt = r.date_ts
                print(f"{i:>2}. [{r.type}] {r.subject} — {r.from_email} ({r.account}/{r.folder})")
                print(f"    id={r.message_id} score={r.score:.3f}")
                if r.snippet:
                    print(f"    {r.snippet}")
        return 0

    if ns.cmd == "embed":
        vectors_path = ns.vectors or (root / "vectors" / "mailmind_hnsw.bin")
        # Embed missing chunks
        import sqlite3

        con = sqlite3.connect(str(db_path))
        cur = con.cursor()
        # Rebuild: clear mapping and delete index file(s)
        if ns.rebuild:
            cur.execute("DELETE FROM chunk_vectors")
            con.commit()
            vectors_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                if vectors_path.exists():
                    vectors_path.unlink()
                lp = vectors_path.with_suffix(".labels.txt")
                if lp.exists():
                    lp.unlink()
            except Exception:
                pass
        cur.execute(
            "SELECT id, text FROM chunks WHERE id NOT IN (SELECT chunk_id FROM chunk_vectors) ORDER BY id"
        )
        rows = cur.fetchall()
        if ns.limit and ns.limit > 0:
            rows = rows[: ns.limit]
        texts = [r[1] for r in rows]
        chunk_ids = [int(r[0]) for r in rows]
        if not rows:
            print("No new chunks to embed.")
        else:
            from .progress import ProgressReporter, ProgressConfig, resolve_progress_mode
            emb = Embedder(model=ns.model, dim=ns.dim, backend=ns.backend)
            # Build or extend ANN index
            ann = HnswIndex(dim=ns.dim, path=vectors_path)
            ann_saved = False
            if ann.available():
                try:
                    ann.load()
                except Exception:
                    ann.init(capacity=max(len(texts) * 2, 10000))
            else:
                print("hnswlib not installed; embeddings computed but ANN index unavailable.")
            reporter = ProgressReporter(
                ProgressConfig(
                    mode=resolve_progress_mode(ns.progress),
                    root=root,
                    key="embed",
                    total=len(texts),
                    desc="Embed chunks",
                )
            )
            total_embedded = 0
            for i in range(0, len(texts), ns.batch):
                batch_texts = texts[i : i + ns.batch]
                batch_ids = chunk_ids[i : i + ns.batch]
                batch_vecs = emb.encode(batch_texts, batch_size=ns.batch)
                total_embedded += len(batch_vecs)
                if ann.available():
                    try:
                        ann.add(batch_vecs, batch_ids)
                        ann_saved = True
                    except Exception:
                        pass
                reporter.update(len(batch_vecs))
            if ann_saved:
                try:
                    ann.save()
                except Exception:
                    pass
            reporter.close()
            # Record mapping in DB
            cur.executemany(
                "INSERT OR IGNORE INTO chunk_vectors (chunk_id, model, dim) VALUES (?, ?, ?)",
                [(cid, ns.model, int(ns.dim)) for cid in chunk_ids],
            )
            con.commit()
            if ann_saved:
                print(f"Embedded {total_embedded} chunks. ANN index saved to {vectors_path}.")
            else:
                print(f"Embedded {total_embedded} chunks. ANN index not built (install hnswlib).")
        con.close()
        return 0

    if ns.cmd == "search-hybrid":
        cfg_h = HybridConfig(
            vectors_path=(ns.root / "vectors" / "mailmind_hnsw.bin"),
            model=ns.model or "google/embeddinggemma-300m",
            dim=ns.dim or 256,
            fts_k=ns.fts_k,
            ann_k=ns.ann_k,
        )
        if ns.nl:
            from .hybrid import hybrid_search_nl
            plan, results = hybrid_search_nl(db_path=db_path, query=ns.query, cfg=cfg_h)
            print("Recognized constraints:")
            print(f"  semantic_queries: {', '.join(plan.semantic_queries[:3])}")
            if plan.concept:
                print(f"  concept: {plan.concept}")
            if plan.hard.date_from or plan.hard.date_to:
                print(f"  date: {plan.hard.date_from or ''}..{plan.hard.date_to or ''}")
            if plan.hard.has_attachment:
                print("  has_attachment: true")
            if plan.hard.folder:
                print(f"  folder: {plan.hard.folder}")
            if plan.hard.from_name or plan.hard.from_email:
                print(f"  from: {plan.hard.from_name or plan.hard.from_email}")
        else:
            results = hybrid_search(db_path=db_path, query=ns.query, cfg=cfg_h)
        for i, r in enumerate(results[: ns.limit], 1):
            print(f"{i:>2}. [{r.type}] {r.subject} — {r.from_email} ({r.account}/{r.folder})")
            print(f"    id={r.message_id}")
            if r.snippet:
                print(f"    {r.snippet}")
        return 0

    if ns.cmd == "summarize":
        text = summarize_query(db_path=db_path, root=root, query=ns.query, top_k=ns.top)
        print(text)
        return 0

    if ns.cmd == "process-attachments":
        root.mkdir(parents=True, exist_ok=True)
        (root / "attachments").mkdir(parents=True, exist_ok=True)
        init_db(db_path)
        total, processed, chunks = process_attachments(
            db_path=db_path,
            attachments_root=root / "attachments",
            langs=ns.langs,
            min_chars_per_page=ns.min_chars_per_page,
            limit=ns.limit,
            reprocess=ns.reprocess,
            progress_mode=ns.progress,
            progress_root=root,
        )
        print(f"Attachments processed: {processed}/{total}. Chunks added: {chunks}.")
        return 0

    if ns.cmd == "worker":
        if ns.worker_cmd == "attachments":
            taken, ok, chunks = worker_attachments_once(
                db_path=db_path,
                attachments_root=(root / "attachments"),
                langs=ns.langs,
                min_chars_per_page=ns.min_chars_per_page,
                max_jobs=ns.max,
                progress_mode=ns.progress,
                progress_root=root,
            )
            print(f"Attachment jobs: taken={taken} ok={ok} chunks_added={chunks}")
            return 0
        if ns.worker_cmd == "embeddings":
            # Auto-detect model/dim from DB if not provided
            model = ns.model
            dim = ns.dim
            if not model or not dim:
                import sqlite3

                con = sqlite3.connect(str(db_path))
                row = con.execute(
                    "SELECT model, dim, COUNT(*) as n FROM chunk_vectors GROUP BY model, dim ORDER BY n DESC LIMIT 1"
                ).fetchone()
                con.close()
                if row is not None and row[0]:
                    model = model or row[0]
                    dim = dim or int(row[1])
                else:
                    model = model or "intfloat/multilingual-e5-small"
                    dim = dim or 256
            taken, embedded = worker_embeddings_once(
                db_path=db_path,
                vectors_path=(root / "vectors" / "mailmind_hnsw.bin"),
                model=model,
                dim=dim,
                backend=ns.backend,
                batch_size=ns.batch,
                max_jobs=ns.max,
                progress_mode=ns.progress,
                progress_root=root,
            )
            print(f"Embedding jobs: taken={taken} embedded={embedded}")
            return 0

    if ns.cmd == "extract-tables":
        metrics = [s.strip() for s in ns.metrics.split(",")] if ns.metrics else None
        taken, tables, rows = extract_tables_by_query(
            db_path=db_path,
            root=root,
            query=ns.query,
            out_csv=ns.out,
            since_year=ns.since,
            metrics=metrics,
            progress_mode=ns.progress,
        )
        print(f"Tables: attachments_examined={taken} tables_found={tables} rows_written={rows}")
        if metrics and rows:
            print(f"Aggregated totals written to {ns.out.with_suffix('.agg.csv')}")
        return 0

    ap.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
