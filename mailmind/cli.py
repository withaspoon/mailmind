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
    p.add_argument("--model", type=str, default="google/embeddinggemma-300m", help="Sentence-Transformers model name")
    p.add_argument("--dim", type=int, default=256, help="Embedding dim to serve (MRL truncation)")
    p.add_argument("--batch", type=int, default=128, help="Embedding batch size")
    p.add_argument("--vectors", type=Path, default=None, help="Path for ANN index file (default: <root>/vectors/mailmind_hnsw.bin)")
    p.add_argument("--limit", type=int, default=0, help="Only embed first N chunks (0 = all)")
    p.set_defaults(cmd="embed")


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser("mailmind", description="Local email indexer and searcher (offline)")
    sub = ap.add_subparsers(dest="cmd")
    _add_init(sub)
    _add_index(sub)
    _add_index_mbox(sub)
    _add_search(sub)
    _add_embed(sub)
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
            emb = Embedder(model=ns.model, dim=ns.dim)
            vecs = emb.encode(texts, batch_size=ns.batch)
            # Build or extend ANN index
            ann = HnswIndex(dim=ns.dim, path=vectors_path)
            if ann.available():
                try:
                    ann.load()
                except Exception:
                    ann.init(capacity=max(len(vecs) * 2, 10000))
            else:
                print("hnswlib not installed; embeddings computed but ANN index unavailable.")
            if ann.available():
                ann.add(vecs, chunk_ids)
                ann.save()
            # Record mapping in DB
            cur.executemany(
                "INSERT OR IGNORE INTO chunk_vectors (chunk_id, model, dim) VALUES (?, ?, ?)",
                [(cid, ns.model, int(ns.dim)) for cid in chunk_ids],
            )
            con.commit()
            print(f"Embedded {len(vecs)} chunks. ANN index saved to {vectors_path}.")
        con.close()
        return 0

    ap.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
