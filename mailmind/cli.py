from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import load_config
from .db import init_db
from .indexer import index_maildir


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


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser("mailmind", description="Local email indexer and searcher (offline)")
    sub = ap.add_subparsers(dest="cmd")
    _add_init(sub)
    _add_index(sub)
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

    ap.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

