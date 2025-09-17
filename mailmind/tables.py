from __future__ import annotations

import csv
import math
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from .hybrid import HybridConfig, hybrid_search
from .progress import ProgressReporter, ProgressConfig, resolve_progress_mode


@dataclass
class TableRow:
    message_id: str
    attachment_path: str
    page: int
    table_index: int
    row_index: int
    data: Dict[str, str]
    year: Optional[int]


def _normalize_header(h: str) -> str:
    h = (h or "").strip()
    h = re.sub(r"\s+", " ", h)
    h = h.strip(" :;,.|-")
    return h


def _try_parse_number(s: str) -> Optional[float]:
    if s is None:
        return None
    t = s.strip()
    if not t:
        return None
    # Remove currency symbols and spaces
    t = re.sub(r"[\s$€£¥]", "", t)
    # Handle thousands separators and decimal commas
    if re.match(r"^-?[0-9]{1,3}(?:[.,][0-9]{3})*(?:[.,][0-9]+)?$", t):
        # Normalize to dot decimal
        if "," in t and "." in t:
            # Assume dot thousands, comma decimal (e.g., 1.234,56)
            t = t.replace(".", "").replace(",", ".")
        elif "," in t and t.count(",") == 1 and t.rfind(",") > len(t) - 4:
            # Single comma near end, treat as decimal
            t = t.replace(",", ".")
        else:
            t = t.replace(",", "")
    try:
        return float(t)
    except Exception:
        return None


def _extract_year(cells: Dict[str, str], default_year: Optional[int]) -> Optional[int]:
    # Prefer an explicit Year column if present
    for key in cells.keys():
        k = key.lower()
        if k in {"year", "år", "ano"}:
            v = re.findall(r"\d{4}", cells[key] or "")
            if v:
                return int(v[0])
    # Try to find a 4-digit year anywhere in the row
    joined = " ".join((cells.get(k) or "") for k in cells)
    m = re.search(r"(19|20)\d{2}", joined)
    if m:
        return int(m.group(0))
    return default_year


def extract_pdf_tables(pdf_path: Path) -> List[Tuple[int, int, List[List[str]]]]:
    """Extract tables from a PDF using pdfplumber.

    Returns a list of (page_number, table_index, rows) where rows is a list of string lists.
    """
    try:
        import pdfplumber  # type: ignore
    except Exception:
        raise RuntimeError("pdfplumber not installed. Install with: pip install pdfplumber")

    out: List[Tuple[int, int, List[List[str]]]] = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for i, page in enumerate(pdf.pages, 1):
            try:
                tables = page.extract_tables({
                    "vertical_strategy": "lines",
                    "horizontal_strategy": "lines",
                    "snap_tolerance": 3,
                    "join_tolerance": 3,
                    "min_words_vertical": 2,
                    "min_words_horizontal": 2,
                })
            except Exception:
                tables = []
            for t_idx, rows in enumerate(tables or []):
                # Clean cells to strings
                cleaned: List[List[str]] = []
                for r in rows:
                    cleaned.append([("" if c is None else str(c)).strip() for c in r])
                if cleaned:
                    out.append((i, t_idx, cleaned))
    return out


def flatten_tables(
    tables: List[Tuple[int, int, List[List[str]]]],
    default_year: Optional[int],
) -> List[Tuple[int, int, List[str], List[str], Optional[int]]]:
    """Flatten tables into rows with headers.

    Returns list of (page, table_index, headers, row, year).
    """
    flat: List[Tuple[int, int, List[str], List[str], Optional[int]]] = []
    for page, t_idx, rows in tables:
        if not rows:
            continue
        headers = [
            _normalize_header(h if h is not None else "")
            for h in (rows[0] if rows else [])
        ]
        # If header row looks empty, fabricate generic headers
        if not any(headers):
            headers = [f"col_{i+1}" for i in range(len(rows[0]))]
        for ridx, row in enumerate(rows[1:], 1):
            # Pad/truncate row to header length
            if len(row) < len(headers):
                row = row + [""] * (len(headers) - len(row))
            elif len(row) > len(headers):
                row = row[: len(headers)]
            cells = {headers[i]: (row[i] or "").strip() for i in range(len(headers))}
            year = _extract_year(cells, default_year)
            flat.append((page, t_idx, headers, row, year))
    return flat


def extract_tables_by_query(
    db_path: Path,
    root: Path,
    query: str,
    out_csv: Path,
    since_year: Optional[int] = None,
    metrics: Optional[List[str]] = None,
    progress_mode: str | None = None,
) -> Tuple[int, int, int]:
    """Find PDF attachments for messages matching query, extract tables, and write CSV.

    If metrics are provided, include only rows where at least one metric column is present
    and numeric; also output a second CSV with aggregated totals per year per metric.

    Returns (attachments_examined, tables_found, rows_written).
    """
    cfg = HybridConfig(vectors_path=root / "vectors" / "mailmind_hnsw.bin")
    results = hybrid_search(db_path, query, cfg)
    con = sqlite3.connect(str(db_path))
    con.row_factory = sqlite3.Row
    cur = con.cursor()

    def message_pdfs(mid: str) -> List[sqlite3.Row]:
        return list(
            cur.execute(
                "SELECT a.*, m.date_ts FROM attachments a JOIN messages m ON m.id = a.message_id WHERE m.message_id=? AND (mime LIKE 'application/pdf%' OR path_original LIKE '%.pdf')",
                (mid,),
            )
        )

    # Prepare CSV writers
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    f_rows = out_csv.open("w", newline="", encoding="utf-8")
    rows_writer = csv.writer(f_rows)
    rows_writer.writerow([
        "message_id",
        "attachment_path",
        "page",
        "table_index",
        "row_index",
        "year",
        "column",
        "value",
    ])

    agg_writer = None
    f_agg = None
    agg: Dict[Tuple[int, str], float] = {}
    metric_keys = [m.lower() for m in (metrics or [])]

    attachments_examined = 0
    tables_found = 0
    rows_written = 0

    reporter = ProgressReporter(
        ProgressConfig(
            mode=resolve_progress_mode(progress_mode),
            root=root,
            key="extract-tables",
            total=len(results) if results else None,
            desc="Tables",
        )
    )
    try:
        for r in results:
            pdfs = message_pdfs(r.message_id)
            for a in pdfs:
                attachments_examined += 1
                p = Path(a["path_original"]) if a["path_original"] else None
                if not p or not p.exists():
                    continue
                default_year = int(a["date_ts"]) if a["date_ts"] else None
                if default_year:
                    from datetime import datetime

                    default_year = datetime.utcfromtimestamp(default_year).year

                try:
                    tables = extract_pdf_tables(p)
                except Exception:
                    continue
                if not tables:
                    continue
                tables_found += len(tables)
                flat = flatten_tables(tables, default_year)
                for (page, t_idx, headers, row, year) in flat:
                    if since_year and year and year < since_year:
                        continue
                    # Emit every non-empty numeric cell; if metrics specified, filter by header name
                    for col_name, cell in zip(headers, row):
                        val = _try_parse_number(cell)
                        if val is None:
                            continue
                        if metric_keys and not any(m in col_name.lower() for m in metric_keys):
                            continue
                        rows_writer.writerow(
                            [
                                r.message_id,
                                str(p),
                                page,
                                t_idx,
                                # row_index reported per table
                                flat.index((page, t_idx, headers, row, year)),
                                year or "",
                                col_name,
                                val,
                            ]
                        )
                        rows_written += 1
                        if year is not None and metric_keys:
                            for m in metric_keys:
                                if m in col_name.lower():
                                    agg[(year, m)] = agg.get((year, m), 0.0) + val
            reporter.update(1)
    finally:
        f_rows.close()
        con.close()
        reporter.close()

    # Write aggregation if requested
    if metric_keys and rows_written:
        agg_path = out_csv.with_suffix(".agg.csv")
        f_agg = agg_path.open("w", newline="", encoding="utf-8")
        agg_writer = csv.writer(f_agg)
        agg_writer.writerow(["year", "metric", "total"])
        for (year, metric), total in sorted(agg.items()):
            agg_writer.writerow([year, metric, total])
        f_agg.close()

    return attachments_examined, tables_found, rows_written
