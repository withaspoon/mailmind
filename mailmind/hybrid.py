from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from .search import SearchResult
from .embedder import get_embedder
from .ann import HnswIndex
from .utils.fts import nl_to_fts_query
from .planner import plan_query, Plan
from .utils.date_nlp import detect_date_range
from .constraints import extract_constraints


@dataclass
class HybridConfig:
    vectors_path: Path
    model: str = "google/embeddinggemma-300m"
    dim: int = 256
    fts_k: int = 50
    ann_k: int = 100
    rrf_k: int = 60


def _fts_messages(db_path: Path, q: str, k: int) -> List[SearchResult]:
    import sqlite3

    con = sqlite3.connect(str(db_path))
    con.row_factory = sqlite3.Row
    out: List[SearchResult] = []
    try:
        sql = (
            """
            SELECT m.account, m.folder, m.message_id AS mid, m.subject, m.from_email, m.to_emails, m.date_ts,
                   snippet(messages_fts, 5, '[', ']', '…', 8) AS snip,
                   bm25(messages_fts) AS score
            FROM messages_fts
            JOIN messages m ON m.message_id = messages_fts.message_id
            WHERE messages_fts MATCH ?
            ORDER BY score LIMIT ?
            """
        )
        q_fts = nl_to_fts_query(q)
        for row in con.execute(sql, (q_fts, k)):
            out.append(
                SearchResult(
                    type="message",
                    score=float(row["score"]),
                    message_id=row["mid"],
                    subject=row["subject"] or "",
                    from_email=row["from_email"] or "",
                    to_emails=row["to_emails"] or "",
                    date_ts=int(row["date_ts"]) if row["date_ts"] is not None else 0,
                    folder=row["folder"] or "",
                    account=row["account"] or "",
                    snippet=row["snip"] or "",
                )
            )
    finally:
        con.close()
    return out


def _ann_chunks(db_path: Path, cfg: HybridConfig, q: str) -> List[Tuple[int, float]]:
    # Returns list of (chunk_id, distance)
    emb = get_embedder(model=cfg.model, dim=cfg.dim)
    qvec = emb.encode([q])[0]
    ann = HnswIndex(dim=cfg.dim, path=cfg.vectors_path)
    if not ann.available():
        return []
    try:
        ann.load()
    except Exception:
        return []
    labels, dists = ann.search([qvec], k=cfg.ann_k)
    if not labels:
        return []
    labs = labels[0]
    ds = dists[0]
    return list(zip(labs, ds))


def _fetch_chunk_contexts(db_path: Path, chunk_ids: List[int]) -> Dict[int, Tuple[str, str, str, int, str, str, str]]:
    # chunk_id -> (message_id, subject, from_email, date_ts, folder, account, text)
    con = sqlite3.connect(str(db_path))
    con.row_factory = sqlite3.Row
    out: Dict[int, Tuple[str, str, str, int, str, str, str]] = {}
    try:
        if not chunk_ids:
            return out
        # SQLite limits params; chunk into groups of 999
        for i in range(0, len(chunk_ids), 900):
            part = chunk_ids[i : i + 900]
            qmarks = ",".join(["?"] * len(part))
            sql = f"""
                SELECT c.id AS chunk_id, m.message_id AS mid, m.subject, m.from_email, m.date_ts, m.folder, m.account, c.text
                FROM chunks c
                JOIN messages m ON m.id = c.message_id
                WHERE c.id IN ({qmarks})
            """
            for row in con.execute(sql, part):
                out[int(row["chunk_id"])] = (
                    row["mid"],
                    row["subject"] or "",
                    row["from_email"] or "",
                    int(row["date_ts"]) if row["date_ts"] is not None else 0,
                    row["folder"] or "",
                    row["account"] or "",
                    row["text"] or "",
                )
    finally:
        con.close()
    return out


def hybrid_search(db_path: Path, query: str, cfg: HybridConfig) -> List[SearchResult]:
    # Get FTS message results and ANN chunk results
    # Auto-resolve model/dim from DB mapping if available
    try:
        con = sqlite3.connect(str(db_path))
        row = con.execute(
            "SELECT model, dim, COUNT(*) as n FROM chunk_vectors GROUP BY model, dim ORDER BY n DESC LIMIT 1"
        ).fetchone()
        con.close()
        if row is not None and row[0]:
            cfg.model = row[0]
            try:
                cfg.dim = int(row[1])
            except Exception:
                pass
    except Exception:
        pass

    fts = _fts_messages(db_path, query, cfg.fts_k)
    ann_pairs = _ann_chunks(db_path, cfg, query)
    chunk_ids = [cid for (cid, _dist) in ann_pairs]
    chunk_ctx = _fetch_chunk_contexts(db_path, chunk_ids)

    # Build pseudo-results for ANN (mapped to message_id)
    ann_results: List[SearchResult] = []
    for (cid, dist) in ann_pairs:
        ctx = chunk_ctx.get(int(cid))
        if not ctx:
            continue
        mid, subject, from_email, date_ts, folder, account, text = ctx
        ann_results.append(
            SearchResult(
                type="chunk",
                score=float(dist),  # smaller is better for L2
                message_id=mid,
                subject=subject,
                from_email=from_email,
                to_emails="",
                date_ts=date_ts,
                folder=folder,
                account=account,
                snippet=text[:200] + ("…" if len(text) > 200 else ""),
                chunk_id=int(cid),
            )
        )

    # Reciprocal Rank Fusion by message_id
    def rrf(ranks: Dict[str, int], k: int) -> Dict[str, float]:
        return {mid: 1.0 / (k + r) for mid, r in ranks.items()}

    def ranks_by_mid(items: List[SearchResult]) -> Dict[str, int]:
        ranks: Dict[str, int] = {}
        for idx, it in enumerate(items, 1):
            if it.message_id not in ranks:
                ranks[it.message_id] = idx
        return ranks

    r_fts = ranks_by_mid(fts)
    r_ann = ranks_by_mid(ann_results)
    s_fts = rrf(r_fts, cfg.rrf_k)
    s_ann = rrf(r_ann, cfg.rrf_k)

    # Merge scores
    merged: Dict[str, float] = {}
    for mid, s in s_fts.items():
        merged[mid] = merged.get(mid, 0.0) + s
    for mid, s in s_ann.items():
        merged[mid] = merged.get(mid, 0.0) + s

    # Pick representative SearchResult per message_id: prefer FTS, else ANN
    pick: Dict[str, SearchResult] = {}
    for it in fts + ann_results:
        if it.message_id not in pick:
            pick[it.message_id] = it

    # Rank by merged score descending
    mids_sorted = sorted(merged.items(), key=lambda kv: kv[1], reverse=True)
    out: List[SearchResult] = []
    for mid, _score in mids_sorted:
        out.append(pick[mid])
    return out


def _ann_chunks_multi(db_path: Path, cfg: HybridConfig, queries: List[str]) -> List[Tuple[int, float]]:
    if not queries:
        return []
    emb = get_embedder(model=cfg.model, dim=cfg.dim)
    qvecs = emb.encode(queries[:5])
    ann = HnswIndex(dim=cfg.dim, path=cfg.vectors_path)
    if not ann.available():
        return []
    try:
        ann.load()
    except Exception:
        return []
    all_pairs: List[Tuple[int, float]] = []
    for qv in qvecs:
        labels, dists = ann.search([qv], k=cfg.ann_k)
        if not labels:
            continue
        labs = labels[0]
        ds = dists[0]
        all_pairs.extend(zip(labs, ds))
    best: Dict[int, float] = {}
    for cid, dist in all_pairs:
        if cid not in best or dist < best[cid]:
            best[cid] = dist
    return list(best.items())


def _parse_ts(iso: str) -> int:
    from datetime import datetime, timezone
    try:
        if len(iso) == 4 and iso.isdigit():
            return int(datetime(int(iso), 1, 1, tzinfo=timezone.utc).timestamp())
        return int(datetime.fromisoformat(iso).replace(tzinfo=timezone.utc).timestamp())
    except Exception:
        return 0


def hybrid_search_nl(db_path: Path, query: str, cfg: HybridConfig) -> Tuple[Plan, List[SearchResult]]:
    # Auto-resolve model/dim from DB mapping
    try:
        con = sqlite3.connect(str(db_path))
        row = con.execute(
            "SELECT model, dim, COUNT(*) as n FROM chunk_vectors GROUP BY model, dim ORDER BY n DESC LIMIT 1"
        ).fetchone()
        con.close()
        if row is not None and row[0]:
            cfg.model = row[0]
            try:
                cfg.dim = int(row[1])
            except Exception:
                pass
    except Exception:
        pass

    # Fast constraints (tiny LLM) for temporal and soft hints
    fast = extract_constraints(query)
    plan = plan_query(query)
    # Merge temporal hypotheses (soft)
    if fast.get("date_hypotheses"):
        plan.temporal_hypotheses = fast.get("date_hypotheses")
    # Merge soft hints
    try:
        plan.has_attachment_prob = float(fast.get("has_attachment_prob", 0.0) or 0.0)
    except Exception:
        plan.has_attachment_prob = 0.0
    atp = fast.get("attachment_type_probs", {}) or {}
    if isinstance(atp, dict):
        # ensure floats
        plan.attachment_type_probs = {str(k): float(atp.get(k) or 0.0) for k in atp.keys()}
    plan.folder_hint = fast.get("folder_hint")
    plan.from_hint = fast.get("from_hint")
    plan.org_hint = fast.get("org_hint")
    # Fallback date parsing (multilingual) if planner omitted it
    if not (plan.hard.date_from or plan.hard.date_to) and not plan.temporal_hypotheses:
        df, dt = detect_date_range(query)
        if df and dt:
            plan.hard.date_from = df
            plan.hard.date_to = dt
    qlist: List[str] = []
    if query:
        qlist.append(query)
    if plan.concept:
        qlist.append(plan.concept)
    for s in plan.semantic_queries:
        if s and s not in qlist:
            qlist.append(s)
    # Add soft hints as query expansions (no hard-coded vocab)
    for key in ("folder_hint", "from_hint", "org_hint"):
        hv = getattr(plan, key)
        if hv and isinstance(hv, str) and hv not in qlist:
            qlist.append(hv)
    # Retrieval
    fts = _fts_messages(db_path, query, plan.fts_k)
    ann_pairs = _ann_chunks_multi(db_path, cfg, qlist)
    chunk_ids = [cid for (cid, _dist) in ann_pairs]
    chunk_ctx = _fetch_chunk_contexts(db_path, chunk_ids)
    ann_results: List[SearchResult] = []
    for (cid, dist) in ann_pairs:
        ctx = chunk_ctx.get(int(cid))
        if not ctx:
            continue
        mid, subject, from_email, date_ts, folder, account, text = ctx
        ann_results.append(
            SearchResult(
                type="chunk",
                score=float(dist),
                message_id=mid,
                subject=subject,
                from_email=from_email,
                to_emails="",
                date_ts=date_ts,
                folder=folder,
                account=account,
                snippet=text[:200] + ("…" if len(text) > 200 else ""),
                chunk_id=int(cid),
            )
        )

    # RRF fusion
    def ranks_by_mid(items: List[SearchResult]) -> Dict[str, int]:
        ranks: Dict[str, int] = {}
        for idx, it in enumerate(items, 1):
            if it.message_id not in ranks:
                ranks[it.message_id] = idx
        return ranks

    def rrf(ranks: Dict[str, int], k: int) -> Dict[str, float]:
        return {mid: 1.0 / (k + r) for mid, r in ranks.items()}

    r_fts = ranks_by_mid(fts)
    r_ann = ranks_by_mid(ann_results)
    merged: Dict[str, float] = {}
    for mid, s in rrf(r_fts, getattr(plan, "rrf_k", 60)).items():
        merged[mid] = merged.get(mid, 0.0) + s
    for mid, s in rrf(r_ann, getattr(plan, "rrf_k", 60)).items():
        merged[mid] = merged.get(mid, 0.0) + s

    pick: Dict[str, SearchResult] = {}
    for it in fts + ann_results:
        if it.message_id not in pick:
            pick[it.message_id] = it
    # Temporal soft scoring: turn temporal hypotheses into a soft boost, not a filter
    def temporal_boost(ts: int) -> float:
        if not ts:
            return 0.0
        if not plan.temporal_hypotheses and not (plan.hard.date_from or plan.hard.date_to):
            return 0.0
        import time as _time
        from datetime import datetime
        # Convert ts to date
        try:
            d = datetime.utcfromtimestamp(ts).date()
        except Exception:
            return 0.0
        hyps = plan.temporal_hypotheses[:3] if plan.temporal_hypotheses else []
        if not hyps and (plan.hard.date_from or plan.hard.date_to):
            hyps = [{"from": plan.hard.date_from, "to": plan.hard.date_to, "weight": 1.0}]
        score = 0.0
        for h in hyps:
            try:
                df = h.get("from")
                dt = h.get("to")
                w = float(h.get("weight", 1.0))
                if not df or not dt:
                    continue
                from datetime import date
                df_d = date.fromisoformat(df)
                dt_d = date.fromisoformat(dt)
                # If within range → full weight; otherwise decay exponentially by month distance
                if df_d <= d <= dt_d:
                    score = max(score, w)
                else:
                    # months distance approx via days/30
                    delta_days = min(abs((d - df_d).days), abs((d - dt_d).days))
                    months = delta_days / 30.0
                    softness = max(0.1, float(getattr(plan, "temporal_softness", 1.0)))
                    score = max(score, w * (2.71828 ** (-months / (2.0 * softness))))
            except Exception:
                continue
        return score

    # Apply temporal boosts to merged scores; also soft boosts for attachments/types and folder/from hints
    for mid in list(merged.keys()):
        r = pick.get(mid)
        if not r:
            continue
        tb = temporal_boost(r.date_ts)
        if tb:
            merged[mid] += tb
        # Soft attachment/type boosts
        try:
            con = sqlite3.connect(str(db_path))
            con.row_factory = sqlite3.Row
            row = con.execute("SELECT id, folder, from_email FROM messages WHERE message_id=?", (r.message_id,)).fetchone()
            if row is not None:
                mid_int = int(row["id"])
                att_count = con.execute("SELECT COUNT(*) FROM attachments WHERE message_id=?", (mid_int,)).fetchone()[0]
                if att_count and plan.has_attachment_prob > 0:
                    merged[mid] += 0.1 * float(plan.has_attachment_prob)
                atp = plan.attachment_type_probs or {}
                if atp.get("pdf"):
                    # any pdf?
                    pdf_cnt = con.execute("SELECT COUNT(*) FROM attachments WHERE message_id=? AND (mime LIKE 'application/pdf%' OR path_original LIKE '%.pdf')", (mid_int,)).fetchone()[0]
                    if pdf_cnt:
                        merged[mid] += 0.1 * float(atp.get("pdf", 0.0))
                # Soft folder/from hints
                fh = plan.folder_hint
                if fh and (row["folder"] or "").lower().endswith(str(fh).lower()):
                    merged[mid] += 0.05
                frh = plan.from_hint
                if frh and (row["from_email"] or "").lower().find(str(frh).lower()) != -1:
                    merged[mid] += 0.05
        except Exception:
            pass

    mids_sorted = sorted(merged.items(), key=lambda kv: kv[1], reverse=True)
    results: List[SearchResult] = [pick[mid] for mid, _ in mids_sorted]

    # Apply structural filters (keep temporal as soft only)
    h = plan.hard
    if h and (h.has_attachment or h.folder or h.from_name or h.from_email):
        con = sqlite3.connect(str(db_path))
        con.row_factory = sqlite3.Row
        try:
            allowed: Dict[str, bool] = {}
            mids = [r.message_id for r in results]
            for i in range(0, len(mids), 900):
                part = mids[i : i + 900]
                qmarks = ",".join(["?"] * len(part))
                sql = f"SELECT message_id, date_ts, from_email, folder, id FROM messages WHERE message_id IN ({qmarks})"
                for row in con.execute(sql, part):
                    ok = True
                    if ok and h.folder:
                        ok = ok and (row["folder"] or "").lower().endswith(h.folder.lower())
                    if ok and (h.from_name or h.from_email):
                        f = (row["from_email"] or "").lower()
                        if h.from_email:
                            ok = ok and h.from_email.lower() in f
                        if h.from_name and ok:
                            ok = ok and h.from_name.lower() in f
                    if ok and h.has_attachment:
                        cnt = con.execute("SELECT COUNT(*) FROM attachments WHERE message_id=?", (row["id"],)).fetchone()[0]
                        ok = ok and cnt > 0
                    allowed[row["message_id"]] = ok
            results = [r for r in results if allowed.get(r.message_id, True)]
        finally:
            con.close()

    return plan, results
