from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional


def _today_iso() -> str:
    return datetime.utcnow().strftime("%Y-%m-%d")


def _ollama_available() -> bool:
    return shutil.which("ollama") is not None


def _prompt_for(query: str, today: str) -> str:
    return (
        "You are a fast constraint extractor. Output only valid JSON (no text outside JSON).\n"
        "Given 'today' and a user query in any language, produce soft constraints.\n"
        "Fields: date_hypotheses (array of {from,to,weight}), has_attachment_prob (0..1),\n"
        "attachment_type_probs (map: pdf/image/any -> 0..1), folder_hint (or null), from_hint (or null), org_hint (or null).\n"
        "Prefer speed over precision; include 1-3 plausible date ranges for relative expressions.\n"
        + json.dumps({
            "today": today,
            "user_query": query,
        })
    )


def _schema_json() -> str:
    # Schema coercing outputs into our expected keys
    return json.dumps(
        {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "properties": {
                "date_hypotheses": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "from": {"type": "string"},
                            "to": {"type": "string"},
                            "weight": {"type": "number"},
                        },
                        "required": ["from", "to"],
                    },
                },
                "has_attachment_prob": {"type": "number"},
                "attachment_type_probs": {
                    "type": "object",
                    "additionalProperties": {"type": "number"},
                },
                "folder_hint": {"type": ["string", "null"]},
                "from_hint": {"type": ["string", "null"]},
                "org_hint": {"type": ["string", "null"]},
            },
            "required": [],
        }
    )


def _iso_date_only(s: str) -> Optional[str]:
    try:
        # Extract first YYYY-MM-DD
        import re

        m = re.search(r"(\d{4}-\d{2}-\d{2})", s)
        if m:
            return m.group(1)
    except Exception:
        pass
    return None


def _coerce_date_hypotheses(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    dh = data.get("date_hypotheses") or []
    out: List[Dict[str, Any]] = []
    # Accept alternate keys like from_date/to_date at top level
    if not dh:
        fd = data.get("from") or data.get("from_date")
        td = data.get("to") or data.get("to_date")
        if isinstance(fd, str) or isinstance(td, str):
            fd_iso = _iso_date_only(str(fd)) if fd else None
            td_iso = _iso_date_only(str(td)) if td else None
            if fd_iso and td_iso:
                out.append({"from": fd_iso, "to": td_iso, "weight": 1.0})
    # If dh present, normalize
    if isinstance(dh, list):
        for h in dh:
            if not isinstance(h, dict):
                continue
            f = h.get("from") or h.get("from_date")
            t = h.get("to") or h.get("to_date")
            f_iso = _iso_date_only(str(f)) if f else None
            t_iso = _iso_date_only(str(t)) if t else None
            if f_iso and t_iso:
                try:
                    w = float(h.get("weight", 1.0))
                except Exception:
                    w = 1.0
                out.append({"from": f_iso, "to": t_iso, "weight": w})
    return out


def extract_constraints(query: str) -> Dict[str, Any]:
    """Extract soft constraints using a tiny local LLM (Ollama). Falls back to no constraints.

    Returns a dict with keys:
      - date_hypotheses: list of {from, to, weight}
      - has_attachment_prob: float
      - attachment_type_probs: dict
      - folder_hint, from_hint, org_hint: optional strings
    """
    model = os.getenv("MAILMIND_FAST_LLM_MODEL", "gemma3:4b")
    today = _today_iso()
    if _ollama_available():
        prompt = _prompt_for(query, today)
        # Try with schema formatting first; pass prompt via stdin for CLI compatibility
        args = ["ollama", "run", model, "--format", _schema_json()]
        try:
            proc = subprocess.run(args, input=prompt.encode("utf-8"), stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
            out = proc.stdout.decode("utf-8", errors="ignore").strip()
            # Attempt to locate JSON in output
            start = out.find("{")
            end = out.rfind("}")
            if start != -1 and end != -1 and end > start:
                out = out[start : end + 1]
            data = json.loads(out)
            # Coerce to our format
            dh = _coerce_date_hypotheses(data)
            return {
                "date_hypotheses": dh,
                "has_attachment_prob": float(data.get("has_attachment_prob", 0.0) or 0.0),
                "attachment_type_probs": data.get("attachment_type_probs", {}) or {},
                "folder_hint": data.get("folder_hint"),
                "from_hint": data.get("from_hint"),
                "org_hint": data.get("org_hint"),
            }
        except Exception:
            # Retry without schema if formatting not supported
            try:
                args = ["ollama", "run", model]
                proc = subprocess.run(args, input=prompt.encode("utf-8"), stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
                out = proc.stdout.decode("utf-8", errors="ignore").strip()
                start = out.find("{")
                end = out.rfind("}")
                if start != -1 and end != -1 and end > start:
                    out = out[start : end + 1]
                data = json.loads(out)
                dh = _coerce_date_hypotheses(data)
                return {
                    "date_hypotheses": dh,
                    "has_attachment_prob": float(data.get("has_attachment_prob", 0.0) or 0.0),
                    "attachment_type_probs": data.get("attachment_type_probs", {}) or {},
                    "folder_hint": data.get("folder_hint"),
                    "from_hint": data.get("from_hint"),
                    "org_hint": data.get("org_hint"),
                }
            except Exception:
                pass
    # Fallback: no constraints
    return {
        "date_hypotheses": [],
        "has_attachment_prob": 0.0,
        "attachment_type_probs": {},
        "folder_hint": None,
        "from_hint": None,
        "org_hint": None,
    }


def extract_constraints_debug(query: str) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """Same as extract_constraints, but also returns debug metadata.

    Meta includes: backend (ollama/not_found), model, args, returncode, stdout_sample, stderr_sample.
    """
    meta: Dict[str, Any] = {"backend": None, "model": os.getenv("MAILMIND_FAST_LLM_MODEL", "gemma3:4b")}
    today = _today_iso()
    if not _ollama_available():
        meta.update({"backend": "ollama:not_found"})
        return extract_constraints(query), meta
    meta["backend"] = "ollama"
    prompt = _prompt_for(query, today)
    args = ["ollama", "run", meta["model"], "--format", _schema_json()]
    meta["args"] = [args[0], args[1], args[2], "--format", "<schema>", "<stdin>"]
    meta["prompt"] = prompt
    meta["schema"] = _schema_json()
    try:
        proc = subprocess.run(args, input=prompt.encode("utf-8"), stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        out = proc.stdout.decode("utf-8", errors="ignore").strip()
        err = proc.stderr.decode("utf-8", errors="ignore").strip()
        rc = proc.returncode
        meta.update({
            "returncode": rc,
            "stdout_sample": out[:400],
            "stderr_sample": err[:200],
        })
        start = out.find("{")
        end = out.rfind("}")
        if start != -1 and end != -1 and end > start:
            out = out[start : end + 1]
        data = json.loads(out)
        dh = _coerce_date_hypotheses(data)
        parsed = {
            "date_hypotheses": dh,
            "has_attachment_prob": float(data.get("has_attachment_prob", 0.0) or 0.0),
            "attachment_type_probs": data.get("attachment_type_probs", {}) or {},
            "folder_hint": data.get("folder_hint"),
            "from_hint": data.get("from_hint"),
            "org_hint": data.get("org_hint"),
        }
        return parsed, meta
    except Exception as e:
        meta.update({"error": str(e)[:200]})
        # fallback try without schema
        try:
            args2 = ["ollama", "run", meta["model"]]
            proc = subprocess.run(args2, input=prompt.encode("utf-8"), stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
            out = proc.stdout.decode("utf-8", errors="ignore").strip()
            err = proc.stderr.decode("utf-8", errors="ignore").strip()
            rc = proc.returncode
            meta.update({
                "args2": [args2[0], args2[1], args2[2], "<stdin>"],
                "returncode2": rc,
                "stdout_sample2": out[:400],
                "stderr_sample2": err[:200],
            })
            start = out.find("{")
            end = out.rfind("}")
            if start != -1 and end != -1 and end > start:
                out = out[start : end + 1]
            data = json.loads(out)
            dh = _coerce_date_hypotheses(data)
            parsed = {
                "date_hypotheses": dh,
                "has_attachment_prob": float(data.get("has_attachment_prob", 0.0) or 0.0),
                "attachment_type_probs": data.get("attachment_type_probs", {}) or {},
                "folder_hint": data.get("folder_hint"),
                "from_hint": data.get("from_hint"),
                "org_hint": data.get("org_hint"),
            }
            return parsed, meta
        except Exception as e2:
            meta.update({"error2": str(e2)[:200]})
            return extract_constraints(query), meta
