from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

from .llm import load_llm_from_env


@dataclass
class HardFilters:
    date_from: Optional[str] = None
    date_to: Optional[str] = None
    has_attachment: Optional[bool] = None
    attachment_types: List[str] = field(default_factory=list)  # e.g., ["pdf"]
    folder: Optional[str] = None  # e.g., Sent
    from_name: Optional[str] = None
    from_email: Optional[str] = None
    org_hint: Optional[str] = None


@dataclass
class Plan:
    semantic_queries: List[str]
    concept: Optional[str]  # one short descriptor string; optional
    hard: HardFilters
    fts_k: int = 50
    ann_k: int = 100
    top_k: int = 50

    def to_json(self) -> str:
        return json.dumps(
            {
                "semantic_queries": self.semantic_queries,
                "concept": self.concept,
                "hard": asdict(self.hard),
                "fts_k": self.fts_k,
                "ann_k": self.ann_k,
                "top_k": self.top_k,
            },
            ensure_ascii=False,
            indent=2,
        )


def _fallback_plan(query: str) -> Plan:
    q = query.strip()
    hard = HardFilters()
    return Plan(
        semantic_queries=[q] if q else [],
        concept=None,
        hard=hard,
        fts_k=50,
        ann_k=100,
        top_k=50,
    )


def plan_query(query: str) -> Plan:
    """Generate a plan using local LLM when available; fallback to heuristics.

    The LLM is prompted to produce a compact JSON with fields: semantic_queries[], hard{}, soft_predicates{}, boosts{}, fts_k, ann_k, top_k
    """
    llm = load_llm_from_env()
    prompt = f"""
Turn the user's request into a compact JSON plan for fuzzy, multilingual email retrieval.
Fields:
- semantic_queries: array of 1-3 short paraphrases (include the original); can be in any language.
- concept: one short descriptor sentence capturing the intent (e.g., "email about medical imaging from a professional"), or null.
- hard: {{date_from, date_to, has_attachment, attachment_types, folder, from_name, from_email, org_hint}} (omit fields if unknown)
- fts_k, ann_k, top_k: integers
Only output JSON, no comments.

User query: "{query}"
"""
    try:
        out = llm.generate(prompt)
        data = json.loads(out)
        hard = data.get("hard", {})
        plan = Plan(
            semantic_queries=list(dict.fromkeys(data.get("semantic_queries", [query]))),
            concept=data.get("concept"),
            hard=HardFilters(
                date_from=hard.get("date_from"),
                date_to=hard.get("date_to"),
                has_attachment=hard.get("has_attachment"),
                attachment_types=hard.get("attachment_types", []) or [],
                folder=hard.get("folder"),
                from_name=hard.get("from_name"),
                from_email=hard.get("from_email"),
                org_hint=hard.get("org_hint"),
            ),
            fts_k=int(data.get("fts_k", 50)),
            ann_k=int(data.get("ann_k", 100)),
            top_k=int(data.get("top_k", 50)),
        )
        if not plan.semantic_queries:
            plan.semantic_queries = [query]
        return plan
    except Exception:
        # Fallback to minimal plan with only the original query
        return _fallback_plan(query)
