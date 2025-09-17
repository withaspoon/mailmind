from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from typing import Optional


@dataclass
class LLMConfig:
    backend: str = "ollama"
    model: str = "llama3.1:8b-instruct-q4_K_M"
    temperature: float = 0.2
    top_p: float = 0.95
    max_tokens: int = 800


class LLMBackend:
    def generate(self, prompt: str, system: Optional[str] = None, cfg: Optional[LLMConfig] = None) -> str:  # pragma: no cover - thin wrapper
        raise NotImplementedError


class OllamaBackend(LLMBackend):
    def __init__(self, model: str) -> None:
        self.model = model

    def generate(self, prompt: str, system: Optional[str] = None, cfg: Optional[LLMConfig] = None) -> str:
        # Prefer CLI to avoid HTTP dependencies; send prompt via stdin for compatibility
        if system:
            # Many models ignore a separate system input; prepend to prompt deterministically
            prompt = f"[SYSTEM]\n{system}\n\n[USER]\n{prompt}"
        args = ["ollama", "run", self.model]
        try:
            proc = subprocess.run(
                args,
                input=prompt.encode("utf-8"),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
        except FileNotFoundError:
            # If Ollama CLI is not present, fallback to extractive summary
            return SimpleExtractiveBackend().generate(prompt)
        out = proc.stdout.decode("utf-8", errors="ignore").strip()
        if proc.returncode != 0 or not out:
            # On any runtime error or empty output, fallback to extractive summary
            return SimpleExtractiveBackend().generate(prompt)
        return out


class SimpleExtractiveBackend(LLMBackend):
    """Fallback summarizer if no local LLM is available.

    Produces a concise bullet list by selecting representative sentences.
    """

    def generate(self, prompt: str, system: Optional[str] = None, cfg: Optional[LLMConfig] = None) -> str:
        # naive sentence selection: pick top lines, compress whitespace
        import re

        text = prompt
        # Heuristic: lines that look like facts
        lines = []
        for ln in text.splitlines():
            ln = ln.strip()
            if not ln:
                continue
            if ln.startswith("-") or ln.endswith(".") or len(ln) > 60:
                lines.append(ln)
        # Deduplicate and limit
        seen = set()
        uniq = []
        for ln in lines:
            key = re.sub(r"\W+", " ", ln.lower())[:120]
            if key in seen:
                continue
            seen.add(key)
            uniq.append(ln)
            if len(uniq) >= 12:
                break
        return "\n".join(f"- {ln}" for ln in uniq)


def load_llm_from_env() -> LLMBackend:
    import shutil

    backend = os.getenv("MAILMIND_LLM_BACKEND", "ollama").lower()
    model = os.getenv("MAILMIND_LLM_MODEL", "llama3.1:8b-instruct-q4_K_M")
    if backend == "ollama":
        if shutil.which("ollama") is not None:
            return OllamaBackend(model)
        # Fallback silently to extractive if ollama not available
        return SimpleExtractiveBackend()
    return SimpleExtractiveBackend()
