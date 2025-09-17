from __future__ import annotations

from typing import Iterable, List, Optional, Dict, Tuple
import os


def _select_device() -> str:
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


class Embedder:
    """Pluggable embedding backend.

    Tries Sentence-Transformers if available; otherwise falls back to a lightweight
    hash-based embedding (development-only) so the pipeline can run without network.
    """

    def __init__(self, model: str = "intfloat/multilingual-e5-small", dim: int = 256, backend: str = "auto") -> None:
        self.model_name = model
        self.dim = dim
        self._st = None
        self._tfm = None
        self._backend = backend
        # Reduce file descriptor pressure from HF tokenizers when processes fork
        try:
            os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        except Exception:
            pass
        try:
            if backend in ("auto", "st"):
                from sentence_transformers import SentenceTransformer  # type: ignore

                self._st = SentenceTransformer(model)
                # Get native dimension if available
                try:
                    native_dim = self._st.get_sentence_embedding_dimension()
                    if self.dim > native_dim:
                        self.dim = native_dim
                except Exception:
                    pass
        except Exception:
            self._st = None

        # Transformers backend (for models like EmbeddingGemma)
        if self._st is None or backend in ("auto", "transformers"):
            try:
                import torch  # type: ignore
                from transformers import AutoModel, AutoTokenizer  # type: ignore

                device = _select_device()
                tok = AutoTokenizer.from_pretrained(model, local_files_only=True)
                mod = AutoModel.from_pretrained(model, local_files_only=True)
                mod.eval()
                if device != "cpu":
                    mod.to(device)
                self._tfm = (tok, mod, device)
                # Derive dim from config if available
                try:
                    hidden = getattr(mod.config, "hidden_size", None) or getattr(mod.config, "d_model", None)
                    if isinstance(hidden, int) and self.dim > hidden:
                        self.dim = hidden
                except Exception:
                    pass
            except Exception:
                self._tfm = None

    def _normalize(self, vec: List[float]) -> List[float]:
        import math

        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def _hash_embed(self, texts: List[str]) -> List[List[float]]:
        # Very simple, deterministic hash projection into dim (dev fallback)
        out: List[List[float]] = []
        mod = self.dim
        for t in texts:
            v = [0.0] * mod
            for i, ch in enumerate(t):
                h = (ord(ch) * 1315423911 + i * 2654435761) % mod
                v[h] += 1.0
            out.append(self._normalize(v))
        return out

    def encode(self, texts: List[str], batch_size: int = 128) -> List[List[float]]:
        # Prefer ST when available and selected
        if self._st is not None and self._backend in ("auto", "st"):
            vecs = self._st.encode(
                texts,
                batch_size=batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            try:
                import numpy as np  # type: ignore

                if isinstance(vecs, np.ndarray):
                    vecs = vecs[:, : self.dim]
                    return [row.tolist() for row in vecs]
            except Exception:
                pass
            return [list(v)[: self.dim] for v in vecs]

        # Transformers fallback/explicit backend
        if self._tfm is not None:
            tok, mod, device = self._tfm
            import torch  # type: ignore

            all_vecs: List[List[float]] = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                enc = tok(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=1024,
                    return_tensors="pt",
                )
                if device != "cpu":
                    enc = {k: v.to(device) for k, v in enc.items()}  # type: ignore
                with torch.no_grad():
                    out = mod(**enc)
                    last = out.last_hidden_state  # (B, T, H)
                    mask = enc.get("attention_mask")
                    mask = mask.unsqueeze(-1).to(last.dtype)  # (B, T, 1)
                    summed = (last * mask).sum(dim=1)
                    counts = mask.sum(dim=1).clamp(min=1e-9)
                    embs = summed / counts
                    # Truncate dims
                    if embs.shape[1] > self.dim:
                        embs = embs[:, : self.dim]
                    # L2 normalize
                    embs = torch.nn.functional.normalize(embs, p=2, dim=1)
                    all_vecs.extend(embs.detach().cpu().tolist())
            return all_vecs

        # Deterministic development-only fallback
        return self._hash_embed(texts)


# Simple global cache to avoid repeatedly loading models (reduces open FDs)
_EMBEDDER_CACHE: Dict[Tuple[str, int, str], Embedder] = {}


def get_embedder(model: str = "intfloat/multilingual-e5-small", dim: int = 256, backend: str = "auto") -> Embedder:
    key: Tuple[str, int, str] = (model, int(dim), backend)
    emb = _EMBEDDER_CACHE.get(key)
    if emb is None:
        emb = Embedder(model=model, dim=dim, backend=backend)
        _EMBEDDER_CACHE[key] = emb
    return emb
