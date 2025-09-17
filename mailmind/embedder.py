from __future__ import annotations

from typing import Iterable, List, Optional


class Embedder:
    """Pluggable embedding backend.

    Tries Sentence-Transformers if available; otherwise falls back to a lightweight
    hash-based embedding (development-only) so the pipeline can run without network.
    """

    def __init__(self, model: str = "google/embeddinggemma-300m", dim: int = 256) -> None:
        self.model_name = model
        self.dim = dim
        self._st = None
        try:
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
        if self._st is None:
            return self._hash_embed(texts)
        # Use ST; slice to dim to emulate MRL truncation when needed
        vecs = self._st.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        # vecs could be numpy arrays; convert to lists and truncate
        try:
            import numpy as np  # type: ignore

            if isinstance(vecs, np.ndarray):
                vecs = vecs[:, : self.dim]
                return [row.tolist() for row in vecs]
        except Exception:
            pass
        # Fallback generic slice
        return [list(v)[: self.dim] for v in vecs]

