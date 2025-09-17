from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Tuple


class HnswIndex:
    """Thin wrapper around hnswlib, optional at runtime.

    Uses L2 space with normalized vectors (cosine equivalent). Persists to a single
    .bin file next to a labels.txt for mapping.
    """

    def __init__(self, dim: int, path: Path, ef: int = 200, M: int = 32) -> None:
        self.dim = dim
        self.path = path
        self._index = None
        self._labels: List[int] = []
        try:
            import hnswlib  # type: ignore

            self._hnswlib = hnswlib
        except Exception:
            self._hnswlib = None

    def available(self) -> bool:
        return self._hnswlib is not None

    def _labels_path(self) -> Path:
        return self.path.with_suffix(".labels.txt")

    def init(self, capacity: int = 10000) -> None:
        if not self.available():
            raise RuntimeError("hnswlib not installed. Install it to use ANN search.")
        p = self.path
        p.parent.mkdir(parents=True, exist_ok=True)
        import hnswlib  # type: ignore

        self._index = hnswlib.Index(space="l2", dim=self.dim)
        self._index.init_index(max_elements=capacity, ef_construction=200, M=32)
        self._index.set_ef(200)

    def load(self) -> None:
        if not self.available():
            raise RuntimeError("hnswlib not installed.")
        import hnswlib  # type: ignore

        self._index = hnswlib.Index(space="l2", dim=self.dim)
        self._index.load_index(str(self.path))
        # Load labels
        self._labels = []
        lp = self._labels_path()
        if lp.exists():
            with lp.open("r", encoding="utf-8") as f:
                for line in f:
                    self._labels.append(int(line.strip()))

    def current_count(self) -> int:
        if self._index is None:
            return 0
        try:
            return int(self._index.get_current_count())  # type: ignore[attr-defined]
        except Exception:
            # Fallback to labels length if available
            return len(self._labels)

    def max_elements(self) -> int:
        if self._index is None:
            return 0
        try:
            return int(self._index.get_max_elements())  # type: ignore[attr-defined]
        except Exception:
            # Unknown
            return 0

    def ensure_capacity(self, needed_extra: int) -> None:
        if self._index is None:
            # Initialize with a generous capacity if not present
            self.init(capacity=max(needed_extra * 2, 10000))
            return
        try:
            cur = self.current_count()
            mx = self.max_elements()
            if mx and (cur + needed_extra) > mx:
                target = max(cur + needed_extra, int(mx * 2))
                try:
                    self._index.resize_index(target)  # type: ignore[attr-defined]
                except Exception:
                    # As a last resort, re-init a fresh index (labels lost). Caller may decide to skip ANN if this fails.
                    self.init(capacity=target)
        except Exception:
            # Silent best-effort
            pass

    def save(self) -> None:
        if self._index is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._index.save_index(str(self.path))
        # Save labels mapping
        with self._labels_path().open("w", encoding="utf-8") as f:
            for lab in self._labels:
                f.write(f"{lab}\n")

    def add(self, vectors: List[List[float]], labels: List[int]) -> None:
        if self._index is None:
            self.init(capacity=max(len(labels) * 2, 10000))
        # Ensure enough room
        self.ensure_capacity(len(labels))
        self._index.add_items(vectors, labels)  # type: ignore[attr-defined]
        self._labels.extend(labels)

    def search(self, query_vecs: List[List[float]], k: int = 10) -> Tuple[List[List[int]], List[List[float]]]:
        if self._index is None:
            self.load()
        labels, dists = self._index.knn_query(query_vecs, k=k)  # type: ignore[attr-defined]
        return labels.tolist(), dists.tolist()  # type: ignore[return-value]
