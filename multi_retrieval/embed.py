"""Dense vectors for the semantic route.

Two backends behind one protocol:

* ``SentenceTransformerEmbedder`` — the real one. A pretrained sentence encoder,
  held in memory. Permitted by the track rules: dense retrieval is explicitly in
  scope, only "heavy external industrial vector DB clusters" are excluded, and a
  pretrained encoder is not the "full-parameter fine-tuning" that is out of
  scope. 50,000 x 384 float32 is 77 MB resident and one query is a single
  matmul, which is comfortably "in-memory for light execution".

* ``HashingEmbedder`` — a dependency-free fallback so this package imports and
  its tests run on a machine with nothing installed. It is lexical underneath,
  so it cannot do the cross-category semantic matching the real encoder can.
  A fallback, never the default.

Catalog vectors are expensive to compute and never change, so they are cached to
disk and keyed by a fingerprint of the catalog file plus the model name. The
stored id order is verified before the cache is trusted.
"""

from __future__ import annotations

import hashlib
import json
import zlib
from pathlib import Path
from typing import Protocol

import numpy as np

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_CACHE = ".cache/multi_retrieval"
HASHING_DIMENSION = 512


class Embedder(Protocol):
    name: str
    dimension: int

    def encode(self, texts: list[str]) -> np.ndarray:
        """Return an (n, dimension) float32 array of L2-normalised rows."""
        ...


def _normalise(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return (matrix / norms).astype(np.float32)


class SentenceTransformerEmbedder:
    """Real semantic embeddings. Needs `pip install sentence-transformers`."""

    def __init__(self, model_name: str = DEFAULT_MODEL, *, batch_size: int = 256) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as error:                      # pragma: no cover
            raise ImportError(
                "sentence-transformers is not installed. Either "
                "`pip install -r requirements-multi-retrieval.txt`, or pass "
                "HashingEmbedder() to run without it."
            ) from error
        self._model = SentenceTransformer(model_name)
        self.name = model_name
        self.batch_size = batch_size
        # renamed in sentence-transformers 6; support both so the package works
        # across versions rather than emitting a deprecation warning per run
        getter = getattr(self._model, "get_embedding_dimension", None) or \
            self._model.get_sentence_embedding_dimension
        self.dimension = int(getter())

    def encode(self, texts: list[str]) -> np.ndarray:
        vectors = self._model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vectors.astype(np.float32)


class HashingEmbedder:
    """Deterministic bag-of-words hashing. No dependencies beyond numpy.

    Uses crc32 rather than the builtin ``hash``: Python randomises string
    hashing per process, which would make vectors differ between runs and
    silently invalidate every cache.
    """

    def __init__(self, dimension: int = HASHING_DIMENSION) -> None:
        self.name = f"hashing-{dimension}"
        self.dimension = dimension

    def encode(self, texts: list[str]) -> np.ndarray:
        matrix = np.zeros((len(texts), self.dimension), dtype=np.float32)
        for row, text in enumerate(texts):
            for word in text.lower().split():
                bucket = zlib.crc32(word.encode("utf-8")) % self.dimension
                matrix[row, bucket] += 1.0
        return _normalise(matrix)


class VectorStore:
    """Catalog vectors, computed once and cached on disk."""

    def __init__(
        self,
        embedder: Embedder,
        *,
        cache_dir: str | Path = DEFAULT_CACHE,
        catalog_path: str | Path | None = None,
    ) -> None:
        self.embedder = embedder
        self.cache_dir = Path(cache_dir)
        self.catalog_path = Path(catalog_path) if catalog_path else None
        self.matrix: np.ndarray | None = None
        self.loaded_from_cache = False

    # ------------------------------------------------------------------ cache

    def _fingerprint(self, ids: list[str]) -> str:
        parts = [self.embedder.name, str(self.embedder.dimension), str(len(ids))]
        if self.catalog_path and self.catalog_path.exists():
            stat = self.catalog_path.stat()
            parts += [str(stat.st_size), str(int(stat.st_mtime))]
        digest = hashlib.blake2b("|".join(parts).encode("utf-8"), digest_size=8).hexdigest()
        return digest

    def _paths(self, key: str) -> tuple[Path, Path]:
        return self.cache_dir / f"{key}.npy", self.cache_dir / f"{key}.json"

    def _load(self, key: str, ids: list[str]) -> np.ndarray | None:
        vectors_path, meta_path = self._paths(key)
        if not (vectors_path.exists() and meta_path.exists()):
            return None
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            # Verify the id ORDER, not just the count: a cache whose rows line up
            # with a different product order is worse than no cache at all.
            if meta.get("first") != ids[0] or meta.get("last") != ids[-1]:
                return None
            if meta.get("count") != len(ids):
                return None
            matrix = np.load(vectors_path)
        except (OSError, ValueError, json.JSONDecodeError, KeyError, IndexError):
            return None
        if matrix.shape[0] != len(ids):
            return None
        return matrix

    def _save(self, key: str, ids: list[str], matrix: np.ndarray) -> None:
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            vectors_path, meta_path = self._paths(key)
            np.save(vectors_path, matrix)
            meta_path.write_text(json.dumps({
                "model": self.embedder.name,
                "count": len(ids),
                "first": ids[0],
                "last": ids[-1],
            }) + "\n", encoding="utf-8")
        except OSError:
            pass                              # a cache we cannot write is not an error

    # ------------------------------------------------------------------ build

    def build(self, texts: list[str], ids: list[str]) -> np.ndarray:
        key = self._fingerprint(ids)
        cached = self._load(key, ids)
        if cached is not None:
            self.matrix = cached
            self.loaded_from_cache = True
            return cached
        matrix = self.embedder.encode(texts)
        self._save(key, ids, matrix)
        self.matrix = matrix
        self.loaded_from_cache = False
        return matrix

    def similarity(self, query: np.ndarray) -> np.ndarray:
        if self.matrix is None:
            raise RuntimeError("build() must run before similarity()")
        return self.matrix @ query


__all__ = [
    "Embedder", "SentenceTransformerEmbedder", "HashingEmbedder",
    "VectorStore", "DEFAULT_MODEL", "DEFAULT_CACHE",
]
