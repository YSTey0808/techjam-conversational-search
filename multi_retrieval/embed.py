"""Dense vectors for the semantic route.

The catalog side is precomputed offline with ``nomic-ai/nomic-embed-text-v1.5``
(768-dim, 2048-token context) and shipped as ``data/embeddings/v2_nomic.npy`` --
50,000 x 768 L2-normalised float32, row-aligned with the catalog. Encoding it
live takes ~40 minutes, so it is never rebuilt; ``VectorStore.load_precomputed``
adopts it directly.

Only the query is embedded at request time. Backends behind one protocol:

* ``NomicEmbedder`` — the real one. Wraps the same nomic model the catalog
  vectors were built with, and always applies the ``search_query:`` task prefix
  the model expects (omitting it silently halves relevance). Needs
  ``sentence-transformers`` + ``einops``.

* ``HashingEmbedder`` — a dependency-free fallback so this package imports and
  its tests run on a machine with nothing installed. Lexical underneath, so it
  cannot do the cross-category semantic matching the real encoder can, and it
  is dimension-incompatible with the precomputed matrix. A fallback for the
  live-build path only, never used against ``v2_nomic.npy``.

* ``SentenceTransformerEmbedder`` — a generic wrapper kept for
  ``scripts/score_multi_retrieval.py``'s ``--vector sentence-transformers``.

Live-built catalog vectors are cached to disk keyed by a fingerprint of the
catalog file plus the model name; the stored id order is verified before the
cache is trusted.
"""

from __future__ import annotations

import hashlib
import json
import zlib
from pathlib import Path
from typing import Protocol

import numpy as np

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
NOMIC_MODEL = "nomic-ai/nomic-embed-text-v1.5"
NOMIC_QUERY_PREFIX = "search_query: "
DEFAULT_EMBEDDINGS_DIR = "data/embeddings"
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


class NomicEmbedder:
    """Query encoder matching the precomputed nomic-embed-text-v1.5 catalog.

    Only ever encodes the query, so it unconditionally prepends the
    ``search_query:`` prefix. The catalog vectors used ``search_document:`` when
    they were built; the two prefixes are what make asymmetric retrieval work.
    """

    def __init__(self, model_name: str = NOMIC_MODEL, *, device: str | None = None) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as error:                      # pragma: no cover
            raise ImportError(
                "sentence-transformers (and einops) are needed for the nomic "
                "query encoder. `pip install -r requirements-multi-retrieval.txt`, "
                "or run without the vector route."
            ) from error
        self._model = SentenceTransformer(model_name, trust_remote_code=True, device=device)
        self.name = model_name
        getter = getattr(self._model, "get_embedding_dimension", None) or \
            self._model.get_sentence_embedding_dimension
        self.dimension = int(getter())

    def encode(self, texts: list[str]) -> np.ndarray:
        prefixed = [NOMIC_QUERY_PREFIX + text for text in texts]
        vectors = self._model.encode(
            prefixed,
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

    def load_precomputed(
        self,
        matrix_path: str | Path,
        ids_path: str | Path,
        index_ids: list[str],
    ) -> np.ndarray:
        """Adopt a catalog matrix built offline instead of encoding one now.

        The stored ids are matched against the index order and the rows are
        permuted to agree; a stored id set that does not cover the catalog is
        an error, not a silent partial match.
        """
        matrix = np.load(matrix_path)
        stored = json.loads(Path(ids_path).read_text(encoding="utf-8"))
        if matrix.shape[0] != len(stored):
            raise RuntimeError(
                f"{Path(matrix_path).name}: {matrix.shape[0]} rows but "
                f"{Path(ids_path).name} lists {len(stored)} ids"
            )
        if stored != list(index_ids):
            position = {asin: row for row, asin in enumerate(stored)}
            try:
                order = [position[asin] for asin in index_ids]
            except KeyError as error:
                raise RuntimeError(
                    f"precomputed embeddings do not cover product "
                    f"{error.args[0]!r}"
                ) from error
            matrix = matrix[order]
        self.matrix = np.ascontiguousarray(matrix, dtype=np.float32)
        self.loaded_from_cache = True
        return self.matrix

    def similarity(self, query: np.ndarray) -> np.ndarray:
        if self.matrix is None:
            raise RuntimeError("build() must run before similarity()")
        return self.matrix @ query


__all__ = [
    "Embedder", "NomicEmbedder", "SentenceTransformerEmbedder", "HashingEmbedder",
    "VectorStore", "DEFAULT_MODEL", "NOMIC_MODEL", "DEFAULT_EMBEDDINGS_DIR",
    "DEFAULT_CACHE",
]
