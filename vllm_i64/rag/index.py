"""
vllm_i64/rag/index.py
FAISS-based vector index.
Stores chunks + their vectors, persists to disk.
"""

from __future__ import annotations
import json
import numpy as np
from pathlib import Path


class VectorIndex:
    """
    Wraps FAISS for storing and searching chunk embeddings.

    Usage:
        idx = VectorIndex(dim=384)
        idx.add(chunks, vectors)
        idx.save("./rag_index")

        idx2 = VectorIndex.load("./rag_index")
        results = idx2.search(query_vector, k=3)
    """

    def __init__(self, dim: int) -> None:
        try:
            import faiss
        except ImportError:
            raise ImportError("pip install faiss-cpu")
        import faiss as _faiss
        self.dim = dim
        self.index = _faiss.IndexFlatL2(dim)
        self.chunks: list[str] = []

    def add(self, chunks: list[str], vectors: np.ndarray) -> None:
        """
        Add chunks and their embeddings to the index.
        vectors : (N, dim) float32
        """
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have same length")
        self.index.add(vectors.astype(np.float32))
        self.chunks.extend(chunks)

    def search(self, query_vector: np.ndarray, k: int = 3) -> list[dict]:
        """
        Find top-k closest chunks.
        query_vector : (1, dim) or (dim,) float32
        Returns list of {"chunk": str, "score": float}
        """
        if query_vector.ndim == 1:
            query_vector = query_vector[np.newaxis, :]
        distances, indices = self.index.search(query_vector.astype(np.float32), k)
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0:
                continue
            results.append({
                "chunk": self.chunks[idx],
                "score": float(dist),
            })
        return results

    def save(self, directory: str) -> None:
        import faiss
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path / "index.faiss"))
        with open(path / "chunks.json", "w", encoding="utf-8") as f:
            json.dump(self.chunks, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, directory: str) -> "VectorIndex":
        import faiss
        path = Path(directory)
        with open(path / "chunks.json", encoding="utf-8") as f:
            chunks = json.load(f)
        index = faiss.read_index(str(path / "index.faiss"))
        obj = cls.__new__(cls)
        obj.dim = index.d
        obj.index = index
        obj.chunks = chunks
        return obj
