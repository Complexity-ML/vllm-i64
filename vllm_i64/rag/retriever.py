"""
vllm_i64/rag/retriever.py
High-level RAG interface.

Usage:
    # Build index from docs
    retriever = Retriever()
    retriever.index_file("./docs/dossier_eic.pdf")
    retriever.index_file("./docs/pacific_prime_paper.pdf")
    retriever.save("./rag_index")

    # Load and query
    retriever = Retriever.load("./rag_index")
    context = retriever.get_context("C'est quoi le budget WP2 ?")

    # Inject into Pacific Chat
    prompt = f"Context:\n{context}\n\nQuestion: C'est quoi le budget WP2 ?"
"""

from __future__ import annotations
import logging
from pathlib import Path

from .chunker import chunk_file, chunk_text
from .embedder import get_embedder, Embedder
from .index import VectorIndex

log = logging.getLogger(__name__)


class Retriever:

    def __init__(self, embedder: Embedder | None = None) -> None:
        self.embedder = embedder or get_embedder()
        self.vector_index: VectorIndex | None = None

    # ── Indexing ─────────────────────────────────────────────────────────────

    def index_file(
        self,
        path: str,
        chunk_size: int = 200,
        overlap: int = 50,
    ) -> int:
        """Index a .pdf or .txt file. Returns number of chunks added."""
        log.info("Indexing %s ...", path)
        chunks = chunk_file(path, chunk_size=chunk_size, overlap=overlap)
        return self._add_chunks(chunks)

    def index_text(
        self,
        text: str,
        chunk_size: int = 200,
        overlap: int = 50,
    ) -> int:
        """Index raw text. Returns number of chunks added."""
        chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
        return self._add_chunks(chunks)

    def _add_chunks(self, chunks: list[str]) -> int:
        vectors = self.embedder.encode(chunks)
        if self.vector_index is None:
            self.vector_index = VectorIndex(dim=vectors.shape[1])
        self.vector_index.add(chunks, vectors)
        log.info("Added %d chunks (total: %d)", len(chunks), len(self.vector_index.chunks))
        return len(chunks)

    # ── Retrieval ────────────────────────────────────────────────────────────

    def retrieve(self, query: str, k: int = 3) -> list[dict]:
        """
        Return top-k relevant chunks for a query.
        Each result: {"chunk": str, "score": float}
        """
        if self.vector_index is None:
            raise RuntimeError("Index is empty — call index_file() first")
        q_vec = self.embedder.encode([query])
        return self.vector_index.search(q_vec, k=k)

    def get_context(self, query: str, k: int = 3, separator: str = "\n---\n") -> str:
        """
        Retrieve top-k chunks and join them into a single context string.
        Ready to inject into Pacific Chat prompt.
        """
        results = self.retrieve(query, k=k)
        return separator.join(r["chunk"] for r in results)

    # ── Persistence ──────────────────────────────────────────────────────────

    def save(self, directory: str) -> None:
        if self.vector_index is None:
            raise RuntimeError("Nothing to save — index is empty")
        self.vector_index.save(directory)
        log.info("Index saved to %s", directory)

    @classmethod
    def load(cls, directory: str, embedder: Embedder | None = None) -> "Retriever":
        obj = cls(embedder=embedder)
        obj.vector_index = VectorIndex.load(directory)
        log.info("Index loaded from %s (%d chunks)", directory, len(obj.vector_index.chunks))
        return obj
