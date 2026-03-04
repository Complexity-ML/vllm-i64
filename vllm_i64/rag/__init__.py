"""
vllm-i64 :: RAG (Retrieval-Augmented Generation)

Native RAG pipeline integrated into the inference server.
Chunking, embedding, FAISS indexing, and context retrieval.

INL - 2025
"""

from .retriever import Retriever
from .embedder import get_embedder, SentenceTransformerEmbedder, I64Embedder
from .chunker import chunk_text, chunk_file
from .index import VectorIndex

__all__ = [
    "Retriever",
    "get_embedder",
    "SentenceTransformerEmbedder",
    "I64Embedder",
    "chunk_text",
    "chunk_file",
    "VectorIndex",
]
