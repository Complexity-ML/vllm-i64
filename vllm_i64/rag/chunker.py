"""
vllm_i64/rag/chunker.py
Split raw text into overlapping chunks.
"""

from __future__ import annotations


def chunk_text(
    text: str,
    chunk_size: int = 200,
    overlap: int = 50,
) -> list[str]:
    """
    Split text into word-based chunks with overlap.

    chunk_size : number of words per chunk
    overlap    : number of words shared between consecutive chunks
    """
    words = text.split()
    chunks = []
    step = chunk_size - overlap

    for i in range(0, len(words), step):
        chunk = " ".join(words[i : i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)

    return chunks


def chunk_file(path: str, chunk_size: int = 200, overlap: int = 50) -> list[str]:
    """
    Load a .txt or .pdf file and return chunks.
    Requires PyMuPDF (fitz) for PDFs.
    """
    if path.endswith(".pdf"):
        try:
            import fitz
        except ImportError:
            raise ImportError("pip install pymupdf")
        doc = fitz.open(path)
        text = "\n".join(page.get_text() for page in doc)
    else:
        with open(path, encoding="utf-8") as f:
            text = f.read()

    return chunk_text(text, chunk_size=chunk_size, overlap=overlap)
