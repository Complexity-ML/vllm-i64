"""
vllm_i64/rag/embedder.py
Swappable embedder interface.

Now  : sentence-transformers (float32)
Later: i64-embed (int64) — swap by changing RAG_EMBEDDER env var
"""

from __future__ import annotations
import os
import numpy as np


class Embedder:
    """Base interface — all embedders must implement encode()."""

    def encode(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError


class SentenceTransformerEmbedder(Embedder):
    """
    Float32 embedder using sentence-transformers.
    Default model : all-MiniLM-L6-v2 (384 dim, fast, good quality)
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError("pip install sentence-transformers")
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()

    def encode(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)


class I64Embedder(Embedder):
    """
    i64-embed native embedder — plug in once trained.
    Expects a DeepForEmbedding checkpoint.
    """

    def __init__(self, checkpoint: str, tokenizer_path: str, embed_dim: int = 256) -> None:
        import torch
        from transformers import PreTrainedTokenizerFast
        from complexity_deep.models.embed import DeepForEmbedding, EmbedConfig

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        cfg = EmbedConfig(embed_dim=embed_dim)
        self.model = DeepForEmbedding(cfg)
        state = torch.load(checkpoint, map_location="cpu")
        self.model.load_state_dict(state)
        self.model.eval().to(self.device)
        self.dim = embed_dim

    def encode(self, texts: list[str]) -> np.ndarray:
        import torch
        enc = self.tokenizer(
            texts, padding=True, truncation=True,
            max_length=128, return_tensors="pt"
        )
        input_ids     = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)
        with torch.no_grad():
            emb = self.model(input_ids, attention_mask)       # float32
            emb_i64 = self.model.quantize(emb).cpu().numpy()  # int64
        return emb_i64


def get_embedder() -> Embedder:
    """
    Factory — controlled by RAG_EMBEDDER env var.
    RAG_EMBEDDER=sentence_transformers  (default)
    RAG_EMBEDDER=i64                    (once trained)
    """
    backend = os.getenv("RAG_EMBEDDER", "sentence_transformers")
    if backend == "i64":
        checkpoint    = os.environ["I64_EMBED_CHECKPOINT"]
        tokenizer     = os.environ.get("I64_TOKENIZER", "./tokenizer/tokenizer.json")
        embed_dim     = int(os.environ.get("I64_EMBED_DIM", "256"))
        return I64Embedder(checkpoint, tokenizer, embed_dim)
    return SentenceTransformerEmbedder()
