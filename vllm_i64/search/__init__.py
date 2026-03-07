"""
vllm-i64 :: Web Search (Perplexity-style)

Search-augmented generation: web search → scrape → cite → generate.

INL - 2025
"""

from .web_search import WebSearcher, SearchResult

__all__ = ["WebSearcher", "SearchResult"]
