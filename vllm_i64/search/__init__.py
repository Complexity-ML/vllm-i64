"""
vllm-i64 :: Web Search (Perplexity-style)

Search-augmented generation: web search → cite → generate.
Partitioned history with token-routed isolation per API key.

INL - 2025
"""

from .web_search import WebSearcher, SearchResult
from .history import PartitionedSearchHistory

__all__ = ["WebSearcher", "SearchResult", "PartitionedSearchHistory"]
