"""
vllm-i64 :: Web Search

Brave Search API client for Perplexity-style search-augmented generation.
Fetches web results (snippets) and formats them as numbered sources
for citation injection into the model prompt.

Usage:
    searcher = WebSearcher(api_key="BSA...")
    results = await searcher.search("What is token routing in MoE?")
    context = searcher.format_context(results)
    # → "[1] Token routing assigns... (source: arxiv.org)\n[2] ..."

INL - 2025
"""

import os
import re
import logging
from dataclasses import dataclass
from typing import List, Optional
from urllib.parse import urlparse

import aiohttp

logger = logging.getLogger("vllm_i64.search")

# ── Data ──────────────────────────────────────────────────────────────────────

@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str
    favicon: str = ""

    @property
    def domain(self) -> str:
        return urlparse(self.url).netloc


# ── Brave Search API ──────────────────────────────────────────────────────────

BRAVE_SEARCH_URL = "https://api.search.brave.com/res/v1/web/search"


class WebSearcher:
    """
    Async web searcher using Brave Search API (snippet-only, no scraping).

    Environment variable: BRAVE_SEARCH_API_KEY
    Free tier: 2,000 queries/month.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        max_results: int = 5,
    ):
        self.api_key = api_key or os.environ.get("BRAVE_SEARCH_API_KEY", "")
        self.max_results = max_results

        if not self.api_key:
            logger.warning("No BRAVE_SEARCH_API_KEY set — web search will be unavailable")

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    async def search(self, query: str, count: Optional[int] = None) -> List[SearchResult]:
        """
        Search the web via Brave Search API.
        Returns list of SearchResult with title, url, snippet.
        """
        if not self.api_key:
            raise RuntimeError("BRAVE_SEARCH_API_KEY not configured")

        count = count or self.max_results
        headers = {
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "X-Subscription-Token": self.api_key,
        }
        params = {
            "q": query,
            "count": str(count),
        }

        async with aiohttp.ClientSession() as session:
            async with session.get(
                BRAVE_SEARCH_URL, headers=headers, params=params, timeout=aiohttp.ClientTimeout(total=10)
            ) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    logger.error("Brave Search error %d: %s", resp.status, text[:200])
                    raise RuntimeError(f"Brave Search API error: {resp.status}")
                data = await resp.json()

        web_results = data.get("web", {}).get("results", [])
        results = []
        for item in web_results[:count]:
            results.append(SearchResult(
                title=item.get("title", ""),
                url=item.get("url", ""),
                snippet=_clean_html(item.get("description", "")),
                favicon=item.get("profile", {}).get("img", ""),
            ))

        logger.info("Web search '%s' → %d results", query, len(results))
        return results

    def format_context(self, results: List[SearchResult], max_chars: int = 3000) -> str:
        """
        Format search results as numbered sources for prompt injection.

        Returns:
            "[1] Title — domain.com\nSnippet text...\n\n[2] ..."
        """
        if not results:
            return ""
        parts = []
        total = 0
        for i, r in enumerate(results, 1):
            entry = f"[{i}] {r.title} — {r.domain}\n{r.snippet}"
            if total + len(entry) > max_chars:
                break
            parts.append(entry)
            total += len(entry)
        return "\n\n".join(parts)

    def format_sources(self, results: List[SearchResult]) -> List[dict]:
        """
        Format sources as structured data for the API response.

        Returns list of:
            {"index": 1, "title": "...", "url": "...", "domain": "...", "favicon": "..."}
        """
        return [
            {
                "index": i,
                "title": r.title,
                "url": r.url,
                "domain": r.domain,
                "favicon": r.favicon,
            }
            for i, r in enumerate(results, 1)
        ]


# ── HTML utilities ────────────────────────────────────────────────────────────

def _clean_html(text: str) -> str:
    """Remove HTML tags and decode entities from Brave snippets."""
    text = re.sub(r"<[^>]+>", "", text)
    text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    text = text.replace("&quot;", '"').replace("&#39;", "'").replace("&nbsp;", " ")
    return text.strip()
