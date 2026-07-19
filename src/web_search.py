"""
web_search.py
-------------
Lightweight web search wrapper used to pull in recent sports information
that may not yet be in the vector database (e.g. a tournament that just
concluded). Uses the `duckduckgo-search` package, which needs no API key,
so the project can run out of the box.

If you'd rather use a paid, higher-quality search API (SerpAPI, Bing, Tavily,
Google Custom Search, etc.), swap the implementation of `search_web` below --
the rest of the app only depends on its return shape:
    List[{"title": str, "snippet": str, "url": str}]
"""

from typing import List, Dict

from duckduckgo_search import DDGS


def search_web(query: str, max_results: int = 5) -> List[Dict]:
    """Run a web search and return a list of lightweight result dicts."""
    results = []
    try:
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                results.append(
                    {
                        "title": r.get("title", ""),
                        "snippet": r.get("body", ""),
                        "url": r.get("href", ""),
                    }
                )
    except Exception as e:
        # Network issues / rate limits shouldn't crash quiz generation --
        # the agent will just fall back to ChromaDB-only context.
        print(f"[web_search] search failed: {e}")
    return results


def results_to_context(results: List[Dict]) -> str:
    """Format search results into plain text suitable for prompt injection."""
    lines = []
    for r in results:
        if r["snippet"]:
            lines.append(f"- {r['snippet']} (source: {r['url']})")
    return "\n".join(lines)


if __name__ == "__main__":
    res = search_web("latest cricket world cup winner")
    print(results_to_context(res))
