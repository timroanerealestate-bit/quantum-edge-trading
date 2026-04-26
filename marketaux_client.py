"""
MarketAux API client.
Endpoints used:
  /v1/news/all                    — latest news articles with per-entity sentiment
  /v1/entity/stats/aggregation   — aggregated bullish/bearish counts per symbol

Docs: https://www.marketaux.com/documentation
Free tier: 100 requests/day, no per-minute limit.
"""
import time
import requests
from datetime import datetime, timedelta, timezone
from config import MARKETAUX_API_TOKEN, MARKETAUX_BASE_URL, MARKETAUX_NEWS_HOURS, MARKETAUX_ARTICLE_LIMIT

_last_call: float = 0.0
_MIN_DELAY = 1.2   # gentle throttle — 1 call/second max


def _get(endpoint: str, params: dict) -> dict:
    """Rate-limited GET to MarketAux API."""
    global _last_call
    elapsed = time.time() - _last_call
    if elapsed < _MIN_DELAY:
        time.sleep(_MIN_DELAY - elapsed)

    params["api_token"] = MARKETAUX_API_TOKEN
    url = f"{MARKETAUX_BASE_URL}{endpoint}"
    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    _last_call = time.time()

    data = resp.json()
    if "error" in data:
        raise ValueError(f"MarketAux error: {data['error']}")
    return data


def _published_after(hours: int = MARKETAUX_NEWS_HOURS) -> str:
    """ISO-8601 timestamp for N hours ago (UTC), as MarketAux expects."""
    dt = datetime.now(timezone.utc) - timedelta(hours=hours)
    return dt.strftime("%Y-%m-%dT%H:%M")


def get_news(symbol: str, limit: int = MARKETAUX_ARTICLE_LIMIT) -> dict:
    """
    Fetch latest news articles for a symbol with entity-level sentiment scores.
    Each article has an 'entities' list — find the matching symbol to get its
    individual sentiment_score and sentiment_label.
    """
    return _get("/news/all", {
        "symbols":        symbol,
        "filter_entities": "true",
        "limit":          limit,
        "published_after": _published_after(),
        "sort":           "published_at",
        "sort_order":     "desc",
        "language":       "en",
    })


def get_entity_sentiment(symbol: str, limit: int = MARKETAUX_ARTICLE_LIMIT) -> dict:
    """
    Fetch pre-aggregated sentiment stats for a symbol.
    Returns bullish_count, bearish_count, neutral_count, sentiment_avg,
    article_count, and overall sentiment_label for the look-back window.
    """
    return _get("/entity/stats/aggregation", {
        "symbols":        symbol,
        "group_by":       "symbol",
        "limit":          limit,
        "published_after": _published_after(),
    })


def parse_entity_stats(data: dict, symbol: str) -> dict | None:
    """
    Extract the sentiment record for a specific symbol from the aggregation response.
    Returns a dict or None if not found.
    """
    for item in data.get("data", []):
        sym = (
            item.get("symbol")
            or item.get("ticker")
            or item.get("entity_symbol", "")
        )
        if sym.upper() == symbol.upper():
            return item
    # If group_by=symbol returned just one record, return it directly
    records = data.get("data", [])
    if len(records) == 1:
        return records[0]
    return None


def parse_news_sentiments(data: dict, symbol: str) -> list[dict]:
    """
    From a /news/all response, extract per-article sentiment entries
    that match the requested symbol. Returns list of:
    {title, published_at, sentiment_score, sentiment_label, url}
    """
    results = []
    for article in data.get("data", []):
        title        = article.get("title", "")
        published_at = article.get("published_at", "")
        url          = article.get("url", "")
        entities     = article.get("entities", [])

        matched_score = None
        matched_label = None

        for ent in entities:
            if ent.get("symbol", "").upper() == symbol.upper():
                matched_score = ent.get("sentiment_score")
                matched_label = ent.get("sentiment", "neutral")
                break

        # If no per-entity match, use article-level sentiment if present
        if matched_score is None:
            matched_score = article.get("sentiment_score", 0.0)
            matched_label = article.get("sentiment", "neutral")

        results.append({
            "title":           title,
            "published_at":    published_at,
            "sentiment_score": float(matched_score) if matched_score is not None else 0.0,
            "sentiment_label": matched_label or "neutral",
            "url":             url,
        })

    return results
