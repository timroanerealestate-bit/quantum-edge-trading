"""
News & sentiment scoring — three-tier source priority:

  1. MarketAux (primary)   — entity-level scored sentiment + aggregated stats
  2. Alpha Vantage          — per-ticker scored sentiment (25 req/day limit)
  3. yfinance headlines     — keyword-based fallback, no quota

MarketAux gives the richest signal: article counts by bullish/bearish/neutral,
average sentiment score, and per-article entity sentiment all in one call.
"""
import yfinance as yf
from marketaux_client import (
    get_news, get_entity_sentiment, parse_entity_stats, parse_news_sentiments
)
from alpha_vantage_client import get_news_sentiment as av_news


# ── yfinance keyword fallback ───────────────────────────────────────────────
BULLISH_WORDS = [
    "surge", "soar", "rally", "beat", "record", "upgrade", "outperform",
    "buy", "bull", "gain", "rise", "profit", "growth", "strong", "positive",
    "breakout", "momentum", "opportunity", "boost", "expand", "jump",
]
BEARISH_WORDS = [
    "drop", "fall", "plunge", "miss", "downgrade", "underperform", "sell",
    "bear", "loss", "decline", "weak", "negative", "crash", "cut", "layoff",
    "lawsuit", "warning", "risk", "concern", "investigation", "tumble",
]


def _yf_sentiment(symbol: str) -> dict:
    """Keyword-based sentiment from yfinance headlines — last-resort fallback."""
    result = {
        "sentiment_score": 50, "label": "Neutral",
        "article_count": 0, "top_headlines": [],
        "source": "yfinance-keywords",
        "bullish_count": 0, "bearish_count": 0, "neutral_count": 0,
    }
    try:
        news = yf.Ticker(symbol).news or []
        headlines = [
            (a.get("content", {}) or {}).get("title", "") or a.get("title", "")
            for a in news[:20]
        ]
        headlines = [h for h in headlines if h]
        result["article_count"] = len(headlines)
        result["top_headlines"] = headlines[:5]

        bull = sum(sum(1 for w in BULLISH_WORDS if w in h.lower()) for h in headlines)
        bear = sum(sum(1 for w in BEARISH_WORDS if w in h.lower()) for h in headlines)
        total = bull + bear
        if total:
            ratio = bull / total
            result["sentiment_score"] = max(20, min(80, int(ratio * 100)))
            result["label"] = "Bullish" if ratio > 0.6 else ("Bearish" if ratio < 0.4 else "Neutral")
            result["bullish_count"] = bull
            result["bearish_count"] = bear
    except Exception as e:
        result["error"] = str(e)
    return result


def _marketaux_sentiment(symbol: str) -> dict | None:
    """
    Pull sentiment from MarketAux using both endpoints:
      - /entity/stats/aggregation → aggregate bull/bear/neutral counts + avg score
      - /news/all                 → per-article detail + headlines
    Returns None if MarketAux is unavailable or returns no data.
    """
    try:
        # ── Aggregated stats (primary signal) ──────────────────────────────
        agg_data   = get_entity_sentiment(symbol)
        agg_record = parse_entity_stats(agg_data, symbol)

        # ── Individual articles (for headlines + per-article scores) ────────
        news_data  = get_news(symbol)
        articles   = parse_news_sentiments(news_data, symbol)

        if not agg_record and not articles:
            return None

        # Parse aggregate record
        bullish_count  = int(agg_record.get("bullish_count",  0) or 0) if agg_record else 0
        bearish_count  = int(agg_record.get("bearish_count",  0) or 0) if agg_record else 0
        neutral_count  = int(agg_record.get("neutral_count",  0) or 0) if agg_record else 0
        article_count  = int(agg_record.get("article_count",  0) or 0) if agg_record else len(articles)
        sentiment_avg  = float(agg_record.get("sentiment_avg", 0) or 0) if agg_record else 0.0

        # If aggregation gives no useful data, compute from article list
        if article_count == 0 and articles:
            scores = [a["sentiment_score"] for a in articles]
            sentiment_avg = sum(scores) / len(scores) if scores else 0.0
            bullish_count = sum(1 for a in articles if a["sentiment_label"] == "positive")
            bearish_count = sum(1 for a in articles if a["sentiment_label"] == "negative")
            neutral_count = len(articles) - bullish_count - bearish_count
            article_count = len(articles)

        total_articles = bullish_count + bearish_count + neutral_count
        if total_articles == 0:
            total_articles = max(article_count, 1)

        # Convert avg sentiment [-1,+1] → 0–100
        normalized_score = int((sentiment_avg + 1) * 50)
        normalized_score = max(0, min(100, normalized_score))

        # Determine label from both avg score AND bull/bear ratio
        bull_ratio = bullish_count / total_articles if total_articles else 0.5
        if sentiment_avg > 0.15 or bull_ratio > 0.55:
            label = "Bullish"
        elif sentiment_avg < -0.15 or bull_ratio < 0.35:
            label = "Bearish"
        else:
            label = "Neutral"

        headlines = [a["title"] for a in articles if a["title"]][:5]

        return {
            "sentiment_score": normalized_score,
            "label":           label,
            "article_count":   article_count,
            "top_headlines":   headlines,
            "bullish_count":   bullish_count,
            "bearish_count":   bearish_count,
            "neutral_count":   neutral_count,
            "sentiment_avg":   round(sentiment_avg, 3),
            "source":          "marketaux",
        }

    except Exception:
        return None


def _av_sentiment(symbol: str) -> dict | None:
    """Alpha Vantage sentiment — secondary source."""
    try:
        data     = av_news(symbol)
        articles = data.get("feed", [])
        if not articles:
            return None

        ticker_scores, headlines = [], []
        bull = bear = neutral = 0
        for article in articles[:20]:
            title = article.get("title", "")
            if title:
                headlines.append(title)
            for ts in article.get("ticker_sentiment", []):
                if ts.get("ticker", "").upper() == symbol.upper():
                    val   = float(ts.get("ticker_sentiment_score", 0))
                    lbl   = ts.get("ticker_sentiment_label", "Neutral").lower()
                    ticker_scores.append(val)
                    if "bull" in lbl:   bull    += 1
                    elif "bear" in lbl: bear    += 1
                    else:               neutral += 1

        if not ticker_scores:
            return None

        avg  = sum(ticker_scores) / len(ticker_scores)
        norm = max(0, min(100, int((avg + 1) * 50)))
        return {
            "sentiment_score": norm,
            "label": "Bullish" if avg > 0.15 else ("Bearish" if avg < -0.15 else "Neutral"),
            "article_count": len(articles),
            "top_headlines": headlines[:5],
            "bullish_count": bull, "bearish_count": bear, "neutral_count": neutral,
            "sentiment_avg": round(avg, 3),
            "source": "alpha-vantage",
        }
    except Exception:
        return None


def score_sentiment(symbol: str) -> dict:
    """
    Score news sentiment for a symbol using the best available source.
    Priority: MarketAux → Alpha Vantage → yfinance keywords
    """
    default = {
        "sentiment_score": 50, "label": "Neutral",
        "article_count": 0, "top_headlines": [],
        "bullish_count": 0, "bearish_count": 0, "neutral_count": 0,
        "sentiment_avg": 0.0, "source": "none",
    }

    result = _marketaux_sentiment(symbol)
    if result:
        return result

    result = _av_sentiment(symbol)
    if result:
        return result

    return _yf_sentiment(symbol)
