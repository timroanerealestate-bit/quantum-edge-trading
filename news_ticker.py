"""
Market Pulse News Ticker
========================
Fetches high-impact global financial news for the scrolling ticker.

Primary source  : Finnhub general news API
Fallback source : Alpha Vantage NEWS_SENTIMENT endpoint

Scheduled refresh windows (ET):
  08:00 AM — Pre-market
  09:30 AM — Market Open
  12:00 PM — Midday
  04:30 PM — Market Close
  07:30 PM — Post-market

Keys are injected by dashboard.py before any call is made.
"""
from __future__ import annotations

import time
import requests
from datetime import datetime, date

# ── Keys injected by dashboard.py from st.secrets ────────────────────────────
FINNHUB_API_KEY: str = ""
AV_API_KEY:      str = ""

# ── High-impact keyword filter ────────────────────────────────────────────────
_HIGH_IMPACT: list[str] = [
    # Fed / macro policy
    "federal reserve", "fed ", "fomc", "rate hike", "rate cut", "interest rate",
    "inflation", " cpi", " pce", " gdp", "recession", "stagflation",
    "jobs report", "nonfarm payroll", "unemployment", "jobless claims",
    "consumer price", "producer price",
    # Fiscal / political
    "debt ceiling", "government shutdown", "congress", "treasury",
    "tariff", "trade war", "sanction", "stimulus", "budget",
    # Geopolitics / armed conflict
    "war ", "conflict", "military strike", "invasion", "airstrike",
    "russia", "ukraine", "china ", "taiwan", "iran ", "north korea",
    "strait of hormuz", "middle east", "oil supply disruption",
    # Commodities / energy
    "crude oil", "opec", "natural gas", "energy crisis", "supply shock",
    "brent", "wti", "gas prices",
    # Big-cap market movers
    "nvidia", "nvda", "apple ", "aapl", "microsoft", "msft", "meta ",
    "alphabet", "google", "amazon", "amzn", "tesla", "tsla",
    "earnings beat", "earnings miss", "revenue miss", "guidance cut", "guidance raise",
    "profit warning", "downgrade", "upgrade",
    # Systemic financial risk
    "bank failure", "credit crisis", "liquidity crunch", "default",
    "bankruptcy", "yield curve", "10-year yield", "bond market", "spread widens",
    # Crypto / regulation
    "bitcoin", "crypto crash", "sec charges", "regulation",
    # Market structure events
    "market crash", "circuit breaker", "flash crash", "volatility spike",
    "merger", "acquisition", "buyout", "ipo",
]


# ── Sentiment mapping ─────────────────────────────────────────────────────────

def _sentiment_emoji(score: float) -> str:
    """Map a -1..+1 sentiment score to a color-coded emoji."""
    if score > 0.12:
        return "🟢"
    if score < -0.12:
        return "🔴"
    return "⚪"


def _is_high_impact(title: str, summary: str = "") -> bool:
    text = (title + " " + summary).lower()
    return any(kw in text for kw in _HIGH_IMPACT)


# ── Refresh bucket ─────────────────────────────────────────────────────────────

def get_refresh_bucket() -> str:
    """
    Returns a stable string key that only changes at the five scheduled ET
    refresh windows.  Falls back to 5-minute epoch buckets if pytz is absent.
    """
    _WINDOWS = [(8, 0), (9, 30), (12, 0), (16, 30), (19, 30)]
    try:
        import pytz
        et  = pytz.timezone("America/New_York")
        now = datetime.now(et)
        ds  = now.strftime("%Y-%m-%d")
        for h, m in reversed(_WINDOWS):
            window_time = now.replace(hour=h, minute=m, second=0, microsecond=0)
            if now >= window_time:
                return f"{ds}_{h:02d}{m:02d}"
        return f"{ds}_premarket"
    except Exception:
        return str(int(time.time() // 300))   # 5-minute fallback buckets


# ── Public entry point ────────────────────────────────────────────────────────

def fetch_ticker_items(limit: int = 15) -> list[dict]:
    """
    Return up to `limit` high-impact news items.
    Each item: {"emoji": str, "headline": str, "source": str}
    Primary: Finnhub. Fallback: Alpha Vantage. Final: static placeholder.
    """
    items = _from_finnhub(limit)
    if len(items) < 3:
        items = items + _from_alphavantage(limit)

    # Deduplicate on first 60 chars of headline (case-insensitive)
    seen: set[str] = set()
    unique: list[dict] = []
    for it in items:
        key = it["headline"][:60].lower()
        if key not in seen:
            seen.add(key)
            unique.append(it)

    return unique[:12] if unique else _fallback_items()


# ── Finnhub source ────────────────────────────────────────────────────────────

def _from_finnhub(limit: int) -> list[dict]:
    if not FINNHUB_API_KEY:
        return []
    try:
        url  = (f"https://finnhub.io/api/v1/news"
                f"?category=general&token={FINNHUB_API_KEY}")
        data = requests.get(url, timeout=8).json()
        items: list[dict] = []
        for art in (data if isinstance(data, list) else []):
            title   = art.get("headline", "").strip()
            summary = art.get("summary", "")
            if not title or not _is_high_impact(title, summary):
                continue
            items.append({
                "emoji":    "⚪",   # Finnhub general news has no per-article score
                "headline": title[:140],
                "source":   art.get("source", "Finnhub"),
            })
        return items
    except Exception:
        return []


# ── Alpha Vantage fallback ────────────────────────────────────────────────────

def _from_alphavantage(limit: int) -> list[dict]:
    if not AV_API_KEY:
        return []
    try:
        url = (
            "https://www.alphavantage.co/query"
            "?function=NEWS_SENTIMENT"
            "&topics=financial_markets,economy_macro,economy_fiscal,monetary_policy"
            f"&sort=LATEST&limit=30&apikey={AV_API_KEY}"
        )
        data = requests.get(url, timeout=10).json()
        items: list[dict] = []
        for art in data.get("feed", []):
            title   = art.get("title", "").strip()
            summary = art.get("summary", "")
            if not title or not _is_high_impact(title, summary):
                continue
            score = float(art.get("overall_sentiment_score", 0))
            items.append({
                "emoji":    _sentiment_emoji(score),
                "headline": title[:140],
                "source":   art.get("source", "Alpha Vantage"),
            })
        return items
    except Exception:
        return []


# ── Static fallback ───────────────────────────────────────────────────────────

def _fallback_items() -> list[dict]:
    """Shown when both APIs return no high-impact headlines."""
    today = date.today().strftime("%B %d, %Y")
    return [
        {
            "emoji":    "⚪",
            "headline": f"Market session active — {today}. No major high-impact macro headlines detected at this time.",
            "source":   "Quantum Edge",
        },
        {
            "emoji":    "⚪",
            "headline": "Monitor the VIX level and sector rotation charts for intraday directional signals.",
            "source":   "Quantum Edge",
        },
        {
            "emoji":    "⚪",
            "headline": "Ask the AI Research Agent for live validated trade setups across small, mid, and large cap.",
            "source":   "Quantum Edge",
        },
    ]
