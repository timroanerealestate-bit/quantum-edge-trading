import os

# ── Local .env support (optional — not available on Streamlit Cloud) ───────────
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ── Streamlit Cloud: pull secrets into os.environ ─────────────────────────────
def _load_cloud_secrets() -> None:
    """If running on Streamlit Cloud, copy st.secrets → os.environ."""
    try:
        import streamlit as st
        _keys = [
            "GROQ_API_KEY", "ALPHA_VANTAGE_API_KEY",
            "MARKETAUX_API_TOKEN", "ANTHROPIC_API_KEY",
        ]
        for k in _keys:
            if k in st.secrets and not os.getenv(k):
                os.environ[k] = str(st.secrets[k])
    except Exception:
        pass

_load_cloud_secrets()

ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY", "")
ALPHA_VANTAGE_BASE_URL = "https://www.alphavantage.co/query"

# MarketAux — primary news & sentiment source
MARKETAUX_API_TOKEN   = os.getenv("MARKETAUX_API_TOKEN", "")
MARKETAUX_BASE_URL    = "https://api.marketaux.com/v1"
MARKETAUX_NEWS_HOURS  = 48   # look back window for news (hours)
MARKETAUX_ARTICLE_LIMIT = 10  # articles per symbol per call (free tier friendly)

# SEC EDGAR
SEC_EDGAR_BASE_URL = "https://data.sec.gov"
SEC_HEADERS = {"User-Agent": "TrainingBot research@trainingbot.local"}

# Alpha Vantage free tier: 25 req/day — fallback sentiment only

# Watchlist — symbols to scan by default (50 stocks across all cap sizes)
DEFAULT_WATCHLIST = [
    # Mega-cap tech
    "AAPL", "MSFT", "NVDA", "GOOGL", "META", "AMZN", "TSLA", "AVGO",
    # Financials
    "JPM", "GS", "BAC", "MS", "V", "MA", "BRK-B", "BLK",
    # Healthcare / Biotech
    "UNH", "LLY", "JNJ", "ABBV", "MRK", "AMGN", "GILD", "MRNA",
    # Energy
    "XOM", "CVX", "OXY", "SLB",
    # Consumer / Retail
    "WMT", "COST", "TGT", "HD", "NKE", "SBUX",
    # Industrials / Defence
    "CAT", "DE", "LMT", "RTX",
    # Semiconductors
    "AMD", "INTC", "MU", "QCOM", "ARM",
    # Cloud / Software
    "CRM", "ORCL", "NOW", "SNOW", "PLTR",
    # ETFs (market pulse)
    "SPY", "QQQ", "IWM",
]

# Signal weights (must sum to 1.0)
# Sentiment weight raised now that MarketAux gives richer entity-level data
SIGNAL_WEIGHTS = {
    "institutional_flow": 0.32,
    "technical_momentum": 0.28,
    "volume_surge":       0.18,
    "news_sentiment":     0.22,   # increased: MarketAux gives real scored sentiment
}

# Thresholds
MIN_SCORE_TO_REPORT = 40   # out of 100 (55 is too strict for stable large-caps)
VOLUME_SURGE_MULTIPLIER = 1.5  # current vol vs 20-day avg
RSI_OVERSOLD = 35
RSI_OVERBOUGHT = 70
