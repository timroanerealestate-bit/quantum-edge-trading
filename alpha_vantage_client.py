"""
Alpha Vantage API wrapper — used ONLY for News & Sentiment.
All price/technical data now comes from yfinance (free, unlimited).
Free tier: 25 calls/day. At 1 call/symbol for sentiment, supports 25 symbols/day.
"""
import time
import requests
from config import ALPHA_VANTAGE_API_KEY, ALPHA_VANTAGE_BASE_URL

# Only 1 AV call per symbol now (news sentiment), so 3s gap is plenty safe
AV_DELAY_SECONDS = 3

_last_call_time = 0.0


def _get(params: dict) -> dict:
    """Rate-limited GET wrapper."""
    global _last_call_time
    elapsed = time.time() - _last_call_time
    if elapsed < AV_DELAY_SECONDS:
        time.sleep(AV_DELAY_SECONDS - elapsed)

    params["apikey"] = ALPHA_VANTAGE_API_KEY
    resp = requests.get(ALPHA_VANTAGE_BASE_URL, params=params, timeout=15)
    resp.raise_for_status()
    _last_call_time = time.time()
    data = resp.json()

    if "Error Message" in data:
        raise ValueError(f"Alpha Vantage error: {data['Error Message']}")
    if "Note" in data:
        raise RuntimeError(f"Alpha Vantage rate limit hit: {data['Note']}")
    if "Information" in data:
        raise RuntimeError(f"Alpha Vantage limit: {data['Information']}")
    return data


def get_daily_adjusted(symbol: str, outputsize: str = "compact") -> dict:
    """Daily OHLCV + adjusted close for a symbol."""
    return _get({
        "function": "TIME_SERIES_DAILY",
        "symbol": symbol,
        "outputsize": outputsize,
    })


def get_quote(symbol: str) -> dict:
    """Real-time quote snapshot."""
    data = _get({"function": "GLOBAL_QUOTE", "symbol": symbol})
    return data.get("Global Quote", {})


def get_rsi(symbol: str, interval: str = "daily", period: int = 14) -> dict:
    return _get({
        "function": "RSI",
        "symbol": symbol,
        "interval": interval,
        "time_period": period,
        "series_type": "close",
    })


def get_macd(symbol: str, interval: str = "daily") -> dict:
    return _get({
        "function": "MACD",
        "symbol": symbol,
        "interval": interval,
        "series_type": "close",
    })


def get_bbands(symbol: str, interval: str = "daily", period: int = 20) -> dict:
    return _get({
        "function": "BBANDS",
        "symbol": symbol,
        "interval": interval,
        "time_period": period,
        "series_type": "close",
    })


def get_adx(symbol: str, interval: str = "daily", period: int = 14) -> dict:
    return _get({
        "function": "ADX",
        "symbol": symbol,
        "interval": interval,
        "time_period": period,
    })


def get_obv(symbol: str, interval: str = "daily") -> dict:
    return _get({
        "function": "OBV",
        "symbol": symbol,
        "interval": interval,
    })


def get_news_sentiment(symbol: str, limit: int = 50) -> dict:
    """News & sentiment from Alpha Vantage News API."""
    return _get({
        "function": "NEWS_SENTIMENT",
        "tickers": symbol,
        "limit": limit,
        "sort": "LATEST",
    })


def get_overview(symbol: str) -> dict:
    """Company fundamentals overview."""
    return _get({"function": "OVERVIEW", "symbol": symbol})


def get_earnings(symbol: str) -> dict:
    """Quarterly and annual earnings."""
    return _get({"function": "EARNINGS", "symbol": symbol})
