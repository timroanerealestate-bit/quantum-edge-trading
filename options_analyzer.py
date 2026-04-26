"""
Options chain analyzer using yfinance.
Identifies best call/put opportunities based on current signals.
"""
import yfinance as yf
import pandas as pd


def _safe_price(ticker: yf.Ticker) -> float | None:
    try:
        return ticker.fast_info.last_price
    except Exception:
        return None


def find_best_calls(symbol: str, composite_score: float, current_price: float | None = None) -> list[dict]:
    """
    Find the best call options for a bullish trade idea.
    Returns list of opportunities sorted by volume (highest liquidity first).
    """
    ideas = []
    try:
        ticker = yf.Ticker(symbol)
        expirations = ticker.options
        if not expirations:
            return []

        price = current_price or _safe_price(ticker)
        if not price:
            return []

        # Look at nearest 3 expirations
        for exp in expirations[:3]:
            try:
                chain = ticker.option_chain(exp)
                calls = chain.calls
            except Exception:
                continue

            if calls.empty:
                continue

            # Near-the-money to slightly OTM calls (95% – 112% of price)
            mask = (calls["strike"] >= price * 0.95) & (calls["strike"] <= price * 1.12)
            atm = calls[mask].copy()

            if atm.empty:
                continue

            for _, row in atm.iterrows():
                ask = float(row.get("ask", row.get("lastPrice", 0)) or 0)
                if ask <= 0:
                    continue

                strike = float(row["strike"])
                iv = float(row.get("impliedVolatility", 0) or 0)
                oi = int(row.get("openInterest", 0) or 0)
                vol = int(row.get("volume", 0) or 0)

                ideas.append({
                    "symbol":               symbol,
                    "type":                 "CALL",
                    "strike":               strike,
                    "expiration":           exp,
                    "ask":                  round(ask, 2),
                    "iv_pct":               round(iv * 100, 1) if iv else None,
                    "open_interest":        oi,
                    "volume":               vol,
                    "upside_to_strike_pct": round((strike - price) / price * 100, 1),
                    "breakeven":            round(strike + ask, 2),
                    "current_price":        round(price, 2),
                    "composite_score":      composite_score,
                })

        ideas.sort(key=lambda x: x["volume"], reverse=True)
        return ideas[:6]

    except Exception:
        return []


def find_best_puts(symbol: str, composite_score: float, current_price: float | None = None) -> list[dict]:
    """
    Find the best put options for a bearish/hedge trade.
    Returns list of opportunities sorted by volume.
    """
    ideas = []
    try:
        ticker = yf.Ticker(symbol)
        expirations = ticker.options
        if not expirations:
            return []

        price = current_price or _safe_price(ticker)
        if not price:
            return []

        for exp in expirations[:3]:
            try:
                chain = ticker.option_chain(exp)
                puts = chain.puts
            except Exception:
                continue

            if puts.empty:
                continue

            # ATM to slightly OTM puts (88% – 103% of price)
            mask = (puts["strike"] >= price * 0.88) & (puts["strike"] <= price * 1.03)
            atm = puts[mask].copy()

            if atm.empty:
                continue

            for _, row in atm.iterrows():
                ask = float(row.get("ask", row.get("lastPrice", 0)) or 0)
                if ask <= 0:
                    continue

                strike = float(row["strike"])
                iv = float(row.get("impliedVolatility", 0) or 0)
                oi = int(row.get("openInterest", 0) or 0)
                vol = int(row.get("volume", 0) or 0)

                ideas.append({
                    "symbol":                   symbol,
                    "type":                     "PUT",
                    "strike":                   strike,
                    "expiration":               exp,
                    "ask":                      round(ask, 2),
                    "iv_pct":                   round(iv * 100, 1) if iv else None,
                    "open_interest":            oi,
                    "volume":                   vol,
                    "downside_protection_pct":  round((price - strike) / price * 100, 1),
                    "breakeven":                round(strike - ask, 2),
                    "current_price":            round(price, 2),
                    "composite_score":          composite_score,
                })

        ideas.sort(key=lambda x: x["volume"], reverse=True)
        return ideas[:6]

    except Exception:
        return []


def summarize_options(symbol: str, composite_score: float) -> dict:
    """
    Full options summary for a symbol based on composite score.
    Returns calls (bullish bias) and/or puts (bearish/hedge bias).
    """
    try:
        ticker = yf.Ticker(symbol)
        price = _safe_price(ticker)

        bias = "BULLISH" if composite_score >= 60 else ("BEARISH" if composite_score <= 40 else "NEUTRAL")

        calls = find_best_calls(symbol, composite_score, price) if composite_score >= 45 else []
        puts  = find_best_puts(symbol, composite_score, price) if composite_score <= 60 else []

        return {
            "symbol":          symbol,
            "current_price":   round(price, 2) if price else None,
            "bias":            bias,
            "composite_score": composite_score,
            "calls":           calls,
            "puts":            puts,
        }

    except Exception as e:
        return {"symbol": symbol, "error": str(e), "calls": [], "puts": []}
