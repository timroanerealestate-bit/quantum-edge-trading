"""
Technical analysis engine — uses yfinance (free, unlimited) for all price data
and the `ta` library to calculate indicators locally.
Zero Alpha Vantage calls — preserves the 25/day quota for news sentiment only.
"""
import yfinance as yf
import pandas as pd

try:
    import ta
    HAS_TA = True
except ImportError:
    HAS_TA = False

from config import RSI_OVERSOLD, RSI_OVERBOUGHT, VOLUME_SURGE_MULTIPLIER


def analyze_symbol(symbol: str) -> dict:
    """
    Download OHLCV via yfinance, compute all indicators locally, return scored dict.
    """
    result = {
        "symbol": symbol,
        "rsi": None,
        "macd_signal": None,
        "bb_position": None,
        "adx": None,
        "volume_ratio": None,
        "trend_direction": "neutral",
        "tech_score": 0,
        "details": [],
    }
    score = 0
    details = []

    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period="6mo")

        if df.empty or len(df) < 30:
            details.append(f"Insufficient price history ({len(df)} bars)")
            result["details"] = details
            return result

        closes = df["Close"]
        volumes = df["Volume"]
        highs = df["High"]
        lows = df["Low"]
        current_close = float(closes.iloc[-1])

        # ── Volume ──────────────────────────────────────────────────────────
        current_vol = float(volumes.iloc[-1])
        avg_vol = float(volumes.iloc[-21:-1].mean())
        volume_ratio = current_vol / avg_vol if avg_vol > 0 else 1.0
        result["volume_ratio"] = round(volume_ratio, 2)

        if volume_ratio >= VOLUME_SURGE_MULTIPLIER * 2:
            score += 20
            details.append(f"Volume surge {volume_ratio:.1f}x avg (strong)")
        elif volume_ratio >= VOLUME_SURGE_MULTIPLIER:
            score += 12
            details.append(f"Volume elevated {volume_ratio:.1f}x avg")
        else:
            details.append(f"Volume {volume_ratio:.1f}x avg (normal)")

        # ── SMA Trend ────────────────────────────────────────────────────────
        sma20 = float(closes.iloc[-20:].mean())
        sma50 = float(closes.iloc[-50:].mean()) if len(closes) >= 50 else sma20

        if current_close > sma20 * 1.02:
            result["trend_direction"] = "bullish"
            score += 10
            details.append(f"Price above 20-SMA by {((current_close/sma20)-1)*100:.1f}%")
        elif current_close < sma20 * 0.98:
            result["trend_direction"] = "bearish"
            details.append(f"Price below 20-SMA by {((1-current_close/sma20))*100:.1f}%")

        # Golden/death cross bonus
        if sma20 > sma50 * 1.01:
            score += 5
            details.append("20-SMA above 50-SMA (golden cross region)")
        elif sma20 < sma50 * 0.99:
            details.append("20-SMA below 50-SMA (death cross region)")

        # ── RSI ──────────────────────────────────────────────────────────────
        if HAS_TA:
            rsi_series = ta.momentum.RSIIndicator(closes, window=14).rsi()
            rsi = float(rsi_series.iloc[-1])
            result["rsi"] = round(rsi, 1)

            if rsi <= RSI_OVERSOLD:
                score += 20
                details.append(f"RSI oversold at {rsi:.1f} (potential bounce)")
            elif rsi >= RSI_OVERBOUGHT:
                score -= 5
                details.append(f"RSI overbought at {rsi:.1f} (caution)")
            elif 40 <= rsi <= 60:
                score += 8
                details.append(f"RSI neutral at {rsi:.1f}")
            else:
                score += 12
                details.append(f"RSI bullish momentum at {rsi:.1f}")
        else:
            details.append("RSI: ta library not installed")

        # ── MACD ─────────────────────────────────────────────────────────────
        if HAS_TA:
            macd_ind = ta.trend.MACD(closes, window_slow=26, window_fast=12, window_sign=9)
            macd_line = macd_ind.macd()
            signal_line = macd_ind.macd_signal()
            hist = macd_ind.macd_diff()

            hist_clean = hist.dropna()
            if len(hist_clean) >= 2:
                hist_curr = float(hist_clean.iloc[-1])
                hist_prev = float(hist_clean.iloc[-2])
                macd_val = float(macd_line.iloc[-1])
                sig_val = float(signal_line.iloc[-1])

                if macd_val > sig_val and hist_curr > 0:
                    result["macd_signal"] = "bullish"
                    score += 15
                    details.append(f"MACD bullish (line {macd_val:+.2f} above signal)")
                elif hist_curr > hist_prev and hist_prev < 0:
                    result["macd_signal"] = "crossing_up"
                    score += 20
                    details.append("MACD histogram crossing up (momentum shift)")
                elif hist_curr > 0 and hist_curr > hist_prev:
                    result["macd_signal"] = "strengthening"
                    score += 10
                    details.append("MACD momentum strengthening")
                elif macd_val < sig_val:
                    result["macd_signal"] = "bearish"
                    details.append(f"MACD bearish (line {macd_val:+.2f} below signal)")
                else:
                    result["macd_signal"] = "neutral"
                    score += 5
                    details.append("MACD neutral")

        # ── Bollinger Bands ───────────────────────────────────────────────────
        if HAS_TA:
            bb = ta.volatility.BollingerBands(closes, window=20, window_dev=2)
            upper = float(bb.bollinger_hband().iloc[-1])
            lower = float(bb.bollinger_lband().iloc[-1])
            band_width = upper - lower
            position = (current_close - lower) / band_width if band_width > 0 else 0.5
            result["bb_position"] = round(position, 2)

            if position < 0.2:
                score += 18
                details.append(f"Price near lower Bollinger Band ({position:.0%}) — oversold zone")
            elif position > 0.8:
                score -= 5
                details.append(f"Price near upper Bollinger Band ({position:.0%}) — overbought zone")
            else:
                score += 5
                details.append(f"Price mid-band at {position:.0%}")

        # ── ADX (trend strength) ──────────────────────────────────────────────
        if HAS_TA and len(df) >= 28:
            adx_ind = ta.trend.ADXIndicator(highs, lows, closes, window=14)
            adx = float(adx_ind.adx().iloc[-1])
            result["adx"] = round(adx, 1)

            if adx >= 30:
                score += 10
                details.append(f"Strong trend confirmed (ADX={adx:.1f})")
            elif adx >= 20:
                score += 5
                details.append(f"Developing trend (ADX={adx:.1f})")
            else:
                details.append(f"Weak/no trend (ADX={adx:.1f})")

    except Exception as e:
        details.append(f"Analysis error: {e}")

    result["tech_score"] = max(0, min(100, score))
    result["details"] = details
    return result


def get_current_quote(symbol: str) -> tuple[str, str]:
    """Return (price_str, change_pct_str) via yfinance — no AV calls."""
    try:
        info = yf.Ticker(symbol).fast_info
        price = info.last_price
        prev = info.previous_close
        if price and prev:
            pct = (price - prev) / prev * 100
            return f"{price:.2f}", f"{pct:+.4f}%"
        return f"{price:.2f}" if price else "N/A", "N/A"
    except Exception:
        return "N/A", "N/A"
