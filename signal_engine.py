"""
Signal engine — combines institutional flow, technicals, and sentiment
into a single composite score and trade recommendation.

Institutional data now sourced from yfinance (institutional_holders + major_holders).
No SEC EDGAR parsing required — yfinance aggregates the same 13F data automatically.
"""
import yfinance as yf
from config import SIGNAL_WEIGHTS, MIN_SCORE_TO_REPORT
from technical_analysis import analyze_symbol, get_current_quote
from news_sentiment import score_sentiment


def _clamp(val, lo=0, hi=100):
    return max(lo, min(hi, val))


def _read_major(major, index_key: str) -> float | None:
    """
    Read one value from yfinance major_holders.
    Actual format (confirmed from debug):
        Single column 'Value' (float64), labels are in the DataFrame INDEX.
        e.g. index = ['insidersPercentHeld', 'institutionsPercentHeld', ...]
    Values are fractions (0.652 → 65.2%).
    """
    try:
        if "Value" in major.columns and index_key in major.index:
            v = float(major.loc[index_key, "Value"])
            return v * 100.0 if v <= 1.5 else v
    except Exception:
        pass
    return None


# Well-known institutional names that signal "smart money" conviction
MARQUEE_INSTITUTIONS = [
    "vanguard", "blackrock", "state street", "fidelity", "berkshire",
    "citadel", "point72", "renaissance", "two sigma", "millennium",
    "bridgewater", "tiger global", "coatue", "viking", "de shaw",
    "jpmorgan", "goldman sachs", "morgan stanley", "t. rowe", "capital group",
]


def score_institutional_flow(symbol: str) -> tuple[float, list[str]]:
    """
    Score institutional positioning using yfinance's institutional_holders
    and major_holders data (sourced from the same 13F filings, pre-aggregated).

    Returns (score 0–100, detail strings).

    Scoring breakdown (max 100):
      • Institutional ownership %     → up to 50 pts
      • Marquee name holders          → up to 30 pts
      • Net buying flow (pctChange)   → up to 15 pts
      • Insider ownership bonus       →      5 pts
    """
    details = []
    score = 0.0

    try:
        ticker = yf.Ticker(symbol)
        major  = ticker.major_holders
        inst_df = ticker.institutional_holders

        # ── 1. Institutional ownership % ─────────────────────────────────────
        inst_pct   = _read_major(major, "institutionsPercentHeld")
        inst_count = _read_major(major, "institutionsCount")

        if inst_pct is not None:
            if inst_pct >= 80:
                score += 50
                details.append(f"Institutional ownership: {inst_pct:.1f}% — very high conviction")
            elif inst_pct >= 65:
                score += 38
                details.append(f"Institutional ownership: {inst_pct:.1f}% — strong")
            elif inst_pct >= 50:
                score += 25
                details.append(f"Institutional ownership: {inst_pct:.1f}% — moderate")
            elif inst_pct >= 30:
                score += 12
                details.append(f"Institutional ownership: {inst_pct:.1f}% — low")
            else:
                score += 3
                details.append(f"Institutional ownership: {inst_pct:.1f}% — minimal")

            if inst_count is not None:
                details.append(f"Total institutions holding: {int(inst_count):,}")
        else:
            details.append("Institutional ownership % unavailable")

        # ── 2. Marquee name holders ───────────────────────────────────────────
        if inst_df is not None and not inst_df.empty and "Holder" in inst_df.columns:
            top_holders = inst_df["Holder"].head(15).tolist()
            marquee_found = [
                h for h in top_holders
                if any(m in h.lower() for m in MARQUEE_INSTITUTIONS)
            ]
            if marquee_found:
                bonus = min(30, len(marquee_found) * 8)
                score += bonus
                details.append(f"Marquee holders ({len(marquee_found)}): {', '.join(marquee_found[:4])}")
            else:
                details.append(f"Top holders: {', '.join(h[:28] for h in top_holders[:3])}")

            # ── 3. Net buying / selling flow from pctChange ─────────────────
            if "pctChange" in inst_df.columns:
                recent = inst_df.head(20)
                buyers  = int((recent["pctChange"] > 0.01).sum())   # >1% increase
                sellers = int((recent["pctChange"] < -0.01).sum())  # >1% decrease
                new_pos = int((recent["pctChange"] > 0.99).sum())   # new positions

                if new_pos > 0:
                    score += 10
                    details.append(f"{new_pos} new institutional positions opened")

                if buyers > sellers:
                    flow_score = min(15, (buyers - sellers) * 3)
                    score += flow_score
                    details.append(f"Net institutional buying: {buyers} increasing vs {sellers} reducing")
                elif sellers > buyers:
                    details.append(f"Net institutional selling: {sellers} reducing vs {buyers} increasing")
                else:
                    details.append(f"Institutional flow mixed ({buyers} buying / {sellers} selling)")
        else:
            details.append("Holder detail list unavailable")

        # ── 4. Insider ownership (skin-in-the-game bonus) ────────────────────
        insider_pct = _read_major(major, "insidersPercentHeld")
        if insider_pct is not None:
            if insider_pct >= 10:
                score += 5
                details.append(f"Insider ownership: {insider_pct:.1f}% (strong alignment)")
            elif insider_pct >= 3:
                score += 3
                details.append(f"Insider ownership: {insider_pct:.1f}%")

    except Exception as e:
        details.append(f"Institutional data error: {e}")
        return 0.0, details

    return _clamp(score), details


def score_volume_surge(volume_ratio: float | None) -> tuple[float, list[str]]:
    if volume_ratio is None:
        return 50.0, ["Volume data unavailable"]
    if volume_ratio >= 3.0:
        return 100.0, [f"Extreme volume surge: {volume_ratio:.1f}x average"]
    if volume_ratio >= 2.0:
        return 80.0, [f"High volume: {volume_ratio:.1f}x average"]
    if volume_ratio >= 1.5:
        return 60.0, [f"Above-average volume: {volume_ratio:.1f}x"]
    if volume_ratio >= 1.0:
        return 40.0, [f"Normal volume: {volume_ratio:.1f}x"]
    return 20.0, [f"Low volume: {volume_ratio:.1f}x average"]


def _convergence_bonus(inst_score: float, sentiment: dict, tech: dict) -> tuple[float, list[str]]:
    """
    Award a bonus when multiple independent signals agree (confluence).
    Convergence = institutional + sentiment + technicals all pointing the same way.
    Max bonus: +8 pts (added to composite before clamping).
    """
    bonus = 0.0
    notes = []

    sent_label    = sentiment.get("label", "Neutral")
    trend         = tech.get("trend_direction", "neutral")
    macd_signal   = tech.get("macd_signal", "neutral")
    inst_bullish  = inst_score >= 55        # meaningful institutional presence
    sent_bullish  = sent_label == "Bullish"
    sent_bearish  = sent_label == "Bearish"
    tech_bullish  = trend == "bullish" and macd_signal in ("bullish", "strengthening", "crossing_up")
    tech_bearish  = trend == "bearish" and macd_signal == "bearish"

    # Full convergence — all three sources agree bullish
    if inst_bullish and sent_bullish and tech_bullish:
        bonus += 8
        notes.append("⭐ Full convergence: institutional + sentiment + technicals all bullish")
    # Two-way convergence bonuses
    elif inst_bullish and sent_bullish:
        bonus += 5
        notes.append("Institutional + sentiment convergence (bullish)")
    elif sent_bullish and tech_bullish:
        bonus += 4
        notes.append("Sentiment + technical convergence (bullish)")
    elif inst_bullish and tech_bullish:
        bonus += 4
        notes.append("Institutional + technical convergence (bullish)")
    # Bearish divergence warning (sentiment bearish but inst holding strong)
    elif sent_bearish and inst_bullish:
        notes.append("⚠ Divergence: institutions holding but sentiment bearish — monitor closely")

    return bonus, notes


def build_composite_score(symbol: str) -> dict:
    """
    Run all signal modules for a symbol and return a composite scored result.
    """
    print(f"\n  Analyzing {symbol}...")

    # --- Technical analysis (yfinance + ta library) ---
    tech = analyze_symbol(symbol)

    # --- News & sentiment (MarketAux → AV → yfinance fallback) ---
    sentiment = score_sentiment(symbol)

    # --- Institutional flow (yfinance institutional_holders + major_holders) ---
    inst_score, inst_details = score_institutional_flow(symbol)

    # --- Volume score ---
    vol_score, vol_details = score_volume_surge(tech.get("volume_ratio"))

    # --- Base composite weighted score ---
    w = SIGNAL_WEIGHTS
    composite = (
        inst_score           * w["institutional_flow"]
        + tech["tech_score"] * w["technical_momentum"]
        + vol_score          * w["volume_surge"]
        + sentiment["sentiment_score"] * w["news_sentiment"]
    )

    # --- Convergence bonus ---
    conv_bonus, conv_notes = _convergence_bonus(inst_score, sentiment, tech)
    composite = _clamp(composite + conv_bonus)

    # --- Current quote (yfinance) ---
    price, change_pct = get_current_quote(symbol)

    # --- Recommendation ---
    if composite >= 75:
        recommendation = "STRONG BUY"
    elif composite >= 60:
        recommendation = "BUY"
    elif composite >= 45:
        recommendation = "WATCH"
    elif composite >= 30:
        recommendation = "NEUTRAL"
    else:
        recommendation = "AVOID"

    return {
        "symbol":           symbol,
        "price":            price,
        "change_pct":       change_pct,
        "composite_score":  round(composite, 1),
        "recommendation":   recommendation,
        "convergence_notes": conv_notes,
        "scores": {
            "institutional": round(inst_score, 1),
            "technical":     tech["tech_score"],
            "volume":        round(vol_score, 1),
            "sentiment":     sentiment["sentiment_score"],
        },
        "tech":         tech,
        "sentiment":    sentiment,
        "inst_details": inst_details,
        "vol_details":  vol_details,
    }


def run_scan(watchlist: list[str]) -> list[dict]:
    """
    Score every symbol in the watchlist, return sorted by composite score.
    Results below MIN_SCORE_TO_REPORT are hidden unless nothing qualifies.
    """
    results = []
    for symbol in watchlist:
        try:
            results.append(build_composite_score(symbol))
        except Exception as e:
            print(f"  Error analyzing {symbol}: {e}")

    results.sort(key=lambda x: x["composite_score"], reverse=True)
    filtered = [r for r in results if r["composite_score"] >= MIN_SCORE_TO_REPORT]
    return filtered if filtered else results
