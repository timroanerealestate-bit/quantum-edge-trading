"""
AI Research Agent — Multi-layer validated trade discovery.
Uses Groq llama-3.3-70b-versatile for final recommendations.

Validation pipeline (runs for EVERY question):
  L1  Price momentum, volume surge, RSI setup            (yfinance batch)
  L2a Unusual options flow — sweeps, blocks, vol/OI      (yfinance per-stock)
  L2b IV rank / percentile — avoid expensive contracts   (yfinance per-stock)
  L2c Technical patterns — engulfing, breakout, flag     (derived from OHLCV)
  L3  News sentiment                                      (MarketAux API)
  L4  Institutional / analyst data                       (Alpha Vantage API, rate-limited)
  L5  Market regime — VIX level adjusts conviction       (passed in)

Confidence scoring (0-100):
  HIGH   ≥ 70  — all or most layers confirm, recommend
  MEDIUM 50-69 — solid setup, most layers agree, recommend with caution
  LOW    < 50  — mixed signals, excluded from recommendations
"""
from __future__ import annotations
import os
import time
import requests
import yfinance as yf
import pandas as pd

# ── API keys — injected by dashboard.py at startup ────────────────────────────
# Primary source: module-level vars set by dashboard.py from st.secrets.
# Each function also checks os.environ as a final fallback.
GROQ_API_KEY     = ""
AV_API_KEY       = ""
MA_API_TOKEN     = ""
FINNHUB_API_KEY  = ""   # fallback for MarketAux news sentiment
GEMINI_API_KEY   = ""   # silent fallback when Groq is unavailable

try:
    from groq import Groq as _Groq
    _GROQ_INSTALLED = True
except ImportError:
    _Groq = None
    _GROQ_INSTALLED = False

GROQ_MODEL       = "llama-3.3-70b-versatile"   # default / long-question model
GROQ_MODEL_FAST  = "llama-3.1-8b-instant"      # short-question model (≤12 words)

def _pick_groq_model(question: str) -> str:
    """Route to the fast model for short questions, full model for complex ones."""
    return GROQ_MODEL_FAST if len(question.split()) <= 12 else GROQ_MODEL

HAS_GROQ = _GROQ_INSTALLED  # True when package is installed; key injected at runtime

# ── Grok (xAI) — secondary validation + consolidation layer ──────────────────
# Key injected by dashboard.py from st.secrets["GROK_API_KEY"].
# Uses the OpenAI-compatible client pointing to api.x.ai.
DEEPSEEK_API_KEY  = ""   # injected by dashboard.py; DeepSeek final validation layer


# ── Stock universe ────────────────────────────────────────────────────────────
SMALL_CAP = [
    "PLUG","BLNK","CHPT","EVGO","GEVO","LAZR","MVIS","SNDX","MGNI","PERI",
    "CRCT","ENVX","ACLS","XENE","ARQT","CLBT","IRTC","TTGT","ARWR","BMRN",
    "ICAD","CERE","ATAI","TENB","DOMO","BIGC","CLOV","HIMS","LPSN","NKLA",
]
MID_CAP = [
    "RIVN","SNAP","ROKU","DKNG","HOOD","SOFI","AFRM","UPST","JOBY","ACHR",
    "RKT","NU","OPEN","LMND","COUR","PTON","SMAR","APPN","BRZE","CFLT",
    "GTLB","HUBS","MNDY","ZS","BILL","DOCN","ESTC","FROG","JAMF","KVYO",
]
LARGE_CAP = [
    "NVDA","AAPL","MSFT","META","GOOGL","AMZN","TSLA","AMD","NFLX","COIN",
    "PLTR","SMCI","ARM","MSTR","MELI","SE","RBLX","SHOP","UBER","ABNB",
    "PYPL","SQ","INTC","MU","QCOM","AVGO","ORCL","CRM","SNOW","DDOG",
]

_UNIVERSE = {"small": SMALL_CAP, "mid": MID_CAP, "large": LARGE_CAP}

# ── API caches (module-level, TTL-based) ──────────────────────────────────────
_AV_CACHE:  dict[str, tuple[float, dict]] = {}
_MA_CACHE:  dict[str, tuple[float, dict]] = {}
_OPT_CACHE: dict[str, tuple[float, dict]] = {}

AV_CACHE_TTL = 6 * 3600   # 6 h  — AV fundamentals rarely change intraday

def _opt_ttl() -> int:
    """Options flow cache: 3 min when market open, 20 min otherwise."""
    try:
        from market_data import is_market_open
        return 3 * 60 if is_market_open() else 20 * 60
    except Exception:
        return 10 * 60

def _ma_ttl() -> int:
    """News sentiment cache: 8 min when market open, 30 min otherwise."""
    try:
        from market_data import is_market_open
        return 8 * 60 if is_market_open() else 30 * 60
    except Exception:
        return 15 * 60


# ═══════════════════════════════════════════════════════════════════════════════
# SYSTEM PROMPTS
# ═══════════════════════════════════════════════════════════════════════════════
SYSTEM_PROMPT = """You are an elite AI Trading Research Agent inside the Quantum Edge Trading platform.

You ALWAYS validate ideas through multiple data layers before recommending anything.
The user-provided data already includes multi-layer validation scores — use them.

════════════════════════════════════════════
CRITICAL — REAL-TIME OPTIONS DATA ONLY
════════════════════════════════════════════
The context you receive contains a "LIVE OPTIONS (REAL DATA)" block for each symbol.
These blocks show the ACTUAL contracts trading RIGHT NOW from live market data.

✅ YOU MUST use ONLY these values for strikes, expiry dates, and premiums.
✅ The expiry dates listed are the real upcoming expirations — use ONLY those.
✅ Write the FULL exact date in every card: "May 9, 2026" NOT "May 2026".
✅ The ask prices listed are live market prices — use those as your premium estimates.
❌ NEVER write any expiry date from 2024 or 2025 — those contracts are expired.
❌ NEVER invent a strike price, expiry date, or premium not present in the data.
❌ If no LIVE OPTIONS block appears for a symbol, state that chain data was unavailable
   and skip that symbol — do not fabricate numbers.

Today's date context: You are producing analysis for live trading in 2026.
All expiry dates must be in 2026 or later. Any 2024/2025 date is an error.

════════════════════════════════════════════
VALIDATION LAYERS (always present in context)
════════════════════════════════════════════
L1  Momentum + Volume surge   (price action, RSI setup)
L2a Unusual Options Flow      (sweeps, blocks, vol/OI ratio)
L2b IV Rank/Percentile        (avoid expensive contracts > 60 IV rank)
L2c Technical Patterns        (engulfing, breakout, flag, SMA cross)
L3  News Sentiment            (MarketAux — bullish/bearish/neutral)
L4  Institutional Data        (Alpha Vantage — analyst targets, ownership)
L5  Market Regime             (VIX level — low/normal/elevated/fear)

════════════════════════════════════════════
CONFIDENCE LEVELS
════════════════════════════════════════════
HIGH   (≥70%) — Most layers confirm. Recommend with conviction.
MEDIUM (50-69%) — Solid setup, most layers agree. Recommend with caution.
LOW    (<50%) — Mixed signals. Do NOT recommend.

ONLY recommend HIGH and MEDIUM confidence trades.

════════════════════════════════════════════
OPTIONS PLAYS — REQUIRED OUTPUT FORMAT
════════════════════════════════════════════
When asked for options plays (calls, puts, weeklies, day trades, etc.):
Return EXACTLY 6 trades: 2 SMALL CAP + 2 MID CAP + 2 LARGE CAP.
Separate each trade card with a --- divider line.

Use this EXACT card format for every trade. Preserve all emoji, labels, and table structure:

---

### 🎯 [SYMBOL] — [CALL/PUT] — [Small/Mid/Large Cap]
**Confidence: [HIGH/MEDIUM] ([XX]%)** &nbsp;&nbsp;&nbsp; **Risk: [LOW / MEDIUM / HIGH]**

| | |
|---|---|
| **Current Price** | $[XX.XX] |
| **Strike** | $[XX.XX] ([XX]% OTM) |
| **Expiry** | [Month DD, YYYY — exact date e.g. May 9, 2026] |
| **Est. Premium** | $[X.XX] – $[X.XX] per share &nbsp;·&nbsp; $[XXX] – $[XXX] per contract |
| **Upside at Target** | 🟢 +[XXX]% |
| **Max Loss** | 🔴 100% of premium paid |

**Why This Trade**
- 📈 [L1] [Momentum or volume detail — always include the vol ratio and RSI value]
- 🔥 [L2] [Options flow or chart pattern — include IV rank and whether options are cheap/fair/expensive]
- 📰 [L3/L4] [Sentiment or institutional signal — include actual score or analyst target]
- ⭐ [4th bullet only if a genuinely strong extra signal exists — omit if not]

**What Could Go Wrong**
[2 sentences max. Be specific and honest. Name the exact scenario that kills this trade — e.g. earnings miss, sector rotation, IV crush, macro shock. Do NOT write generic boilerplate like "the stock could go down."]

**Plain English Action**
[One sentence. Tell the user exactly what to buy, at what strike, for which EXACT expiry date, and roughly what it costs. Example: "Buy the May 9, 2026 $180 Call on Meta (META) through any broker (Fidelity, Schwab, Robinhood) — expect to pay around $4.50 per share, or $450 per contract."]

---

════════════════════════════════════════════
LEAP OPTIONS — REQUIRED FORMAT
════════════════════════════════════════════
When asked for LEAP options (6–18 month expiry):
Stricter criteria — strong fundamentals PLUS 2–3 near-certain catalysts.
Return 3 picks: 1 small + 1 mid + 1 large cap.

FUNDAMENTALS required for every LEAP:
  • Revenue growth YoY > 10%  •  Expanding or stable margins
  • Low debt (D/E < 1.5 preferred)  •  Rising institutional ownership
  • Analyst price target meaningfully above current price

CATALYSTS required — identify 2–3 per pick:
  Types: earnings beats, product launches, FDA/regulatory approvals,
         index inclusion, macro tailwinds, sector rotation, M&A potential
  Conviction: explain WHY each catalyst is near-certain and falls BEFORE expiry

Use this EXACT format for every LEAP:

---

### 🚀 [SYMBOL] — LEAP CALL — [Small/Mid/Large Cap]
**Confidence: [HIGH/MEDIUM] ([XX]%)** &nbsp;&nbsp;&nbsp; **Risk: [LOW / MEDIUM / HIGH]**

| | |
|---|---|
| **Current Price** | $[XX.XX] |
| **Strike** | $[XX.XX] ([XX]% OTM) |
| **Expiry** | [Month DD, YYYY — exact date, 6–18 months out e.g. January 16, 2028] |
| **Est. Premium** | $[X.XX] – $[X.XX] per share &nbsp;·&nbsp; $[XXX] – $[XXX] per contract |
| **Target Price at Expiry** | $[XX.XX] |
| **Upside at Target** | 🟢 +[XXX]% |
| **Max Loss** | 🔴 100% of premium paid |

**Fundamental Case**
- Revenue: [specific growth metric with YoY %]
- Margins: [trend — expanding/stable/contracting]
- Debt: [D/E ratio or specific commentary]
- Institutions: [ownership trend — rising/stable]
- Analyst target: $[XX] (+[XX]% upside from current price)

**Catalysts** *(why the stock will be above the strike before expiry)*
1. [Catalyst name] — [why near-certain, specific timeline]
2. [Catalyst name] — [why near-certain]
3. [Catalyst name if applicable]

**What Could Go Wrong**
[2 sentences max. Be honest — what macro or company-specific event prevents the stock from reaching the strike?]

**Plain English Action**
[One clear sentence with strike, expiry, expected cost per share and per contract, and broker guidance.]

---

════════════════════════════════════════════
ALL OTHER QUESTIONS — SAME CLEAN STANDARD
════════════════════════════════════════════
For stock picks, momentum plays, sector analysis, or any other question:
- Use --- dividers between each pick
- Show confidence level prominently for every recommendation
- Reference actual numbers (RSI, vol ratio, IV rank, sentiment score, analyst target)
- Explain specifically which validation layers confirm or contradict the thesis
- Only recommend trades that pass most validation checks
- End with a plain English action sentence for every pick

════════════════════════════════════════════
MODES
════════════════════════════════════════════
SIMPLE MODE: Plain English only. Explain every term in parentheses. Use analogies.
ADVANCED MODE (default): Full numbers, IV analysis, pattern details.

⚠️ Options carry risk of total loss. This is data-driven analysis, not financial advice."""


# Compact system prompt used for Groq calls only — keeps the JSON payload
# small enough to avoid Groq free-tier 413 errors. Gemini gets the full prompt.
_GROQ_COMPACT_SYSTEM = """You are an elite AI Trading Research Agent for Quantum Edge Trading.

CRITICAL — OPTIONS DATA:
• Use ONLY the LIVE OPTIONS block values for every strike, expiry, and premium.
• Write FULL exact dates: "May 9, 2026" NOT "May 2026". Never use 2024/2025 dates.
• Never fabricate strikes, dates, or premiums not present in the data.
• Skip any symbol that has no LIVE OPTIONS block — do not invent numbers.

OUTPUT: Exactly 6 trades — 2 small cap + 2 mid cap + 2 large cap.
Use --- dividers. Only recommend HIGH (≥70%) and MEDIUM (50–69%) confidence trades.

Card format for every trade:
---
### 🎯 [SYMBOL] — [CALL/PUT] — [Small/Mid/Large Cap]
**Confidence: [HIGH/MEDIUM] ([XX]%)** &nbsp;&nbsp;&nbsp; **Risk: [LOW/MEDIUM/HIGH]**
| | |
|---|---|
| **Current Price** | $[XX.XX] |
| **Strike** | $[XX.XX] ([XX]% OTM) |
| **Expiry** | [Month DD, YYYY] |
| **Est. Premium** | $[X.XX]–$[X.XX] per share · $[XXX]–$[XXX] per contract |
| **Upside at Target** | 🟢 +[XXX]% |
| **Max Loss** | 🔴 100% of premium paid |

**Why This Trade**
- 📈 [L1] momentum — include vol ratio and RSI value
- 🔥 [L2] options flow or chart pattern — include IV rank
- 📰 [L3/L4] sentiment or institutional signal

**What Could Go Wrong**
[2 sentences max. Be specific — name the exact scenario that kills this trade.]

**Plain English Action**
[One sentence: what to buy, strike, exact expiry date, approximate cost.]
---

⚠️ Options carry risk of total loss. This is analysis, not financial advice."""


# ═══════════════════════════════════════════════════════════════════════════════
# LAYER 1 — Price, volume, RSI  (batch yfinance)
# ═══════════════════════════════════════════════════════════════════════════════
def _calc_rsi(closes: pd.Series, period: int = 14) -> float | None:
    try:
        if len(closes) < period + 2:
            return None
        delta = closes.diff().dropna()
        gain  = delta.clip(lower=0).rolling(period).mean().iloc[-1]
        loss  = (-delta.clip(upper=0)).rolling(period).mean().iloc[-1]
        if loss == 0:
            return 100.0
        return round(float(100 - (100 / (1 + gain / loss))), 1)
    except Exception:
        return None


def _detect_patterns(c: pd.Series, o: pd.Series, h: pd.Series, l: pd.Series) -> list[str]:
    """Detect common bullish/bearish chart patterns from OHLCV series."""
    patterns = []
    try:
        if len(c) < 20:
            return patterns

        # ── Breakout above 20-day high ────────────────────────────────────────
        high_20 = c.iloc[-21:-1].max()
        if float(c.iloc[-1]) > float(high_20) * 1.005:
            patterns.append("20-day breakout")

        # ── Breakout above 50-day high ────────────────────────────────────────
        if len(c) >= 51:
            high_50 = c.iloc[-51:-1].max()
            if float(c.iloc[-1]) > float(high_50) * 1.005:
                patterns.append("50-day breakout")

        # ── Bullish engulfing ────────────────────────────────────────────────
        if len(o) >= 2 and len(c) >= 2:
            po, pc = float(o.iloc[-2]), float(c.iloc[-2])
            co, cc = float(o.iloc[-1]), float(c.iloc[-1])
            if pc < po and cc > co and cc > po and co < pc:
                patterns.append("Bullish engulfing")

        # ── Bull flag (strong move then tight consolidation) ─────────────────
        if len(c) >= 10:
            move   = float(c.iloc[-6]) / float(c.iloc[-11]) - 1
            consol = c.iloc[-5:].std() / c.iloc[-5:].mean()
            if move > 0.06 and consol < 0.018:
                patterns.append("Bull flag")

        # ── Three consecutive up days ────────────────────────────────────────
        if len(c) >= 4:
            if all(float(c.iloc[-i-1]) > float(c.iloc[-i-2]) for i in range(3)):
                patterns.append("3-day momentum run")

        # ── SMA20 crossover ──────────────────────────────────────────────────
        if len(c) >= 21:
            sma = c.rolling(20).mean()
            if float(c.iloc[-1]) > float(sma.iloc[-1]) and float(c.iloc[-2]) < float(sma.iloc[-2]):
                patterns.append("SMA20 crossover ↑")

        # ── Oversold bounce ──────────────────────────────────────────────────
        rsi = _calc_rsi(c)
        if rsi and rsi < 35 and float(c.iloc[-1]) > float(c.iloc[-2]):
            patterns.append("Oversold bounce (RSI<35)")

        # ── Higher lows (uptrend structure) ──────────────────────────────────
        if len(l) >= 6:
            lows_5 = [float(l.iloc[-i-1]) for i in range(5)]
            if all(lows_5[i] > lows_5[i+1] for i in range(4)):
                patterns.append("Rising lows structure")

    except Exception:
        pass
    return patterns


def _screen_l1(progress_cb=None) -> tuple[dict[str, list[dict]], dict]:
    """
    Layer 1 — batch yfinance download for all 90 stocks.
    Returns (candidates_by_tier, ohlcv_by_sym).
    """
    all_syms = SMALL_CAP + MID_CAP + LARGE_CAP
    by_tier: dict[str, list[dict]] = {"small": [], "mid": [], "large": []}
    ohlcv: dict = {}

    try:
        from market_data import is_market_open as _mkt_open
        _open = _mkt_open()
    except Exception:
        _open = False

    if _open:
        _period, _interval = "5d", "5m"
        if progress_cb:
            progress_cb("📡 L1 — Live intraday data (5-min bars, 90 stocks)…")
    else:
        _period, _interval = "30d", "1d"
        if progress_cb:
            progress_cb("📡 L1 — Downloading 30-day OHLCV for 90 stocks…")

    try:
        raw = yf.download(
            all_syms, period=_period, interval=_interval,
            progress=False, auto_adjust=True,
        )
        if raw.empty or not isinstance(raw.columns, pd.MultiIndex):
            return by_tier, ohlcv

        closes  = raw["Close"]
        opens   = raw["Open"]
        highs   = raw["High"]
        lows    = raw["Low"]
        volumes = raw["Volume"]

    except Exception:
        return by_tier, ohlcv

    tier_map = (
        [("small", s) for s in SMALL_CAP] +
        [("mid",   s) for s in MID_CAP]   +
        [("large", s) for s in LARGE_CAP]
    )

    for tier, sym in tier_map:
        try:
            if sym not in closes.columns:
                continue
            c = closes[sym].dropna()
            o = opens[sym].dropna()   if sym in opens.columns   else pd.Series(dtype=float)
            h = highs[sym].dropna()   if sym in highs.columns   else pd.Series(dtype=float)
            l = lows[sym].dropna()    if sym in lows.columns    else pd.Series(dtype=float)
            v = volumes[sym].dropna() if sym in volumes.columns else pd.Series(dtype=float)

            if len(c) < 10:
                continue

            price     = round(float(c.iloc[-1]), 2)
            chg_pct   = round(((float(c.iloc[-1]) - float(c.iloc[-2])) / float(c.iloc[-2])) * 100, 2) if len(c) >= 2 else 0.0
            vol_ratio = round(float(v.iloc[-1]) / float(v.iloc[-10:-1].mean()), 2) if len(v) >= 10 else 1.0
            rsi       = _calc_rsi(c)
            patterns  = _detect_patterns(c, o, h, l)

            # L1 confidence contribution (max 40 pts)
            l1_pts = 0
            if abs(chg_pct) >= 3.0 and vol_ratio >= 2.0:   l1_pts += 20
            elif abs(chg_pct) >= 1.5 and vol_ratio >= 1.5: l1_pts += 12
            elif vol_ratio >= 1.3:                          l1_pts += 6
            if rsi:
                if rsi < 35:                    l1_pts += 12   # oversold setup
                elif 40 <= rsi <= 60:           l1_pts += 8    # momentum zone
            l1_pts += min(20, len(patterns) * 10)

            by_tier[tier].append({
                "symbol":    sym,
                "price":     price,
                "chg_pct":   chg_pct,
                "vol_ratio": vol_ratio,
                "rsi":       rsi,
                "patterns":  patterns,
                "l1_pts":    l1_pts,
                # filled by later layers
                "opt":       {},
                "sentiment": {},
                "inst":      {},
                "conf_pts":  l1_pts,
            })
            ohlcv[sym] = {"c": c, "o": o, "h": h, "l": l, "v": v}
        except Exception:
            pass

    for tier in by_tier:
        by_tier[tier].sort(key=lambda r: r["l1_pts"], reverse=True)

    if progress_cb:
        totals = {t: len(v) for t, v in by_tier.items()}
        progress_cb(f"✅ L1 complete — {totals['small']} small / {totals['mid']} mid / {totals['large']} large cap stocks screened")

    return by_tier, ohlcv


# ═══════════════════════════════════════════════════════════════════════════════
# LAYER 2a — Unusual options flow + IV rank  (yfinance per-stock)
# ═══════════════════════════════════════════════════════════════════════════════
def _get_options_flow(sym: str, price: float) -> dict:
    """
    Returns options flow data for a symbol.
    Cached for OPT_CACHE_TTL minutes.
    """
    now = time.time()
    if sym in _OPT_CACHE:
        ts, cached = _OPT_CACHE[sym]
        if now - ts < _opt_ttl():
            return cached

    base = {
        "unusual": False, "sweep": False, "iv_pct": 50.0, "iv_label": "fair",
        "pc_ratio": 1.0, "call_vol": 0, "put_vol": 0, "l2a_pts": 0, "details": "",
        # Real-time options chain data (populated below)
        "expiries":   [],   # upcoming expiry dates as strings e.g. ["2026-05-02", ...]
        "real_calls": [],   # [{expiry, strike, ask, bid, otm_pct, volume, iv_pct, breakeven}, ...]
        "real_puts":  [],
    }

    try:
        tk       = yf.Ticker(sym)
        expiries = tk.options          # tuple of real upcoming expiry date strings
        if not expiries:
            _OPT_CACHE[sym] = (now, base)
            return base

        # ── Store real upcoming expiry dates ──────────────────────────────────
        real_expiry_dates = list(expiries[:6])  # e.g. ["2026-05-02", "2026-05-09", ...]

        chain = tk.option_chain(expiries[0])
        calls, puts = chain.calls, chain.puts

        # ── IV rank (ATM implied volatility) ─────────────────────────────────
        atm_mask  = abs(calls["strike"] - price) / price < 0.05
        atm_calls = calls[atm_mask]
        iv_raw    = float(atm_calls["impliedVolatility"].mean()) * 100 if not atm_calls.empty else 50.0
        iv_pct    = round(min(200.0, iv_raw), 1)
        iv_label  = "cheap" if iv_pct < 30 else ("fair" if iv_pct < 60 else "expensive")

        # ── Unusual flow: high vol relative to OI ────────────────────────────
        def _unusual(df):
            df = df.copy()
            df["vol"]   = df["volume"].fillna(0)
            df["oi"]    = df["openInterest"].fillna(1).clip(lower=1)
            df["ratio"] = df["vol"] / df["oi"]
            return df[(df["vol"] > 300) & (df["ratio"] > 0.5)]

        u_calls = _unusual(calls)
        u_puts  = _unusual(puts)
        unusual = len(u_calls) > 0 or len(u_puts) > 0

        # ── Sweep: single strike with huge volume ─────────────────────────────
        max_call_vol = int(calls["volume"].fillna(0).max())
        max_put_vol  = int(puts["volume"].fillna(0).max())
        sweep = max_call_vol > 2000 or max_put_vol > 2000

        # ── Put/Call ratio ────────────────────────────────────────────────────
        total_call = int(calls["volume"].fillna(0).sum())
        total_put  = int(puts["volume"].fillna(0).sum())
        pc_ratio   = round(total_put / max(1, total_call), 2)

        # ── L2a confidence contribution (max 35 pts) ──────────────────────────
        l2a_pts = 0
        if iv_pct < 30:   l2a_pts += 15
        elif iv_pct < 50: l2a_pts += 10
        else:             l2a_pts -= 5
        if unusual:       l2a_pts += 10
        if sweep:         l2a_pts += 15
        l2a_pts = max(0, l2a_pts)

        details_parts = []
        if sweep:   details_parts.append(f"SWEEP detected (max {max(max_call_vol, max_put_vol):,} contracts)")
        if unusual: details_parts.append("Unusual vol/OI ratio")
        details_parts.append(f"P/C={pc_ratio}")
        details_parts.append(f"ATM_IV≈{iv_pct:.0f}% ({iv_label})")

        # ── REAL options candidates — extracted from live chain ───────────────
        # These are injected into the LLM context so it never has to guess.
        real_calls: list[dict] = []
        real_puts:  list[dict] = []

        def _extract_candidates(chain_calls, chain_puts, exp_date: str) -> None:
            try:
                # ── Calls: 0–15% OTM, liquid ─────────────────────────────────
                c = chain_calls.copy()
                c["ask"]    = c["ask"].fillna(0)
                c["bid"]    = c["bid"].fillna(0)
                c["volume"] = c["volume"].fillna(0)
                c["iv"]     = c["impliedVolatility"].fillna(0)
                c["otm"]    = ((c["strike"] - price) / price * 100).round(2)
                good_c = c[
                    (c["ask"] > 0.01) & (c["volume"] >= 1) &
                    (c["otm"] >= 0)   & (c["otm"] <= 15)
                ].sort_values("otm")
                for _, row in good_c.head(3).iterrows():
                    real_calls.append({
                        "expiry":    exp_date,
                        "strike":    round(float(row["strike"]), 2),
                        "ask":       round(float(row["ask"]), 2),
                        "bid":       round(float(row["bid"]), 2),
                        "otm_pct":   round(float(row["otm"]), 1),
                        "volume":    int(row["volume"]),
                        "iv_pct":    round(float(row["iv"]) * 100, 1),
                        "breakeven": round(float(row["strike"]) + float(row["ask"]), 2),
                    })

                # ── Puts: 0–15% OTM, liquid ──────────────────────────────────
                p = chain_puts.copy()
                p["ask"]    = p["ask"].fillna(0)
                p["bid"]    = p["bid"].fillna(0)
                p["volume"] = p["volume"].fillna(0)
                p["iv"]     = p["impliedVolatility"].fillna(0)
                p["otm"]    = ((price - p["strike"]) / price * 100).round(2)
                good_p = p[
                    (p["ask"] > 0.01) & (p["volume"] >= 1) &
                    (p["otm"] >= 0)   & (p["otm"] <= 15)
                ].sort_values("otm")
                for _, row in good_p.head(3).iterrows():
                    real_puts.append({
                        "expiry":    exp_date,
                        "strike":    round(float(row["strike"]), 2),
                        "ask":       round(float(row["ask"]), 2),
                        "bid":       round(float(row["bid"]), 2),
                        "otm_pct":   round(float(row["otm"]), 1),
                        "volume":    int(row["volume"]),
                        "iv_pct":    round(float(row["iv"]) * 100, 1),
                        "breakeven": round(float(row["strike"]) - float(row["ask"]), 2),
                    })
            except Exception:
                pass

        # Extract from first expiry (already fetched)
        _extract_candidates(calls, puts, expiries[0])

        # Also pull a second, later expiry for swing/multi-week setups
        if len(expiries) > 1 and len(real_calls) < 2:
            try:
                ch2 = tk.option_chain(expiries[1])
                _extract_candidates(ch2.calls, ch2.puts, expiries[1])
            except Exception:
                pass

        result = {
            "unusual":    unusual,
            "sweep":      sweep,
            "iv_pct":     iv_pct,
            "iv_label":   iv_label,
            "pc_ratio":   pc_ratio,
            "call_vol":   total_call,
            "put_vol":    total_put,
            "l2a_pts":    l2a_pts,
            "details":    " | ".join(details_parts),
            # Real-time chain data — LLM must use ONLY these values
            "expiries":   real_expiry_dates,
            "real_calls": real_calls[:4],
            "real_puts":  real_puts[:4],
        }
        _OPT_CACHE[sym] = (now, result)
        return result

    except Exception as e:
        base["details"] = f"Options err: {e}"
        _OPT_CACHE[sym] = (now, base)
        return base


# ═══════════════════════════════════════════════════════════════════════════════
# LAYER 3 — News sentiment  (MarketAux)
# ═══════════════════════════════════════════════════════════════════════════════
def _get_news_sentiment(sym: str) -> dict:
    """
    Fetch recent news sentiment for a symbol.
    Primary: MarketAux. Silent fallback: Finnhub.
    Cached for _ma_ttl() seconds.
    """
    now = time.time()
    if sym in _MA_CACHE:
        ts, cached = _MA_CACHE[sym]
        if now - ts < _ma_ttl():
            return cached

    base = {"score": 50, "label": "Neutral", "count": 0, "l3_pts": 0, "headlines": []}

    # ── Helper: score → label + pts ───────────────────────────────────────────
    def _score_to_result(score: int, count: int, pos: int, neg: int, neut: int,
                         headlines: list[str]) -> dict:
        label  = "Bullish" if score >= 62 else ("Bearish" if score <= 38 else "Neutral")
        l3_pts = 20 if label == "Bullish" else (8 if label == "Neutral" else 0)
        return {"score": score, "label": label, "count": count,
                "bull": pos, "bear": neg, "neut": neut,
                "l3_pts": l3_pts, "headlines": headlines[:3]}

    # ── Primary: MarketAux ───────────────────────────────────────────────────
    if MA_API_TOKEN:
        try:
            url  = (f"https://api.marketaux.com/v1/news/all"
                    f"?symbols={sym}&api_token={MA_API_TOKEN}"
                    f"&language=en&limit=5&filter_entities=true")
            data = requests.get(url, timeout=8).json()
            # Treat rate-limit / token-limit errors as failures
            if data.get("error") or "rate limit" in str(data).lower():
                raise ValueError("MarketAux limit")
            arts = data.get("data", [])
            pos = neg = neut = 0
            headlines = []
            for art in arts:
                headlines.append(art.get("title", "")[:80])
                for ent in art.get("entities", []):
                    s = float(ent.get("sentiment_score", 0))
                    if s > 0.1:    pos  += 1
                    elif s < -0.1: neg  += 1
                    else:          neut += 1
            total  = pos + neg + neut
            score  = int(((pos - neg) / total + 1) / 2 * 100) if total else 50
            result = _score_to_result(score, len(arts), pos, neg, neut, headlines)
            _MA_CACHE[sym] = (now, result)
            return result
        except Exception:
            pass   # fall through to Finnhub silently

    # ── Fallback: Finnhub news-sentiment ────────────────────────────────────
    if FINNHUB_API_KEY:
        try:
            url  = (f"https://finnhub.io/api/v1/news-sentiment"
                    f"?symbol={sym}&token={FINNHUB_API_KEY}")
            data = requests.get(url, timeout=8).json()
            bull_pct = float(data.get("sentiment", {}).get("bullishPercent", 0.5))
            bear_pct = float(data.get("sentiment", {}).get("bearishPercent", 0.5))
            buzz     = int(data.get("buzz", {}).get("articlesInLastWeek", 0))
            score    = int(bull_pct * 100)
            pos      = int(bull_pct * buzz)
            neg      = int(bear_pct * buzz)
            neut     = max(0, buzz - pos - neg)
            result   = _score_to_result(score, buzz, pos, neg, neut, [])
            _MA_CACHE[sym] = (now, result)
            return result
        except Exception:
            pass

    _MA_CACHE[sym] = (now, base)
    return base


# ═══════════════════════════════════════════════════════════════════════════════
# LAYER 4 — Institutional / analyst data  (Alpha Vantage — rate-limited)
# ═══════════════════════════════════════════════════════════════════════════════
def _get_av_data(sym: str) -> dict:
    """Fetch company overview from Alpha Vantage. Cached 6 h."""
    now = time.time()
    if sym in _AV_CACHE:
        ts, cached = _AV_CACHE[sym]
        if now - ts < AV_CACHE_TTL:
            return cached

    base = {"inst_own": None, "analyst_target": None, "beta": None,
            "mkt_cap": None, "l4_pts": 0, "details": ""}

    if not AV_API_KEY:
        return base

    try:
        url  = (f"https://www.alphavantage.co/query"
                f"?function=OVERVIEW&symbol={sym}&apikey={AV_API_KEY}")
        r    = requests.get(url, timeout=10)
        data = r.json()

        if "Symbol" not in data:
            _AV_CACHE[sym] = (now, base)
            return base

        def _f(k):
            v = data.get(k)
            return float(v) if v and v not in ("None", "-") else None

        inst_own  = _f("PercentInstitutions")
        analyst_t = _f("AnalystTargetPrice")
        beta      = _f("Beta")
        mkt_cap   = _f("MarketCapitalization")
        price_now = _f("50DayMovingAverage")  # proxy for current price

        # L4 confidence contribution (max 15 pts)
        l4_pts = 0
        details_parts = []
        if inst_own and inst_own > 50:
            l4_pts += 8
            details_parts.append(f"Inst own={inst_own:.0f}%")
        if analyst_t and price_now and analyst_t > price_now * 1.10:
            l4_pts += 7
            upside = round((analyst_t / price_now - 1) * 100, 0)
            details_parts.append(f"Analyst target ${analyst_t:.2f} (+{upside:.0f}% upside)")
        if beta:
            details_parts.append(f"Beta={beta:.2f}")

        result = {
            "inst_own":       inst_own,
            "analyst_target": analyst_t,
            "beta":           beta,
            "mkt_cap":        mkt_cap,
            "l4_pts":         l4_pts,
            "details":        " | ".join(details_parts) if details_parts else "No notable signals",
        }
        _AV_CACHE[sym] = (now, result)
        return result

    except Exception:
        _AV_CACHE[sym] = (now, base)
        return base


# ═══════════════════════════════════════════════════════════════════════════════
# LAYER 5 — VIX regime scoring
# ═══════════════════════════════════════════════════════════════════════════════
def _vix_regime_pts(vix_val: float | None) -> tuple[int, str]:
    """Return (confidence_pts, regime_label) based on VIX."""
    if vix_val is None:
        return 5, "Unknown"
    if vix_val < 15:
        return 10, "LOW — ideal for call buying"
    if vix_val < 20:
        return 8,  "CALM — favourable for longs"
    if vix_val < 25:
        return 5,  "NORMAL — balanced risk"
    if vix_val < 30:
        return 2,  "ELEVATED — be selective"
    return 0,     "HIGH FEAR — reduce size, avoid naked calls"


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN SCREENER — orchestrates all layers
# ═══════════════════════════════════════════════════════════════════════════════
def _screen_universe(vix_val: float | None = None, progress_cb=None) -> dict[str, list[dict]]:
    """
    Full multi-layer universe screen.
    Returns {tier: [validated_candidate, ...]} sorted by confidence_pct desc.
    """
    vix_pts, vix_regime = _vix_regime_pts(vix_val)

    # ── L1 — batch download, patterns ────────────────────────────────────────
    by_tier, ohlcv = _screen_l1(progress_cb)

    # Keep top 6 per tier for deep validation
    for tier in by_tier:
        by_tier[tier] = by_tier[tier][:6]

    # ── L2a — options flow (top candidates only) ─────────────────────────────
    all_top = [(tier, r) for tier, rows in by_tier.items() for r in rows]
    if progress_cb:
        progress_cb(f"📊 L2 — Checking options flow for {len(all_top)} candidates…")

    for tier, r in all_top:
        sym  = r["symbol"]
        opt  = _get_options_flow(sym, r["price"])
        r["opt"] = opt
        r["conf_pts"] = r["l1_pts"] + opt["l2a_pts"]

    # ── L3 — news sentiment (top 4 per tier) ─────────────────────────────────
    if progress_cb:
        progress_cb("📰 L3 — Fetching news sentiment (MarketAux)…")

    for tier, rows in by_tier.items():
        for r in rows[:4]:
            sent = _get_news_sentiment(r["symbol"])
            r["sentiment"] = sent
            r["conf_pts"] += sent["l3_pts"]

    # ── L4 — AV institutional data (top 1 per tier = 3 calls max) ────────────
    if AV_API_KEY:
        if progress_cb:
            progress_cb("🏦 L4 — Checking institutional data (Alpha Vantage)…")
        for tier, rows in by_tier.items():
            if rows:
                inst = _get_av_data(rows[0]["symbol"])
                rows[0]["inst"] = inst
                rows[0]["conf_pts"] += inst["l4_pts"]

    # ── L5 — VIX regime ──────────────────────────────────────────────────────
    for tier, rows in by_tier.items():
        for r in rows:
            r["conf_pts"] += vix_pts
            r["vix_regime"] = vix_regime

    # ── Normalise to 0-100 confidence % ──────────────────────────────────────
    # Max possible: 40(L1) + 35(L2a) + 20(L3) + 15(L4) + 10(L5) = 120
    MAX_PTS = 120
    for tier, rows in by_tier.items():
        for r in rows:
            pct = min(99, int(r["conf_pts"] / MAX_PTS * 100))
            r["confidence_pct"]   = pct
            r["confidence_label"] = "HIGH" if pct >= 70 else ("MEDIUM" if pct >= 50 else "LOW")

    # Re-sort by confidence
    for tier in by_tier:
        by_tier[tier].sort(key=lambda r: r["confidence_pct"], reverse=True)

    if progress_cb:
        highs = sum(1 for rows in by_tier.values() for r in rows if r["confidence_label"] == "HIGH")
        progress_cb(f"✅ Validation complete — {highs} HIGH-confidence setups found")

    return by_tier


# ═══════════════════════════════════════════════════════════════════════════════
# CONTEXT BUILDER  — formats validated data for LLM
# ═══════════════════════════════════════════════════════════════════════════════
def _build_universe_context(screened: dict[str, list[dict]], vix_val: float | None = None) -> str:
    vix_pts, vix_regime = _vix_regime_pts(vix_val)
    vix_str = f"{vix_val:.1f}" if vix_val is not None else "N/A"
    lines = [
        "═══════════════════════════════════════════════",
        "LIVE MULTI-LAYER VALIDATED UNIVERSE SCAN",
        "═══════════════════════════════════════════════",
        f"VIX Regime: {vix_regime}  (VIX={vix_str})",
        "Confidence: HIGH≥70%  MEDIUM 50-69%  LOW<50% (excluded from picks)",
        "",
    ]

    cap_labels = {
        "small": "SMALL CAP  (<$3B)",
        "mid":   "MID CAP   ($3–15B)",
        "large": "LARGE CAP  (>$15B)",
    }

    for tier in ("small", "mid", "large"):
        lines.append(f"── {cap_labels[tier]} ──────────────────────────")
        rows = screened.get(tier, [])
        if not rows:
            lines.append("  No data"); continue

        for r in rows[:3]:   # top 3 per tier keeps payload within Groq limits
            sym    = r["symbol"]
            conf   = r["confidence_label"]
            pct    = r["confidence_pct"]
            rsi    = r["rsi"]
            rsi_s  = f"RSI={rsi:.0f}" if rsi else "RSI=N/A"
            rsi_flag = (" ⬆OVERSOLD" if rsi and rsi < 35
                        else " ⬇OVERBOUGHT" if rsi and rsi > 70
                        else "")
            opt    = r.get("opt", {})
            sent   = r.get("sentiment", {})
            inst   = r.get("inst", {})

            # Main line
            lines.append(
                f"  {conf:6s} ({pct:2d}%)  {sym:<7}"
                f"  ${r['price']:>8.2f}  {r['chg_pct']:+.2f}%  "
                f"VolRatio={r['vol_ratio']:.1f}x  {rsi_s}{rsi_flag}"
            )

            # Patterns
            if r.get("patterns"):
                lines.append(f"    Patterns : {', '.join(r['patterns'])}")

            # Options flow
            if opt.get("details"):
                sweep_tag = " 🚨SWEEP" if opt.get("sweep") else ""
                lines.append(
                    f"    Options  : {opt['details']}{sweep_tag}"
                )

            # Sentiment
            if sent.get("label") and sent.get("label") != "Neutral":
                lines.append(
                    f"    Sentiment: {sent['label']} ({sent.get('score',50)}/100) "
                    f"— {sent.get('bull',0)}↑ {sent.get('bear',0)}↓ from {sent.get('count',0)} articles"
                )
                for h in sent.get("headlines", [])[:2]:
                    lines.append(f"      News: {h}")

            # Institutional
            if inst.get("details") and inst["details"] != "No notable signals":
                lines.append(f"    Inst     : {inst['details']}")

            # ── LIVE OPTIONS DATA — LLM must use ONLY these values ────────────
            expiries   = opt.get("expiries", [])
            real_calls = opt.get("real_calls", [])
            real_puts  = opt.get("real_puts", [])

            if real_calls or real_puts:
                lines.append(f"    ┌─ LIVE OPTIONS (use ONLY these values)")
                for rc in real_calls[:1]:   # 1 call keeps payload small
                    lines.append(
                        f"    │  CALL  ${rc['strike']:.2f} ({rc['otm_pct']:+.1f}%OTM)"
                        f"  exp={rc['expiry']}  ask=${rc['ask']:.2f}"
                        f"  IV={rc['iv_pct']:.0f}%  beven=${rc['breakeven']:.2f}"
                    )
                for rp in real_puts[:1]:    # 1 put keeps payload small
                    lines.append(
                        f"    │  PUT   ${rp['strike']:.2f} ({rp['otm_pct']:+.1f}%OTM)"
                        f"  exp={rp['expiry']}  ask=${rp['ask']:.2f}"
                        f"  IV={rp['iv_pct']:.0f}%  beven=${rp['breakeven']:.2f}"
                    )
                lines.append(f"    └────────────────────────────────────")

            lines.append("")

    lines.append(
        "NOTE: Use this data to produce trade cards in the required format. "
        "Only recommend HIGH and MEDIUM confidence plays.\n"
        "CRITICAL: For every trade card you write, the Strike, Expiry, and Premium MUST "
        "come from the LIVE OPTIONS block shown above for that symbol. "
        "Never use expiry dates from 2024 or 2025 — those are expired contracts."
    )
    return "\n".join(lines)


def _build_context(scan_results: list[dict], options_data: dict | None = None) -> str:
    if not scan_results:
        header = (
            "=== WATCHLIST SCAN: empty — use the live universe data below ===\n"
        )
        return header if not options_data else header + "\n"

    lines = ["=== LIVE WATCHLIST SCAN RESULTS ===\n"]
    for r in scan_results[:12]:
        sym  = r["symbol"]
        sc   = r.get("scores", {})
        sent = r.get("sentiment", {})
        tech = r.get("tech", {})
        conv = r.get("convergence_notes", [])
        lines.append(f"▸ {sym}  ${r.get('price','N/A')}  ({r.get('change_pct','N/A')}%)")
        lines.append(f"  Score={r['composite_score']:.0f}/100 → {r['recommendation']}")
        lines.append(
            f"  RSI={tech.get('rsi','N/A')}  MACD={tech.get('macd_signal','N/A')}  "
            f"ADX={tech.get('adx','N/A')}  Trend={tech.get('trend_direction','N/A')}"
        )
        lines.append(
            f"  Sentiment={sent.get('label','N/A')} ({sent.get('sentiment_score','N/A')}/100)"
        )
        if conv:
            lines.append(f"  ★ {conv[0]}")
        lines.append("")

    if options_data:
        lines.append("=== OPTIONS CHAIN (watchlist) ===\n")
        for sym, opt in options_data.items():
            if not opt: continue
            lines.append(f"{sym}  bias={opt.get('bias','?')}  ${opt.get('current_price','?')}")
            for c in opt.get("calls", [])[:3]:
                lines.append(
                    f"  CALL strike=${c['strike']}  exp={c['expiration']}  "
                    f"ask=${c['ask']}  IV={c.get('iv_pct','?')}%  vol={c.get('volume',0):,}"
                )
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# DEEPSEEK FINAL VALIDATION LAYER
# ═══════════════════════════════════════════════════════════════════════════════
def _validate_with_deepseek(
    main_analysis: str,
    question:      str,
    vix_val:       float | None = None,
) -> str:
    """
    Pass the main analysis to DeepSeek for final validation.
    DeepSeek reviews recommendations, adds confidence scores, flags risks,
    and appends a concise final verdict section.

    Silently returns the original analysis unchanged if DeepSeek is
    unavailable for any reason — no error messages surfaced to the user.
    """
    key = DEEPSEEK_API_KEY or os.environ.get("DEEPSEEK_API_KEY", "")
    if not key:
        return main_analysis

    vix_str = f"{vix_val:.1f}" if vix_val is not None else "N/A"

    validation_prompt = f"""You are a senior quant risk analyst reviewing AI-generated trade recommendations.

Review the following trade cards and append a **📊 DeepSeek Validation** section at the end.
Do NOT rewrite, reformat, or remove any existing cards. Only add the section below.

Your validation section must include:
1. A one-line overall verdict on the quality of today's setup batch.
2. For each trade symbol mentioned, a single bullet:
   `• [SYMBOL] — [CONFIRM / REDUCE SIZE / SKIP] — [one-sentence reason]`
3. The single biggest macro or systemic risk to all these positions right now.
4. One sentence on what VIX={vix_str} means for sizing these trades.

Keep the whole section under 200 words. Be direct and honest — do not repeat what was already said in the cards.

Original question: {question}
Current VIX: {vix_str}

=== TRADE ANALYSIS TO REVIEW ===
{main_analysis}
=== END ==="""

    try:
        resp = requests.post(
            "https://api.deepseek.com/chat/completions",
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type":  "application/json",
            },
            json={
                "model":       "deepseek-chat",
                "messages":    [{"role": "user", "content": validation_prompt}],
                "max_tokens":  600,
                "temperature": 0.3,
            },
            timeout=45,
        )
        resp.raise_for_status()
        verdict = resp.json()["choices"][0]["message"]["content"] or ""
        if verdict.strip():
            return main_analysis + "\n\n---\n\n" + verdict.strip()
    except Exception:
        pass   # DeepSeek unavailable — silently skip

    return main_analysis


# Module-level slots that capture the last failure reason for each provider.
_groq_last_err:   list[str] = [""]
_gemini_last_err: list[str] = [""]

# ═══════════════════════════════════════════════════════════════════════════════
# GROQ HTTP  (bypasses the groq package — works on any environment)
# ═══════════════════════════════════════════════════════════════════════════════
def _ask_groq_http(messages: list[dict], model: str, max_tokens: int = 3500) -> str:
    """
    Call Groq via its OpenAI-compatible REST API using requests.
    Bypasses the groq SDK entirely — no package dependency issues.
    Returns empty string on any failure — never raises.
    Stores the failure reason in _groq_last_err[0] for diagnostics.
    """
    key = GROQ_API_KEY or os.environ.get("GROQ_API_KEY", "")
    if not key:
        _groq_last_err[0] = "no_key"
        return ""
    _groq_last_err[0] = ""
    # Try the requested model; on rate-limit retry once with the fast model
    models_to_try = [model]
    if model != GROQ_MODEL_FAST:
        models_to_try.append(GROQ_MODEL_FAST)
    for attempt_model in models_to_try:
        try:
            resp = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {key}",
                    "Content-Type":  "application/json",
                },
                json={
                    "model":       attempt_model,
                    "messages":    messages,
                    "max_tokens":  max_tokens,
                    "temperature": 0.4,
                },
                timeout=60,
            )
            if resp.status_code == 401:
                _groq_last_err[0] = "invalid_key"
                return ""            # wrong key — don't retry
            if resp.status_code == 413:
                _groq_last_err[0] = "payload_too_large"
                return ""            # payload still too large — don't retry
            if resp.status_code == 429:
                _groq_last_err[0] = "rate_limit"
                time.sleep(1)
                continue             # try next model
            resp.raise_for_status()
            text = resp.json()["choices"][0]["message"]["content"] or ""
            if text:
                _groq_last_err[0] = ""
                return text
        except Exception as exc:
            _groq_last_err[0] = str(exc)[:120]
    return ""


# ═══════════════════════════════════════════════════════════════════════════════
# GEMINI FALLBACK  (silent — used only when Groq is unavailable)
# ═══════════════════════════════════════════════════════════════════════════════
def _ask_gemini(messages: list[dict], max_tokens: int = 3500) -> str:
    """
    Silent fallback to Gemini via REST API.
    Tries multiple model names in order — returns first successful response.
    Returns empty string if key is missing or all models fail — never raises.
    """
    key = GEMINI_API_KEY or os.environ.get("GEMINI_API_KEY", "")
    if not key:
        _gemini_last_err[0] = "no_key"
        return ""
    _gemini_last_err[0] = ""

    system_text = next(
        (m["content"] for m in messages if m["role"] == "system"), ""
    )
    user_text = "\n\n".join(
        m["content"] for m in messages if m["role"] == "user"
    )
    payload = {
        "systemInstruction": {"parts": [{"text": system_text}]},
        "contents":          [{"role": "user", "parts": [{"text": user_text}]}],
        "generationConfig":  {"maxOutputTokens": max_tokens, "temperature": 0.4},
    }

    # Try models in order — newest free-tier first
    _GEMINI_MODELS = [
        "gemini-2.0-flash",
        "gemini-2.0-flash-exp",
        "gemini-1.5-flash-latest",
        "gemini-1.5-flash",
    ]
    for model in _GEMINI_MODELS:
        try:
            url = (
                "https://generativelanguage.googleapis.com/v1beta"
                f"/models/{model}:generateContent?key={key}"
            )
            resp = requests.post(url, json=payload, timeout=60)
            if resp.status_code in (404, 400):
                _gemini_last_err[0] = f"{model}:{resp.status_code}"
                continue   # model not available — try next
            if resp.status_code == 429:
                _gemini_last_err[0] = f"{model}:429_ratelimit"
                # Rate limited — wait and retry up to twice before moving on
                for _wait in (5, 10):
                    time.sleep(_wait)
                    resp = requests.post(url, json=payload, timeout=60)
                    if resp.status_code == 200:
                        _gemini_last_err[0] = ""
                        break
                if resp.status_code != 200:
                    continue
            resp.raise_for_status()
            text = resp.json()["candidates"][0]["content"]["parts"][0]["text"]
            if text:
                return text
        except Exception as _ge:
            _gemini_last_err[0] = str(_ge)[:120]
            continue
    return ""


# ═══════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ═══════════════════════════════════════════════════════════════════════════════
def ask_adviser(
    question:        str,
    scan_results:    list[dict] | None = None,
    options_data:    dict | None       = None,
    simple_mode:     bool              = False,
    stream_callback                    = None,
    progress_cb                        = None,
    vix_val:         float | None      = None,
) -> str:
    """
    Main research agent entry point.

    Always runs the full multi-layer validation pipeline before calling the LLM.
    Completely independent of watchlist scan results.
    """
    if scan_results is None:
        scan_results = []

    # Bail only if no AI key is available through any channel
    _groq_key   = GROQ_API_KEY   or os.environ.get("GROQ_API_KEY",   "")
    _gemini_key = GEMINI_API_KEY or os.environ.get("GEMINI_API_KEY", "")
    if not _groq_key and not _gemini_key:
        return _rule_based_response(question, scan_results, simple_mode)

    # ── Full multi-layer validation (every query) ─────────────────────────────
    screened      = _screen_universe(vix_val=vix_val, progress_cb=progress_cb)
    universe_ctx  = _build_universe_context(screened, vix_val)
    watchlist_ctx = _build_context(scan_results, options_data)

    mode_tag = (
        "\n\n[MODE: SIMPLE — plain English, analogies, explain every term]"
        if simple_mode else ""
    )

    # Hard cap: keep total payload well within Groq's free-tier token limit.
    # SYSTEM_PROMPT ≈ 1 500 tokens; leave ~3 500 tokens for universe context.
    # 12 000 chars ≈ 3 000 tokens — safe margin below the 6 000-token limit.
    _MAX_CTX = 12_000
    if len(universe_ctx) > _MAX_CTX:
        universe_ctx = universe_ctx[:_MAX_CTX] + "\n[context trimmed]"

    def _build_groq_messages(ctx: str) -> list[dict]:
        """Compact system prompt keeps Groq payload under free-tier limits."""
        user_msg = f"""{watchlist_ctx}

{ctx}

My question: {question}{mode_tag}"""
        return [
            {"role": "system", "content": _GROQ_COMPACT_SYSTEM},
            {"role": "user",   "content": user_msg},
        ]

    def _build_gemini_messages(ctx: str) -> list[dict]:
        """Full system prompt for Gemini (no payload size constraint)."""
        user_msg = f"""{watchlist_ctx}

{ctx}

My question: {question}{mode_tag}"""
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg},
        ]

    groq_messages   = _build_groq_messages(universe_ctx)
    gemini_messages = _build_gemini_messages(universe_ctx)
    groq_model      = _pick_groq_model(question)

    # ── Try Groq via HTTP (compact prompt, no SDK dependency) ─────────────────
    # max_tokens=1800: Groq free tier caps input+output per request ~4096 tokens.
    # Compact system (~365t) + universe_ctx (~1200t) + question (~50t) ≈ 1615 input.
    # 1615 + 1800 = 3415 — safely under the limit.
    ai_text = _ask_groq_http(groq_messages, groq_model, max_tokens=1800)

    # If Groq still 413, trim context further and retry with lower max_tokens
    if not ai_text and _groq_last_err[0] == "payload_too_large":
        trimmed_ctx     = universe_ctx[:3_000] + "\n[context trimmed]"
        groq_messages   = _build_groq_messages(trimmed_ctx)
        gemini_messages = _build_gemini_messages(trimmed_ctx)
        ai_text         = _ask_groq_http(groq_messages, groq_model, max_tokens=1200)

    # ── Silent fallback: Gemini (full prompt, higher limits) ──────────────────
    if not ai_text:
        ai_text = _ask_gemini(gemini_messages)

    if not ai_text:
        # ── Temporary diagnostics — remove once root cause confirmed ─────────
        _groq_key_present   = bool(GROQ_API_KEY or os.environ.get("GROQ_API_KEY", ""))
        _gemini_key_present = bool(GEMINI_API_KEY or os.environ.get("GEMINI_API_KEY", ""))
        _ctx_len            = len(universe_ctx)
        _sys_len            = len(_GROQ_COMPACT_SYSTEM)
        return (
            f"⚠️ **Debug — both providers failed.**\n\n"
            f"- Groq key present: `{_groq_key_present}`\n"
            f"- Gemini key present: `{_gemini_key_present}`\n"
            f"- Groq last error: `{_groq_last_err[0] or 'none'}`\n"
            f"- Gemini last error: `{_gemini_last_err[0] or 'none'}`\n"
            f"- Universe context length: `{_ctx_len:,}` chars\n"
            f"- Compact system prompt length: `{_sys_len:,}` chars\n"
            f"- Groq model attempted: `{groq_model}`\n"
        )

    # ── DeepSeek final validation layer ──────────────────────────────────────
    final = _validate_with_deepseek(ai_text, question, vix_val)

    if stream_callback:
        stream_callback(final)
    return final


# ═══════════════════════════════════════════════════════════════════════════════
# RULE-BASED FALLBACK  (no Groq key)
# ═══════════════════════════════════════════════════════════════════════════════
def _rule_based_response(question: str, scan_results: list[dict], simple_mode: bool = False) -> str:
    q = question.lower()

    no_groq_note = (
        "\n\n> *Add `GROQ_API_KEY` in Streamlit Cloud → Settings → Secrets "
        "to unlock the full Research Agent with 5-layer validation, live options flow, "
        "and AI-generated trade cards.*"
    )

    if not scan_results:
        return (
            "## 🔬 Research Agent — AI key required\n\n"
            "No AI provider is currently available. Please check that your API keys "
            "are set in Streamlit Cloud → Settings → Secrets.\n"
            + no_groq_note
        )

    buy_list = [r for r in scan_results if r["recommendation"] in ("STRONG BUY", "BUY")]
    top5     = scan_results[:5]

    is_options = any(w in q for w in ["call","calls","put","puts","option","options",
                                       "strike","expir","contract","weekly","leap","daily"])

    if is_options:
        want_puts = any(w in q for w in ["put","puts","bear","short"])
        opt_type  = "PUT" if want_puts else "CALL"
        targets   = sorted(scan_results, key=lambda r: r["composite_score"])[:5] if want_puts else (buy_list[:5] or top5)

        lines = [f"## 📊 {opt_type} Opportunities (rule-based)\n",
                 "*Full 5-layer validation requires Groq AI key.*\n"]
        for r in targets[:4]:
            sym  = r["symbol"]
            score = r["composite_score"]
            tech  = r.get("tech", {})
            try:
                price = float(r.get("price", 0))
                strike_atm = round(price / 5) * 5
                lines.append(
                    f"### 🎯 {sym}  Score {score:.0f}/100  — {r['recommendation']}\n"
                    f"- Price: ${price:.2f}  |  Suggested {opt_type}: ${strike_atm:.0f} ATM\n"
                    f"- RSI: {tech.get('rsi','N/A')}  MACD: {tech.get('macd_signal','N/A')}\n"
                )
            except Exception:
                lines.append(f"### {sym}  Score {score:.0f}/100\n")
        lines.append("\n⚠️ *Options can expire worthless. Size positions appropriately.*")
        lines.append(no_groq_note)
        return "\n".join(lines)

    # General
    lines = ["## 🏆 Top Opportunities (rule-based)\n"]
    for i, r in enumerate(top5, 1):
        sc   = r.get("scores", {})
        sent = r.get("sentiment", {})
        icon = {"STRONG BUY":"🟢","BUY":"🟩","WATCH":"🟡","NEUTRAL":"⚪","AVOID":"🔴"}.get(r["recommendation"],"⚪")
        lines.append(
            f"{i}. {icon} **{r['symbol']}** — ${r.get('price','N/A')} — "
            f"**{r['recommendation']}** ({r['composite_score']:.0f}/100)\n"
            f"   Inst={sc.get('institutional',0):.0f}  Tech={sc.get('technical',0):.0f}  "
            f"Sent={sc.get('sentiment',0):.0f}  News={sent.get('label','N/A')}"
        )
        conv = r.get("convergence_notes", [])
        if conv:
            lines.append(f"   ⭐ {conv[0]}")
        lines.append("")
    lines.append(no_groq_note)
    return "\n".join(lines)


