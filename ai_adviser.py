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
from dotenv import load_dotenv

load_dotenv()

def _get_secret(key: str) -> str:
    """Read from Streamlit secrets (Cloud) with fallback to .env (local)."""
    # Try Streamlit secrets first (works on Cloud and locally with secrets.toml)
    try:
        import streamlit as st
        val = st.secrets.get(key, "")
        if val:
            return val
    except Exception:
        pass
    # Fall back to environment variable (.env loaded above)
    return os.getenv(key, "")

# Keys are re-fetched at call time inside each function — module-level values
# are only used as a fast-path check; actual API calls always call _get_secret().
# Keys are injected by dashboard.py via _load_keys() after Streamlit starts.
# These defaults are only used if ai_adviser is imported outside of dashboard.py.
GROQ_API_KEY  = os.getenv("GROQ_API_KEY", "")
AV_API_KEY    = os.getenv("ALPHA_VANTAGE_API_KEY", "")
MA_API_TOKEN  = os.getenv("MARKETAUX_API_TOKEN", "")

try:
    from groq import Groq as _Groq
    _GROQ_INSTALLED = True
except ImportError:
    _Groq = None
    _GROQ_INSTALLED = False

def _groq_key() -> str:
    """Always fetch the freshest key — works both locally and on Streamlit Cloud."""
    return _get_secret("GROQ_API_KEY")

def _has_groq() -> bool:
    return _GROQ_INSTALLED and bool(_groq_key())
GROQ_MODEL    = "llama-3.3-70b-versatile"

HAS_GROQ = _GROQ_INSTALLED  # package installed; key checked at call time


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
MEDIUM (50-69%) — Solid setup, some layers agree. Recommend with caution.
LOW    (<50%) — Mixed signals. Do NOT recommend.

ONLY recommend HIGH and MEDIUM confidence trades.

════════════════════════════════════════════
LEAP OPTIONS — SPECIAL FRAMEWORK
════════════════════════════════════════════
When asked for LEAP options (6-18 month expiry), apply this STRICTER framework:

FUNDAMENTALS required for every LEAP pick:
  • Revenue growth YoY > 10% (or strong profitability trend)
  • Expanding or stable operating margins
  • Low/manageable debt (D/E < 1.5 preferred)
  • Rising institutional ownership (bullish smart-money signal)
  • Analyst price target meaningfully above current price

CATALYSTS required — must identify 2-3 per pick:
  Types: earnings beats, product launch, FDA/regulatory approval,
         index inclusion, macro tailwind, sector rotation, M&A potential
  Conviction: explain WHY each catalyst is near-certain or highly probable
  Timeline: confirm catalyst falls BEFORE the expiry date

LEAP CARD FORMAT (return 3 picks: 1 small + 1 mid + 1 large cap):

═══════════════════════════════════════
🚀 [SYMBOL] — LEAP CALL — [CAP TIER]
Confidence: HIGH ([XX]%)
═══════════════════════════════════════
💰 Current Price: $XX.XX
🎯 Strike: $XX (XX% OTM)  |  Expiry: [month year, 6-18 mo out]
💵 Est. Premium: $X.XX–$X.XX per contract
📈 Target Price at Expiry: $XX  |  Options Gain: +XX%
⚠️  Max Loss: 100% of premium paid  |  Risk: [LOW/MEDIUM]

📊 Fundamental Case:
  • Revenue: [specific growth metric]
  • Margins: [trend]
  • Debt: [D/E or coverage ratio]
  • Institutions: [ownership % trend]
  • Analyst target: $XX (+XX% upside)

🔥 Catalysts (why the stock WILL be above strike):
  1. [Catalyst name] — [why near-certain, timeline]
  2. [Catalyst name] — [why near-certain, timeline]
  3. [Catalyst name] — [why near-certain, timeline]  (if applicable)

✅ Validation Layers:
  • [L1/L4] Fundamental + institutional signal
  • [L3] Sentiment signal
  • [L5] VIX regime suitability

💡 Plain English: [one sentence]
═══════════════════════════════════════

════════════════════════════════════════════
OPTIONS PLAYS — STRICT OUTPUT FORMAT
════════════════════════════════════════════
When asked for options plays (daily, weeklies, calls, puts, etc.):
Return EXACTLY 6 trades: 2 SMALL CAP + 2 MID CAP + 2 LARGE CAP.

For each trade use this exact card format:

─────────────────────────────────────────
🎯 [SYMBOL] — [CALL/PUT] — [CAP TIER]
Confidence: [HIGH/MEDIUM] ([XX]%)
─────────────────────────────────────────
💰 Price: $XX.XX  |  Strike: $XX  |  Expiry: [date]
📈 Upside: +XX%   |  Breakeven: $XX.XX
⚠️  Risk Level: [LOW/MEDIUM/HIGH]  |  Max Loss: cost of premium
IV Rank: XX% (cheap/fair/expensive)

✅ Why This Trade:
  • [L1] Momentum/Volume reason
  • [L2] Options flow / pattern reason
  • [L3/L4] Sentiment / institutional reason

❌ What Could Go Wrong:
  • [specific risk]

💡 Plain English: [one sentence for beginners]
─────────────────────────────────────────

════════════════════════════════════════════
ALL OTHER QUESTIONS — SAME FRAMEWORK
════════════════════════════════════════════
For stock picks, day trades, sector analysis, or any question:
- Use the validation data provided
- Show confidence level for each recommendation
- Reference actual numbers (RSI, IV, sentiment score, vol ratio)
- Explain WHY each layer does or does not confirm
- Only recommend what passes most validation checks

════════════════════════════════════════════
MODES
════════════════════════════════════════════
SIMPLE MODE: Plain English, explain every term, analogies welcome.
ADVANCED MODE (default): Full numbers, IV analysis, pattern details.

⚠️ Options carry risk of total loss. This is data-driven analysis, not financial advice."""


UI_SYSTEM_PROMPT = """You are a senior UI/UX expert and trading platform designer.
Critically analyse the dashboard below and propose concrete improvements,
comparing it to TradingView, Bloomberg Terminal, thinkorswim, and Webull.

Rules:
1. Identify real weaknesses honestly — minimum 6 proposals, maximum 10.
2. Compare each to how top platforms solve it.
3. Number each proposal. Each must use EXACTLY this format:

### [N]. [TITLE]
**Impact:** HIGH / MEDIUM / LOW
**Current State:** [what it looks like now]
**Proposed Change:** [specific change — mention colours, sizes, layout]
**Why:** [one sentence reasoning referencing a named platform]

4. After all proposals, include a section called:
### CSS ARTIFACT
Provide a complete, copy-paste-ready CSS block that implements ALL proposals at once.
The CSS block must be fenced in triple backticks with the css language tag.
Use real Streamlit CSS selectors (.stApp, .stButton > button, [data-testid="stSidebar"], etc.)
Keep the dark theme. Use the existing palette: #0f1014 bg, #1a1b22 card, #00d26a emerald, #8b56f6 purple.

5. End with a 2–3 sentence ### Overall Verdict.

Be direct, specific, and actionable."""


UI_CSS_PROMPT = """You are a CSS expert for Streamlit trading dashboards.
Generate production-ready CSS for the approved UI improvements listed.

Rules:
1. Use real Streamlit CSS class selectors:
   .stApp, .main .block-container, [data-testid="stSidebar"],
   .stButton > button, .stTabs [data-baseweb="tab"],
   .stTextInput > div > div > input, .stMetric, etc.
2. Use the existing colour palette — do not break the dark theme:
   Background: #0f1014   Card: #1a1b22   Emerald: #00d26a
   Purple: #8b56f6        Red: #ff4040    Text: #f0f0f5
3. Each change must have a comment: /* === Proposal N: Title === */
4. Changes must be purely additive — enhance, don't break existing styles.
5. Return ONLY valid CSS inside a single ```css ... ``` fenced block.
   No explanations, no markdown prose outside the fence."""

CURRENT_DASHBOARD_DESC = """
Dashboard: Quantum Edge Trading
Stack: Streamlit + Plotly + custom CSS

Layout:
  • Full-width dark app, charcoal background (#0f1014)
  • Fixed header bar: logo, nav pills, live timestamp
  • Sidebar: watchlist textarea, Run Scan button, Quick Analyze, top-7 results list
  • Main: VIX box → market heat map → summary metrics → 6 tabs
  • Tabs: AI Research Agent | Results | Whale Activity | Market News | Charts | UI Improvements

Colors: emerald #00d26a, purple #8b56f6, red #ff4040, gold #ffc107
Fonts: Inter (UI), JetBrains Mono (timestamps)
Charts: candlestick, RSI, MACD, score bar, radar, gauge, treemap heatmap
Buttons: 3D elevated box-shadow

Known weaknesses to evaluate:
  • Sidebar always visible (22% screen width consumed)
  • No auto-refresh — manual scan trigger only
  • No keyboard shortcuts
  • No watchlist save/load
  • Charts have no drawing tools
  • No order ticket integration
  • Dark mode only
"""


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

    base = {"unusual": False, "sweep": False, "iv_pct": 50.0, "iv_label": "fair",
            "pc_ratio": 1.0, "call_vol": 0, "put_vol": 0, "l2a_pts": 0, "details": ""}

    try:
        tk      = yf.Ticker(sym)
        expiries = tk.options
        if not expiries:
            _OPT_CACHE[sym] = (now, base)
            return base

        chain = tk.option_chain(expiries[0])
        calls, puts = chain.calls, chain.puts

        # ── IV rank (ATM implied volatility) ─────────────────────────────────
        atm_mask = abs(calls["strike"] - price) / price < 0.05
        atm_calls = calls[atm_mask]
        iv_raw = float(atm_calls["impliedVolatility"].mean()) * 100 if not atm_calls.empty else 50.0
        iv_pct = round(min(200.0, iv_raw), 1)
        iv_label = "cheap" if iv_pct < 30 else ("fair" if iv_pct < 60 else "expensive")

        # ── Unusual flow: high vol relative to OI ────────────────────────────
        def _unusual(df):
            df = df.copy()
            df["vol"]  = df["volume"].fillna(0)
            df["oi"]   = df["openInterest"].fillna(1).clip(lower=1)
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
        if iv_pct < 30:    l2a_pts += 15  # cheap options = better risk/reward
        elif iv_pct < 50:  l2a_pts += 10  # fair
        else:              l2a_pts -= 5   # expensive, penalise
        if unusual:        l2a_pts += 10  # smart money positioning
        if sweep:          l2a_pts += 15  # big institutional sweep
        l2a_pts = max(0, l2a_pts)

        details_parts = []
        if sweep:    details_parts.append(f"SWEEP detected (max {max(max_call_vol, max_put_vol):,} contracts)")
        if unusual:  details_parts.append("Unusual vol/OI ratio")
        details_parts.append(f"P/C={pc_ratio}")
        details_parts.append(f"ATM_IV≈{iv_pct:.0f}% ({iv_label})")

        result = {
            "unusual":   unusual,
            "sweep":     sweep,
            "iv_pct":    iv_pct,
            "iv_label":  iv_label,
            "pc_ratio":  pc_ratio,
            "call_vol":  total_call,
            "put_vol":   total_put,
            "l2a_pts":   l2a_pts,
            "details":   " | ".join(details_parts),
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
    """Fetch recent sentiment from MarketAux. Cached 30 min."""
    now = time.time()
    if sym in _MA_CACHE:
        ts, cached = _MA_CACHE[sym]
        if now - ts < _ma_ttl():
            return cached

    base = {"score": 50, "label": "Neutral", "count": 0, "l3_pts": 0, "headlines": []}

    if not MA_API_TOKEN:
        return base

    try:
        url = (
            f"https://api.marketaux.com/v1/news/all"
            f"?symbols={sym}&api_token={MA_API_TOKEN}"
            f"&language=en&limit=5&filter_entities=true"
        )
        r    = requests.get(url, timeout=8)
        data = r.json()
        arts = data.get("data", [])

        pos = neg = neut = 0
        headlines = []
        for art in arts:
            headlines.append(art.get("title", "")[:80])
            for ent in art.get("entities", []):
                s = float(ent.get("sentiment_score", 0))
                if s > 0.1:   pos  += 1
                elif s < -0.1: neg += 1
                else:          neut += 1

        total = pos + neg + neut
        score = int(((pos - neg) / total + 1) / 2 * 100) if total else 50
        label = "Bullish" if score >= 62 else ("Bearish" if score <= 38 else "Neutral")

        # L3 confidence contribution (max 20 pts)
        l3_pts = 0
        if label == "Bullish":  l3_pts = 20
        elif label == "Neutral": l3_pts = 8
        else:                    l3_pts = 0

        result = {
            "score":     score,
            "label":     label,
            "count":     len(arts),
            "bull":      pos, "bear": neg, "neut": neut,
            "l3_pts":    l3_pts,
            "headlines": headlines[:3],
        }
        _MA_CACHE[sym] = (now, result)
        return result

    except Exception:
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

        for r in rows:
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

            lines.append("")

    lines.append(
        "NOTE: Use this data to produce trade cards in the required format. "
        "Only recommend HIGH and MEDIUM confidence plays."
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

    if not HAS_GROQ or not GROQ_API_KEY:
        return _rule_based_response(question, scan_results, simple_mode)

    # ── Full multi-layer validation (every query) ─────────────────────────────
    screened     = _screen_universe(vix_val=vix_val, progress_cb=progress_cb)
    universe_ctx = _build_universe_context(screened, vix_val)
    watchlist_ctx = _build_context(scan_results, options_data)

    mode_tag = (
        "\n\n[MODE: SIMPLE — plain English, analogies, explain every term]"
        if simple_mode else ""
    )

    user_msg = f"""{watchlist_ctx}

{universe_ctx}

My question: {question}{mode_tag}"""

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_msg},
    ]

    try:
        client = _Groq(api_key=GROQ_API_KEY)

        if stream_callback:
            full_text = ""
            stream = client.chat.completions.create(
                model=GROQ_MODEL, messages=messages,
                max_tokens=3500, temperature=0.4, stream=True,
            )
            for chunk in stream:
                text = chunk.choices[0].delta.content or ""
                if text:
                    full_text += text
                    stream_callback(text)
            return full_text

        resp = client.chat.completions.create(
            model=GROQ_MODEL, messages=messages,
            max_tokens=3500, temperature=0.4, stream=False,
        )
        return resp.choices[0].message.content or "No response generated."

    except Exception as e:
        fallback = _rule_based_response(question, scan_results, simple_mode)
        return f"⚠️ AI error: {e}\n\n---\n\n{fallback}"


def ask_ui_adviser(stream_callback=None) -> str:
    """UI/UX improvement analysis agent — returns proposals + embedded CSS artifact."""
    if not HAS_GROQ or not GROQ_API_KEY:
        return _ui_rule_based()

    messages = [
        {"role": "system", "content": UI_SYSTEM_PROMPT},
        {"role": "user",   "content":
            f"Analyse this trading dashboard and return your improvement proposals "
            f"followed by a CSS ARTIFACT block:\n\n{CURRENT_DASHBOARD_DESC}"},
    ]

    try:
        client = _Groq(api_key=GROQ_API_KEY)

        if stream_callback:
            full_text = ""
            stream = client.chat.completions.create(
                model=GROQ_MODEL, messages=messages,
                max_tokens=4000, temperature=0.35, stream=True,
            )
            for chunk in stream:
                text = chunk.choices[0].delta.content or ""
                if text:
                    full_text += text
                    stream_callback(text)
            return full_text

        resp = client.chat.completions.create(
            model=GROQ_MODEL, messages=messages,
            max_tokens=4000, temperature=0.35, stream=False,
        )
        return resp.choices[0].message.content or "No response generated."

    except Exception as e:
        return f"⚠️ UI adviser error: {e}\n\n---\n\n{_ui_rule_based()}"


def generate_ui_css(approved_titles: list[str], analysis: str,
                    stream_callback=None) -> str:
    """
    Generate targeted CSS for a specific subset of approved proposals.
    Called when the user approves individual items and clicks Apply.
    Returns a CSS string (without fences) ready to write to custom_ui.css.
    """
    if not HAS_GROQ or not GROQ_API_KEY:
        return "/* Groq API key required for CSS generation */"

    approved_str = "\n".join(f"- {t}" for t in approved_titles)
    messages = [
        {"role": "system", "content": UI_CSS_PROMPT},
        {"role": "user",   "content":
            f"Original dashboard analysis:\n{analysis}\n\n"
            f"The user has approved ONLY these proposals — generate CSS for them:\n"
            f"{approved_str}\n\n"
            f"Return a single ```css ... ``` block with all changes."},
    ]

    try:
        client = _Groq(api_key=GROQ_API_KEY)

        if stream_callback:
            full_text = ""
            stream = client.chat.completions.create(
                model=GROQ_MODEL, messages=messages,
                max_tokens=2500, temperature=0.2, stream=True,
            )
            for chunk in stream:
                text = chunk.choices[0].delta.content or ""
                if text:
                    full_text += text
                    stream_callback(text)
            return full_text

        resp = client.chat.completions.create(
            model=GROQ_MODEL, messages=messages,
            max_tokens=2500, temperature=0.2, stream=False,
        )
        return resp.choices[0].message.content or "/* No CSS generated */"

    except Exception as e:
        return f"/* CSS generation error: {e} */"


def extract_css_from_response(text: str) -> str:
    """Pull the raw CSS out of a markdown ```css ... ``` fence."""
    import re
    m = re.search(r"```css\s*([\s\S]*?)```", text, re.IGNORECASE)
    return m.group(1).strip() if m else ""


# ═══════════════════════════════════════════════════════════════════════════════
# RULE-BASED FALLBACK  (no Groq key)
# ═══════════════════════════════════════════════════════════════════════════════
def _rule_based_response(question: str, scan_results: list[dict], simple_mode: bool = False) -> str:
    q = question.lower()

    no_groq_note = (
        "\n\n> *Add `GROQ_API_KEY` to `.env` to unlock the full Research Agent with "
        "5-layer validation, live options flow, and AI-generated trade cards.*"
    )

    if not scan_results:
        return (
            "## 🔬 Research Agent — Groq key required\n\n"
            "The multi-layer validation pipeline needs the Groq AI key to run.\n"
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


def _ui_rule_based() -> str:
    return """## 🖌 UI Improvement Proposals

*Add a GROQ_API_KEY for AI-powered comparison against TradingView, Bloomberg, and thinkorswim.*

**1. Collapsible Sidebar** — MEDIUM IMPACT
Sidebar consumes 22% screen width permanently. A collapse toggle reclaims space for charts.

**2. Watchlist Save/Load** — HIGH IMPACT
No persistence between sessions. Named watchlist profiles would improve daily workflow.

**3. Auto-Refresh Toggle** — MEDIUM IMPACT
TradingView refreshes automatically. Optional auto-scan every 15–30 min keeps signals current.

**4. Keyboard Shortcuts** — LOW IMPACT
`/` to focus search, `S` to scan, arrow keys to navigate symbols.

**5. Chart Drawing Tools** — HIGH IMPACT
Charts are view-only. Trend lines and support/resistance annotations would be a major upgrade."""
