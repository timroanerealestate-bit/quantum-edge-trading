"""
Market data utilities — VIX score & sector / stock heat-map.
Quantum Edge Trading  •  yfinance-backed, no paid API required.

Real-time strategy:
  • Market OPEN  → intraday intervals (2 m), fast_info for VIX, short caches
  • Market CLOSED → daily bars, standard caches
"""
from __future__ import annotations
from datetime import datetime, time as dtime
from zoneinfo import ZoneInfo
import pandas as pd
import yfinance as yf

_ET = ZoneInfo("America/New_York")

# ── Market-hours helpers ──────────────────────────────────────────────────────
def is_market_open() -> bool:
    """Return True if US equities market is currently open (9:30–16:00 ET, Mon–Fri)."""
    now = datetime.now(_ET)
    if now.weekday() >= 5:                          # Saturday / Sunday
        return False
    t = now.time()
    return dtime(9, 30) <= t <= dtime(16, 0)


def is_premarket() -> bool:
    """4:00–9:30 ET weekdays."""
    now = datetime.now(_ET)
    if now.weekday() >= 5:
        return False
    t = now.time()
    return dtime(4, 0) <= t < dtime(9, 30)


def is_afterhours() -> bool:
    """16:00–20:00 ET weekdays."""
    now = datetime.now(_ET)
    if now.weekday() >= 5:
        return False
    t = now.time()
    return dtime(16, 0) < t <= dtime(20, 0)


def get_market_status() -> dict:
    """Return market status info for UI display."""
    if is_market_open():
        return {"label": "MARKET OPEN",  "color": "#00d26a", "dot": "🟢", "open": True}
    if is_premarket():
        return {"label": "PRE-MARKET",   "color": "#ffc107", "dot": "🟡", "open": False}
    if is_afterhours():
        return {"label": "AFTER HOURS",  "color": "#8b56f6", "dot": "🟣", "open": False}
    return     {"label": "MARKET CLOSED","color": "#8892a4", "dot": "⚫", "open": False}


def cache_ttl(open_sec: int, closed_sec: int) -> int:
    """Return the right TTL (seconds) based on whether market is open."""
    return open_sec if is_market_open() else closed_sec


# ── Sector ETF map ────────────────────────────────────────────────────────────
SECTOR_ETFS: dict[str, str] = {
    "Technology":   "XLK",
    "Financials":   "XLF",
    "Energy":       "XLE",
    "Health Care":  "XLV",
    "Industrials":  "XLI",
    "Comm Svcs":    "XLC",
    "Cons Discr":   "XLY",
    "Cons Staples": "XLP",
    "Materials":    "XLB",
    "Real Estate":  "XLRE",
    "Utilities":    "XLU",
}

SECTOR_STOCKS: dict[str, list[str]] = {
    "Technology":   ["AAPL", "MSFT", "NVDA", "AMD",  "AVGO", "ORCL", "CRM",  "INTC"],
    "Financials":   ["JPM",  "BAC",  "GS",   "MS",   "WFC",  "C",    "AXP",  "BLK"],
    "Energy":       ["XOM",  "CVX",  "COP",  "SLB",  "EOG",  "MPC",  "PSX",  "VLO"],
    "Health Care":  ["UNH",  "LLY",  "JNJ",  "ABBV", "MRK",  "PFE",  "TMO",  "ABT"],
    "Comm Svcs":    ["GOOGL","META", "NFLX", "DIS",  "T",    "VZ",   "CMCSA","EA"],
    "Cons Discr":   ["AMZN", "TSLA", "HD",   "MCD",  "NKE",  "LOW",  "SBUX", "GM"],
    "Industrials":  ["CAT",  "BA",   "HON",  "GE",   "LMT",  "UPS",  "RTX",  "DE"],
    "Cons Staples": ["WMT",  "PG",   "KO",   "PEP",  "COST", "PM",   "MO",   "CL"],
    "Materials":    ["LIN",  "FCX",  "NEM",  "APD",  "ECL",  "NUE",  "CF",   "ALB"],
    "Real Estate":  ["AMT",  "PLD",  "EQIX", "CCI",  "PSA",  "DLR",  "O",    "WELL"],
    "Utilities":    ["NEE",  "DUK",  "SO",   "AEP",  "D",    "EXC",  "SRE",  "XEL"],
}

_MEGA  = {"AAPL","MSFT","NVDA","GOOGL","AMZN","META","TSLA","AVGO","LLY","UNH",
          "JPM","XOM","V","MA","JNJ","PG","HD","MRK","ABBV","WMT","COST"}
_LARGE = {"AMD","INTC","ORCL","BAC","WFC","MS","C","GS","AXP","BLK","CVX","COP",
          "SLB","EOG","MPC","PSX","VLO","NFLX","DIS","T","VZ","CMCSA","EA",
          "MCD","NKE","LOW","SBUX","GM","CAT","BA","HON","GE","LMT","UPS","RTX",
          "DE","KO","PEP","PM","MO","CL","LIN","FCX","NEM","APD","ECL","NUE",
          "CF","ALB","AMT","PLD","EQIX","CCI","PSA","DLR","O","WELL","NEE","DUK",
          "SO","AEP","D","EXC","SRE","XEL","TMO","ABT","PFE","CRM"}


def _weight(sym: str) -> int:
    if sym in _MEGA:  return 35
    if sym in _LARGE: return 18
    return 8


# ── Batch price-change fetch ──────────────────────────────────────────────────
def _batch_changes(symbols: list[str]) -> dict[str, float]:
    """
    Return {symbol: pct_change_today}.

    Market OPEN  → 2-minute intraday bars (today vs previous close).
    Market CLOSED → daily bars (latest close vs prior close).
    """
    if not symbols:
        return {}

    open_ = is_market_open()

    try:
        if open_:
            # Intraday: compare latest tick to previous session close
            raw = yf.download(
                symbols, period="2d", interval="2m",
                progress=False, auto_adjust=True,
            )
        else:
            raw = yf.download(
                symbols, period="5d", interval="1d",
                progress=False, auto_adjust=True,
            )

        if raw.empty:
            return {s: 0.0 for s in symbols}

        if isinstance(raw.columns, pd.MultiIndex):
            closes = raw["Close"]
        else:
            closes = raw[["Close"]].rename(columns={"Close": symbols[0]})

        result: dict[str, float] = {}
        for sym in symbols:
            try:
                if sym not in closes.columns:
                    result[sym] = 0.0
                    continue
                col = closes[sym].dropna()
                if len(col) >= 2:
                    c, p = float(col.iloc[-1]), float(col.iloc[-2])
                    result[sym] = round(((c - p) / p) * 100, 2) if p else 0.0
                else:
                    result[sym] = 0.0
            except Exception:
                result[sym] = 0.0
        return result

    except Exception:
        return {s: 0.0 for s in symbols}


# ── VIX ───────────────────────────────────────────────────────────────────────
def get_vix() -> dict:
    """
    Fetch the current CBOE VIX.

    Market OPEN  → yf.Ticker.fast_info (most recent tick, ~15 min delay on free).
    Market CLOSED → daily history.
    """
    base_err = {"value": None, "change": None, "change_pct": None,
                "label": "Unavailable", "color": "#8892a4", "tier": "⚪",
                "as_of": "—"}
    try:
        tk = yf.Ticker("^VIX")

        if is_market_open() or is_afterhours() or is_premarket():
            # fast_info gives the most recent available price
            fi   = tk.fast_info
            curr = round(float(fi.last_price), 2)
            prev = round(float(fi.previous_close), 2)
        else:
            hist = tk.history(period="5d", interval="1d")
            if hist.empty:
                return base_err
            curr = round(float(hist["Close"].iloc[-1]), 2)
            prev = float(hist["Close"].iloc[-2]) if len(hist) >= 2 else curr

        chg     = round(curr - prev, 2)
        chg_pct = round((chg / prev) * 100, 2) if prev else 0.0
        as_of   = datetime.now(_ET).strftime("%H:%M:%S ET")

        if curr < 15:   label, color, tier = "LOW VOLATILITY", "#00d26a", "🟢"
        elif curr < 20: label, color, tier = "CALM",           "#34d399", "🟢"
        elif curr < 25: label, color, tier = "NORMAL",         "#ffc107", "🟡"
        elif curr < 30: label, color, tier = "ELEVATED",       "#ff8c00", "🟠"
        else:           label, color, tier = "HIGH FEAR",      "#ff4040", "🔴"

        return {"value": curr, "change": chg, "change_pct": chg_pct,
                "label": label, "color": color, "tier": tier, "as_of": as_of}

    except Exception:
        return base_err


# ── Heat map ──────────────────────────────────────────────────────────────────
def get_heatmap_data() -> dict:
    """
    Fetch % changes for all sectors + representative stocks.
    Uses intraday bars during market hours for near-real-time colours.
    """
    all_syms = list(SECTOR_ETFS.values()) + list(
        {s for syms in SECTOR_STOCKS.values() for s in syms}
    )
    chg_map = _batch_changes(all_syms)

    sectors = [
        {"name": name, "symbol": etf, "change_pct": chg_map.get(etf, 0.0)}
        for name, etf in SECTOR_ETFS.items()
    ]
    stocks = {
        sector: [
            {"symbol": s, "change_pct": chg_map.get(s, 0.0), "weight": _weight(s)}
            for s in syms
        ]
        for sector, syms in SECTOR_STOCKS.items()
    }
    return {"sectors": sectors, "stocks": stocks}
