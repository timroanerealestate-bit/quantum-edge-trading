"""
Paper trading portfolio tracker — persists to portfolio.json.
Handles stocks, call options, and put options with live P&L via yfinance.
"""
import json
import uuid
from datetime import datetime
from pathlib import Path
import yfinance as yf

_PORTFOLIO_PATH = Path(__file__).parent / "portfolio.json"


def load_portfolio() -> list[dict]:
    if _PORTFOLIO_PATH.exists():
        try:
            return json.loads(_PORTFOLIO_PATH.read_text(encoding="utf-8"))
        except Exception:
            return []
    return []


def save_portfolio(trades: list[dict]) -> None:
    _PORTFOLIO_PATH.write_text(
        json.dumps(trades, indent=2, default=str),
        encoding="utf-8",
    )


def add_trade(
    trades: list[dict],
    symbol: str,
    trade_type: str,      # "Stock" / "Call" / "Put"
    direction: str,       # "Long" / "Short"
    quantity: float,
    entry_price: float,
    entry_date: str,
    strike: float | None = None,
    expiry: str | None = None,
    notes: str = "",
) -> list[dict]:
    trade = {
        "id":          str(uuid.uuid4())[:8],
        "symbol":      symbol.upper(),
        "trade_type":  trade_type,
        "direction":   direction,
        "quantity":    quantity,
        "entry_price": entry_price,
        "entry_date":  entry_date,
        "strike":      strike,
        "expiry":      expiry,
        "notes":       notes,
        "closed":      False,
        "added_at":    datetime.now().isoformat(),
    }
    return trades + [trade]


def remove_trade(trades: list[dict], trade_id: str) -> list[dict]:
    return [t for t in trades if t.get("id") != trade_id]


def close_trade(trades: list[dict], trade_id: str, exit_price: float) -> list[dict]:
    result = []
    for t in trades:
        if t.get("id") == trade_id:
            t = dict(t)
            t["closed"]     = True
            t["exit_price"] = exit_price
            t["closed_at"]  = datetime.now().isoformat()
        result.append(t)
    return result


def _live_price(symbol: str) -> float | None:
    try:
        return yf.Ticker(symbol).fast_info.last_price
    except Exception:
        return None


def enrich_trades(trades: list[dict]) -> list[dict]:
    """Attach live_price, unrealized_pnl / realized_pnl, pnl_pct to every trade."""
    enriched = []
    for t in trades:
        t = dict(t)
        entry = float(t.get("entry_price", 0) or 0)
        qty   = float(t.get("quantity", 1) or 1)
        mul   = 100 if t.get("trade_type") in ("Call", "Put") else 1
        dir_  = t.get("direction", "Long")

        if t.get("closed"):
            exit_ = float(t.get("exit_price", entry) or entry)
            raw   = (exit_ - entry) if dir_ == "Long" else (entry - exit_)
            t["realized_pnl"] = round(raw * qty * mul, 2)
            t["pnl_pct"]      = round(raw / entry * 100, 2) if entry else 0.0
            enriched.append(t)
            continue

        live = _live_price(t.get("symbol", ""))
        t["live_price"] = round(live, 2) if live else None

        if live and entry:
            raw = (live - entry) if dir_ == "Long" else (entry - live)
            t["unrealized_pnl"] = round(raw * qty * mul, 2)
            t["pnl_pct"]        = round(raw / entry * 100, 2)
        else:
            t["unrealized_pnl"] = None
            t["pnl_pct"]        = None

        enriched.append(t)
    return enriched


def portfolio_summary(enriched: list[dict]) -> dict:
    open_   = [t for t in enriched if not t.get("closed")]
    closed_ = [t for t in enriched if t.get("closed")]
    open_pnls   = [t["unrealized_pnl"] for t in open_   if t.get("unrealized_pnl") is not None]
    closed_pnls = [t["realized_pnl"]   for t in closed_ if t.get("realized_pnl")   is not None]
    all_pnls    = open_pnls + closed_pnls
    winners     = sum(1 for p in all_pnls if p > 0)
    losers      = sum(1 for p in all_pnls if p <= 0)
    return {
        "open_count":   len(open_),
        "closed_count": len(closed_),
        "open_pnl":     round(sum(open_pnls),   2),
        "realized_pnl": round(sum(closed_pnls), 2),
        "total_pnl":    round(sum(all_pnls),    2),
        "winners":      winners,
        "losers":       losers,
        "win_rate":     round(winners / len(all_pnls) * 100, 1) if all_pnls else 0.0,
    }
