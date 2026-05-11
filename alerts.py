"""
Alert engine — fires when scan scores cross thresholds.
Persists history to alerts.json. Optional SMTP email via Gmail/any provider.
"""
import json
import smtplib
from datetime import datetime
from email.mime.text import MIMEText
from pathlib import Path

_ALERTS_PATH = Path(__file__).parent / "alerts.json"


def load_alert_history() -> list[dict]:
    if _ALERTS_PATH.exists():
        try:
            return json.loads(_ALERTS_PATH.read_text(encoding="utf-8"))[-500:]
        except Exception:
            return []
    return []


def save_alert_history(history: list[dict]) -> None:
    _ALERTS_PATH.write_text(
        json.dumps(history[-500:], indent=2, default=str),
        encoding="utf-8",
    )


def check_alerts(scan_results: list[dict], config: dict) -> list[dict]:
    """
    Evaluate scan results against alert rules.

    config keys:
      score_threshold   int   (default 75)  — alert if composite_score >= this
      rsi_oversold      float (default 30)  — alert if RSI <= this
      macd_crossover    bool  (default True) — alert on MACD histogram crossing up
      strong_buy_only   bool  (default False) — only alert STRONG BUY signals
    """
    thresh    = int(config.get("score_threshold", 75))
    rsi_ov    = float(config.get("rsi_oversold", 30))
    macd_x    = bool(config.get("macd_crossover", True))
    sb_only   = bool(config.get("strong_buy_only", False))
    triggered = []

    for r in scan_results:
        rec  = r.get("recommendation", "")
        if sb_only and rec != "STRONG BUY":
            continue

        tech    = r.get("tech", {})
        score   = r["composite_score"]
        rsi     = tech.get("rsi")
        macd    = tech.get("macd_signal")
        reasons = []

        if score >= thresh:
            reasons.append(f"Score {score:.0f} ≥ threshold {thresh}")
        if rsi is not None and rsi <= rsi_ov:
            reasons.append(f"RSI oversold at {rsi:.1f} ≤ {rsi_ov}")
        if macd_x and macd == "crossing_up":
            reasons.append("MACD histogram crossing up (momentum shift)")

        if reasons:
            triggered.append({
                "symbol":         r["symbol"],
                "score":          score,
                "recommendation": rec,
                "price":          r.get("price", "N/A"),
                "reasons":        reasons,
                "fired_at":       datetime.now().isoformat(),
                "read":           False,
            })

    return triggered


def send_email(alerts: list[dict], smtp_cfg: dict) -> tuple[bool, str]:
    """
    Send alert email via SMTP (supports Gmail + any TLS provider).

    smtp_cfg keys: host, port, user, password, recipient
    Returns (success: bool, error_message: str).
    """
    if not alerts:
        return True, ""
    user = smtp_cfg.get("user", "").strip()
    pwd  = smtp_cfg.get("password", "").strip()
    if not user or not pwd:
        return False, "SMTP credentials not configured"

    lines = [f"Quantum Edge Trading — {len(alerts)} alert(s) fired\n"]
    for a in alerts:
        lines.append(
            f"[{a['score']:.0f}/100]  {a['symbol']}  —  {a['recommendation']}"
            f"  @ ${a['price']}"
        )
        for r in a["reasons"]:
            lines.append(f"  • {r}")
        lines.append("")
    lines.append("---\nQuantum Edge Trading Dashboard")

    msg            = MIMEText("\n".join(lines))
    msg["Subject"] = f"QE Alert: {', '.join(a['symbol'] for a in alerts[:5])}"
    msg["From"]    = user
    msg["To"]      = smtp_cfg.get("recipient", user)

    try:
        host = smtp_cfg.get("host", "smtp.gmail.com")
        port = int(smtp_cfg.get("port", 587))
        with smtplib.SMTP(host, port) as s:
            s.ehlo()
            s.starttls()
            s.login(user, pwd)
            s.sendmail(user, msg["To"], msg.as_string())
        return True, ""
    except Exception as e:
        return False, str(e)
