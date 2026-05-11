"""
Chart Analyst — AI-powered options trade analysis using Groq vision.
Model: meta-llama/llama-4-maverick-17b-128e-instruct
  • Vision-capable (reads chart images)
  • 131K token context window
  • Fast inference via Groq — uses your existing GROQ_API_KEY

Upload chart screenshots; get structured DECISION / ENTRY / TARGET / STOP analysis.
Ported from the React artifact app.
"""
import base64
import os
from datetime import datetime
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Best Groq vision model: 128-expert Llama 4 Maverick, 131K context
GROQ_VISION_MODEL = "meta-llama/llama-4-maverick-17b-128e-instruct"
# Fallback if Maverick is at capacity
GROQ_VISION_FALLBACK = "meta-llama/llama-4-scout-17b-16e-instruct"

STRATEGIES = [
    {
        "id": "momentum", "label": "Momentum", "icon": "📈", "best": False,
        "short": "Ride strong price moves backed by volume.",
        "how": (
            "You're jumping on a stock that's already moving hard in one direction. "
            "If a stock is surging up with big volume, you buy a call. If it's tanking fast, "
            "you buy a put. You're not predicting — you're reacting to what's already happening."
        ),
        "best_for": "High-volume, volatile market hours (open & power hour)",
        "risk": "Moves can reverse fast — you need tight stops",
    },
    {
        "id": "breakout", "label": "Breakout", "icon": "🚀", "best": False,
        "short": "Enter when price breaks through a key level.",
        "how": (
            "A stock sits below resistance (a ceiling price) for a while, then punches through it "
            "with conviction. That breakout signals buyers are in control. You buy a call on the break. "
            "Works the same in reverse for puts on support breaks."
        ),
        "best_for": "Pre-market movers, earnings day, news catalysts",
        "risk": "Fake breakouts are common — wait for a candle close above the level",
    },
    {
        "id": "scalping", "label": "Scalping", "icon": "⚡", "best": False,
        "short": "Fast in-and-out trades capturing small moves.",
        "how": (
            "You're not looking for big swings. You enter, grab a quick 20–50% options gain "
            "on a small price move, and get out. Trades can last 2–10 minutes. "
            "Requires sharp focus and fast execution."
        ),
        "best_for": "1-min and 2-min charts, experienced traders",
        "risk": "High frequency = more commissions, more mistakes if distracted",
    },
    {
        "id": "trend", "label": "Trend Following", "icon": "📊", "best": True,
        "short": "Trade with the dominant market direction.",
        "how": (
            "If the overall trend is up (higher highs, higher lows), you only buy calls on dips. "
            "If it's down, you only buy puts on bounces. You're not fighting the market — "
            "you're going with it. Simple and effective."
        ),
        "best_for": "Any skill level — especially good for beginners",
        "risk": "Trends end without warning — watch for reversal signals",
    },
    {
        "id": "meanrevert", "label": "Mean Reversion", "icon": "↔️", "best": False,
        "short": "Bet on price snapping back after going too far.",
        "how": (
            "When a stock has moved way too far up or down too fast, it often pulls back toward "
            "its average. You're fading the extreme move. RSI overbought/oversold signals are your guide."
        ),
        "best_for": "Choppy, ranging markets with no clear trend",
        "risk": "Can get crushed if the move keeps going — hardest strategy to time",
    },
    {
        "id": "snr", "label": "Support & Resistance", "icon": "📏", "best": False,
        "short": "Trade bounces or breaks at key price levels.",
        "how": (
            "Support is a floor price where buyers keep stepping in. Resistance is a ceiling where "
            "sellers take over. You buy calls when price bounces off support, buy puts when it "
            "bounces off resistance."
        ),
        "best_for": "All skill levels — very clear visual setups",
        "risk": "Levels break — always use a stop below/above the level",
    },
]

TIMEFRAMES = ["1-min", "2-min", "5-min", "15-min", "1-hour", "Daily"]


def get_strategy(strategy_id: str) -> dict | None:
    return next((s for s in STRATEGIES if s["id"] == strategy_id), None)


def get_time_context() -> str:
    h = datetime.now().hour
    if h < 10: return "market open (high volatility window)"
    if h < 12: return "mid-morning"
    if h < 14: return "midday (typically low volume/chop)"
    if h < 15: return "early afternoon"
    return "power hour (high volatility window)"


def analyze_chart(
    images: list[tuple[bytes, str]],  # [(raw_bytes, mime_type), ...]
    strategy_id: str,
) -> str:
    """
    Send chart image(s) + strategy context to Groq Llama 4 Maverick (vision).
    Uses your existing GROQ_API_KEY — no extra cost or signup needed.
    Returns structured analysis text with DECISION / ENTRY / TARGET / STOP sections.
    """
    try:
        from groq import Groq as _Groq
    except ImportError:
        return "ERROR: Run `pip install groq` to enable chart analysis."

    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        return "ERROR: GROQ_API_KEY not found in .env file."

    strat    = get_strategy(strategy_id)
    time_ctx = get_time_context()

    prompt = f"""You are an expert options trading coach helping a trader build skill fast.

Strategy: {strat['label']} — {strat['how']}
Time of day: {time_ctx}
Charts provided: {len(images)}

Analyze the chart(s) and respond in this exact format:

DECISION
[BUY CALL / BUY PUT / NO TRADE / NEED MORE DATA]

ENTRY
[Premium price range to pay for the contract]

TARGET
[Specific price or % to take profit]

STOP / EXIT SIGNAL
[What price action means get out now]

HOLD VS EXIT
[What to watch mid-trade]

TEACHING MOMENT
[2-3 sentences max. Plain language. Explain exactly what you saw in the chart that drove this call — the specific pattern, candle, indicator, or level. Help the trader recognize it next time on their own.]

If NEED MORE DATA: skip ENTRY/TARGET/STOP and instead say exactly which timeframe to pull and why. Be direct. No filler."""

    # Build multimodal content — images first, then the text prompt
    content: list[dict] = []
    for raw, mime in images:
        b64 = base64.standard_b64encode(raw).decode("utf-8")
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:{mime};base64,{b64}"},
        })
    content.append({"type": "text", "text": prompt})

    client = _Groq(api_key=api_key)

    # Try Maverick first, fall back to Scout if at capacity
    for model in (GROQ_VISION_MODEL, GROQ_VISION_FALLBACK):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": content}],
                max_tokens=1500,
                temperature=0.2,   # low temp = more precise, less hallucination
            )
            return resp.choices[0].message.content or "No response returned."
        except Exception as e:
            err = str(e)
            if "model" in err.lower() or "capacity" in err.lower() or "rate" in err.lower():
                continue   # try fallback model
            return f"ERROR: {err}"

    return "ERROR: Both Groq vision models are currently unavailable. Try again in a moment."
