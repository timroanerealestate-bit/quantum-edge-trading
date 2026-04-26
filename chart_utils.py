"""
Plotly chart builders for the Training Bot dashboard.
Premium charcoal/emerald/purple theme — Quantum Edge Trading.
"""
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots

try:
    import ta
    HAS_TA = True
except ImportError:
    HAS_TA = False

# ── Palette — Quantum Edge Trading ───────────────────────────────────────────
C_BG       = "#0f1014"       # deepest background — charcoal
C_PAPER    = "#0f1014"       # chart paper — charcoal
C_GRID     = "#1a1b22"       # grid lines — slightly lighter charcoal
C_TEXT     = "#f0f0f5"       # primary text — off-white
C_MUTED    = "#8892a4"       # muted text / axis ticks — light gray
C_BULL     = "#00d26a"       # emerald green  — bullish candles / positive
C_BEAR     = "#ff4040"       # vivid red      — bearish candles / negative
C_CYAN     = "#00d26a"       # emerald        — primary accent (SMA20, RSI)
C_VIOLET   = "#8b56f6"       # purple         — MACD line / AI section
C_AMBER    = "#ffc107"       # gold           — SMA50 / signal line
C_WATCH    = "#ffc107"       # alias
C_NEUTRAL  = "#8892a4"       # slate
C_FILL_BUL = "rgba(0,210,106,0.06)"
C_FILL_RSI = "rgba(0,210,106,0.06)"
C_FILL_RAD = "rgba(0,210,106,0.10)"


# ── Theme helper ──────────────────────────────────────────────────────────────
def _dark_layout(**overrides) -> dict:
    base = dict(
        template="plotly_dark",
        paper_bgcolor=C_PAPER,
        plot_bgcolor=C_BG,
        font=dict(color=C_TEXT, family="Inter, Arial, sans-serif", size=12),
        margin=dict(l=50, r=24, t=52, b=32),
        legend=dict(
            bgcolor="rgba(15,16,20,0.9)",
            bordercolor="rgba(0,210,106,0.12)",
            borderwidth=1,
            font=dict(size=11, color="rgba(240,240,245,0.6)"),
        ),
        xaxis=dict(
            gridcolor=C_GRID,
            zerolinecolor="rgba(0,210,106,0.08)",
            tickfont=dict(color=C_MUTED, size=10),
            linecolor="rgba(0,210,106,0.08)",
        ),
        yaxis=dict(
            gridcolor=C_GRID,
            zerolinecolor="rgba(0,210,106,0.08)",
            tickfont=dict(color=C_MUTED, size=10),
            linecolor="rgba(0,210,106,0.08)",
        ),
    )
    base.update(overrides)
    return base


def _empty_fig(msg: str, height: int = 220) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        x=0.5, y=0.5, xref="paper", yref="paper",
        text=msg, showarrow=False,
        font=dict(color=C_MUTED, size=13),
    )
    fig.update_layout(**_dark_layout(height=height))
    return fig


# ── Price charts ──────────────────────────────────────────────────────────────
def candlestick_chart(symbol: str, period: str = "3mo") -> go.Figure:
    """Candlestick with volume bars and SMA overlays."""
    try:
        df = yf.Ticker(symbol).history(period=period)
        if df.empty:
            return _empty_fig(f"No price data for {symbol}")

        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            row_heights=[0.72, 0.28],
            vertical_spacing=0.02,
        )

        # Candlestick
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df["Open"], high=df["High"],
            low=df["Low"],   close=df["Close"],
            name=symbol,
            increasing=dict(line=dict(color=C_BULL, width=1), fillcolor=C_BULL),
            decreasing=dict(line=dict(color=C_BEAR, width=1), fillcolor=C_BEAR),
        ), row=1, col=1)

        closes = df["Close"]

        # SMA 20 — cyan
        if len(df) >= 20:
            fig.add_trace(go.Scatter(
                x=df.index, y=closes.rolling(20).mean(),
                name="SMA 20",
                line=dict(color=C_CYAN, width=1.8, dash="solid"),
                opacity=0.85,
            ), row=1, col=1)

        # SMA 50 — amber
        if len(df) >= 50:
            fig.add_trace(go.Scatter(
                x=df.index, y=closes.rolling(50).mean(),
                name="SMA 50",
                line=dict(color=C_AMBER, width=1.8, dash="dot"),
                opacity=0.75,
            ), row=1, col=1)

        # Volume bars — colored by direction
        bar_colors = [
            f"rgba(16,185,129,0.55)" if c >= o else f"rgba(239,68,68,0.55)"
            for c, o in zip(df["Close"], df["Open"])
        ]
        fig.add_trace(go.Bar(
            x=df.index, y=df["Volume"],
            name="Volume",
            marker_color=bar_colors,
            showlegend=False,
        ), row=2, col=1)

        fig.update_layout(
            **_dark_layout(
                title=dict(
                    text=f"<b>{symbol}</b>  •  Price Chart ({period})",
                    font=dict(size=15, color=C_TEXT),
                    x=0.01,
                ),
                height=500,
                xaxis_rangeslider_visible=False,
                hovermode="x unified",
            )
        )
        fig.update_yaxes(title_text="Price ($)", row=1, col=1,
                         title_font=dict(color=C_MUTED, size=11))
        fig.update_yaxes(title_text="Volume", row=2, col=1,
                         title_font=dict(color=C_MUTED, size=11))
        return fig

    except Exception as e:
        return _empty_fig(f"Chart error: {e}")


def rsi_chart(symbol: str, period: str = "3mo") -> go.Figure:
    """RSI (14) with gradient fill and overbought/oversold bands."""
    try:
        if not HAS_TA:
            return _empty_fig("Install 'ta' library for RSI")

        df = yf.Ticker(symbol).history(period=period)
        if df.empty:
            return _empty_fig(f"No data for {symbol}")

        rsi = ta.momentum.RSIIndicator(df["Close"], window=14).rsi()

        fig = go.Figure()

        # Shaded zones
        fig.add_hrect(y0=70, y1=100, fillcolor="rgba(239,68,68,0.06)",
                      line_width=0, layer="below")
        fig.add_hrect(y0=0, y1=30, fillcolor="rgba(16,185,129,0.06)",
                      line_width=0, layer="below")

        # RSI line with fill
        fig.add_trace(go.Scatter(
            x=df.index, y=rsi,
            name="RSI (14)",
            line=dict(color=C_CYAN, width=2.2),
            fill="tozeroy",
            fillcolor=C_FILL_RSI,
        ))

        # Overbought / oversold reference lines
        fig.add_hline(y=70, line_dash="dash", line_color="rgba(239,68,68,0.6)",
                      line_width=1.2,
                      annotation_text="Overbought 70",
                      annotation_font=dict(color="rgba(239,68,68,0.7)", size=10),
                      annotation_position="top right")
        fig.add_hline(y=30, line_dash="dash", line_color="rgba(16,185,129,0.6)",
                      line_width=1.2,
                      annotation_text="Oversold 30",
                      annotation_font=dict(color="rgba(16,185,129,0.7)", size=10),
                      annotation_position="bottom right")
        fig.add_hline(y=50, line_dash="dot", line_color="rgba(255,255,255,0.08)",
                      line_width=1)

        fig.update_layout(
            **_dark_layout(
                title=dict(text=f"<b>{symbol}</b>  •  RSI (14)",
                           font=dict(size=14, color=C_TEXT), x=0.01),
                height=270,
                yaxis=dict(range=[0, 100], gridcolor=C_GRID,
                           tickfont=dict(color=C_MUTED)),
            )
        )
        return fig

    except Exception as e:
        return _empty_fig(f"RSI error: {e}")


def macd_chart(symbol: str, period: str = "3mo") -> go.Figure:
    """MACD with violet/amber lines and colored histogram."""
    try:
        if not HAS_TA:
            return _empty_fig("Install 'ta' library for MACD")

        df = yf.Ticker(symbol).history(period=period)
        if df.empty:
            return _empty_fig(f"No data for {symbol}")

        macd_ind  = ta.trend.MACD(df["Close"], window_slow=26, window_fast=12, window_sign=9)
        macd_line = macd_ind.macd()
        signal    = macd_ind.macd_signal()
        hist      = macd_ind.macd_diff()

        # Color histogram bars by value
        hist_colors = [
            "rgba(16,185,129,0.75)" if v >= 0 else "rgba(239,68,68,0.75)"
            for v in hist
        ]

        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            row_heights=[0.52, 0.48],
            vertical_spacing=0.04,
        )

        # MACD line — violet
        fig.add_trace(go.Scatter(
            x=df.index, y=macd_line, name="MACD",
            line=dict(color=C_VIOLET, width=2.2),
        ), row=1, col=1)

        # Signal line — amber
        fig.add_trace(go.Scatter(
            x=df.index, y=signal, name="Signal",
            line=dict(color=C_AMBER, width=1.8, dash="dot"),
        ), row=1, col=1)

        # Histogram
        fig.add_trace(go.Bar(
            x=df.index, y=hist, name="Momentum",
            marker_color=hist_colors,
        ), row=2, col=1)

        fig.update_layout(
            **_dark_layout(
                title=dict(text=f"<b>{symbol}</b>  •  MACD (12/26/9)",
                           font=dict(size=14, color=C_TEXT), x=0.01),
                height=330,
                hovermode="x unified",
            )
        )
        return fig

    except Exception as e:
        return _empty_fig(f"MACD error: {e}")


# ── Signal charts ─────────────────────────────────────────────────────────────
def score_bar_chart(results: list[dict]) -> go.Figure:
    """Horizontal ranked bar chart — gradient colored by recommendation."""
    if not results:
        return _empty_fig("No results yet — run a scan")

    color_map = {
        "STRONG BUY": C_BULL,
        "BUY":        "#34d399",
        "WATCH":      C_AMBER,
        "NEUTRAL":    C_NEUTRAL,
        "AVOID":      C_BEAR,
    }

    symbols = [r["symbol"]          for r in results]
    scores  = [r["composite_score"] for r in results]
    recs    = [r["recommendation"]  for r in results]
    prices  = [r.get("price", "")   for r in results]
    colors  = [color_map.get(r, C_NEUTRAL) for r in recs]
    labels  = [f"  {s:.0f}  •  {r}" for s, r in zip(scores, recs)]

    fig = go.Figure(go.Bar(
        x=scores,
        y=symbols,
        orientation="h",
        marker=dict(
            color=scores,
            colorscale=[
                [0.00, "#ff4040"],
                [0.30, "#ffc107"],
                [0.55, "#00d26a"],
                [0.75, "#00d26a"],
                [1.00, "#00ff88"],
            ],
            cmin=0, cmax=100,
            colorbar=dict(
                title=dict(text="Score", font=dict(color=C_MUTED, size=11)),
                tickfont=dict(color=C_MUTED, size=10),
                bgcolor="rgba(0,0,0,0)",
                bordercolor="rgba(0,210,106,0.12)",
                thickness=12,
                len=0.8,
            ),
            line=dict(width=0),
        ),
        text=labels,
        textposition="outside",
        textfont=dict(size=11, color=C_TEXT),
        hovertemplate="<b>%{y}</b><br>Score: %{x:.1f}/100<extra></extra>",
    ))

    fig.update_layout(
        **_dark_layout(
            title=dict(
                text="<b>Ranked Opportunities</b>",
                font=dict(size=15, color=C_TEXT), x=0.01,
            ),
            height=max(300, len(symbols) * 42 + 90),
            xaxis=dict(range=[0, 115], title="Composite Score / 100",
                       gridcolor=C_GRID, tickfont=dict(color=C_MUTED)),
            yaxis=dict(autorange="reversed", tickfont=dict(color=C_TEXT, size=12)),
        ),
        bargap=0.35,
    )
    return fig


def score_radar(scores: dict, symbol: str) -> go.Figure:
    """Radar / spider chart — cyan fill, violet border."""
    cats = ["Institutional", "Technical", "Volume", "Sentiment"]
    vals = [
        scores.get("institutional", 0),
        scores.get("technical",     0),
        scores.get("volume",        0),
        scores.get("sentiment",     0),
    ]
    cats_loop = cats + [cats[0]]
    vals_loop = vals + [vals[0]]

    fig = go.Figure()

    # Filled area
    fig.add_trace(go.Scatterpolar(
        r=vals_loop,
        theta=cats_loop,
        fill="toself",
        fillcolor="rgba(0,210,106,0.10)",
        line=dict(color=C_CYAN, width=2.5),
        marker=dict(size=7, color=C_CYAN,
                    line=dict(color=C_BG, width=2)),
        name=symbol,
        hovertemplate="%{theta}: <b>%{r:.0f}</b><extra></extra>",
    ))

    fig.update_layout(
        **_dark_layout(
            title=dict(
                text=f"<b>{symbol}</b>  •  Signal Radar",
                font=dict(size=13, color=C_TEXT), x=0.5, xanchor="center",
            ),
            height=310,
        ),
        polar=dict(
            radialaxis=dict(
                visible=True, range=[0, 100],
                color=C_MUTED,
                gridcolor="rgba(255,255,255,0.06)",
                tickfont=dict(size=9, color=C_MUTED),
                tickvals=[25, 50, 75, 100],
            ),
            angularaxis=dict(
                color=C_TEXT,
                gridcolor="rgba(255,255,255,0.06)",
                tickfont=dict(size=11, color="#94a3b8"),
            ),
            bgcolor=C_BG,
        ),
        showlegend=False,
    )
    return fig


def sentiment_gauge(score: float, label: str = "") -> go.Figure:
    """Gauge chart — gradient colored by score."""
    if score >= 60:
        bar_color, glow = C_BULL, "rgba(16,185,129,0.3)"
    elif score <= 40:
        bar_color, glow = C_BEAR, "rgba(239,68,68,0.3)"
    else:
        bar_color, glow = C_AMBER, "rgba(245,158,11,0.3)"

    title_text = "Sentiment" + (f"<br><span style='font-size:11px;color:#94a3b8'>{label}</span>" if label else "")

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        title={"text": title_text, "font": {"size": 13, "color": C_TEXT}},
        number={"suffix": "/100", "font": {"size": 26, "color": bar_color}},
        gauge={
            "axis": {
                "range": [0, 100],
                "tickcolor": C_MUTED,
                "tickfont": {"size": 9, "color": C_MUTED},
            },
            "bar":     {"color": bar_color, "thickness": 0.22},
            "bgcolor": "#1a1b22",
            "borderwidth": 1,
            "bordercolor": "rgba(255,255,255,0.06)",
            "steps": [
                {"range": [0,  40],  "color": "rgba(239,68,68,0.1)"},
                {"range": [40, 60],  "color": "rgba(245,158,11,0.08)"},
                {"range": [60, 100], "color": "rgba(16,185,129,0.1)"},
            ],
            "threshold": {
                "line":  {"color": bar_color, "width": 3},
                "value": score,
            },
        },
    ))
    fig.update_layout(**_dark_layout(height=235))
    return fig


def mini_score_gauge(score: float, title: str = "Score") -> go.Figure:
    """Compact gauge for embedding in columns."""
    if score >= 60:
        color = C_BULL
    elif score < 40:
        color = C_BEAR
    else:
        color = C_AMBER

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        title={"text": title, "font": {"size": 12, "color": C_MUTED}},
        number={"font": {"size": 22, "color": color}},
        gauge={
            "axis":  {"range": [0, 100], "tickcolor": C_MUTED,
                      "tickfont": {"size": 8}},
            "bar":   {"color": color, "thickness": 0.25},
            "bgcolor": "#1a1b22",
            "borderwidth": 1,
            "bordercolor": "rgba(255,255,255,0.05)",
            "steps": [
                {"range": [0,  40],  "color": "rgba(239,68,68,0.08)"},
                {"range": [40, 60],  "color": "rgba(245,158,11,0.06)"},
                {"range": [60, 100], "color": "rgba(16,185,129,0.08)"},
            ],
        },
    ))
    fig.update_layout(
        **_dark_layout(height=185, margin=dict(l=20, r=20, t=40, b=10))
    )
    return fig


# ── Market heat map ───────────────────────────────────────────────────────────
def market_heatmap_chart(heatmap_data: dict) -> go.Figure:
    """
    FinViz-style treemap heat map.
    heatmap_data = {"sectors": [...], "stocks": {sector: [...]}}
    Colors: deep-red (≤-3%) → neutral (#1a1b22) → emerald (≥+3%)
    """
    sectors       = heatmap_data.get("sectors", [])
    stocks_by_sec = heatmap_data.get("stocks",  {})

    labels  = ["Market"]
    parents = [""]
    values  = [0]
    colors  = [0.0]
    texts   = [""]

    # Sector nodes
    for s in sectors:
        nm, chg = s["name"], s["change_pct"]
        sign = "+" if chg >= 0 else ""
        labels.append(nm);  parents.append("Market"); values.append(100)
        colors.append(chg); texts.append(f"<b>{nm}</b><br>{sign}{chg:.2f}%")

    # Stock leaf nodes
    for sector, stock_list in stocks_by_sec.items():
        for stk in stock_list:
            sym, chg, w = stk["symbol"], stk["change_pct"], stk.get("weight", 10)
            sign = "+" if chg >= 0 else ""
            labels.append(sym);  parents.append(sector); values.append(w)
            colors.append(chg);  texts.append(f"<b>{sym}</b><br>{sign}{chg:.2f}%")

    # Red → neutral → emerald color scale
    colorscale = [
        [0.00, "#6b0000"],   # deep red    ≤ −3 %
        [0.28, "#c0392b"],   # red         −2 %
        [0.44, "#e74c3c"],   # light red   −0.5 %
        [0.50, "#1a1b22"],   # neutral      0 %
        [0.56, "#196f3d"],   # light green +0.5 %
        [0.72, "#27ae60"],   # green       +2 %
        [1.00, "#00d26a"],   # emerald     ≥ +3 %
    ]

    fig = go.Figure(go.Treemap(
        labels   = labels,
        parents  = parents,
        values   = values,
        text     = texts,
        textinfo = "text",
        hovertemplate = "%{text}<extra></extra>",
        marker   = dict(
            colors     = colors,
            colorscale = colorscale,
            cmid       = 0,
            cmin       = -3,
            cmax       = 3,
            showscale  = True,
            colorbar   = dict(
                title       = dict(text="% Chg", font=dict(color=C_MUTED, size=10)),
                tickfont    = dict(color=C_MUTED, size=9),
                tickformat  = "+.1f",
                thickness   = 10,
                len         = 0.75,
                bgcolor     = "rgba(0,0,0,0)",
                bordercolor = "rgba(255,255,255,0.05)",
            ),
            line = dict(width=0.5, color=C_BG),
        ),
        textfont = dict(family="Inter, sans-serif", size=11, color="#ffffff"),
        pathbar  = dict(visible=False),
        tiling   = dict(packing="squarify", pad=2),
    ))

    fig.update_layout(
        **_dark_layout(
            height=520,
            margin=dict(l=0, r=60, t=8, b=0),
        ),
    )
    return fig
