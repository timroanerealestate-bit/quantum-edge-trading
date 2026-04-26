"""
Training Bot — Streamlit Dashboard
AI Trading Adviser | Institutional Flow | Charts | News | Options
Run with:  streamlit run dashboard.py
"""
import streamlit as st
import pandas as pd
from datetime import datetime

# ─── Page config MUST be first Streamlit call ─────────────────────────────────
st.set_page_config(
    page_title="Quantum Edge Trading",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Local imports ─────────────────────────────────────────────────────────────
from signal_engine import build_composite_score
from config import DEFAULT_WATCHLIST
import chart_utils as cu
import ai_adviser as adviser
import market_data as md
from options_analyzer import summarize_options

# ─── Inject API keys from Streamlit secrets (Cloud) or .env (local) ───────────
import os as _os_keys

def _load_keys():
    groq_key = ""
    av_key   = ""
    ma_key   = ""

    # Try st.secrets (Streamlit Cloud dashboard)
    try:
        groq_key = st.secrets["GROQ_API_KEY"]
    except Exception:
        pass
    try:
        av_key = st.secrets["ALPHA_VANTAGE_API_KEY"]
    except Exception:
        pass
    try:
        ma_key = st.secrets["MARKETAUX_API_TOKEN"]
    except Exception:
        pass

    # Fall back to .env (local)
    if not groq_key:
        groq_key = _os_keys.getenv("GROQ_API_KEY", "")
    if not av_key:
        av_key   = _os_keys.getenv("ALPHA_VANTAGE_API_KEY", "")
    if not ma_key:
        ma_key   = _os_keys.getenv("MARKETAUX_API_TOKEN", "")

    adviser.GROQ_API_KEY  = groq_key
    adviser.AV_API_KEY    = av_key
    adviser.MA_API_TOKEN  = ma_key
    adviser.HAS_GROQ      = adviser._GROQ_INSTALLED and bool(groq_key)

_load_keys()

# ── DEBUG: remove after confirming keys load on Cloud ─────────────────────────
with st.sidebar:
    with st.expander("🔑 Key Debug", expanded=False):
        st.write("Groq installed:", adviser._GROQ_INSTALLED)
        st.write("Groq key found:", bool(adviser.GROQ_API_KEY))
        st.write("HAS_GROQ:", adviser.HAS_GROQ)
        st.write("Key preview:", adviser.GROQ_API_KEY[:8] + "..." if adviser.GROQ_API_KEY else "EMPTY")

# ─── CSS: Quantum Edge — Charcoal / Emerald / Purple premium theme ───────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600&display=swap');

/* ════════════════════════════════════════════════════
   GLOBAL RESET & BASE
   Palette:
     BG main   #0f1014   charcoal black
     BG card   #1a1b22   dark panel
     BG raised #21222c   elevated surface
     Emerald   #00d26a   gains / active / primary
     Purple    #8b56f6   secondary / AI
     Red       #ff4040   losses / danger
     Gold      #f5a623   watch / neutral
     Text-1    #f0f0f5   primary
     Text-2    #8892a4   secondary
     Border    rgba(255,255,255,0.06)
═══════════════════════════════════════════════════ */
*, *::before, *::after { box-sizing: border-box; }

html, body, [class*="css"], .stApp {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
}

.stApp {
    background-color: #0f1014 !important;
    background-image:
        radial-gradient(ellipse 70% 50% at 15% 0%,  rgba(0,210,106,0.04)  0%, transparent 55%),
        radial-gradient(ellipse 55% 45% at 85% 95%, rgba(139,86,246,0.04) 0%, transparent 55%) !important;
    min-height: 100vh;
}

.main .block-container {
    padding-top: 0 !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
    max-width: 1600px !important;
}

/* ════════════════════════════════════════════════════
   KEYFRAMES
═══════════════════════════════════════════════════ */
@keyframes shimmer {
    0%   { background-position: -600px 0; }
    100% { background-position: 600px 0; }
}
@keyframes live-dot {
    0%, 100% { opacity: 1;    transform: scale(1);    }
    50%       { opacity: 0.3; transform: scale(0.85); }
}
@keyframes glow-pulse {
    0%, 100% { box-shadow: 0 0 10px rgba(0,210,106,0.3), 0 4px 0 #007535; }
    50%       { box-shadow: 0 0 24px rgba(0,210,106,0.55), 0 4px 0 #007535; }
}
@keyframes border-breathe {
    0%, 100% { border-color: rgba(0,210,106,0.25); }
    50%       { border-color: rgba(0,210,106,0.6);  }
}
@keyframes float-up {
    from { opacity: 0; transform: translateY(8px); }
    to   { opacity: 1; transform: translateY(0);   }
}
@keyframes card-in {
    from { opacity: 0; transform: translateY(12px) scale(0.98); }
    to   { opacity: 1; transform: translateY(0)    scale(1);    }
}

/* ════════════════════════════════════════════════════
   TOP HEADER BANNER
═══════════════════════════════════════════════════ */
.qe-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: #13141a;
    border-bottom: 1px solid rgba(255,255,255,0.06);
    padding: 0 32px;
    height: 64px;
    margin: -1rem -2rem 2rem -2rem;
    position: relative;
    overflow: hidden;
}
.qe-header::after {
    content: '';
    position: absolute;
    bottom: 0; left: 0; right: 0; height: 1px;
    background: linear-gradient(90deg,
        transparent 0%, rgba(0,210,106,0.5) 30%,
        rgba(0,210,106,0.9) 50%, rgba(0,210,106,0.5) 70%,
        transparent 100%);
}
.qe-brand { display: flex; align-items: center; gap: 12px; }
.qe-logo-mark {
    width: 36px; height: 36px;
    background: linear-gradient(135deg, #00d26a, #00a850);
    border-radius: 9px;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.9rem; font-weight: 900; color: #0f1014;
    letter-spacing: -1px;
    box-shadow: 0 0 18px rgba(0,210,106,0.4), 0 2px 8px rgba(0,0,0,0.5);
}
.qe-title {
    font-size: 1.05rem;
    font-weight: 800;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    color: #f0f0f5;
}
.qe-title span { color: #00d26a; }
.qe-subtitle {
    font-size: 0.62rem;
    color: rgba(255,255,255,0.25);
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-top: 1px;
    font-weight: 500;
}
.qe-nav {
    display: flex; align-items: center; gap: 6px;
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.05);
    border-radius: 20px; padding: 4px 6px;
}
.qe-nav-pill {
    padding: 5px 14px; border-radius: 16px;
    font-size: 0.72rem; font-weight: 600;
    letter-spacing: 0.5px; color: rgba(255,255,255,0.35);
    cursor: default; transition: all 0.2s;
}
.qe-nav-pill.active {
    background: rgba(0,210,106,0.15);
    color: #00d26a;
    border: 1px solid rgba(0,210,106,0.25);
}
.qe-meta { display: flex; align-items: center; gap: 20px; }
.qe-badge {
    display: flex; align-items: center; gap: 6px;
    background: rgba(0,210,106,0.08);
    border: 1px solid rgba(0,210,106,0.2);
    border-radius: 20px; padding: 5px 12px;
}
.qe-live-dot {
    width: 6px; height: 6px; border-radius: 50%;
    background: #00d26a;
    box-shadow: 0 0 6px #00d26a;
    animation: live-dot 2s ease-in-out infinite;
}
.qe-live-text {
    font-size: 0.68rem; font-weight: 700;
    letter-spacing: 1.5px; color: #00d26a;
    text-transform: uppercase;
}
.qe-timestamp {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem; color: rgba(255,255,255,0.22);
    letter-spacing: 0.5px;
}

/* ════════════════════════════════════════════════════
   SIDEBAR
═══════════════════════════════════════════════════ */
[data-testid="stSidebar"] {
    background: #13141a !important;
    border-right: 1px solid rgba(255,255,255,0.05) !important;
}
[data-testid="stSidebar"] > div:first-child { padding-top: 1.2rem; }
[data-testid="stSidebar"] .stMarkdown h2 {
    font-size: 0.64rem !important;
    font-weight: 700 !important;
    letter-spacing: 3px !important;
    text-transform: uppercase !important;
    color: rgba(255,255,255,0.2) !important;
    padding: 4px 0 8px 0 !important;
    border-bottom: 1px solid rgba(255,255,255,0.05) !important;
    margin-bottom: 14px !important;
}
[data-testid="stSidebar"] .stMarkdown h3 {
    font-size: 0.62rem !important;
    font-weight: 700 !important;
    letter-spacing: 2.5px !important;
    text-transform: uppercase !important;
    color: rgba(255,255,255,0.18) !important;
    margin-top: 1.4rem !important;
    margin-bottom: 10px !important;
}
[data-testid="stSidebar"] hr {
    border-color: rgba(255,255,255,0.05) !important;
    margin: 14px 0 !important;
}

/* ════════════════════════════════════════════════════
   METRIC CARDS  (glass morphism — charcoal/emerald)
═══════════════════════════════════════════════════ */
[data-testid="stMetric"] {
    background: #1a1b22 !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-top: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 16px !important;
    padding: 22px 24px !important;
    position: relative;
    overflow: hidden;
    transition: all 0.25s ease !important;
    animation: card-in 0.35s ease forwards;
    box-shadow: 0 4px 24px rgba(0,0,0,0.35), 0 1px 0 rgba(255,255,255,0.05) inset !important;
}
[data-testid="stMetric"]::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, transparent, rgba(0,210,106,0.5), transparent);
}
[data-testid="stMetric"]:hover {
    border-color: rgba(0,210,106,0.2) !important;
    transform: translateY(-3px) !important;
    box-shadow: 0 12px 36px rgba(0,0,0,0.45), 0 0 0 1px rgba(0,210,106,0.12), 0 1px 0 rgba(255,255,255,0.06) inset !important;
}
[data-testid="stMetricLabel"] > div {
    color: rgba(255,255,255,0.3) !important;
    font-size: 0.66rem !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 2px !important;
}
[data-testid="stMetricValue"] > div {
    color: #f0f0f5 !important;
    font-size: 1.8rem !important;
    font-weight: 700 !important;
    font-variant-numeric: tabular-nums !important;
    letter-spacing: -0.5px !important;
}

/* ════════════════════════════════════════════════════
   TABS — charcoal / emerald
═══════════════════════════════════════════════════ */
.stTabs [data-baseweb="tab-list"] {
    background: #13141a !important;
    border-radius: 14px !important;
    padding: 5px !important;
    gap: 3px !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
    margin-bottom: 1.8rem !important;
    box-shadow: 0 2px 12px rgba(0,0,0,0.3) !important;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 10px !important;
    color: rgba(255,255,255,0.3) !important;
    font-weight: 600 !important;
    font-size: 0.82rem !important;
    letter-spacing: 0.3px !important;
    padding: 10px 22px !important;
    transition: all 0.2s ease !important;
    border: 1px solid transparent !important;
}
.stTabs [data-baseweb="tab"]:hover {
    color: rgba(255,255,255,0.7) !important;
    background: rgba(255,255,255,0.04) !important;
}
.stTabs [aria-selected="true"] {
    background: rgba(0,210,106,0.1) !important;
    color: #00d26a !important;
    border-color: rgba(0,210,106,0.22) !important;
    box-shadow: 0 0 14px rgba(0,210,106,0.1), 0 1px 0 rgba(0,210,106,0.15) inset !important;
    text-shadow: 0 0 10px rgba(0,210,106,0.35) !important;
}

/* ════════════════════════════════════════════════════
   BUTTONS — elevated / 3D (no more flat!)
═══════════════════════════════════════════════════ */

/* Primary — emerald raised button */
.stButton > button[kind="primary"] {
    background: linear-gradient(180deg, #00df72 0%, #00c062 50%, #00a850 100%) !important;
    border: none !important;
    border-bottom: none !important;
    border-radius: 10px !important;
    color: #0a1a0f !important;
    font-weight: 800 !important;
    font-size: 0.84rem !important;
    letter-spacing: 0.8px !important;
    text-transform: uppercase !important;
    padding: 11px 24px !important;
    position: relative !important;
    transition: all 0.12s ease !important;
    box-shadow:
        0 1px 0 rgba(255,255,255,0.25) inset,
        0 -2px 0 rgba(0,0,0,0.25) inset,
        0 5px 0 #006b32,
        0 8px 20px rgba(0,180,80,0.35) !important;
    transform: translateY(0) !important;
}
.stButton > button[kind="primary"]:hover {
    background: linear-gradient(180deg, #00f07e 0%, #00d068 50%, #00b558 100%) !important;
    transform: translateY(-2px) !important;
    box-shadow:
        0 1px 0 rgba(255,255,255,0.25) inset,
        0 -2px 0 rgba(0,0,0,0.25) inset,
        0 7px 0 #006b32,
        0 12px 28px rgba(0,180,80,0.45) !important;
}
.stButton > button[kind="primary"]:active {
    transform: translateY(4px) !important;
    box-shadow:
        0 1px 0 rgba(255,255,255,0.2) inset,
        0 -1px 0 rgba(0,0,0,0.2) inset,
        0 1px 0 #006b32,
        0 2px 8px rgba(0,180,80,0.2) !important;
}

/* Secondary — glass raised button */
.stButton > button[kind="secondary"] {
    background: linear-gradient(180deg, #272830 0%, #21222c 100%) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 10px !important;
    color: rgba(255,255,255,0.65) !important;
    font-weight: 600 !important;
    font-size: 0.82rem !important;
    padding: 10px 20px !important;
    transition: all 0.15s ease !important;
    box-shadow:
        0 1px 0 rgba(255,255,255,0.08) inset,
        0 4px 0 #0f1014,
        0 6px 16px rgba(0,0,0,0.35) !important;
    transform: translateY(0) !important;
}
.stButton > button[kind="secondary"]:hover {
    background: linear-gradient(180deg, #2d2e38 0%, #262730 100%) !important;
    border-color: rgba(255,255,255,0.18) !important;
    color: #f0f0f5 !important;
    transform: translateY(-2px) !important;
    box-shadow:
        0 1px 0 rgba(255,255,255,0.1) inset,
        0 6px 0 #0f1014,
        0 10px 22px rgba(0,0,0,0.4) !important;
}
.stButton > button[kind="secondary"]:active {
    transform: translateY(3px) !important;
    box-shadow:
        0 1px 0 rgba(255,255,255,0.06) inset,
        0 1px 0 #0f1014,
        0 2px 8px rgba(0,0,0,0.25) !important;
}

/* Sidebar stock buttons */
[data-testid="stSidebar"] .stButton > button {
    background: #1e1f28 !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 8px !important;
    color: rgba(255,255,255,0.4) !important;
    font-size: 0.78rem !important;
    font-weight: 500 !important;
    text-align: left !important;
    transition: all 0.15s ease !important;
    padding: 9px 12px !important;
    box-shadow: 0 2px 0 #0a0b0e, 0 3px 10px rgba(0,0,0,0.25) !important;
    transform: translateY(0) !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: #252630 !important;
    border-color: rgba(0,210,106,0.25) !important;
    color: #00d26a !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 3px 0 #0a0b0e, 0 5px 14px rgba(0,0,0,0.3) !important;
}
[data-testid="stSidebar"] .stButton > button:active {
    transform: translateY(1px) !important;
    box-shadow: 0 1px 0 #0a0b0e, 0 2px 6px rgba(0,0,0,0.2) !important;
}

/* ════════════════════════════════════════════════════
   EXPANDERS
═══════════════════════════════════════════════════ */
.streamlit-expanderHeader, [data-testid="stExpander"] summary {
    background: #1a1b22 !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 12px !important;
    color: rgba(255,255,255,0.6) !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    padding: 14px 18px !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.2) !important;
}
.streamlit-expanderHeader:hover, [data-testid="stExpander"] summary:hover {
    border-color: rgba(0,210,106,0.25) !important;
    background: #1e1f28 !important;
    color: #f0f0f5 !important;
}
[data-testid="stExpander"] details[open] summary {
    border-radius: 12px 12px 0 0 !important;
    border-color: rgba(0,210,106,0.15) !important;
    border-bottom-color: rgba(255,255,255,0.04) !important;
}
.streamlit-expanderContent, [data-testid="stExpander"] details > div {
    background: #13141a !important;
    border: 1px solid rgba(255,255,255,0.05) !important;
    border-top: none !important;
    border-radius: 0 0 12px 12px !important;
    padding: 18px !important;
}

/* ════════════════════════════════════════════════════
   DATA TABLES
═══════════════════════════════════════════════════ */
[data-testid="stDataFrame"] {
    background: #1a1b22 !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 14px !important;
    overflow: hidden !important;
    box-shadow: 0 6px 28px rgba(0,0,0,0.35) !important;
}
[data-testid="stDataFrame"] iframe { border-radius: 14px !important; }

/* ════════════════════════════════════════════════════
   INPUTS
═══════════════════════════════════════════════════ */
.stTextInput input, .stTextArea textarea {
    background: #13141a !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 10px !important;
    color: #f0f0f5 !important;
    font-size: 0.88rem !important;
    transition: all 0.2s ease !important;
    padding: 10px 14px !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.2) inset !important;
}
.stTextInput input:focus, .stTextArea textarea:focus {
    border-color: rgba(0,210,106,0.4) !important;
    box-shadow: 0 0 0 3px rgba(0,210,106,0.06), 0 2px 8px rgba(0,0,0,0.2) inset !important;
}
.stTextInput input::placeholder, .stTextArea textarea::placeholder {
    color: rgba(255,255,255,0.18) !important;
}

/* ════════════════════════════════════════════════════
   SELECTBOX
═══════════════════════════════════════════════════ */
.stSelectbox > div > div {
    background: #13141a !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 10px !important;
    color: rgba(255,255,255,0.75) !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.2) !important;
}
.stSelectbox > div > div:focus-within {
    border-color: rgba(0,210,106,0.35) !important;
    box-shadow: 0 0 0 3px rgba(0,210,106,0.06) !important;
}

/* ════════════════════════════════════════════════════
   TOGGLE
═══════════════════════════════════════════════════ */
[data-baseweb="toggle"] > div { background: #00d26a !important; }

/* ════════════════════════════════════════════════════
   DIVIDERS
═══════════════════════════════════════════════════ */
hr {
    border: none !important;
    border-top: 1px solid rgba(255,255,255,0.05) !important;
    margin: 1.5rem 0 !important;
}

/* ════════════════════════════════════════════════════
   BADGES
═══════════════════════════════════════════════════ */
.rec-badge {
    display: inline-block;
    padding: 4px 13px;
    border-radius: 20px;
    font-size: 0.67rem;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    box-shadow: 0 2px 6px rgba(0,0,0,0.25);
}
.badge-sb {
    background: rgba(0,210,106,0.14);
    color: #00d26a;
    border: 1px solid rgba(0,210,106,0.35);
    box-shadow: 0 0 14px rgba(0,210,106,0.18), 0 2px 6px rgba(0,0,0,0.2);
}
.badge-b {
    background: rgba(0,210,106,0.07);
    color: #00b558;
    border: 1px solid rgba(0,210,106,0.2);
}
.badge-w {
    background: rgba(245,166,35,0.1);
    color: #f5a623;
    border: 1px solid rgba(245,166,35,0.28);
}
.badge-n {
    background: rgba(255,255,255,0.04);
    color: rgba(255,255,255,0.3);
    border: 1px solid rgba(255,255,255,0.08);
}
.badge-av {
    background: rgba(255,64,64,0.1);
    color: #ff4040;
    border: 1px solid rgba(255,64,64,0.28);
    box-shadow: 0 0 12px rgba(255,64,64,0.1), 0 2px 6px rgba(0,0,0,0.2);
}

/* ════════════════════════════════════════════════════
   AI ADVISER — COCKPIT COMMAND CENTER
═══════════════════════════════════════════════════ */
.command-center-header {
    background: linear-gradient(135deg, #16171f 0%, #1a1b24 100%);
    border: 1px solid rgba(139,86,246,0.15);
    border-radius: 18px;
    text-align: center;
    padding: 28px 32px 22px;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
    box-shadow: 0 8px 32px rgba(0,0,0,0.35), 0 0 60px rgba(139,86,246,0.04);
}
.command-center-header::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, transparent, #8b56f6 30%, #00d26a 70%, transparent);
}
.command-center-header::after {
    content: '';
    position: absolute;
    bottom: 0; left: 15%; right: 15%; height: 1px;
    background: linear-gradient(90deg, transparent, rgba(139,86,246,0.3), transparent);
}
.cc-title {
    font-size: 1rem;
    font-weight: 800;
    letter-spacing: 5px;
    text-transform: uppercase;
    color: #f0f0f5;
    margin-bottom: 7px;
}
.cc-title span { color: #8b56f6; }
.cc-subtitle {
    font-size: 0.68rem;
    color: rgba(255,255,255,0.25);
    letter-spacing: 2.5px;
    text-transform: uppercase;
}

.user-bubble {
    background: linear-gradient(135deg, rgba(0,210,106,0.07) 0%, rgba(0,210,106,0.02) 100%);
    border: 1px solid rgba(0,210,106,0.15);
    border-left: 3px solid #00d26a;
    border-radius: 0 14px 14px 0;
    padding: 16px 20px;
    margin-bottom: 12px;
    color: rgba(255,255,255,0.8);
    font-size: 0.9rem;
    line-height: 1.65;
    box-shadow: 0 4px 16px rgba(0,0,0,0.2);
    animation: float-up 0.28s ease;
}
.ai-bubble {
    background: linear-gradient(135deg, rgba(139,86,246,0.07) 0%, rgba(26,27,34,0.9) 100%);
    border: 1px solid rgba(139,86,246,0.15);
    border-left: 3px solid #8b56f6;
    border-radius: 0 14px 14px 0;
    padding: 20px 24px;
    margin-bottom: 16px;
    color: rgba(255,255,255,0.88);
    font-size: 0.9rem;
    line-height: 1.8;
    box-shadow: 0 6px 28px rgba(0,0,0,0.28);
    animation: float-up 0.28s ease;
}

/* Chat input — large glowing command bar */
[data-testid="stChatInput"] {
    background: #13141a !important;
    border: 1px solid rgba(139,86,246,0.2) !important;
    border-radius: 14px !important;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3), 0 1px 0 rgba(255,255,255,0.04) inset !important;
    transition: all 0.25s ease !important;
}
[data-testid="stChatInput"]:focus-within {
    border-color: rgba(139,86,246,0.5) !important;
    box-shadow: 0 0 0 3px rgba(139,86,246,0.07),
                0 0 28px rgba(139,86,246,0.12),
                0 4px 20px rgba(0,0,0,0.35) !important;
    animation: border-breathe 2.5s ease infinite !important;
}
[data-testid="stChatInput"] textarea {
    min-height: 54px !important;
    font-size: 0.95rem !important;
    color: #f0f0f5 !important;
    background: transparent !important;
    border: none !important;
    padding: 16px 18px !important;
}

/* ════════════════════════════════════════════════════
   SIMPLE TIP BOX
═══════════════════════════════════════════════════ */
.simple-tip {
    background: linear-gradient(135deg, rgba(0,210,106,0.06) 0%, rgba(0,210,106,0.02) 100%);
    border: 1px solid rgba(0,210,106,0.18);
    border-radius: 14px;
    padding: 16px 20px;
    margin: 12px 0;
    color: rgba(255,255,255,0.55);
    font-size: 0.87rem;
    line-height: 1.65;
    position: relative;
    overflow: hidden;
}
.simple-tip::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; height: 1px;
    background: linear-gradient(90deg, transparent, rgba(0,210,106,0.45), transparent);
}

/* ════════════════════════════════════════════════════
   PROGRESS BAR
═══════════════════════════════════════════════════ */
.stProgress > div > div > div > div {
    background: linear-gradient(90deg, #00d26a, #00a850) !important;
    border-radius: 4px !important;
    box-shadow: 0 0 8px rgba(0,210,106,0.4) !important;
}

/* ════════════════════════════════════════════════════
   ALERTS & NOTIFICATIONS
═══════════════════════════════════════════════════ */
div[data-baseweb="notification"] { border-radius: 12px !important; }
.stSuccess, div[data-testid="stAlert"][data-type="success"] {
    background: rgba(0,210,106,0.07) !important;
    border: 1px solid rgba(0,210,106,0.2) !important;
    border-radius: 12px !important;
}
.stInfo, div[data-testid="stAlert"][data-type="info"] {
    background: rgba(139,86,246,0.07) !important;
    border: 1px solid rgba(139,86,246,0.2) !important;
    border-radius: 12px !important;
}
.stWarning, div[data-testid="stAlert"][data-type="warning"] {
    background: rgba(245,166,35,0.07) !important;
    border: 1px solid rgba(245,166,35,0.22) !important;
    border-radius: 12px !important;
}
.stError, div[data-testid="stAlert"][data-type="error"] {
    background: rgba(255,64,64,0.07) !important;
    border: 1px solid rgba(255,64,64,0.22) !important;
    border-radius: 12px !important;
}

/* ════════════════════════════════════════════════════
   SPINNER & STATUS
═══════════════════════════════════════════════════ */
.stSpinner > div { border-top-color: #00d26a !important; }
[data-testid="stStatus"] {
    background: #1a1b22 !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 12px !important;
}

/* ════════════════════════════════════════════════════
   SECTION HEADERS h3
═══════════════════════════════════════════════════ */
.main h3 {
    font-size: 0.7rem !important;
    font-weight: 700 !important;
    letter-spacing: 3px !important;
    text-transform: uppercase !important;
    color: rgba(255,255,255,0.2) !important;
    margin-bottom: 1rem !important;
    padding-bottom: 8px !important;
    border-bottom: 1px solid rgba(255,255,255,0.05) !important;
}

/* ════════════════════════════════════════════════════
   SCROLLBAR
═══════════════════════════════════════════════════ */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: rgba(0,0,0,0.15); }
::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.1); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: rgba(0,210,106,0.3); }
</style>
""", unsafe_allow_html=True)

# ─── Load persisted custom CSS (from UI Improvements tab) ─────────────────────
import os as _os
_custom_css_path = _os.path.join(_os.path.dirname(__file__), "custom_ui.css")
if _os.path.exists(_custom_css_path):
    try:
        with open(_custom_css_path, "r", encoding="utf-8") as _f:
            _custom_css = _f.read().strip()
        if _custom_css:
            st.markdown(f"<style>{_custom_css}</style>", unsafe_allow_html=True)
    except Exception:
        pass


# ─── Session state defaults ────────────────────────────────────────────────────
_STATE_DEFAULTS = {
    "scan_results":    [],
    "last_scan_time":  None,
    "chat_history":    [],
    "selected_symbol": None,
    "options_cache":   {},
    "simple_mode":     False,
    "pending_q":       "",
    # UI improvements tab
    "ui_analysis":      "",
    "ui_approved":      set(),
    "ui_rejected":      set(),
    "ui_css_artifacts": "",
    "ui_applied":       False,
}
for _k, _v in _STATE_DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ─── Header banner ────────────────────────────────────────────────────────────
_now = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
st.markdown(f"""
<div class="qe-header">
    <div class="qe-brand">
        <div class="qe-logo-mark">QE</div>
        <div>
            <div class="qe-title">QUANTUM <span>EDGE</span></div>
            <div class="qe-subtitle">Institutional Intelligence &nbsp;·&nbsp; Signal Scanner &nbsp;·&nbsp; AI Adviser</div>
        </div>
    </div>
    <div class="qe-nav">
        <div class="qe-nav-pill active">Dashboard</div>
        <div class="qe-nav-pill">Scanner</div>
        <div class="qe-nav-pill">AI Adviser</div>
    </div>
    <div class="qe-meta">
        <div class="qe-badge">
            <div class="qe-live-dot"></div>
            <div class="qe-live-text">Live</div>
        </div>
        <div class="qe-timestamp">{_now}</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ─── Dynamic TTL: short when market is open, longer when closed ───────────────
def _vix_ttl()     -> int: return 45  if md.is_market_open() else 300
def _heatmap_ttl() -> int: return 60  if md.is_market_open() else 300

# Streamlit cache_data TTL is set at decoration time per unique key.
# We embed the current 5-minute window into the cache key so a new
# cache entry is created automatically when the window rolls over.
def _time_bucket(seconds: int) -> int:
    """Returns current epoch divided by `seconds` — changes every N seconds."""
    import time as _t
    return int(_t.time() // seconds)

@st.cache_data(show_spinner=False)
def _load_vix(_bucket: int):           # _bucket forces refresh on schedule
    return md.get_vix()

@st.cache_data(show_spinner=False)
def _load_heatmap(_bucket: int):
    return md.get_heatmap_data()


# ─── VIX Score box ────────────────────────────────────────────────────────────
_mkt_status = md.get_market_status()
_vix = _load_vix(_time_bucket(_vix_ttl()))
_vix_val   = _vix.get("value")
_vix_color = _vix.get("color", "#8892a4")
_vix_label = _vix.get("label", "—")
_vix_chg   = _vix.get("change", 0)
_vix_chgp  = _vix.get("change_pct", 0)
_vix_tier  = _vix.get("tier", "⚪")
_vix_as_of = _vix.get("as_of", "—")

if _vix_val is not None:
    _vix_chg_str  = f"{_vix_chg:+.2f}"
    _vix_chgp_str = f"{_vix_chgp:+.2f}%"
    _vix_display  = f"{_vix_val:.2f}"
else:
    _vix_chg_str = _vix_chgp_str = "—"
    _vix_display  = "—"

_refresh_ttl_str = ("45s" if _mkt_status["open"] else "5m")

st.markdown(f"""
<div style="
    background: linear-gradient(135deg, #1a1b22 0%, #0f1014 100%);
    border: 1px solid {_vix_color}44;
    border-left: 4px solid {_vix_color};
    border-radius: 12px;
    padding: 18px 28px;
    display: flex;
    align-items: center;
    gap: 32px;
    margin-bottom: 6px;
    box-shadow: 0 4px 24px rgba(0,0,0,0.4);
">
    <div style="display:flex;flex-direction:column;min-width:140px;">
        <div style="display:flex;align-items:center;gap:8px;margin-bottom:4px;">
            <span style="font-size:11px;font-weight:600;letter-spacing:2px;
                         color:#8892a4;text-transform:uppercase;">
                CBOE VIX  •  Fear Index
            </span>
            <span style="font-size:10px;font-weight:700;letter-spacing:1px;
                         background:{_mkt_status['color']}22;
                         border:1px solid {_mkt_status['color']}55;
                         color:{_mkt_status['color']};
                         padding:2px 8px;border-radius:20px;">
                {_mkt_status['dot']} {_mkt_status['label']}
            </span>
        </div>
        <div style="font-size:54px;font-weight:800;line-height:1;color:{_vix_color};
                    font-family:'Inter',sans-serif;letter-spacing:-2px;">
            {_vix_display}
        </div>
        <div style="margin-top:6px;display:flex;align-items:center;gap:10px;">
            <span style="font-size:13px;font-weight:700;
                         background:{_vix_color}22;border:1px solid {_vix_color}55;
                         color:{_vix_color};padding:3px 10px;border-radius:20px;
                         letter-spacing:1px;">
                {_vix_tier} {_vix_label}
            </span>
            <span style="font-size:12px;color:#8892a4;">
                {_vix_chg_str} ({_vix_chgp_str} today)
            </span>
        </div>
        <div style="margin-top:5px;font-size:10px;color:#4a5568;">
            Updated {_vix_as_of} &nbsp;•&nbsp; refreshes every {_refresh_ttl_str}
        </div>
    </div>
    <div style="flex:1;border-left:1px solid rgba(255,255,255,0.06);
                padding-left:28px;font-size:12px;color:#8892a4;line-height:1.8;">
        <b style="color:#f0f0f5;">VIX Guide</b><br>
        <span style="color:#00d26a;">▪ &lt;15</span> Low Volatility — markets calm, options cheap<br>
        <span style="color:#ffc107;">▪ 15–25</span> Normal range — typical trading environment<br>
        <span style="color:#ff8c00;">▪ 25–30</span> Elevated — uncertainty rising, hedge cautiously<br>
        <span style="color:#ff4040;">▪ &gt;30</span> High Fear — potential for sharp moves in both directions
    </div>
</div>
""", unsafe_allow_html=True)

# ─── Market heat map ──────────────────────────────────────────────────────────
with st.spinner("Loading market heat map…"):
    _heatmap_data = _load_heatmap(_time_bucket(_heatmap_ttl()))

st.markdown(
    "<div style='font-size:13px;font-weight:600;letter-spacing:1.5px;"
    "color:#8892a4;text-transform:uppercase;margin:8px 0 4px;'>"
    "📊 Market Heat Map  •  Sector &amp; Stock Performance</div>",
    unsafe_allow_html=True,
)
st.plotly_chart(
    cu.market_heatmap_chart(_heatmap_data),
    width="stretch",
    key="main_heatmap",
)

st.divider()


# ─── Helper: recommendation badge HTML ───────────────────────────────────────
def _badge(rec: str) -> str:
    cls = {"STRONG BUY": "sb", "BUY": "b", "WATCH": "w",
           "NEUTRAL": "n", "AVOID": "av"}.get(rec, "n")
    return f'<span class="rec-badge badge-{cls}">{rec}</span>'


# ─── Helper: options renderer ─────────────────────────────────────────────────
def _render_options(opt: dict, simple: bool = False):
    if not opt:
        return
    err = opt.get("error")
    if err:
        st.warning(f"Options: {err}")
        return

    price = opt.get("current_price")
    bias  = opt.get("bias", "NEUTRAL")
    sym   = opt.get("symbol", "")

    price_str = f"${price:.2f}" if price else "N/A"
    st.markdown(f"**{sym}** — Price: {price_str} — Bias: **{bias}**")

    calls = opt.get("calls", [])
    puts  = opt.get("puts", [])

    if calls:
        st.markdown("#### 📈 Call Options")
        if simple:
            st.caption("*Calls = you profit if the stock goes UP past the breakeven price.*")
        rows = []
        for c in calls[:6]:
            rows.append({
                "Strike":    f"${c['strike']:.0f}",
                "Expiry":    c["expiration"],
                "Ask":       f"${c['ask']:.2f}",
                "Breakeven": f"${c['breakeven']:.2f}",
                "IV":        f"{c['iv_pct']:.0f}%" if c.get("iv_pct") else "—",
                "Volume":    f"{c.get('volume', 0):,}",
                "OI":        f"{c.get('open_interest', 0):,}",
                "OTM%":      f"{c.get('upside_to_strike_pct', 0):+.1f}%",
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True, width='stretch')

    if puts:
        st.markdown("#### 📉 Put Options")
        if simple:
            st.caption("*Puts = you profit if the stock goes DOWN past the breakeven price.*")
        rows = []
        for p in puts[:6]:
            rows.append({
                "Strike":    f"${p['strike']:.0f}",
                "Expiry":    p["expiration"],
                "Ask":       f"${p['ask']:.2f}",
                "Breakeven": f"${p['breakeven']:.2f}",
                "IV":        f"{p['iv_pct']:.0f}%" if p.get("iv_pct") else "—",
                "Volume":    f"{p.get('volume', 0):,}",
                "OI":        f"{p.get('open_interest', 0):,}",
                "Downside%": f"{p.get('downside_protection_pct', 0):.1f}%",
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True, width='stretch')

    if not calls and not puts:
        st.info("No liquid options found for this symbol at this time.")


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 📈 Training Bot")

    # ── Market status + refresh ───────────────────────────────────────────────
    _ms = md.get_market_status()
    st.markdown(
        f"<div style='display:flex;align-items:center;justify-content:space-between;"
        f"background:{_ms['color']}15;border:1px solid {_ms['color']}40;"
        f"border-radius:8px;padding:6px 12px;margin-bottom:4px;'>"
        f"<span style='font-size:12px;font-weight:700;color:{_ms['color']};'>"
        f"{_ms['dot']} {_ms['label']}</span>"
        f"<span style='font-size:10px;color:#8892a4;'>"
        f"{'45s refresh' if _ms['open'] else '5m refresh'}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )
    if st.button("🔄 Refresh Market Data", width="stretch", key="refresh_btn"):
        st.cache_data.clear()
        st.rerun()

    st.divider()

    # ── Simple / Advanced toggle ──────────────────────────────────────────────
    st.session_state.simple_mode = st.toggle(
        "🟢 Simple Mode (beginner-friendly)",
        value=st.session_state.simple_mode,
    )
    if st.session_state.simple_mode:
        st.info("📚 Plain-English explanations are ON")

    st.divider()

    # ── Watchlist ─────────────────────────────────────────────────────────────
    st.markdown("### 📋 Watchlist")
    watchlist_raw = st.text_area(
        "Symbols (one per line or comma-separated)",
        value="\n".join(DEFAULT_WATCHLIST),
        height=200,
        key="watchlist_input",
    )
    watchlist = sorted(set(
        s.strip().upper()
        for s in watchlist_raw.replace(",", "\n").split("\n")
        if s.strip()
    ))
    st.caption(f"{len(watchlist)} symbol(s)")

    scan_btn = st.button("🔍 Run Full Scan", type="primary", width='stretch')

    if st.session_state.last_scan_time:
        st.caption(f"Last scan: {st.session_state.last_scan_time.strftime('%H:%M:%S')}")

    st.divider()

    # ── Quick single-symbol analyse ───────────────────────────────────────────
    st.markdown("### ⚡ Quick Analyze")
    quick_sym = st.text_input("Single symbol", placeholder="NVDA").strip().upper()
    if st.button("Analyze", width='stretch') and quick_sym:
        with st.spinner(f"Analyzing {quick_sym}…"):
            try:
                r = build_composite_score(quick_sym)
                existing = {x["symbol"] for x in st.session_state.scan_results}
                if quick_sym in existing:
                    st.session_state.scan_results = [
                        r if x["symbol"] == quick_sym else x
                        for x in st.session_state.scan_results
                    ]
                else:
                    st.session_state.scan_results.insert(0, r)
                st.session_state.scan_results.sort(
                    key=lambda x: x["composite_score"], reverse=True
                )
                st.session_state.selected_symbol = quick_sym
                rec_icon = {"STRONG BUY": "🟢", "BUY": "🟩", "WATCH": "🟡",
                            "NEUTRAL": "⚪", "AVOID": "🔴"}.get(r["recommendation"], "⚪")
                st.success(f"{rec_icon} {quick_sym}: {r['recommendation']} — {r['composite_score']:.0f}/100")
            except Exception as e:
                st.error(f"Error: {e}")

    st.divider()

    # ── Sidebar quick-view ────────────────────────────────────────────────────
    if st.session_state.scan_results:
        st.markdown("### 🏆 Top Results")
        icon_map = {"STRONG BUY": "🟢", "BUY": "🟩", "WATCH": "🟡",
                    "NEUTRAL": "⚪", "AVOID": "🔴"}
        for r in st.session_state.scan_results[:7]:
            sym   = r["symbol"]
            score = r["composite_score"]
            rec   = r["recommendation"]
            icon  = icon_map.get(rec, "⚪")
            if st.button(f"{icon} {sym}  {score:.0f}", key=f"sb_{sym}", width='stretch'):
                st.session_state.selected_symbol = sym


# ═══════════════════════════════════════════════════════════════════════════════
# RUN SCAN
# ═══════════════════════════════════════════════════════════════════════════════
if scan_btn and watchlist:
    progress = st.progress(0, text="Starting scan…")
    results_buf = []
    total = len(watchlist)

    with st.status(f"Scanning {total} symbols…", expanded=True) as status:
        for i, sym in enumerate(watchlist):
            st.write(f"  Analyzing **{sym}**…")
            try:
                r = build_composite_score(sym)
                results_buf.append(r)
            except Exception as e:
                st.write(f"  ⚠ {sym}: {e}")
            progress.progress((i + 1) / total, text=f"{sym} done ({i+1}/{total})")

        status.update(label=f"✅ Scan complete — {len(results_buf)} symbols", state="complete")

    progress.empty()
    results_buf.sort(key=lambda x: x["composite_score"], reverse=True)
    st.session_state.scan_results    = results_buf
    st.session_state.last_scan_time  = datetime.now()
    st.session_state.options_cache   = {}  # clear stale options
    if results_buf:
        st.session_state.selected_symbol = results_buf[0]["symbol"]


# ═══════════════════════════════════════════════════════════════════════════════
# HEADER + SUMMARY METRICS
# ═══════════════════════════════════════════════════════════════════════════════
results = st.session_state.scan_results
simple  = st.session_state.simple_mode

if results:
    strong  = sum(1 for r in results if r["recommendation"] in ("STRONG BUY", "BUY"))
    watches = sum(1 for r in results if r["recommendation"] == "WATCH")
    avoids  = sum(1 for r in results if r["recommendation"] == "AVOID")
    avg_sc  = sum(r["composite_score"] for r in results) / len(results)
    bullish_sent = sum(1 for r in results if r.get("sentiment", {}).get("label") == "Bullish")

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Scanned",      len(results))
    m2.metric("🟢 Buy",       strong)
    m3.metric("🟡 Watch",     watches)
    m4.metric("🔴 Avoid",     avoids)
    m5.metric("Avg Score",    f"{avg_sc:.1f}")
    m6.metric("Bullish News", bullish_sent)
else:
    st.info("👈 Enter symbols in the sidebar and click **Run Full Scan** to start.")

st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════════════════
tab_ai, tab_results, tab_whale, tab_news, tab_charts, tab_ui = st.tabs([
    "🔬  Research Agent",
    "📊  Results",
    "🐋  Whale Activity",
    "📰  Market News",
    "📈  Charts",
    "🖌  UI Improvements",
])


# ════════════════════════════════════════════════════════════════════════════════
# TAB 1 — AI ADVISER
# ════════════════════════════════════════════════════════════════════════════════
with tab_ai:
    st.markdown("""
<div class="command-center-header">
    <div class="cc-title">🔬 AI Research Agent</div>
    <div class="cc-subtitle">
        Powered by Groq &nbsp;•&nbsp; llama-3.3-70b &nbsp;•&nbsp;
        Scans 90-stock universe for options plays &nbsp;•&nbsp; Small / Mid / Large Cap
    </div>
</div>
""", unsafe_allow_html=True)

    if simple:
        st.markdown("""
<div class="simple-tip">
💡 <b>The Research Agent works instantly — no scan needed!</b> Try:<br>
"Find 6 options plays — 2 small, 2 mid, 2 large cap" &nbsp;|&nbsp;
"What are the top 3 stocks to buy today?" &nbsp;|&nbsp;
"Find the best oversold bounce plays right now"
</div>
""", unsafe_allow_html=True)

    # ── Preset questions ──────────────────────────────────────────────────────
    st.markdown("**Quick questions:**")
    # Each entry: (button label, full question sent to agent)
    presets = [
        (
            "Find 6 options plays — 2 small cap, 2 mid cap, 2 large cap",
            "Find 6 options plays — 2 small cap, 2 mid cap, 2 large cap",
        ),
        (
            "What are the top 3 stocks to buy today?",
            "What are the top 3 stocks to buy today?",
        ),
        (
            "Which stocks have the strongest institutional backing?",
            "Which stocks have the strongest institutional backing?",
        ),
        (
            "Find the best call options with high upside potential",
            "Find the best call options with high upside potential",
        ),
        (
            "What's the overall market sentiment right now?",
            "What's the overall market sentiment right now?",
        ),
        (
            "Find 6 put options across small, mid, and large cap",
            "Find 6 put options across small, mid, and large cap",
        ),
        (
            "Find me 3 good LEAP options",
            (
                "Find me the best LEAP options. Use fundamentals: strong revenue growth, "
                "expanding margins, low debt, and rising institutional ownership. "
                "Identify 2-3 upcoming catalysts per pick — earnings beats, product launches, "
                "regulatory approvals, or macro tailwinds — that make it near-certain the stock "
                "will trade above the strike before expiry. Show strike, expiry (6-18 months out), "
                "estimated cost, upside %, and why each catalyst is a near-certainty. "
                "Confidence must be HIGH. Return one small cap, one mid cap, one large cap."
            ),
        ),
        # ── Day-trader focused quick questions ────────────────────────────────
        (
            "🔥 Top 5 momentum breakouts today",
            (
                "Find today's top 5 momentum breakout stocks. Criteria: price up 3%+ with volume ratio "
                "above 2x the 10-day average, breaking above a 20-day or 50-day high, and at least one "
                "confirmed bullish pattern (bull flag, engulfing, SMA crossover). Prefer small & mid cap "
                "for the biggest percentage moves. Show entry, stop-loss, and upside target for each."
            ),
        ),
        (
            "🐋 Biggest unusual options activity now",
            (
                "Which stocks right now have the largest unusual options activity? Look for: large call or "
                "put sweeps (single-strike volume > 2000 contracts), high volume-to-open-interest ratios "
                "(above 0.5x), and ATM IV below 60% (affordable contracts). I want to follow the smart "
                "money. Show the symbol, sweep direction (bullish/bearish), IV rank, and your read on "
                "whether this looks like a hedge or a directional bet."
            ),
        ),
        (
            "📉 Oversold RSI bounce setups",
            (
                "Show me stocks with RSI below 40 that show bullish reversal signals. Look for: oversold "
                "bounce setups (RSI < 35 + price up today), bullish engulfing candles, SMA20 crossovers "
                "from below, or higher-low structure after a pullback. These should be coiled springs — "
                "oversold but showing first signs of recovery. Include risk/reward for each setup."
            ),
        ),
        (
            "⚡ Best scalp trades for today's session",
            (
                "Find the best scalp trade setups for today's session. I need high-volatility stocks with: "
                "clear intraday momentum (vol ratio > 1.5x), RSI between 45-65 (room to run), a tight "
                "price range they just broke out of, and strong options flow supporting the direction. "
                "Give me a specific entry zone, 1-2% stop-loss level, and 3-5% profit target for each. "
                "Prioritize small and mid cap names with beta > 1.2."
            ),
        ),
        (
            "📊 Sector rotation — where's the money flowing?",
            (
                "Analyze where institutional money is rotating right now based on the current VIX regime "
                "and sector momentum data. Which sectors are seeing accumulation (rising institutional "
                "ownership, bullish options flow, positive price action)? Which are being distributed? "
                "Give me 2-3 specific stocks in the hottest sector that are buyable today, plus the "
                "macro thesis driving the rotation."
            ),
        ),
    ]
    p_cols = st.columns(3)
    for idx, (label, question) in enumerate(presets):
        with p_cols[idx % 3]:
            if st.button(label, key=f"pq_{idx}", width='stretch'):
                st.session_state.pending_q = question

    st.divider()

    # ── Chat history ──────────────────────────────────────────────────────────
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(
                f'<div class="user-bubble">🙋 <b>You:</b> {msg["content"]}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="ai-bubble">🤖 <b>AI Adviser:</b><br><br>{msg["content"]}</div>',
                unsafe_allow_html=True,
            )

    # ── Input ─────────────────────────────────────────────────────────────────
    user_input = st.chat_input(
        "Ask anything… 'Find 6 options plays' or 'What are the top 3 stocks to buy today?'"
    )

    # Resolve input: typed > preset button
    active_q = user_input or (st.session_state.pending_q if st.session_state.pending_q else None)
    if st.session_state.pending_q:
        st.session_state.pending_q = ""

    if active_q:
        st.session_state.chat_history.append({"role": "user", "content": active_q})

        _status_placeholder = st.empty()
        def _progress(msg: str):
            _status_placeholder.info(msg)

        with st.spinner("🔬 Research Agent — running 5-layer validation…"):
            response = adviser.ask_adviser(
                question     = active_q,
                scan_results = results,
                options_data = st.session_state.options_cache,
                simple_mode  = simple,
                progress_cb  = _progress,
                vix_val      = _vix.get("value"),   # pass live VIX to regime scoring
            )
        _status_placeholder.empty()

        st.session_state.chat_history.append({"role": "assistant", "content": response})
        st.rerun()

    # Clear chat
    if st.session_state.chat_history:
        if st.button("🗑 Clear Chat", type="secondary"):
            st.session_state.chat_history = []
            st.rerun()

    # API key hint
    if not adviser.HAS_GROQ or not adviser.GROQ_API_KEY:
        st.caption(
            "💡 Add `GROQ_API_KEY=gsk_...` to your `.env` file to enable the "
            "5-layer Research Agent (options flow · IV rank · patterns · sentiment · institutional)."
        )


# ════════════════════════════════════════════════════════════════════════════════
# TAB 2 — RESULTS
# ════════════════════════════════════════════════════════════════════════════════
with tab_results:
    if not results:
        st.info("Run a scan to see results.")
    else:
        st.markdown("### 📊 Ranked Opportunities")
        st.plotly_chart(cu.score_bar_chart(results), width='stretch', key="results_score_bar")

        # ── Simple view ───────────────────────────────────────────────────────
        if simple:
            st.markdown("### What Do These Scores Mean?")
            st.markdown("""
| Score | What it means |
|-------|--------------|
| 75–100 | **Strong Buy** — Almost everything looks good |
| 60–74  | **Buy** — More signals point up than down |
| 45–59  | **Watch** — Mixed signals; monitor closely |
| 30–44  | **Neutral** — Nothing exciting either way |
| 0–29   | **Avoid** — Most signals point down |
""")
            st.divider()

            for i, r in enumerate(results, 1):
                sym   = r["symbol"]
                score = r["composite_score"]
                rec   = r["recommendation"]
                price = r.get("price", "N/A")
                chg   = r.get("change_pct", "N/A")
                sent  = r.get("sentiment", {})
                tech  = r.get("tech", {})
                rsi   = tech.get("rsi")
                conv  = r.get("convergence_notes", [])

                icon = {"STRONG BUY": "🟢", "BUY": "🟩", "WATCH": "🟡",
                        "NEUTRAL": "⚪", "AVOID": "🔴"}.get(rec, "⚪")

                with st.expander(
                    f"{icon} #{i}  **{sym}**  —  ${price}  ({chg})  —  {rec}",
                    expanded=(i <= 3),
                ):
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Score", f"{score:.0f}/100")
                    c2.metric("Today", chg)
                    c3.metric("Sentiment", sent.get("label", "—"))

                    if score >= 70:
                        st.success(
                            f"📈 **{sym}** looks very strong! "
                            "Institutions are buying, news is positive, and the chart is trending up."
                        )
                    elif score >= 55:
                        st.info(f"📊 **{sym}** shows good potential — worth watching closely.")
                    elif score >= 40:
                        st.warning(f"⚖️ **{sym}** is mixed — not a clear buy or sell right now.")
                    else:
                        st.error(f"📉 **{sym}** looks weak — most signals point down.")

                    if rsi:
                        if rsi < 35:
                            st.markdown(
                                f"🔵 **RSI {rsi:.0f}** — Oversold (like a spring pulled tight — often bounces)"
                            )
                        elif rsi > 70:
                            st.markdown(
                                f"🔴 **RSI {rsi:.0f}** — Overbought (may need to cool off first)"
                            )
                        else:
                            st.markdown(f"⚪ **RSI {rsi:.0f}** — Normal range")

                    if conv:
                        st.info(f"⭐ {conv[0]}")

                    headlines = sent.get("top_headlines", [])[:2]
                    if headlines:
                        st.markdown("**Recent News:**")
                        for h in headlines:
                            st.markdown(f"• {h[:100]}")

        # ── Advanced view ─────────────────────────────────────────────────────
        else:
            # Data table
            rows = []
            for i, r in enumerate(results, 1):
                sc   = r.get("scores", {})
                sent = r.get("sentiment", {})
                tech = r.get("tech", {})
                rows.append({
                    "#":       i,
                    "Symbol":  r["symbol"],
                    "Price":   r.get("price", "—"),
                    "Chg%":    r.get("change_pct", "—"),
                    "Score":   r["composite_score"],
                    "Rec":     r["recommendation"],
                    "Inst":    sc.get("institutional", 0),
                    "Tech":    sc.get("technical", 0),
                    "Vol":     sc.get("volume", 0),
                    "Sent":    sc.get("sentiment", 0),
                    "RSI":     tech.get("rsi", "—"),
                    "MACD":    tech.get("macd_signal", "—"),
                    "Trend":   tech.get("trend_direction", "—"),
                    "News":    sent.get("label", "—"),
                    "Source":  sent.get("source", "—"),
                })

            df = pd.DataFrame(rows)
            st.dataframe(
                df, hide_index=True, width='stretch',
                column_config={
                    "Score": st.column_config.ProgressColumn(
                        "Score", min_value=0, max_value=100, format="%.1f"
                    ),
                    "Inst": st.column_config.NumberColumn("Inst", format="%.0f"),
                    "Tech": st.column_config.NumberColumn("Tech", format="%.0f"),
                    "Vol":  st.column_config.NumberColumn("Vol",  format="%.0f"),
                    "Sent": st.column_config.NumberColumn("Sent", format="%.0f"),
                },
            )

            st.divider()

            # ── Deep-dive panel ───────────────────────────────────────────────
            st.markdown("### 🔬 Symbol Deep Dive")
            sym_opts = [r["symbol"] for r in results]
            default_idx = (
                sym_opts.index(st.session_state.selected_symbol)
                if st.session_state.selected_symbol in sym_opts else 0
            )
            sel = st.selectbox("Select symbol", sym_opts, index=default_idx, key="dd_sym")
            st.session_state.selected_symbol = sel

            sel_r = next((r for r in results if r["symbol"] == sel), None)
            if sel_r:
                # Radar | Sentiment gauge | Tech details
                rc1, rc2, rc3 = st.columns(3)
                with rc1:
                    st.plotly_chart(
                        cu.score_radar(sel_r["scores"], sel), width='stretch',
                        key=f"dd_radar_{sel}",
                    )
                with rc2:
                    sent_d = sel_r.get("sentiment", {})
                    st.plotly_chart(
                        cu.sentiment_gauge(
                            sent_d.get("sentiment_score", 50),
                            sent_d.get("label", ""),
                        ),
                        width='stretch',
                        key=f"dd_gauge_{sel}",
                    )
                with rc3:
                    st.markdown("**Technical Signals**")
                    for d in sel_r["tech"].get("details", [])[:7]:
                        st.markdown(f"• {d}")

                # Convergence
                conv = sel_r.get("convergence_notes", [])
                if conv:
                    for c in conv:
                        st.info(c)

                # Institutional details
                with st.expander("📋 Institutional Flow Detail"):
                    for d in sel_r.get("inst_details", []):
                        icon = (
                            "✅" if any(w in d.lower() for w in ["buying", "new position", "strong", "high", "marquee"])
                            else "⚠️" if any(w in d.lower() for w in ["selling", "reducing", "minimal", "low"])
                            else "ℹ️"
                        )
                        st.markdown(f"{icon} {d}")

                # Options loader
                st.divider()
                if st.button(f"📊 Load Options for {sel}", key="load_opt_dd"):
                    with st.spinner("Fetching options chain…"):
                        try:
                            opt = summarize_options(sel, sel_r["composite_score"])
                            st.session_state.options_cache[sel] = opt
                        except Exception as e:
                            st.error(f"Options error: {e}")

                if sel in st.session_state.options_cache:
                    _render_options(st.session_state.options_cache[sel], simple)


# ════════════════════════════════════════════════════════════════════════════════
# TAB 3 — WHALE ACTIVITY
# ════════════════════════════════════════════════════════════════════════════════
with tab_whale:
    st.markdown("### 🐋 Institutional (Whale) Activity")

    if simple:
        st.markdown("""
<div class="simple-tip">
💡 <b>What is Whale Activity?</b><br>
"Whales" are giant investment firms like Vanguard, BlackRock, and hedge funds that manage
hundreds of billions of dollars. When they buy or hold a stock in large quantities,
it's a strong signal of confidence. This tab shows which stocks have the most "smart money" behind them.
</div>
""", unsafe_allow_html=True)

    if not results:
        st.info("Run a scan to see institutional data.")
    else:
        # Summary table
        whale_rows = []
        for r in results:
            inst_s  = r["scores"].get("institutional", 0)
            details = r.get("inst_details", [])

            ownership = marquee = flow = "—"
            for d in details:
                if "Institutional ownership:" in d and ownership == "—":
                    ownership = d.split("Institutional ownership:")[1].strip()[:30]
                if "Marquee holders" in d and marquee == "—":
                    marquee = d.split(":", 1)[1].strip()[:55] if ":" in d else d[:55]
                if "Net institutional" in d and flow == "—":
                    flow = d[:60]

            whale_rows.append({
                "Symbol":          r["symbol"],
                "Inst Score":      inst_s,
                "Ownership":       ownership,
                "Marquee Holders": marquee,
                "Flow Signal":     flow,
                "Rec":             r["recommendation"],
            })

        wdf = pd.DataFrame(whale_rows).sort_values("Inst Score", ascending=False)
        st.dataframe(
            wdf, hide_index=True, width='stretch',
            column_config={
                "Inst Score": st.column_config.ProgressColumn(
                    "Inst Score", min_value=0, max_value=100, format="%.0f"
                ),
            },
        )

        st.divider()
        st.markdown("### 🔍 Institutional Detail")

        whale_syms = [r["symbol"] for r in results]
        w_sel = st.selectbox("Select symbol", whale_syms, key="whale_sel")
        w_r   = next((r for r in results if r["symbol"] == w_sel), None)

        if w_r:
            wc1, wc2 = st.columns([3, 1])
            with wc1:
                st.markdown(f"**{w_sel} — Institutional Flow Details**")
                for d in w_r.get("inst_details", []):
                    icon = (
                        "✅" if any(w in d.lower() for w in ["buying", "new", "strong", "high", "marquee"])
                        else "⚠️" if any(w in d.lower() for w in ["selling", "reducing", "minimal", "low"])
                        else "ℹ️"
                    )
                    st.markdown(f"{icon} {d}")
            with wc2:
                i_score = w_r["scores"].get("institutional", 0)
                st.plotly_chart(
                    cu.mini_score_gauge(i_score, "Inst Score"),
                    width='stretch',
                    key=f"whale_gauge_{w_sel}",
                )
                for c in w_r.get("convergence_notes", []):
                    st.info(c)


# ════════════════════════════════════════════════════════════════════════════════
# TAB 4 — MARKET NEWS
# ════════════════════════════════════════════════════════════════════════════════
with tab_news:
    st.markdown("### 📰 Market News & Sentiment")

    if simple:
        st.markdown("""
<div class="simple-tip">
💡 <b>How News Affects Stocks:</b><br>
Positive news (earnings beats, upgrades, product launches) tends to push stocks <b>up</b>.<br>
Negative news (lawsuits, missed earnings, downgrades) tends to push them <b>down</b>.<br>
Our scanner reads recent articles and scores sentiment from –1 (very bearish) to +1 (very bullish).
</div>
""", unsafe_allow_html=True)

    if not results:
        st.info("Run a scan to see news data.")
    else:
        # Sentiment overview
        bull_c = sum(1 for r in results if r.get("sentiment", {}).get("label") == "Bullish")
        bear_c = sum(1 for r in results if r.get("sentiment", {}).get("label") == "Bearish")
        neut_c = len(results) - bull_c - bear_c

        nc1, nc2, nc3 = st.columns(3)
        nc1.metric("🟢 Bullish", bull_c)
        nc2.metric("⚪ Neutral", neut_c)
        nc3.metric("🔴 Bearish", bear_c)

        st.divider()

        # Per-symbol news cards
        for r in results:
            sym    = r["symbol"]
            sent   = r.get("sentiment", {})
            label  = sent.get("label", "Neutral")
            s_score = sent.get("sentiment_score", 50)
            source = sent.get("source", "unknown")
            bull   = sent.get("bullish_count", 0)
            bear   = sent.get("bearish_count", 0)
            neut   = sent.get("neutral_count", 0)
            avg    = sent.get("sentiment_avg")
            heads  = sent.get("top_headlines", [])

            lbl_icon = "🟢" if label == "Bullish" else ("🔴" if label == "Bearish" else "⚪")

            with st.expander(
                f"{lbl_icon}  **{sym}** — {label}  ({s_score}/100)  via {source}"
            ):
                ncc1, ncc2, ncc3, ncc4 = st.columns(4)
                ncc1.metric("Score",   f"{s_score}/100")
                ncc2.metric("🟢 Bull", bull)
                ncc3.metric("🔴 Bear", bear)
                ncc4.metric("⚪ Neut", neut)

                if avg is not None:
                    pct = max(0.0, min(1.0, (avg + 1) / 2))
                    st.progress(pct, text=f"Avg sentiment: {avg:+.3f}")

                if heads:
                    st.markdown("**Recent Headlines:**")
                    for h in heads[:4]:
                        st.markdown(f"• {h[:105]}")
                else:
                    st.caption("No headlines retrieved for this symbol.")


# ════════════════════════════════════════════════════════════════════════════════
# TAB 5 — CHARTS
# ════════════════════════════════════════════════════════════════════════════════
with tab_charts:
    st.markdown("### 📈 Technical Charts")

    if not results:
        st.info("Run a scan to enable charting.")
    else:
        sym_opts = [r["symbol"] for r in results]
        default_idx = (
            sym_opts.index(st.session_state.selected_symbol)
            if st.session_state.selected_symbol in sym_opts else 0
        )

        ch_col1, ch_col2 = st.columns([2, 1])
        with ch_col1:
            chart_sym = st.selectbox(
                "Symbol to chart", sym_opts, index=default_idx, key="chart_sel"
            )
        with ch_col2:
            chart_period = st.selectbox("Period", ["1mo", "3mo", "6mo", "1y"], index=1)

        # Main candlestick
        st.plotly_chart(
            cu.candlestick_chart(chart_sym, chart_period), width='stretch',
            key=f"chart_candle_{chart_sym}_{chart_period}",
        )

        # RSI + MACD side by side
        ind1, ind2 = st.columns(2)
        with ind1:
            st.plotly_chart(
                cu.rsi_chart(chart_sym, chart_period), width='stretch',
                key=f"chart_rsi_{chart_sym}_{chart_period}",
            )
        with ind2:
            st.plotly_chart(
                cu.macd_chart(chart_sym, chart_period), width='stretch',
                key=f"chart_macd_{chart_sym}_{chart_period}",
            )

        # Signal breakdown
        st.divider()
        chart_r = next((r for r in results if r["symbol"] == chart_sym), None)
        if chart_r:
            if simple:
                st.markdown("### 📖 What Do These Charts Mean?")
                tech = chart_r.get("tech", {})
                rsi  = tech.get("rsi")
                macd = tech.get("macd_signal", "")
                trend = tech.get("trend_direction", "")

                explanations = []
                if rsi:
                    if rsi < 35:
                        explanations.append(
                            f"📉 **RSI {rsi:.0f}** — This stock is oversold. "
                            "Think of it like a rubber band stretched too far — it often snaps back up."
                        )
                    elif rsi > 70:
                        explanations.append(
                            f"📈 **RSI {rsi:.0f}** — This stock is overbought. "
                            "It's been rising fast and may need to cool down first."
                        )
                    else:
                        explanations.append(f"⚪ **RSI {rsi:.0f}** — Normal momentum range (between 35 and 70).")

                macd_labels = {
                    "bullish":       "✅ **MACD** — Momentum is building upward — buyers are in control.",
                    "crossing_up":   "🚀 **MACD** — A crossover is happening! This is often a buy signal.",
                    "strengthening": "📈 **MACD** — Momentum is growing stronger.",
                    "bearish":       "⚠️ **MACD** — Momentum is pointing down — sellers are in control.",
                    "neutral":       "⚪ **MACD** — No clear direction yet.",
                }
                if macd in macd_labels:
                    explanations.append(macd_labels[macd])

                if trend == "bullish":
                    explanations.append("📊 **Trend** — Price is above its 20-day average. The overall direction is UP.")
                elif trend == "bearish":
                    explanations.append("📊 **Trend** — Price is below its 20-day average. The overall direction is DOWN.")

                for exp in explanations:
                    st.markdown(f"• {exp}")

            else:
                cr1, cr2 = st.columns(2)
                with cr1:
                    st.plotly_chart(
                        cu.score_radar(chart_r["scores"], chart_sym), width='stretch',
                        key=f"chart_radar_{chart_sym}",
                    )
                with cr2:
                    st.markdown("**Signal Details**")
                    for d in chart_r["tech"].get("details", []):
                        st.markdown(f"• {d}")

                    vol_d = chart_r.get("vol_details", [])
                    if vol_d:
                        st.markdown("**Volume:**")
                        for d in vol_d:
                            st.markdown(f"• {d}")

                    conv = chart_r.get("convergence_notes", [])
                    if conv:
                        st.markdown("**Convergence:**")
                        for c in conv:
                            st.info(c)


# ════════════════════════════════════════════════════════════════════════════════
# TAB 6 — UI IMPROVEMENTS
# ════════════════════════════════════════════════════════════════════════════════
with tab_ui:
    st.markdown("""
<div class="command-center-header">
    <div class="cc-title">🖌 UI Improvements Agent</div>
    <div class="cc-subtitle">
        AI analyses the dashboard &nbsp;•&nbsp; Compares to TradingView, Bloomberg &amp;
        thinkorswim &nbsp;•&nbsp; Proposes specific changes &nbsp;•&nbsp; You approve before anything changes
    </div>
</div>
""", unsafe_allow_html=True)

    st.markdown("""
**How it works:**
1. Click **Analyse Dashboard** — the AI reviews every aspect of the current design
2. Review the numbered improvement proposals
3. Approve the ones you want — rejected proposals are discarded
4. Approved changes are noted for implementation in the next update
""")

    col_btn1, col_btn2, _ = st.columns([1, 1, 3])
    with col_btn1:
        analyse_btn = st.button(
            "🔍 Analyse Dashboard", type="primary", width="stretch",
            key="ui_analyse_btn",
        )
    with col_btn2:
        if st.session_state.ui_analysis:
            if st.button("🗑 Clear Analysis", type="secondary", width="stretch", key="ui_clear_btn"):
                st.session_state.ui_analysis = ""
                st.session_state.ui_approved = set()
                st.session_state.ui_rejected = set()
                st.rerun()

    if analyse_btn:
        st.session_state.ui_approved = set()
        st.session_state.ui_rejected = set()
        _ui_placeholder = st.empty()
        _ui_chunks      = []

        def _ui_stream(chunk: str):
            _ui_chunks.append(chunk)
            _ui_placeholder.markdown("".join(_ui_chunks))

        with st.spinner("🧠 AI is analysing the dashboard against top platforms…"):
            if adviser.HAS_GROQ:
                full = adviser.ask_ui_adviser(stream_callback=_ui_stream)
            else:
                full = adviser.ask_ui_adviser()

        _ui_placeholder.empty()
        st.session_state.ui_analysis = full
        st.rerun()

    # ── Show analysis + approval UI ───────────────────────────────────────────
    if st.session_state.ui_analysis:
        st.divider()
        st.markdown("### 📋 Analysis Results")
        st.markdown(
            "<div style='background:#1a1b22;border:1px solid rgba(139,86,246,0.25);"
            "border-left:4px solid #8b56f6;border-radius:10px;padding:20px 24px;"
            "margin-bottom:16px;'>"
            + st.session_state.ui_analysis.replace("\n", "<br>")
            + "</div>",
            unsafe_allow_html=True,
        )

        st.divider()
        st.markdown("### ✅ Your Approval")
        st.caption("Mark which improvements you'd like to see implemented:")

        # Parse numbered proposals from the analysis text
        import re as _re
        _proposals = _re.findall(
            r"(\d+)\.\s+\*{0,2}([^\n\*]+)\*{0,2}",
            st.session_state.ui_analysis,
        )

        if _proposals:
            for num_str, title in _proposals[:12]:
                num   = int(num_str)
                is_ap = num in st.session_state.ui_approved
                is_rj = num in st.session_state.ui_rejected

                badge = "✅ Approved" if is_ap else ("❌ Rejected" if is_rj else "⏳ Pending")
                badge_color = "#00d26a" if is_ap else ("#ff4040" if is_rj else "#8892a4")

                with st.container():
                    ac1, ac2, ac3 = st.columns([6, 1, 1])
                    with ac1:
                        st.markdown(
                            f"<span style='color:{badge_color};font-weight:600;'>{badge}</span> &nbsp; "
                            f"**{num}. {title.strip()}**",
                            unsafe_allow_html=True,
                        )
                    with ac2:
                        if not is_ap:
                            if st.button("✅", key=f"ui_ap_{num}", help="Approve this change"):
                                st.session_state.ui_approved.add(num)
                                st.session_state.ui_rejected.discard(num)
                                st.rerun()
                    with ac3:
                        if not is_rj:
                            if st.button("❌", key=f"ui_rj_{num}", help="Reject this change"):
                                st.session_state.ui_rejected.add(num)
                                st.session_state.ui_approved.discard(num)
                                st.rerun()

            # Summary
            st.divider()
            approved = sorted(st.session_state.ui_approved)
            rejected = sorted(st.session_state.ui_rejected)
            pending  = [n for n, _ in _proposals if int(n) not in st.session_state.ui_approved
                        and int(n) not in st.session_state.ui_rejected]

            c1, c2, c3 = st.columns(3)
            c1.metric("✅ Approved", len(approved))
            c2.metric("❌ Rejected", len(rejected))
            c3.metric("⏳ Pending",  len(pending))

            if approved:
                st.success(
                    f"✅ **Proposals #{', #'.join(map(str, approved))} approved.** "
                    "Click **Generate CSS** below to create live styles for your approved changes."
                )

            # ── CSS generation ──────────────────────────────────────────────────
            if approved:
                st.divider()
                st.markdown("### 🎨 Apply Approved Changes")

                gen_col, _, _ = st.columns([2, 1, 3])
                with gen_col:
                    gen_css_btn = st.button(
                        "🎨 Generate CSS for Approved Changes",
                        type="primary",
                        width="stretch",
                        key="ui_gen_css_btn",
                        disabled=not adviser.HAS_GROQ,
                    )

                if not adviser.HAS_GROQ:
                    st.caption(
                        "💡 Add `GROQ_API_KEY` to `.env` to enable CSS generation."
                    )

                if gen_css_btn and adviser.HAS_GROQ:
                    _approved_titles = [
                        title.strip()
                        for num_str, title in _proposals
                        if int(num_str) in st.session_state.ui_approved
                    ]
                    _css_placeholder = st.empty()
                    _css_chunks: list[str] = []

                    def _css_stream(chunk: str):
                        _css_chunks.append(chunk)
                        _css_placeholder.code("".join(_css_chunks), language="css")

                    with st.spinner("🎨 AI generating CSS for approved proposals…"):
                        css_raw = adviser.generate_ui_css(
                            _approved_titles,
                            st.session_state.ui_analysis,
                            stream_callback=_css_stream,
                        )

                    _css_placeholder.empty()
                    extracted = adviser.extract_css_from_response(css_raw)
                    st.session_state.ui_css_artifacts = extracted if extracted else css_raw
                    st.rerun()

                # ── Show CSS artifact card ───────────────────────────────────────
                if st.session_state.ui_css_artifacts:
                    st.markdown(
                        "<div style='background:linear-gradient(135deg,#1a1b22,#0f1014);"
                        "border:1px solid rgba(0,210,106,0.3);"
                        "border-left:4px solid #00d26a;"
                        "border-radius:12px;padding:16px 20px;margin:8px 0 12px;'>"
                        "<div style='font-size:10px;font-weight:700;letter-spacing:2.5px;"
                        "color:#00d26a;text-transform:uppercase;margin-bottom:6px;'>"
                        "✦ CSS Artifact — Ready to Apply</div>"
                        "<div style='font-size:12px;color:#8892a4;'>"
                        "Review the generated CSS below. Click <b style='color:#f0f0f5;'>"
                        "Apply Changes</b> to write it to <code>custom_ui.css</code> and "
                        "activate it immediately.</div>"
                        "</div>",
                        unsafe_allow_html=True,
                    )

                    st.code(st.session_state.ui_css_artifacts, language="css")

                    apply_col, reset_col, _ = st.columns([1, 1, 3])
                    with apply_col:
                        if st.button(
                            "✅ Apply Changes to Dashboard",
                            type="primary",
                            width="stretch",
                            key="ui_apply_btn",
                        ):
                            _css_path = _os.path.join(
                                _os.path.dirname(__file__), "custom_ui.css"
                            )
                            try:
                                with open(_css_path, "w", encoding="utf-8") as _fh:
                                    _fh.write(st.session_state.ui_css_artifacts)
                                # Inject immediately for this session
                                st.markdown(
                                    f"<style>{st.session_state.ui_css_artifacts}</style>",
                                    unsafe_allow_html=True,
                                )
                                st.session_state.ui_applied = True
                                st.success(
                                    "✅ **Changes applied!** CSS is now live and saved to "
                                    "`custom_ui.css` — it will auto-load on every future session."
                                )
                            except Exception as _e:
                                st.error(f"Failed to save CSS: {_e}")

                    with reset_col:
                        if st.session_state.ui_applied:
                            if st.button(
                                "🗑 Remove Applied CSS",
                                type="secondary",
                                width="stretch",
                                key="ui_reset_css_btn",
                            ):
                                _css_path = _os.path.join(
                                    _os.path.dirname(__file__), "custom_ui.css"
                                )
                                if _os.path.exists(_css_path):
                                    _os.remove(_css_path)
                                st.session_state.ui_applied       = False
                                st.session_state.ui_css_artifacts = ""
                                st.rerun()

        else:
            st.info("No numbered proposals detected — review the analysis above and re-run if needed.")

    elif not analyse_btn:
        st.info(
            "👆 Click **Analyse Dashboard** to start. The AI will compare your dashboard "
            "against TradingView, Bloomberg Terminal, and thinkorswim — then propose "
            "specific improvements for your approval."
        )

    # ── Live CSS Editor (always visible at bottom of UI tab) ─────────────────
    st.divider()
    st.markdown("""
<div style='display:flex;align-items:center;gap:12px;margin-bottom:4px;'>
    <div style='font-size:13px;font-weight:700;letter-spacing:1.5px;
                color:#8892a4;text-transform:uppercase;'>
        🖊 Live CSS Editor
    </div>
    <div style='font-size:11px;color:rgba(255,255,255,0.2);'>
        Edit &amp; preview styles instantly — no restart needed
    </div>
</div>
""", unsafe_allow_html=True)

    # Load current custom_ui.css into the editor as the default value
    _editor_css_path = _os.path.join(_os.path.dirname(__file__), "custom_ui.css")
    _editor_default  = ""
    if _os.path.exists(_editor_css_path):
        try:
            with open(_editor_css_path, "r", encoding="utf-8") as _ef:
                _editor_default = _ef.read().strip()
        except Exception:
            pass

    _live_css = st.text_area(
        "CSS Editor",
        value=_editor_default,
        height=300,
        placeholder="/* Paste or type CSS here — click Preview to see it live */\n\n"
                    ".stApp { background-color: #0a0b0f !important; }\n"
                    "[data-testid=\"stSidebar\"] { background: #111216 !important; }",
        key="live_css_editor",
        label_visibility="collapsed",
    )

    _ec1, _ec2, _ec3, _ec4 = st.columns([1, 1, 1, 3])

    with _ec1:
        _preview_btn = st.button(
            "👁 Preview", type="primary", width="stretch", key="css_preview_btn"
        )
    with _ec2:
        _save_btn = st.button(
            "💾 Save & Lock", type="primary", width="stretch", key="css_save_btn"
        )
    with _ec3:
        _clear_btn = st.button(
            "🗑 Clear CSS", type="secondary", width="stretch", key="css_clear_btn"
        )

    # ── Preview: inject CSS into page RIGHT NOW (no save, no rerun) ──────────
    if _preview_btn and _live_css.strip():
        st.markdown(
            f"<style>{_live_css}</style>",
            unsafe_allow_html=True,
        )
        st.success("👁 Preview active — CSS injected. Scroll the page to see changes.")

    # ── Save: write to custom_ui.css + inject immediately ────────────────────
    if _save_btn:
        try:
            with open(_editor_css_path, "w", encoding="utf-8") as _sf:
                _sf.write(_live_css)
            # Inject immediately so changes are visible right now
            if _live_css.strip():
                st.markdown(f"<style>{_live_css}</style>", unsafe_allow_html=True)
            st.session_state.ui_css_artifacts = _live_css
            st.session_state.ui_applied       = True
            st.success(
                "💾 **Saved & locked!** CSS written to `custom_ui.css` and live now. "
                "Changes load automatically on every future launch."
            )
        except Exception as _se:
            st.error(f"Save failed: {_se}")

    # ── Clear: delete custom_ui.css ───────────────────────────────────────────
    if _clear_btn:
        if _os.path.exists(_editor_css_path):
            _os.remove(_editor_css_path)
        st.session_state.ui_applied       = False
        st.session_state.ui_css_artifacts = ""
        st.rerun()

    # ── Quick-reference palette + selectors ──────────────────────────────────
    with st.expander("📖 Selector & Palette Reference", expanded=False):
        st.markdown("""
**Common Streamlit selectors:**
```css
.stApp                                   /* whole page background */
.main .block-container                   /* main content area */
[data-testid="stSidebar"]               /* sidebar */
.stTabs [data-baseweb="tab-list"]       /* tab bar */
.stTabs [aria-selected="true"]          /* active tab */
.stButton > button[kind="primary"]      /* primary buttons */
.stButton > button[kind="secondary"]    /* secondary buttons */
[data-testid="stMetric"]               /* metric cards */
[data-testid="stDataFrame"]            /* data tables */
.stTextInput input                      /* text inputs */
[data-testid="stChatInput"]            /* chat input bar */
```

**Existing colour palette:**
| Name | Hex | Usage |
|------|-----|-------|
| BG Main | `#0f1014` | Page background |
| BG Card | `#1a1b22` | Cards / panels |
| Emerald | `#00d26a` | Gains, active, primary |
| Purple | `#8b56f6` | AI / secondary |
| Red | `#ff4040` | Losses / danger |
| Gold | `#ffc107` | Warnings / watch |
| Text | `#f0f0f5` | Primary text |
| Muted | `#8892a4` | Secondary text |

**Example tweaks to try:**
```css
/* Wider main content */
.main .block-container { max-width: 1800px !important; }

/* Brighter tab highlight */
.stTabs [aria-selected="true"] { color: #00ffaa !important; }

/* Larger metric values */
[data-testid="stMetricValue"] > div { font-size: 2.2rem !important; }

/* Purple primary buttons instead of green */
.stButton > button[kind="primary"] {
    background: linear-gradient(180deg, #9b68ff, #7a40e6) !important;
    box-shadow: 0 5px 0 #4a2090, 0 8px 20px rgba(139,86,246,0.4) !important;
}
```
""")

