import os
import math
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv
import yfinance as yf
from textblob import TextBlob
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Optional packages ──────────────────────────────────────────────────────────
TF_AVAILABLE = True
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
except Exception:
    TF_AVAILABLE = False

XGB_AVAILABLE = True
try:
    from xgboost import XGBRegressor
except ImportError:
    XGB_AVAILABLE = False

LGB_AVAILABLE = True
try:
    import lightgbm as lgb
    from lightgbm import LGBMRegressor
except ImportError:
    LGB_AVAILABLE = False

# ══════════════════════════════════════════════════════════════════════════════
# Page config — must be the first Streamlit call
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="QuantVision — Stock Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

load_dotenv()
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY", "6fe740c75d764b009015e175452500e4")

# ── Indian NSE Stocks ──────────────────────────────────────────────────────────
# Format: "COMPANY NAME (TICKER.NS)"  — displayed in selectbox, ticker extracted on select
INDIAN_STOCKS = [
    "Reliance Industries (RELIANCE.NS)",
    "TCS - Tata Consultancy Services (TCS.NS)",
    "HDFC Bank (HDFCBANK.NS)",
    "ICICI Bank (ICICIBANK.NS)",
    "Infosys (INFY.NS)",
    "State Bank of India (SBIN.NS)",
    "Hindustan Unilever (HINDUNILVR.NS)",
    "Bharti Airtel (BHARTIARTL.NS)",
    "ITC (ITC.NS)",
    "Kotak Mahindra Bank (KOTAKBANK.NS)",
    "Larsen & Toubro (LT.NS)",
    "HCL Technologies (HCLTECH.NS)",
    "Axis Bank (AXISBANK.NS)",
    "Bajaj Finance (BAJFINANCE.NS)",
    "Wipro (WIPRO.NS)",
    "Maruti Suzuki (MARUTI.NS)",
    "Asian Paints (ASIANPAINT.NS)",
    "Sun Pharma (SUNPHARMA.NS)",
    "Ultra Tech Cement (ULTRACEMCO.NS)",
    "Titan Company (TITAN.NS)",
    "Nestle India (NESTLEIND.NS)",
    "Power Grid Corp (POWERGRID.NS)",
    "NTPC (NTPC.NS)",
    "Tech Mahindra (TECHM.NS)",
    "JSW Steel (JSWSTEEL.NS)",
    "Tata Motors (TATAMOTORS.NS)",
    "Tata Steel (TATASTEEL.NS)",
    "Adani Ports (ADANIPORTS.NS)",
    "Adani Enterprises (ADANIENT.NS)",
    "Adani Green Energy (ADANIGREEN.NS)",
    "Bajaj Finserv (BAJAJFINSV.NS)",
    "Bajaj Auto (BAJAJ-AUTO.NS)",
    "M&M - Mahindra & Mahindra (M&M.NS)",
    "ONGC (ONGC.NS)",
    "Coal India (COALINDIA.NS)",
    "Cipla (CIPLA.NS)",
    "Divis Laboratories (DIVISLAB.NS)",
    "Dr. Reddy's Labs (DRREDDY.NS)",
    "Eicher Motors (EICHERMOT.NS)",
    "Grasim Industries (GRASIM.NS)",
    "Hindalco Industries (HINDALCO.NS)",
    "IndusInd Bank (INDUSINDBK.NS)",
    "Hero MotoCorp (HEROMOTOCO.NS)",
    "HDFC Life Insurance (HDFCLIFE.NS)",
    "SBI Life Insurance (SBILIFE.NS)",
    "Britannia Industries (BRITANNIA.NS)",
    "Tata Consumer Products (TATACONSUM.NS)",
    "Apollo Hospitals (APOLLOHOSP.NS)",
    "Zomato (ZOMATO.NS)",
    "Paytm - One97 Communications (PAYTM.NS)",
    "Nykaa - FSN E-Commerce (NYKAA.NS)",
    "Delhivery (DELHIVERY.NS)",
    "Persistent Systems (PERSISTENT.NS)",
    "Mphasis (MPHASIS.NS)",
    "Coforge (COFORGE.NS)",
    "L&T Technology Services (LTTS.NS)",
    "Info Edge (Naukri) (NAUKRI.NS)",
    "Pidilite Industries (PIDILITIND.NS)",
    "Havells India (HAVELLS.NS)",
    "Dabur India (DABUR.NS)",
    "Marico (MARICO.NS)",
    "Colgate-Palmolive India (COLPAL.NS)",
    "Godrej Consumer Products (GODREJCP.NS)",
    "Berger Paints (BERGEPAINT.NS)",
    "Trent (TRENT.NS)",
    "Varun Beverages (VBL.NS)",
    "Interglobe Aviation (IndiGo) (INDIGO.NS)",
    "DLF (DLF.NS)",
    "Shriram Finance (SHRIRAMFIN.NS)",
    "Cholamandalam Finance (CHOLAFIN.NS)",
    "Bank of Baroda (BANKBARODA.NS)",
    "Canara Bank (CANBK.NS)",
    "Punjab National Bank (PNB.NS)",
    "IDFC First Bank (IDFCFIRSTB.NS)",
    "Federal Bank (FEDERALBNK.NS)",
    "Bandhan Bank (BANDHANBNK.NS)",
    "REC Limited (RECLTD.NS)",
    "Power Finance Corp (PFC.NS)",
    "Bharat Electronics (BEL.NS)",
    "HAL - Hindustan Aeronautics (HAL.NS)",
    "Bharat Dynamics (BDL.NS)",
    "SAIL - Steel Authority (SAIL.NS)",
    "Vedanta (VEDL.NS)",
    "Ambuja Cements (AMBUJACEM.NS)",
    "ACC (ACC.NS)",
    "Shree Cement (SHREECEM.NS)",
    "Dalmia Bharat (DALBHARAT.NS)",
    "Voltas (VOLTAS.NS)",
    "Blue Star (BLUESTARCO.NS)",
    "Dixon Technologies (DIXON.NS)",
    "Page Industries (PAGEIND.NS)",
    "Avenue Supermarts (DMart) (DMART.NS)",
    "Tata Power (TATAPOWER.NS)",
    "NHPC (NHPC.NS)",
    "Torrent Power (TORNTPOWER.NS)",
    "Gujarat Gas (GUJARATGAS.NS)",
    "Indraprastha Gas (IGL.NS)",
    "Mahanagar Gas (MGL.NS)",
    "GAIL India (GAIL.NS)",
    "Indian Oil Corp (IOC.NS)",
    "BPCL (BPCL.NS)",
    "HPCL (HPCL.NS)",
    "Petronet LNG (PETRONET.NS)",
    "Motherson Sumi (MOTHERSON.NS)",
    "Bosch India (BOSCHLTD.NS)",
    "Minda Industries (MINDAIND.NS)",
    "Tata Elxsi (TATAELXSI.NS)",
    "Kpit Technologies (KPITTECH.NS)",
    "Zydus Lifesciences (ZYDUSLIFE.NS)",
    "Torrent Pharma (TORNTPHARM.NS)",
    "Lupin (LUPIN.NS)",
    "Aurobindo Pharma (AUROPHARMA.NS)",
    "Ipca Laboratories (IPCALAB.NS)",
    "Alkem Laboratories (ALKEM.NS)",
    "Mankind Pharma (MANKIND.NS)",
    "Max Healthcare (MAXHEALTH.NS)",
    "Fortis Healthcare (FORTIS.NS)",
    "Narayana Hrudayalaya (NH.NS)",
    "P&G India (PGHH.NS)",
    "Emami (EMAMILTD.NS)",
    "United Spirits (MCDOWELL-N.NS)",
    "United Breweries (UBL.NS)",
    "Jubilant Foodworks (JUBLFOOD.NS)",
    "Devyani International (DEVYANI.NS)",
    "Restaurant Brands Asia (RBA.NS)",
    "Indian Railway Catering (IRCTC.NS)",
    "Container Corp (CONCOR.NS)",
    "Adani Total Gas (ATGL.NS)",
    "Torrent Power (TORNTPOWER.NS)",
]
# Remove any duplicates while preserving order
_seen = set()
INDIAN_STOCKS = [x for x in INDIAN_STOCKS if not (x in _seen or _seen.add(x))]

def _ticker_from_label(label: str) -> str:
    """Extract ticker symbol from 'Company Name (TICKER.NS)' format."""
    if "(" in label and label.endswith(")"):
        return label.split("(")[-1].rstrip(")")
    return label

# ── MASTER CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;700&family=Orbitron:wght@400;700;900&display=swap');

/* ═══════════════════════ ROOT & BACKGROUND ═══════════════════════ */
:root {
  --bg-deep:        #020408;
  --bg-dark:        #060d18;
  --bg-card:        rgba(8, 20, 40, 0.85);
  --bg-card-hover:  rgba(10, 28, 58, 0.95);
  --border-dim:     rgba(0, 180, 255, 0.12);
  --border-glow:    rgba(0, 200, 255, 0.45);
  --neon-cyan:      #00d4ff;
  --neon-blue:      #0080ff;
  --neon-purple:    #a855f7;
  --neon-green:     #00ff88;
  --neon-gold:      #ffd700;
  --neon-red:       #ff4060;
  --text-primary:   #e8f4ff;
  --text-secondary: #7ab8d4;
  --text-dim:       #3d6080;
  --glow-cyan:      0 0 20px rgba(0, 212, 255, 0.4), 0 0 60px rgba(0, 212, 255, 0.15);
  --glow-green:     0 0 20px rgba(0, 255, 136, 0.4), 0 0 60px rgba(0, 255, 136, 0.15);
  --glow-purple:    0 0 20px rgba(168, 85, 247, 0.4), 0 0 60px rgba(168, 85, 247, 0.15);
}

/* Animated background */
.stApp {
  background: var(--bg-deep) !important;
  background-image:
    radial-gradient(ellipse 80% 50% at 20% 10%, rgba(0, 100, 200, 0.08) 0%, transparent 60%),
    radial-gradient(ellipse 60% 40% at 80% 80%, rgba(100, 0, 200, 0.06) 0%, transparent 60%),
    radial-gradient(ellipse 40% 30% at 50% 50%, rgba(0, 200, 255, 0.03) 0%, transparent 70%) !important;
  font-family: 'Inter', sans-serif !important;
}

/* Hide Streamlit default elements */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1rem !important; padding-bottom: 2rem !important; }

/* ═══════════════════════ SIDEBAR ═══════════════════════ */
section[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #030810 0%, #050f20 50%, #030810 100%) !important;
  border-right: 1px solid rgba(0, 180, 255, 0.2) !important;
  box-shadow: 4px 0 30px rgba(0, 100, 255, 0.08);
}
section[data-testid="stSidebar"] .stMarkdown * { color: var(--text-secondary) !important; }
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 { color: var(--neon-cyan) !important; }

section[data-testid="stSidebar"] .stTextInput input {
  background: rgba(0, 40, 80, 0.6) !important;
  border: 1px solid rgba(0, 180, 255, 0.25) !important;
  color: var(--text-primary) !important;
  border-radius: 8px !important;
  font-family: 'JetBrains Mono', monospace !important;
  transition: all 0.3s ease;
}
section[data-testid="stSidebar"] .stTextInput input:focus {
  border-color: var(--neon-cyan) !important;
  box-shadow: 0 0 15px rgba(0, 212, 255, 0.25) !important;
}

section[data-testid="stSidebar"] .stCheckbox label,
section[data-testid="stSidebar"] .stSlider label,
section[data-testid="stSidebar"] .stDateInput label { color: var(--text-secondary) !important; }

section[data-testid="stSidebar"] hr {
  border: none !important;
  border-top: 1px solid rgba(0, 180, 255, 0.12) !important;
  margin: 1rem 0 !important;
}

/* ═══════════════════════ HERO HEADER ═══════════════════════ */
.hero-header {
  position: relative;
  padding: 2.5rem 3rem;
  margin-bottom: 2rem;
  border-radius: 20px;
  overflow: hidden;
  background: linear-gradient(135deg,
    rgba(0, 30, 70, 0.9) 0%,
    rgba(0, 15, 45, 0.95) 40%,
    rgba(20, 0, 60, 0.9) 100%
  );
  border: 1px solid rgba(0, 180, 255, 0.2);
  box-shadow:
    0 0 60px rgba(0, 100, 255, 0.12),
    inset 0 1px 0 rgba(255,255,255,0.05);
}
.hero-header::before {
  content: '';
  position: absolute;
  top: -50%;
  left: -50%;
  width: 200%;
  height: 200%;
  background: conic-gradient(
    from 0deg at 50% 50%,
    transparent 0deg,
    rgba(0, 200, 255, 0.03) 60deg,
    transparent 120deg,
    rgba(100, 0, 255, 0.03) 180deg,
    transparent 240deg,
    rgba(0, 200, 255, 0.03) 300deg,
    transparent 360deg
  );
  animation: rotate 20s linear infinite;
  pointer-events: none;
}
@keyframes rotate { to { transform: rotate(360deg); } }

.hero-grid-lines {
  position: absolute;
  inset: 0;
  background-image:
    linear-gradient(rgba(0, 200, 255, 0.04) 1px, transparent 1px),
    linear-gradient(90deg, rgba(0, 200, 255, 0.04) 1px, transparent 1px);
  background-size: 40px 40px;
  pointer-events: none;
}
.hero-logo {
  font-family: 'Orbitron', monospace;
  font-size: 2.6rem;
  font-weight: 900;
  background: linear-gradient(135deg, #00d4ff 0%, #0080ff 40%, #a855f7 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  letter-spacing: 2px;
  margin: 0;
  filter: drop-shadow(0 0 20px rgba(0, 212, 255, 0.5));
  line-height: 1.1;
}
.hero-sub {
  color: var(--text-secondary);
  font-size: 0.95rem;
  margin-top: 0.6rem;
  letter-spacing: 0.5px;
  font-weight: 400;
}
.hero-badge {
  display: inline-block;
  background: rgba(0, 212, 255, 0.1);
  border: 1px solid rgba(0, 212, 255, 0.3);
  color: var(--neon-cyan);
  padding: 0.2rem 0.8rem;
  border-radius: 20px;
  font-size: 0.72rem;
  font-family: 'JetBrains Mono', monospace;
  font-weight: 500;
  letter-spacing: 1px;
  text-transform: uppercase;
  margin-left: 1rem;
  vertical-align: middle;
}
.hero-status {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  margin-top: 1rem;
  font-size: 0.82rem;
  color: var(--text-dim);
  font-family: 'JetBrains Mono', monospace;
}
.hero-dot {
  width: 8px;
  height: 8px;
  background: var(--neon-green);
  border-radius: 50%;
  box-shadow: var(--glow-green);
  animation: pulse-dot 2s ease-in-out infinite;
}
@keyframes pulse-dot {
  0%, 100% { opacity: 1; transform: scale(1); }
  50%       { opacity: 0.5; transform: scale(0.8); }
}

/* ═══════════════════════ SECTION HEADERS ═══════════════════════ */
.sec-title {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  margin: 2rem 0 1.2rem;
  padding-bottom: 0.75rem;
  border-bottom: 1px solid rgba(0, 180, 255, 0.15);
}
.sec-title-icon {
  font-size: 1.2rem;
}
.sec-title-text {
  font-family: 'Orbitron', monospace;
  font-size: 0.9rem;
  font-weight: 700;
  letter-spacing: 2px;
  text-transform: uppercase;
  color: var(--neon-cyan);
  text-shadow: 0 0 20px rgba(0, 212, 255, 0.4);
}
.sec-title-line {
  flex: 1;
  height: 1px;
  background: linear-gradient(90deg, rgba(0, 212, 255, 0.3), transparent);
}

/* ═══════════════════════ LIVE PRICE CARDS ═══════════════════════ */
.price-card {
  position: relative;
  background: var(--bg-card);
  border: 1px solid var(--border-dim);
  border-radius: 16px;
  padding: 1.5rem 1.8rem;
  text-align: left;
  overflow: hidden;
  transition: all 0.4s cubic-bezier(0.23, 1, 0.32, 1);
  backdrop-filter: blur(20px);
  -webkit-backdrop-filter: blur(20px);
  cursor: default;
}
.price-card::before {
  content: '';
  position: absolute;
  top: 0; left: 0; right: 0;
  height: 2px;
  background: linear-gradient(90deg, transparent, var(--neon-cyan), transparent);
  opacity: 0.6;
}
.price-card:hover {
  border-color: var(--border-glow);
  transform: translateY(-3px);
  box-shadow: var(--glow-cyan), 0 20px 40px rgba(0, 0, 0, 0.4);
  background: var(--bg-card-hover);
}
.price-card-symbol {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.78rem;
  font-weight: 700;
  color: var(--neon-cyan);
  letter-spacing: 2px;
  text-transform: uppercase;
  margin-bottom: 0.5rem;
  opacity: 0.85;
}
.price-card-value {
  font-family: 'Orbitron', monospace;
  font-size: 1.9rem;
  font-weight: 700;
  color: var(--text-primary);
  letter-spacing: -1px;
  line-height: 1;
  margin-bottom: 0.4rem;
}
.price-card-delta-pos {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.88rem;
  font-weight: 600;
  color: var(--neon-green);
  text-shadow: 0 0 10px rgba(0, 255, 136, 0.4);
}
.price-card-delta-neg {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.88rem;
  font-weight: 600;
  color: var(--neon-red);
  text-shadow: 0 0 10px rgba(255, 64, 96, 0.4);
}
.price-card-delta-neu {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.88rem;
  color: var(--text-dim);
}
.price-card-corner {
  position: absolute;
  top: 1rem; right: 1rem;
  width: 30px; height: 30px;
  border-top: 1px solid var(--border-glow);
  border-right: 1px solid var(--border-glow);
  border-radius: 0 6px 0 0;
  opacity: 0.4;
}

/* ═══════════════════════ MODEL CARDS ═══════════════════════ */
.model-card {
  position: relative;
  background: rgba(5, 15, 35, 0.9);
  border: 1px solid rgba(0, 180, 255, 0.15);
  border-radius: 14px;
  padding: 1.4rem 1.2rem;
  text-align: center;
  overflow: hidden;
  backdrop-filter: blur(20px);
  transition: all 0.3s ease;
}
.model-card::after {
  content: '';
  position: absolute;
  inset: 0;
  border-radius: 14px;
  padding: 1px;
  background: linear-gradient(135deg, transparent 40%, rgba(0,212,255,0.1) 100%);
  -webkit-mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
  -webkit-mask-composite: xor;
  pointer-events: none;
}
.model-card.best {
  border-color: rgba(0, 255, 136, 0.35);
  box-shadow: 0 0 30px rgba(0, 255, 136, 0.12), inset 0 0 30px rgba(0, 255, 136, 0.03);
}
.model-card.best::before {
  content: '';
  position: absolute;
  top: 0; left: 0; right: 0;
  height: 2px;
  background: linear-gradient(90deg, transparent, var(--neon-green), transparent);
}
.model-name {
  font-family: 'Inter', sans-serif;
  font-weight: 700;
  font-size: 0.88rem;
  margin-bottom: 0.8rem;
  letter-spacing: 0.3px;
}
.model-divider {
  border: none;
  border-top: 1px solid rgba(0, 180, 255, 0.1);
  margin: 0.5rem 0;
}
.model-label {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.65rem;
  color: var(--text-dim);
  text-transform: uppercase;
  letter-spacing: 1.5px;
  margin-top: 0.6rem;
}
.model-value {
  font-family: 'Orbitron', monospace;
  font-size: 1rem;
  font-weight: 700;
  color: var(--text-primary);
  margin-top: 0.1rem;
}
.badge-best {
  display: inline-flex;
  align-items: center;
  gap: 3px;
  background: linear-gradient(135deg, #00ff88, #00cc6a);
  color: #001a0d;
  padding: 0.12rem 0.55rem;
  border-radius: 20px;
  font-size: 0.62rem;
  font-weight: 800;
  letter-spacing: 0.5px;
  vertical-align: middle;
  margin-left: 6px;
  text-transform: uppercase;
  box-shadow: 0 0 12px rgba(0, 255, 136, 0.4);
}

/* ═══════════════════════ INFO BOX ═══════════════════════ */
.info-box {
  display: flex;
  align-items: flex-start;
  gap: 1rem;
  background: rgba(0, 80, 160, 0.1);
  border: 1px solid rgba(0, 150, 255, 0.2);
  border-left: 3px solid var(--neon-cyan);
  border-radius: 0 12px 12px 0;
  padding: 1rem 1.2rem;
  margin: 0.8rem 0 1.2rem;
  backdrop-filter: blur(10px);
}
.info-box-icon { font-size: 1.1rem; margin-top: 0.05rem; flex-shrink: 0; }
.info-box-text {
  font-size: 0.84rem;
  color: var(--text-secondary);
  line-height: 1.6;
}
.info-box-text strong { color: var(--neon-cyan); font-weight: 600; }

/* Sentiment variants */
.info-box.positive { border-left-color: var(--neon-green); background: rgba(0, 255, 136, 0.05); border-color: rgba(0, 255, 136, 0.2); }
.info-box.positive .info-box-text strong { color: var(--neon-green); }
.info-box.negative { border-left-color: var(--neon-red); background: rgba(255, 64, 96, 0.05); border-color: rgba(255, 64, 96, 0.2); }
.info-box.negative .info-box-text strong { color: var(--neon-red); }
.info-box.neutral  { border-left-color: var(--neon-gold); background: rgba(255, 215, 0, 0.05); border-color: rgba(255, 215, 0, 0.2); }
.info-box.neutral .info-box-text strong  { color: var(--neon-gold); }

/* ═══════════════════════ STAT STRIP ═══════════════════════ */
.stat-strip {
  display: flex;
  gap: 0.6rem;
  flex-wrap: wrap;
  margin-bottom: 1.5rem;
}
.stat-pill {
  background: rgba(0, 30, 70, 0.8);
  border: 1px solid rgba(0, 180, 255, 0.2);
  border-radius: 8px;
  padding: 0.45rem 1rem;
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.78rem;
  color: var(--text-secondary);
}
.stat-pill span { color: var(--neon-cyan); font-weight: 600; margin-left: 0.4rem; }

/* ═══════════════════════ FORECAST CARDS ═══════════════════════ */
.forecast-card {
  position: relative;
  background: rgba(5, 15, 35, 0.9);
  border: 1px solid rgba(0, 180, 255, 0.18);
  border-radius: 16px;
  padding: 1.6rem 1.4rem;
  text-align: center;
  overflow: hidden;
  transition: all 0.4s cubic-bezier(0.23, 1, 0.32, 1);
  backdrop-filter: blur(20px);
}
.forecast-card:hover {
  transform: translateY(-4px);
  box-shadow: 0 20px 40px rgba(0,0,0,0.4), var(--glow-purple);
}
.forecast-card.current {
  background: rgba(0, 40, 80, 0.6);
  border-color: rgba(0, 212, 255, 0.35);
  box-shadow: var(--glow-cyan);
}
.forecast-card.best-model {
  border-color: rgba(0, 255, 136, 0.35);
  box-shadow: 0 0 30px rgba(0, 255, 136, 0.12);
}
.forecast-label {
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.68rem;
  color: var(--text-dim);
  text-transform: uppercase;
  letter-spacing: 1.5px;
  margin-bottom: 0.5rem;
}
.forecast-price {
  font-family: 'Orbitron', monospace;
  font-size: 1.6rem;
  font-weight: 700;
  color: var(--text-primary);
  letter-spacing: -0.5px;
  margin-bottom: 0.3rem;
}
.forecast-price.current { color: var(--neon-cyan); text-shadow: 0 0 20px rgba(0, 212, 255, 0.4); }
.forecast-delta-up   { color: var(--neon-green); font-size: 0.9rem; font-weight: 600; font-family: 'JetBrains Mono', monospace; text-shadow: 0 0 10px rgba(0,255,136,0.4); }
.forecast-delta-down { color: var(--neon-red);   font-size: 0.9rem; font-weight: 600; font-family: 'JetBrains Mono', monospace; text-shadow: 0 0 10px rgba(255,64,96,0.4); }
.forecast-model-name {
  font-size: 0.8rem;
  font-weight: 600;
  color: var(--text-secondary);
  margin-bottom: 0.6rem;
}

/* ═══════════════════════ CHART CONTAINER ═══════════════════════ */
.chart-shell {
  background: rgba(4, 12, 28, 0.8);
  border: 1px solid rgba(0, 180, 255, 0.12);
  border-radius: 16px;
  padding: 1rem;
  backdrop-filter: blur(20px);
  margin-bottom: 1rem;
}

/* ═══════════════════════ TABS ═══════════════════════ */
div[data-testid="stTabs"] {
  background: rgba(4, 12, 28, 0.5);
  border-radius: 12px;
  padding: 0.3rem;
  border: 1px solid rgba(0, 180, 255, 0.1);
}
div[data-testid="stTabs"] button[role="tab"] {
  font-family: 'Inter', sans-serif !important;
  font-weight: 500 !important;
  font-size: 0.85rem !important;
  color: var(--text-dim) !important;
  border-radius: 8px !important;
  padding: 0.5rem 1rem !important;
  transition: all 0.2s ease !important;
}
div[data-testid="stTabs"] button[aria-selected="true"] {
  color: var(--neon-cyan) !important;
  background: rgba(0, 212, 255, 0.08) !important;
  border-bottom: none !important;
  box-shadow: 0 0 15px rgba(0, 212, 255, 0.1) !important;
}

/* ═══════════════════════ METRICS OVERRIDE ═══════════════════════ */
[data-testid="stMetricLabel"] {
  font-family: 'JetBrains Mono', monospace !important;
  font-size: 0.7rem !important;
  color: var(--text-dim) !important;
  text-transform: uppercase !important;
  letter-spacing: 1px !important;
}
[data-testid="stMetricValue"] {
  font-family: 'Orbitron', monospace !important;
  color: var(--text-primary) !important;
  font-size: 1.4rem !important;
}
[data-testid="stMetricDelta"] {
  font-family: 'JetBrains Mono', monospace !important;
  font-size: 0.82rem !important;
}

/* ═══════════════════════ EXPANDER ═══════════════════════ */
details {
  background: rgba(4, 12, 28, 0.7) !important;
  border: 1px solid rgba(0, 180, 255, 0.15) !important;
  border-radius: 12px !important;
  padding: 0.2rem 0 !important;
}
details summary {
  color: var(--text-secondary) !important;
  font-family: 'Inter', sans-serif !important;
  font-size: 0.88rem !important;
  font-weight: 500 !important;
  padding: 0.8rem 1rem !important;
}

/* ═══════════════════════ DATAFRAME ═══════════════════════ */
[data-testid="stDataFrame"] {
  border-radius: 10px !important;
  overflow: hidden;
  border: 1px solid rgba(0, 180, 255, 0.15) !important;
}

/* ═══════════════════════ FOOTER CAPTION ═══════════════════════ */
.footer-note {
  text-align: center;
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.72rem;
  color: var(--text-dim);
  margin-top: 2rem;
  padding-top: 1rem;
  border-top: 1px solid rgba(0, 180, 255, 0.08);
  letter-spacing: 0.5px;
}
.footer-note span { color: var(--neon-cyan); }

/* ═══════════════════════ SIDEBAR SECTION LABEL ═══════════════════════ */
.sb-section {
  font-family: 'Orbitron', monospace;
  font-size: 0.65rem;
  font-weight: 700;
  letter-spacing: 2px;
  text-transform: uppercase;
  color: rgba(0, 212, 255, 0.5) !important;
  margin: 1.2rem 0 0.5rem;
}

/* ═══════════════════════ SPINNER / LOADING ═══════════════════════ */
.stSpinner > div { border-top-color: var(--neon-cyan) !important; }

/* ═══════════════════════ SCROLLBAR ═══════════════════════ */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg-deep); }
::-webkit-scrollbar-thumb { background: rgba(0, 180, 255, 0.25); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: rgba(0, 212, 255, 0.5); }

/* ═══════════════════════ WARNING / ERROR ═══════════════════════ */
[data-testid="stAlert"] {
  border-radius: 12px !important;
  font-family: 'Inter', sans-serif !important;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# Constants
# ══════════════════════════════════════════════════════════════════════════════
_PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    plot_bgcolor="rgba(4,12,28,0.0)",
    paper_bgcolor="rgba(4,12,28,0.0)",
    font=dict(family="Inter, sans-serif", color="#7ab8d4", size=11),
    legend=dict(
        bgcolor="rgba(5,15,35,0.85)",
        bordercolor="rgba(0,180,255,0.2)",
        borderwidth=1,
        font=dict(size=11),
    ),
    margin=dict(l=0, r=0, t=30, b=0),
    xaxis=dict(
        gridcolor="rgba(0,180,255,0.06)",
        zeroline=False,
        showspikes=True,
        spikecolor="rgba(0,212,255,0.3)",
        spikethickness=1,
    ),
    yaxis=dict(
        gridcolor="rgba(0,180,255,0.06)",
        zeroline=False,
    ),
    hoverlabel=dict(
        bgcolor="rgba(5,15,35,0.95)",
        bordercolor="rgba(0,212,255,0.4)",
        font=dict(family="JetBrains Mono, monospace", size=12, color="#e8f4ff"),
    ),
)

_MODEL_COLOR = {
    "Linear Regression": "#0090ff",
    "Random Forest":     "#00ff88",
    "XGBoost":           "#ff9f43",
    "LightGBM":          "#a855f7",
    "LSTM":              "#ffd700",
}
_C = {
    "green":  "#00ff88",
    "red":    "#ff4060",
    "blue":   "#0090ff",
    "orange": "#ff9f43",
    "purple": "#a855f7",
    "gold":   "#ffd700",
    "gray":   "#3d6080",
    "cyan":   "#00d4ff",
}


# ══════════════════════════════════════════════════════════════════════════════
# Technical Indicators
# ══════════════════════════════════════════════════════════════════════════════
def compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    if isinstance(series, pd.DataFrame):
        series = series.squeeze()
    delta = series.diff().fillna(0)
    up   = np.where(delta > 0, delta, 0.0)
    down = np.where(delta < 0, -delta, 0.0)
    roll_up   = pd.Series(up,   index=series.index).rolling(window).mean()
    roll_down = pd.Series(down, index=series.index).rolling(window).mean()
    return 100.0 - (100.0 / (1.0 + roll_up / (roll_down + 1e-9)))


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if isinstance(df["Close"], pd.DataFrame):
        df["Close"] = df["Close"].squeeze()
    close = df["Close"]

    df["Return_1d"]     = close.pct_change()
    df["Return_5d"]     = close.pct_change(5)
    df["RollingStd_10"] = close.pct_change().rolling(10).std()

    df["SMA_5"]  = close.rolling(5).mean()
    df["SMA_10"] = close.rolling(10).mean()
    df["SMA_20"] = close.rolling(20).mean()
    df["EMA_10"] = close.ewm(span=10, adjust=False).mean()
    df["EMA_20"] = close.ewm(span=20, adjust=False).mean()

    df["RSI_14"] = compute_rsi(close, 14)

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["MACD"]        = ema12 - ema26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"]   = df["MACD"] - df["MACD_Signal"]

    std20 = close.rolling(20).std()
    df["BB_Upper"] = df["SMA_20"] + 2 * std20
    df["BB_Lower"] = df["SMA_20"] - 2 * std20
    df["BB_Width"] = (df["BB_Upper"] - df["BB_Lower"]) / (df["SMA_20"] + 1e-9)
    df["BB_Pct"]   = (close - df["BB_Lower"]) / (df["BB_Upper"] - df["BB_Lower"] + 1e-9)

    if "High" in df.columns and "Low" in df.columns:
        high, low = df["High"], df["Low"]
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low  - close.shift(1)).abs(),
        ], axis=1).max(axis=1)
        df["ATR_14"] = tr.rolling(14).mean()

    if "Volume" in df.columns:
        direction = np.sign(close.diff().fillna(0))
        df["OBV"] = (direction * df["Volume"]).cumsum()

    return df


# ══════════════════════════════════════════════════════════════════════════════
# Data Fetching
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def fetch_history(symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
    df = yf.download(symbol, start=start, end=end, progress=False, auto_adjust=True)
    if df is None or df.empty:
        return pd.DataFrame()
    df.index = pd.to_datetime(df.index)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


@st.cache_data(show_spinner=False, ttl=60)
def fetch_live_price(symbol: str) -> tuple:
    try:
        hist = yf.Ticker(symbol).history(period="2d", interval="1d")
        if hist is not None and len(hist) >= 2:
            prev = float(hist["Close"].iloc[-2])
            curr = float(hist["Close"].iloc[-1])
            return curr, (curr - prev) / prev * 100
        elif hist is not None and len(hist) == 1:
            return float(hist["Close"].iloc[-1]), float("nan")
    except Exception:
        pass
    return float("nan"), float("nan")


@st.cache_data(show_spinner=False, ttl=300)
def fetch_news_with_dates(symbol: str, query: str, max_items: int = 30) -> list:
    articles = []
    if NEWSAPI_KEY:
        try:
            r = requests.get(
                "https://newsapi.org/v2/everything",
                params={"q": query, "sortBy": "publishedAt", "language": "en",
                        "pageSize": max_items, "apiKey": NEWSAPI_KEY},
                timeout=10,
            )
            for a in r.json().get("articles", []):
                if a.get("title") and a.get("publishedAt"):
                    try:
                        articles.append({"title": a["title"],
                                         "date": pd.to_datetime(a["publishedAt"]).date()})
                    except Exception:
                        pass
        except Exception:
            pass
    if not articles:
        try:
            for n in (yf.Ticker(symbol).news or [])[:max_items]:
                if n.get("title") and n.get("providerPublishTime"):
                    articles.append({
                        "title": n["title"],
                        "date":  pd.to_datetime(n["providerPublishTime"], unit="s").date(),
                    })
        except Exception:
            pass
    return articles


def build_time_aligned_sentiment(df: pd.DataFrame, symbol: str, query: str) -> pd.DataFrame:
    articles = fetch_news_with_dates(symbol, query)
    df = df.copy()
    if not articles:
        df["Sentiment"] = 0.0
        return df
    daily: dict = {}
    for a in articles:
        score = TextBlob(a["title"]).sentiment.polarity
        daily.setdefault(a["date"], []).append(score)
    daily_mean = {pd.Timestamp(d): float(np.mean(v)) for d, v in daily.items()}
    sent = pd.Series(daily_mean).reindex(df.index).ffill().fillna(0.0)
    df["Sentiment"] = sent
    return df


# ══════════════════════════════════════════════════════════════════════════════
# Feature Engineering
# ══════════════════════════════════════════════════════════════════════════════
def prepare_features_labels(df: pd.DataFrame):
    df = df.copy()
    if "Close" not in df.columns:
        raise KeyError("'Close' column missing.")
    if "Volume" not in df.columns:
        df["Volume"] = 0.0
    if "Sentiment" not in df.columns:
        df["Sentiment"] = 0.0

    df["y_next_close"] = df["Close"].shift(-1)

    possible = [
        "SMA_5", "SMA_10", "SMA_20", "EMA_10", "EMA_20",
        "Return_1d", "Return_5d", "RSI_14", "RollingStd_10",
        "MACD", "MACD_Signal", "MACD_Hist",
        "BB_Width", "BB_Pct", "ATR_14", "OBV",
        "Volume", "Sentiment",
    ]
    feature_cols = [f for f in possible if f in df.columns]
    if not feature_cols:
        raise KeyError("No feature columns found.")

    subset = [c for c in feature_cols + ["y_next_close"] if c in df.columns]
    df = df.dropna(subset=subset).copy()
    return df[feature_cols].values, df["y_next_close"].values, feature_cols, df.index


# ══════════════════════════════════════════════════════════════════════════════
# Model Helpers
# ══════════════════════════════════════════════════════════════════════════════
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    mape = float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-9))) * 100)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return {"RMSE": rmse, "MAPE (%)": mape, "R²": 1 - ss_res / (ss_tot + 1e-9)}


def _train_lr(X_tr, y_tr, X_te):
    sc = StandardScaler().fit(X_tr)
    m  = LinearRegression().fit(sc.transform(X_tr), y_tr)
    return m.predict(sc.transform(X_te)), sc, m


def _train_rf(X_tr, y_tr, X_te, n_estimators=200):
    m = RandomForestRegressor(n_estimators=n_estimators, random_state=42, n_jobs=-1)
    m.fit(X_tr, y_tr)
    return m.predict(X_te), m


def _train_xgb(X_tr, y_tr, X_te, y_te):
    m = XGBRegressor(
        n_estimators=300, learning_rate=0.05, max_depth=5,
        subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0,
    )
    m.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)
    return m.predict(X_te), m


def _train_lgb(X_tr, y_tr, X_te, y_te):
    m = LGBMRegressor(
        n_estimators=500, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1,
    )
    try:
        m.fit(X_tr, y_tr, eval_set=[(X_te, y_te)],
              callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(-1)])
    except Exception:
        m.fit(X_tr, y_tr)
    return m.predict(X_te), m


def sequenceify(X: np.ndarray, y: np.ndarray, lookback: int = 20):
    Xs, ys = [], []
    for i in range(lookback, len(X)):
        Xs.append(X[i - lookback:i])
        ys.append(y[i])
    return np.array(Xs), np.array(ys)


def make_lstm_model(input_shape):
    m = Sequential([
        LSTM(64, input_shape=input_shape, return_sequences=True),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dense(1),
    ])
    m.compile(optimizer="adam", loss="mse")
    return m


# ══════════════════════════════════════════════════════════════════════════════
# Plotting — all return go.Figure with consistent dark/neon theme
# ══════════════════════════════════════════════════════════════════════════════
def _L(**kw): return dict(**_PLOTLY_LAYOUT, **kw)


def plot_price_chart(df: pd.DataFrame, show: set) -> go.Figure:
    rows, heights = 1, [0.5]
    if "Volume" in df.columns:
        rows += 1; heights.append(0.12)
    if "MACD" in show and "MACD" in df.columns:
        rows += 1; heights.append(0.19)
    if "RSI" in show and "RSI_14" in df.columns:
        rows += 1; heights.append(0.19)

    total    = sum(heights)
    heights  = [h / total for h in heights]
    fig      = make_subplots(rows=rows, cols=1, shared_xaxes=True,
                              row_heights=heights, vertical_spacing=0.018)

    o  = df["Open"]  if "Open"  in df.columns else df["Close"]
    h  = df["High"]  if "High"  in df.columns else df["Close"]
    lo = df["Low"]   if "Low"   in df.columns else df["Close"]

    fig.add_trace(go.Candlestick(
        x=df.index, open=o, high=h, low=lo, close=df["Close"],
        name="Price",
        increasing_line_color=_C["green"], increasing_fillcolor=_C["green"],
        decreasing_line_color=_C["red"],   decreasing_fillcolor=_C["red"],
    ), row=1, col=1)

    if "SMA/EMA" in show:
        for col_name, color, dash, w in [
            ("SMA_20", _C["cyan"],   "solid", 1.5),
            ("EMA_20", _C["orange"], "dash",  1.5),
        ]:
            if col_name in df.columns:
                fig.add_trace(go.Scatter(
                    x=df.index, y=df[col_name], name=col_name,
                    line=dict(color=color, width=w, dash=dash),
                ), row=1, col=1)

    if "Bollinger Bands" in show and "BB_Upper" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["BB_Upper"], name="BB Upper",
            line=dict(color="rgba(168,85,247,0.5)", width=1, dash="dot"),
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=df.index, y=df["BB_Lower"], name="BB Lower",
            line=dict(color="rgba(168,85,247,0.5)", width=1, dash="dot"),
            fill="tonexty", fillcolor="rgba(168,85,247,0.04)",
        ), row=1, col=1)

    cur = 2
    if "Volume" in df.columns:
        vol_colors = [
            _C["green"] if float(c) >= float(op) else _C["red"]
            for c, op in zip(df["Close"], o)
        ]
        fig.add_trace(go.Bar(
            x=df.index, y=df["Volume"], name="Volume",
            marker_color=vol_colors, opacity=0.45,
        ), row=cur, col=1)
        fig.update_yaxes(title_text="Vol", row=cur, col=1)
        cur += 1

    if "MACD" in show and "MACD" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["MACD"], name="MACD",
            line=dict(color=_C["cyan"], width=1.5),
        ), row=cur, col=1)
        fig.add_trace(go.Scatter(
            x=df.index, y=df["MACD_Signal"], name="Signal",
            line=dict(color=_C["orange"], width=1.5),
        ), row=cur, col=1)
        hist_colors = [
            _C["green"] if v >= 0 else _C["red"]
            for v in df["MACD_Hist"].fillna(0)
        ]
        fig.add_trace(go.Bar(
            x=df.index, y=df["MACD_Hist"], name="Hist",
            marker_color=hist_colors, opacity=0.5,
        ), row=cur, col=1)
        fig.update_yaxes(title_text="MACD", row=cur, col=1)
        cur += 1

    if "RSI" in show and "RSI_14" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["RSI_14"], name="RSI 14",
            line=dict(color=_C["purple"], width=1.8),
        ), row=cur, col=1)
        for level, color in [(70, _C["red"]), (30, _C["green"])]:
            fig.add_trace(go.Scatter(
                x=[df.index[0], df.index[-1]], y=[level, level],
                mode="lines", line=dict(color=color, dash="dot", width=1),
                showlegend=False,
            ), row=cur, col=1)
        fig.add_hrect(y0=70, y1=100,
                      fillcolor="rgba(255,64,96,0.04)", line_width=0,
                      row=cur, col=1)
        fig.add_hrect(y0=0, y1=30,
                      fillcolor="rgba(0,255,136,0.04)", line_width=0,
                      row=cur, col=1)
        fig.update_yaxes(title_text="RSI", range=[0, 100], row=cur, col=1)

    fig.update_layout(
        height=660,
        xaxis_rangeslider_visible=False,
        **_L(),
    )
    return fig


def plot_multi_stock(dfs: dict) -> go.Figure:
    palette = [_C["cyan"], _C["green"], _C["orange"], _C["purple"], _C["gold"]]
    fig = go.Figure()
    for i, (sym, df) in enumerate(dfs.items()):
        if df.empty or "Close" not in df.columns:
            continue
        norm = df["Close"] / df["Close"].iloc[0] * 100
        fig.add_trace(go.Scatter(
            x=df.index, y=norm, name=sym,
            line=dict(color=palette[i % len(palette)], width=2.2),
            fill="tonexty" if i == 0 else "none",
            fillcolor=f"rgba({','.join(str(int(palette[i%len(palette)].lstrip('#')[j:j+2], 16)) for j in (0,2,4))},0.04)",
        ))
    fig.update_layout(
        yaxis_title="Normalised Price (base = 100)",
        height=420, **_L(),
    )
    return fig


def plot_predictions(results: dict, dates_test, y_test) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates_test, y=y_test,
        name="Actual",
        line=dict(color="#e8f4ff", width=2.5),
        fill="tozeroy",
        fillcolor="rgba(232,244,255,0.03)",
    ))
    for name, (preds, _) in results.items():
        n = min(len(dates_test), len(preds))
        fig.add_trace(go.Scatter(
            x=dates_test[-n:], y=preds[-n:],
            name=name,
            line=dict(color=_MODEL_COLOR.get(name, "#fff"), width=2, dash="dash"),
        ))
    fig.update_layout(
        xaxis_title="Date", yaxis_title="Closing Price",
        height=450, **_L(),
    )
    return fig


def plot_metrics_bar(results: dict, best_name: str) -> go.Figure:
    names   = list(results.keys())
    metrics = [results[n][1] for n in names]
    colors  = [
        _MODEL_COLOR.get(n, "#0090ff") if n == best_name else "rgba(0,40,80,0.7)"
        for n in names
    ]

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=["RMSE  ↓ lower is better",
                        "MAPE %  ↓ lower is better",
                        "R²  ↑ higher is better"],
    )
    for col_idx, key in enumerate(["RMSE", "MAPE (%)", "R²"], start=1):
        vals = [m[key] for m in metrics]
        bar_colors = [
            _MODEL_COLOR.get(n, "#0090ff") if n == best_name else "rgba(0,60,120,0.6)"
            for n in names
        ]
        fig.add_trace(go.Bar(
            x=names, y=vals,
            marker=dict(
                color=bar_colors,
                line=dict(
                    color=[_MODEL_COLOR.get(n, "#0090ff") if n == best_name else "rgba(0,180,255,0.2)" for n in names],
                    width=1.5,
                ),
            ),
            text=[f"{v:.3f}" for v in vals],
            textposition="outside",
            textfont=dict(family="JetBrains Mono", size=10, color="#e8f4ff"),
            showlegend=False,
        ), row=1, col=col_idx)

    fig.update_annotations(font=dict(family="Inter", size=11, color="#7ab8d4"))
    fig.update_layout(height=320, **_L())
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# Sidebar
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 1rem 0 0.5rem;">
      <div style="font-family:'Orbitron',monospace; font-size:1.1rem; font-weight:900;
                  background:linear-gradient(135deg,#00d4ff,#a855f7);
                  -webkit-background-clip:text; -webkit-text-fill-color:transparent;
                  background-clip:text; letter-spacing:2px;">QUANTVISION</div>
      <div style="font-size:0.65rem; color:#3d6080; letter-spacing:1.5px;
                  font-family:'JetBrains Mono',monospace; margin-top:0.2rem;">
        INTELLIGENCE TERMINAL
      </div>
    </div>
    <hr style="border:none; border-top:1px solid rgba(0,180,255,0.15); margin:0.8rem 0;">
    """, unsafe_allow_html=True)

    st.markdown('<div class="sb-section">◈ Symbols</div>', unsafe_allow_html=True)
    _primary_label = st.selectbox(
        "Primary Symbol",
        options=INDIAN_STOCKS,
        index=0,
        help="Type to search — by company name or ticker symbol (NSE)",
    )
    primary_symbol = _ticker_from_label(_primary_label)
    _compare_labels = st.multiselect(
        "Compare With",
        options=INDIAN_STOCKS,
        default=[],
        help="Type to search and add multiple stocks for comparison",
    )
    compare_symbols = [_ticker_from_label(l) for l in _compare_labels]

    st.markdown('<div class="sb-section">◈ Date Range</div>', unsafe_allow_html=True)
    dc1, dc2 = st.columns(2)
    with dc1:
        start_date = st.date_input("Start", value=datetime.today() - timedelta(days=730))
    with dc2:
        end_date = st.date_input("End", value=datetime.today())

    st.markdown('<div class="sb-section">◈ Indicators</div>', unsafe_allow_html=True)
    ic1, ic2 = st.columns(2)
    show_sma  = ic1.checkbox("SMA / EMA",      value=True)
    show_rsi  = ic2.checkbox("RSI",             value=True)
    show_macd = ic1.checkbox("MACD",            value=True)
    show_bb   = ic2.checkbox("Bollinger BB",    value=True)

    show_indicators = set()
    if show_sma:  show_indicators.add("SMA/EMA")
    if show_rsi:  show_indicators.add("RSI")
    if show_macd: show_indicators.add("MACD")
    if show_bb:   show_indicators.add("Bollinger Bands")

    st.markdown('<div class="sb-section">◈ Models</div>', unsafe_allow_html=True)
    use_lr   = st.checkbox("Linear Regression",  value=True)
    use_rf   = st.checkbox("Random Forest",       value=True)
    _xgb_lbl = f"XGBoost {'✓' if XGB_AVAILABLE else '✗'}"
    _lgb_lbl = f"LightGBM {'✓' if LGB_AVAILABLE else '✗'}"
    _lstm_lbl = f"LSTM (TF) {'✓' if TF_AVAILABLE else '✗'}"
    use_xgb  = st.checkbox(_xgb_lbl,  value=XGB_AVAILABLE)
    use_lgb  = st.checkbox(_lgb_lbl,  value=LGB_AVAILABLE)
    use_lstm = st.checkbox(_lstm_lbl, value=False)

    if use_xgb  and not XGB_AVAILABLE:  use_xgb  = False
    if use_lgb  and not LGB_AVAILABLE:  use_lgb  = False
    if use_lstm and not TF_AVAILABLE:   use_lstm = False

    st.markdown('<div class="sb-section">◈ Training</div>', unsafe_allow_html=True)
    test_size  = st.slider("Test Split",           0.1, 0.4, 0.2, 0.05)
    rf_trees   = st.slider("RF Trees",              50, 500, 200,   50) if use_rf   else 200
    lookback   = st.slider("LSTM Lookback (days)",  10,  60,  20,    5) if use_lstm else 20
    epochs     = st.slider("LSTM Epochs",           10, 200,  60,   10) if use_lstm else 60
    batch_size = st.slider("LSTM Batch Size",       16, 256,  64,   16) if use_lstm else 64

    st.markdown("""
    <hr style="border:none; border-top:1px solid rgba(0,180,255,0.1); margin:1rem 0;">
    <div style="font-family:'JetBrains Mono',monospace; font-size:0.65rem;
                color:rgba(61,96,128,0.8); text-align:center; line-height:1.8;">
      Educational demo only<br>Not financial advice
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# Derived settings
# ══════════════════════════════════════════════════════════════════════════════
_IS_INDIAN = primary_symbol.endswith((".NS", ".BO"))
CURRENCY   = "₹" if _IS_INDIAN else "$"


@st.cache_data(show_spinner=False, ttl=3600)
def _company_name(symbol: str) -> str:
    try:
        name = yf.Ticker(symbol).info.get("longName", "")
        return name if name else symbol.split(".")[0]
    except Exception:
        return symbol.split(".")[0]


NEWS_QUERY = _company_name(primary_symbol) if _IS_INDIAN else primary_symbol


# ══════════════════════════════════════════════════════════════════════════════
# HERO HEADER
# ══════════════════════════════════════════════════════════════════════════════
now_str = datetime.now().strftime("%H:%M:%S")
tz_label = "IST" if _IS_INDIAN else "local"

st.markdown(f"""
<div class="hero-header">
  <div class="hero-grid-lines"></div>
  <div style="position:relative; z-index:1;">
    <div style="display:flex; align-items:center; gap:1rem; flex-wrap:wrap;">
      <div>
        <h1 class="hero-logo">QUANTVISION</h1>
        <p class="hero-sub">
          Technical Intelligence &nbsp;·&nbsp; News Sentiment &nbsp;·&nbsp;
          Multi-Model ML &nbsp;·&nbsp; Next-Day Forecast
          <span class="hero-badge">BETA</span>
        </p>
      </div>
    </div>
    <div class="hero-status">
      <div class="hero-dot"></div>
      LIVE &nbsp;|&nbsp; {now_str} {tz_label} &nbsp;|&nbsp;
      Streaming {primary_symbol} &nbsp;·&nbsp; Auto-refresh 60s
    </div>
  </div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# Live Price Snapshot
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="sec-title">
  <span class="sec-title-icon">⚡</span>
  <span class="sec-title-text">Live Snapshot</span>
  <div class="sec-title-line"></div>
</div>
""", unsafe_allow_html=True)

all_symbols = [primary_symbol] + compare_symbols
live_cols   = st.columns(len(all_symbols))

for col, sym in zip(live_cols, all_symbols):
    price, pct = fetch_live_price(sym)
    if math.isnan(price):
        card_html = f"""
        <div class="price-card">
          <div class="price-card-corner"></div>
          <div class="price-card-symbol">{sym}</div>
          <div class="price-card-value" style="font-size:1.3rem; color:#3d6080;">N / A</div>
          <div class="price-card-delta-neu">Market closed · API limit</div>
        </div>"""
    else:
        if not math.isnan(pct):
            arrow  = "▲" if pct >= 0 else "▼"
            dclass = "price-card-delta-pos" if pct >= 0 else "price-card-delta-neg"
            delta_html = f'<div class="{dclass}">{arrow} {abs(pct):.2f}%</div>'
        else:
            delta_html = '<div class="price-card-delta-neu">— %</div>'

        card_html = f"""
        <div class="price-card">
          <div class="price-card-corner"></div>
          <div class="price-card-symbol">{sym}</div>
          <div class="price-card-value">{CURRENCY}{price:,.2f}</div>
          {delta_html}
        </div>"""
    with col:
        st.markdown(card_html, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# Fetch & Process Historical Data
# ══════════════════════════════════════════════════════════════════════════════
start_dt = pd.to_datetime(start_date)
end_dt   = pd.to_datetime(end_date) + pd.Timedelta(days=1)

all_dfs: dict = {}
with st.spinner("Fetching historical data…"):
    for sym in all_symbols:
        tmp = fetch_history(sym, start_dt, end_dt)
        if not tmp.empty:
            all_dfs[sym] = add_technical_indicators(tmp)

if primary_symbol not in all_dfs:
    st.error(f"No data returned for **{primary_symbol}**. Check the symbol or date range.")
    st.stop()

df = all_dfs[primary_symbol].copy()

with st.spinner("Fetching news & computing time-aligned sentiment…"):
    df = build_time_aligned_sentiment(df, primary_symbol, NEWS_QUERY)
    all_dfs[primary_symbol] = df

# Quick stats strip
_close = df["Close"]
_hi    = float(_close.max())
_lo    = float(_close.min())
_rows  = len(df)
_ret   = float((_close.iloc[-1] - _close.iloc[0]) / _close.iloc[0] * 100)
_vol   = float(df["Volume"].mean()) if "Volume" in df.columns else 0.0

st.markdown(f"""
<div style="margin-top:1.2rem;">
  <div class="stat-strip">
    <div class="stat-pill">52W HIGH<span>{CURRENCY}{_hi:,.2f}</span></div>
    <div class="stat-pill">52W LOW<span>{CURRENCY}{_lo:,.2f}</span></div>
    <div class="stat-pill">PERIOD RETURN<span style="color:{'#00ff88' if _ret>=0 else '#ff4060'}">{_ret:+.2f}%</span></div>
    <div class="stat-pill">TRADING DAYS<span>{_rows}</span></div>
    <div class="stat-pill">AVG VOLUME<span>{_vol:,.0f}</span></div>
  </div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# Price Charts
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="sec-title">
  <span class="sec-title-icon">📈</span>
  <span class="sec-title-text">Price Chart</span>
  <div class="sec-title-line"></div>
</div>
""", unsafe_allow_html=True)

if len(all_dfs) > 1:
    ct1, ct2 = st.tabs([f"📊  {primary_symbol}  Detail", "⚖️  Multi-Stock Comparison"])
else:
    ct1 = st.container()
    ct2 = None

with ct1:
    st.markdown('<div class="chart-shell">', unsafe_allow_html=True)
    st.plotly_chart(plot_price_chart(df, show_indicators), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

if ct2:
    with ct2:
        st.markdown('<div class="chart-shell">', unsafe_allow_html=True)
        st.plotly_chart(plot_multi_stock(all_dfs), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        closes = pd.DataFrame({s: d["Close"] for s, d in all_dfs.items()}).dropna()
        if len(closes.columns) > 1:
            st.markdown("**Correlation Matrix**")
            st.dataframe(closes.corr().round(3), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# Sentiment
# ══════════════════════════════════════════════════════════════════════════════
if "Sentiment" in df.columns:
    avg = df["Sentiment"].mean()
    if avg > 0.05:
        sent_class, sent_icon, sent_label = "positive", "🟢", "Positive"
    elif avg < -0.05:
        sent_class, sent_icon, sent_label = "negative", "🔴", "Negative"
    else:
        sent_class, sent_icon, sent_label = "neutral",  "🟡", "Neutral"

    st.markdown(f"""
    <div class="info-box {sent_class}">
      <div class="info-box-icon">📰</div>
      <div class="info-box-text">
        News sentiment for <strong>{primary_symbol}</strong>:
        {sent_icon} <strong>{sent_label}</strong>
        &nbsp;(avg polarity {avg:+.4f}) — time-aligned per trading day ·
        NewsAPI 30-day window → yfinance fallback.
      </div>
    </div>
    """, unsafe_allow_html=True)

with st.expander("◈  Data Preview — last 10 rows", expanded=False):
    st.markdown(
        f'<div class="stat-strip" style="margin-bottom:0.8rem;">'
        f'<div class="stat-pill">ROWS<span>{len(df)}</span></div>'
        f'<div class="stat-pill">START<span>{df.index.min().date()}</span></div>'
        f'<div class="stat-pill">END<span>{df.index.max().date()}</span></div>'
        f'</div>',
        unsafe_allow_html=True,
    )
    st.dataframe(df.tail(10), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# Model Training
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="sec-title">
  <span class="sec-title-icon">🤖</span>
  <span class="sec-title-text">Model Training & Evaluation</span>
  <div class="sec-title-line"></div>
</div>
""", unsafe_allow_html=True)

X, y, feature_cols, dates_idx = prepare_features_labels(df)

if len(X) < 200:
    st.warning(f"Only **{len(X)}** rows after feature engineering. Expand the date range (min ~200 needed).")
    st.stop()

split_idx       = int(len(X) * (1 - test_size))
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]
dates_test      = dates_idx[split_idx:]

results: dict = {}
_store = {}

with st.spinner("Training models…"):
    if use_lr:
        preds, sc, m = _train_lr(X_train, y_train, X_test)
        results["Linear Regression"] = (preds, compute_metrics(y_test, preds))
        _store["lr"] = (sc, m)

    if use_rf:
        preds, m = _train_rf(X_train, y_train, X_test, rf_trees)
        results["Random Forest"] = (preds, compute_metrics(y_test, preds))
        _store["rf"] = m

    if use_xgb:
        preds, m = _train_xgb(X_train, y_train, X_test, y_test)
        results["XGBoost"] = (preds, compute_metrics(y_test, preds))
        _store["xgb"] = m

    if use_lgb:
        preds, m = _train_lgb(X_train, y_train, X_test, y_test)
        results["LightGBM"] = (preds, compute_metrics(y_test, preds))
        _store["lgb"] = m

    if use_lstm:
        lstm_sc     = MinMaxScaler()
        X_train_sc  = lstm_sc.fit_transform(X_train)
        X_test_sc   = lstm_sc.transform(X_test)
        X_tr_seq, y_tr_seq = sequenceify(X_train_sc, y_train, lookback)
        X_te_seq, y_te_seq = sequenceify(
            np.vstack([X_train_sc[-lookback:], X_test_sc]),
            np.hstack([y_train[-lookback:], y_test]),
            lookback,
        )
        if len(X_tr_seq) >= 10:
            lstm_m = make_lstm_model((X_tr_seq.shape[1], X_tr_seq.shape[2]))
            es = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
            lstm_m.fit(X_tr_seq, y_tr_seq, validation_split=0.2,
                       epochs=epochs, batch_size=batch_size, verbose=0, callbacks=[es])
            preds = lstm_m.predict(X_te_seq, verbose=0).flatten()
            results["LSTM"] = (preds, compute_metrics(y_te_seq, preds))
            _store["lstm"] = (lstm_sc, lstm_m)

if not results:
    st.warning("No models selected. Enable at least one in the sidebar.")
    st.stop()

best_name = min(results, key=lambda n: results[n][1]["RMSE"])


# ── Model Comparison Cards ─────────────────────────────────────────────────────
card_cols = st.columns(len(results))
for col, (name, (_, metrics)) in zip(card_cols, results.items()):
    is_best = name == best_name
    color   = _MODEL_COLOR.get(name, "#0090ff")
    badge   = '<span class="badge-best">★ BEST</span>' if is_best else ""
    cls     = "model-card best" if is_best else "model-card"
    with col:
        st.markdown(
            f'<div class="{cls}">'
            f'  <div class="model-name" style="color:{color}">{name}{badge}</div>'
            f'  <hr class="model-divider">'
            f'  <div class="model-label">RMSE</div>'
            f'  <div class="model-value">{metrics["RMSE"]:.2f}</div>'
            f'  <div class="model-label">MAPE</div>'
            f'  <div class="model-value">{metrics["MAPE (%)"]:.2f}%</div>'
            f'  <div class="model-label">R²</div>'
            f'  <div class="model-value">{metrics["R²"]:.4f}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

st.markdown("<br>", unsafe_allow_html=True)

st.markdown('<div class="chart-shell">', unsafe_allow_html=True)
st.plotly_chart(plot_metrics_bar(results, best_name), use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown("""
<div style="font-family:'Orbitron',monospace; font-size:0.75rem; color:#7ab8d4;
            letter-spacing:1.5px; text-transform:uppercase; margin:1.5rem 0 0.7rem;">
  Actual vs Predicted — Test Period
</div>
""", unsafe_allow_html=True)
st.markdown('<div class="chart-shell">', unsafe_allow_html=True)
st.plotly_chart(plot_predictions(results, dates_test, y_test), use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# Next-Day Forecast
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="sec-title">
  <span class="sec-title-icon">🔮</span>
  <span class="sec-title-text">Next-Day Forecast</span>
  <div class="sec-title-line"></div>
</div>
""", unsafe_allow_html=True)

X_all, y_all, _, _ = prepare_features_labels(df.dropna().copy())
X_fc_tr, y_fc_tr   = X_all[:-1], y_all[:-1]
X_last              = X_all[-1:]
current_close       = float(df["Close"].dropna().iloc[-1])

fc_cols = st.columns(len(results) + 1)

# Current price card
with fc_cols[0]:
    st.markdown(f"""
    <div class="forecast-card current">
      <div class="forecast-label">Current Close</div>
      <div class="forecast-price current">{CURRENCY}{current_close:,.2f}</div>
      <div style="font-size:0.75rem; color:#3d6080; font-family:'JetBrains Mono',monospace; margin-top:0.3rem;">
        {primary_symbol}
      </div>
    </div>
    """, unsafe_allow_html=True)

# Model forecast cards
for col, name in zip(fc_cols[1:], results.keys()):
    with col:
        pred = float("nan")
        try:
            if name == "Linear Regression":
                sc  = StandardScaler().fit(X_fc_tr)
                m   = LinearRegression().fit(sc.transform(X_fc_tr), y_fc_tr)
                pred = float(m.predict(sc.transform(X_last))[0])
            elif name == "Random Forest":
                m    = RandomForestRegressor(n_estimators=rf_trees, random_state=42, n_jobs=-1).fit(X_fc_tr, y_fc_tr)
                pred = float(m.predict(X_last)[0])
            elif name == "XGBoost":
                m    = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=5,
                                    random_state=42, verbosity=0).fit(X_fc_tr, y_fc_tr)
                pred = float(m.predict(X_last)[0])
            elif name == "LightGBM":
                m    = LGBMRegressor(n_estimators=300, learning_rate=0.05,
                                     random_state=42, verbose=-1).fit(X_fc_tr, y_fc_tr)
                pred = float(m.predict(X_last)[0])
            elif name == "LSTM" and "lstm" in _store:
                sc, lstm_m = _store["lstm"]
                Xseq = sc.transform(X_all[-lookback:]).reshape(1, lookback, -1)
                pred = float(lstm_m.predict(Xseq, verbose=0)[0][0])
        except Exception:
            pass

        is_best = name == best_name

        if math.isnan(pred):
            card = f"""
            <div class="forecast-card">
              <div class="forecast-label">{name}</div>
              <div class="forecast-price" style="color:#3d6080; font-size:1.2rem;">N / A</div>
            </div>"""
        else:
            delta_pct = (pred - current_close) / current_close * 100
            arrow     = "▲" if delta_pct >= 0 else "▼"
            dclass    = "forecast-delta-up" if delta_pct >= 0 else "forecast-delta-down"
            best_cls  = " best-model" if is_best else ""
            star      = "★ " if is_best else ""
            card = f"""
            <div class="forecast-card{best_cls}">
              <div class="forecast-model-name">{star}{name}</div>
              <div class="forecast-label">Predicted Close</div>
              <div class="forecast-price">{CURRENCY}{pred:,.2f}</div>
              <div class="{dclass}">{arrow} {abs(delta_pct):.2f}%</div>
            </div>"""
        st.markdown(card, unsafe_allow_html=True)


# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="footer-note">
  <span>QUANTVISION</span> · Educational demo only ·
  Predictions are <span>not financial advice</span> ·
  Data via yfinance & NewsAPI · Built with Streamlit + Plotly
</div>
""", unsafe_allow_html=True)
