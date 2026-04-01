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

st.set_page_config(
    page_title="QuantVision",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

load_dotenv()
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY", "6fe740c75d764b009015e175452500e4")

# ══════════════════════════════════════════════════════════════════════════════
# INDIAN MARKET UNIVERSE  (name, ticker)
# NSE suffix: .NS  |  BSE suffix: .BO
# ══════════════════════════════════════════════════════════════════════════════
_RAW = [
    # ── NIFTY 50 ──────────────────────────────────────────────────────────────
    ("Adani Enterprises",            "ADANIENT.NS"),
    ("Adani Ports & SEZ",            "ADANIPORTS.NS"),
    ("Apollo Hospitals",             "APOLLOHOSP.NS"),
    ("Asian Paints",                 "ASIANPAINT.NS"),
    ("Axis Bank",                    "AXISBANK.NS"),
    ("Bajaj Auto",                   "BAJAJ-AUTO.NS"),
    ("Bajaj Finance",                "BAJFINANCE.NS"),
    ("Bajaj Finserv",                "BAJAJFINSV.NS"),
    ("Bharat Petroleum (BPCL)",      "BPCL.NS"),
    ("Bharti Airtel",                "BHARTIARTL.NS"),
    ("Britannia Industries",         "BRITANNIA.NS"),
    ("Cipla",                        "CIPLA.NS"),
    ("Coal India",                   "COALINDIA.NS"),
    ("Divi's Laboratories",          "DIVISLAB.NS"),
    ("Dr. Reddy's Laboratories",     "DRREDDY.NS"),
    ("Eicher Motors",                "EICHERMOT.NS"),
    ("Grasim Industries",            "GRASIM.NS"),
    ("HCL Technologies",             "HCLTECH.NS"),
    ("HDFC Bank",                    "HDFCBANK.NS"),
    ("HDFC Life Insurance",          "HDFCLIFE.NS"),
    ("Hero MotoCorp",                "HEROMOTOCO.NS"),
    ("Hindalco Industries",          "HINDALCO.NS"),
    ("Hindustan Unilever (HUL)",     "HINDUNILVR.NS"),
    ("ICICI Bank",                   "ICICIBANK.NS"),
    ("IndusInd Bank",                "INDUSINDBK.NS"),
    ("Infosys",                      "INFY.NS"),
    ("ITC",                          "ITC.NS"),
    ("JSW Steel",                    "JSWSTEEL.NS"),
    ("Kotak Mahindra Bank",          "KOTAKBANK.NS"),
    ("Larsen & Toubro (L&T)",        "LT.NS"),
    ("Mahindra & Mahindra (M&M)",    "M&M.NS"),
    ("Maruti Suzuki",                "MARUTI.NS"),
    ("Nestle India",                 "NESTLEIND.NS"),
    ("NTPC",                         "NTPC.NS"),
    ("ONGC",                         "ONGC.NS"),
    ("Power Grid Corporation",       "POWERGRID.NS"),
    ("Reliance Industries",          "RELIANCE.NS"),
    ("SBI Life Insurance",           "SBILIFE.NS"),
    ("State Bank of India (SBI)",    "SBIN.NS"),
    ("Sun Pharmaceuticals",          "SUNPHARMA.NS"),
    ("Tata Consumer Products",       "TATACONSUM.NS"),
    ("Tata Motors",                  "TATAMOTORS.NS"),
    ("Tata Steel",                   "TATASTEEL.NS"),
    ("Tata Consultancy Services (TCS)", "TCS.NS"),
    ("Tech Mahindra",                "TECHM.NS"),
    ("Titan Company",                "TITAN.NS"),
    ("Trent",                        "TRENT.NS"),
    ("UltraTech Cement",             "ULTRACEMCO.NS"),
    ("Wipro",                        "WIPRO.NS"),
    ("Shriram Finance",              "SHRIRAMFIN.NS"),

    # ── NIFTY NEXT 50 & LARGE CAPS ────────────────────────────────────────────
    ("ABB India",                    "ABB.NS"),
    ("Ambuja Cements",               "AMBUJACEM.NS"),
    ("ACC",                          "ACC.NS"),
    ("Avenue Supermarts (DMart)",    "DMART.NS"),
    ("Berger Paints",                "BERGEPAINT.NS"),
    ("Biocon",                       "BIOCON.NS"),
    ("Bosch India",                  "BOSCHLTD.NS"),
    ("Colgate-Palmolive India",      "COLPAL.NS"),
    ("Container Corp of India",      "CONCOR.NS"),
    ("Dabur India",                  "DABUR.NS"),
    ("DLF",                          "DLF.NS"),
    ("GAIL India",                   "GAIL.NS"),
    ("Godrej Consumer Products",     "GODREJCP.NS"),
    ("Godrej Properties",            "GODREJPROP.NS"),
    ("HDFC AMC",                     "HDFCAMC.NS"),
    ("ICICI Lombard General Insurance","ICICIGI.NS"),
    ("ICICI Prudential Life Insurance","ICICIPRULI.NS"),
    ("Indian Hotels (Taj)",          "INDHOTEL.NS"),
    ("Info Edge (Naukri)",           "NAUKRI.NS"),
    ("IndiGo (InterGlobe Aviation)", "INDIGO.NS"),
    ("Jubilant FoodWorks",           "JUBLFOOD.NS"),
    ("LIC Housing Finance",          "LICHSGFIN.NS"),
    ("LTIMindtree",                  "LTIM.NS"),
    ("Lupin",                        "LUPIN.NS"),
    ("Marico",                       "MARICO.NS"),
    ("Muthoot Finance",              "MUTHOOTFIN.NS"),
    ("NMDC",                         "NMDC.NS"),
    ("Page Industries (Jockey)",     "PAGEIND.NS"),
    ("Petronet LNG",                 "PETRONET.NS"),
    ("PI Industries",                "PIIND.NS"),
    ("Pidilite Industries (Fevicol)","PIDILITIND.NS"),
    ("Piramal Enterprises",          "PEL.NS"),
    ("Punjab National Bank (PNB)",   "PNB.NS"),
    ("Shree Cement",                 "SHREECEM.NS"),
    ("Siemens India",                "SIEMENS.NS"),
    ("Havells India",                "HAVELLS.NS"),
    ("Tata Communications",          "TATACOMM.NS"),
    ("Torrent Pharmaceuticals",      "TORNTPHARM.NS"),
    ("United Breweries (Kingfisher)","UBL.NS"),
    ("Vedanta",                      "VEDL.NS"),
    ("Voltas",                       "VOLTAS.NS"),
    ("Zomato",                       "ZOMATO.NS"),
    ("Nykaa (FSN E-Commerce)",       "NYKAA.NS"),
    ("Paytm (One97 Communications)", "PAYTM.NS"),
    ("Policybazaar (PB Fintech)",    "POLICYBZR.NS"),
    ("United Phosphorus (UPL)",      "UPL.NS"),
    ("Hindustan Petroleum (HPCL)",   "HINDPETRO.NS"),
    ("Indian Oil Corporation (IOC)", "IOC.NS"),
    ("Oracle Financial Services",    "OFSS.NS"),
    ("Mphasis",                      "MPHASIS.NS"),
    ("Persistent Systems",           "PERSISTENT.NS"),
    ("Coforge",                      "COFORGE.NS"),
    ("KPIT Technologies",            "KPITTECH.NS"),

    # ── MIDCAP ────────────────────────────────────────────────────────────────
    ("Astral",                       "ASTRAL.NS"),
    ("AU Small Finance Bank",        "AUBANK.NS"),
    ("Bank of Baroda",               "BANKBARODA.NS"),
    ("Bank of India",                "BANKINDIA.NS"),
    ("Bharat Electronics (BEL)",     "BEL.NS"),
    ("Bharat Forge",                 "BHARATFORG.NS"),
    ("Blue Star",                    "BLUESTAR.NS"),
    ("BSE Ltd",                      "BSE.NS"),
    ("Can Fin Homes",                "CANFINHOME.NS"),
    ("Canara Bank",                  "CANBK.NS"),
    ("CESC",                         "CESC.NS"),
    ("Cholamandalam Finance",        "CHOLAFIN.NS"),
    ("City Union Bank",              "CUB.NS"),
    ("CRISIL",                       "CRISIL.NS"),
    ("Cummins India",                "CUMMINSIND.NS"),
    ("Deepak Nitrite",               "DEEPAKNTR.NS"),
    ("Dixon Technologies",           "DIXON.NS"),
    ("Escorts Kubota",               "ESCORTS.NS"),
    ("Federal Bank",                 "FEDERALBNK.NS"),
    ("HDFC Bank BSE",                "HDFCBANK.BO"),
    ("IDFC First Bank",              "IDFCFIRSTB.NS"),
    ("Indian Bank",                  "INDIANB.NS"),
    ("IndiaMART InterMESH",          "INDIAMART.NS"),
    ("IRCTC",                        "IRCTC.NS"),
    ("IRFC",                         "IRFC.NS"),
    ("JK Cement",                    "JKCEMENT.NS"),
    ("Kaynes Technology",            "KAYNES.NS"),
    ("Laurus Labs",                  "LAURUSLABS.NS"),
    ("Max Healthcare",               "MAXHEALTH.NS"),
    ("Metropolis Healthcare",        "METROPOLIS.NS"),
    ("MRF",                          "MRF.NS"),
    ("Natco Pharma",                 "NATCOPHARM.NS"),
    ("NCC",                          "NCC.NS"),
    ("NHPC",                         "NHPC.NS"),
    ("Pfizer India",                 "PFIZER.NS"),
    ("Polycab India",                "POLYCAB.NS"),
    ("Prestige Estates",             "PRESTIGE.NS"),
    ("PVR Inox",                     "PVRINOX.NS"),
    ("Rail Vikas Nigam (RVNL)",      "RVNL.NS"),
    ("Raymond",                      "RAYMOND.NS"),
    ("Redington",                    "REDINGTON.NS"),
    ("SBI Cards & Payment",          "SBICARD.NS"),
    ("SRF",                          "SRF.NS"),
    ("Star Health Insurance",        "STARHEALTH.NS"),
    ("Supreme Industries",           "SUPREMEIND.NS"),
    ("Syngene International",        "SYNGENE.NS"),
    ("Tata Chemicals",               "TATACHEM.NS"),
    ("Tata Elxsi",                   "TATAELXSI.NS"),
    ("Tata Power",                   "TATAPOWER.NS"),
    ("Torrent Power",                "TORNTPOWER.NS"),
    ("Tube Investments of India",    "TIINDIA.NS"),
    ("Union Bank of India",          "UNIONBANK.NS"),
    ("Varun Beverages",              "VBL.NS"),
    ("Vedant Fashions (Manyavar)",   "MANYAVAR.NS"),
    ("V-Guard Industries",           "VGUARD.NS"),
    ("Yes Bank",                     "YESBANK.NS"),
    ("Zydus Lifesciences",           "ZYDUSLIFE.NS"),
    ("Adani Green Energy",           "ADANIGREEN.NS"),
    ("Adani Total Gas",              "ATGL.NS"),
    ("Angel One",                    "ANGELONE.NS"),
    ("Schaeffler India",             "SCHAEFFLER.NS"),
    ("Sundaram Finance",             "SUNDARMFIN.NS"),
    ("Karnataka Bank",               "KTKBANK.NS"),
    ("Route Mobile",                 "ROUTE.NS"),
    ("Sapphire Foods",               "SAPPHIRE.NS"),
    ("SKF India",                    "SKFINDIA.NS"),
    ("Thomas Cook India",            "THOMASCOOK.NS"),
    ("Timken India",                 "TIMKEN.NS"),
    ("Gujarat Gas",                  "GUJGASLTD.NS"),
    ("Bharat 22 ETF",                "ICICIB22.NS"),
    ("Welspun Corp",                 "WELSPUN.NS"),
    ("Godrej Agrovet",               "GODREJAGRO.NS"),
    ("GNFC",                         "GNFC.NS"),
    ("Motherson Sumi (SAMIL)",       "MOTHERSUMI.NS"),

    # ── NSE ETFs ──────────────────────────────────────────────────────────────
    ("Nippon Nifty BeES ETF",        "NIFTYBEES.NS"),
    ("Nippon Bank BeES ETF",         "BANKBEES.NS"),
    ("Nippon Junior BeES ETF",       "JUNIORBEES.NS"),
    ("Nippon Gold BeES ETF",         "GOLDBEES.NS"),
    ("Nippon Liquid BeES ETF",       "LIQUIDBEES.NS"),
    ("CPSE ETF",                     "CPSEETF.NS"),
    ("Nippon IT BeES ETF",           "ITBEES.NS"),
    ("Nippon PSU Bank BeES ETF",     "PSUBNKBEES.NS"),
    ("HDFC Nifty 50 ETF",            "HDFCNIFTY.NS"),
    ("HDFC Sensex ETF",              "HDFCSENSEX.NS"),
    ("ICICI Pru Nifty 50 ETF",       "ICICINIFTY.NS"),
    ("ICICI Pru Nifty Next 50 ETF",  "ICICINXT50.NS"),
    ("Kotak Nifty 50 ETF",           "KOTAKNIFTY.NS"),
    ("SBI ETF Nifty 50",             "SETFNIF50.NS"),
    ("SBI ETF Sensex",               "SETFBSE500.NS"),
    ("Nippon ETF Nifty 100",         "NETFNIF100.NS"),
    ("UTI Nifty 50 ETF",             "UTINIFTETF.NS"),
    ("Mirae Asset Nifty 50 ETF",     "MAFANG.NS"),
    ("Motilal Oswal Nifty 500 ETF",  "MOM500.NS"),
    ("Nippon ETF Nifty Midcap 150",  "NETFMID150.NS"),
    ("Edelweiss Nifty 100 ETF",      "ENIFTY.NS"),
    ("Franklin Nifty ETF",           "NIFTYIETF.NS"),
    ("DSP Nifty 50 ETF",             "DSPNIFTY.NS"),
    ("Quantum Nifty 50 ETF",         "QNIFTY.NS"),
    ("Nippon ETF Nifty SmallCap 250","NETFSMID250.NS"),
    ("HDFC Nifty Next 50 ETF",       "HDFCNIFTY50.NS"),
    ("Nippon ETF Shariah BeES",      "SHARIABEES.NS"),
    ("Nippon Silver ETF",            "SILVERBEES.NS"),

    # ── BSE / SENSEX COMPONENTS ───────────────────────────────────────────────
    ("Reliance Industries BSE",      "RELIANCE.BO"),
    ("TCS BSE",                      "TCS.BO"),
    ("HDFC Bank BSE",                "HDFCBANK.BO"),
    ("Infosys BSE",                  "INFY.BO"),
    ("ICICI Bank BSE",               "ICICIBANK.BO"),
    ("Hindustan Unilever BSE",       "HINDUNILVR.BO"),
    ("State Bank of India BSE",      "SBIN.BO"),
    ("Bharti Airtel BSE",            "BHARTIARTL.BO"),
    ("ITC BSE",                      "ITC.BO"),
    ("Kotak Mahindra Bank BSE",      "KOTAKBANK.BO"),
    ("L&T BSE",                      "LT.BO"),
    ("HCL Technologies BSE",         "HCLTECH.BO"),
    ("Axis Bank BSE",                "AXISBANK.BO"),
    ("Bajaj Finance BSE",            "BAJFINANCE.BO"),
    ("Wipro BSE",                    "WIPRO.BO"),
    ("Maruti Suzuki BSE",            "MARUTI.BO"),
    ("Asian Paints BSE",             "ASIANPAINT.BO"),
    ("Sun Pharma BSE",               "SUNPHARMA.BO"),
    ("UltraTech Cement BSE",         "ULTRACEMCO.BO"),
    ("Titan BSE",                    "TITAN.BO"),
    ("NTPC BSE",                     "NTPC.BO"),
    ("Power Grid BSE",               "POWERGRID.BO"),
    ("M&M BSE",                      "M&M.BO"),
    ("Tata Motors BSE",              "TATAMOTORS.BO"),
    ("Tata Steel BSE",               "TATASTEEL.BO"),
    ("Bajaj Auto BSE",               "BAJAJ-AUTO.BO"),
    ("Dr. Reddy's BSE",              "DRREDDY.BO"),
    ("Nestle India BSE",             "NESTLEIND.BO"),
    ("IndusInd Bank BSE",            "INDUSINDBK.BO"),
    ("Tech Mahindra BSE",            "TECHM.BO"),
]

# Sorted display list and lookup maps
STOCK_OPTIONS = sorted(
    [f"{name}  ({ticker})" for name, ticker in _RAW],
    key=lambda x: x.lower(),
)
_TICKER_FROM_LABEL = {f"{name}  ({ticker})": ticker for name, ticker in _RAW}
_DEFAULT_LABEL = "Reliance Industries  (RELIANCE.NS)"

# ══════════════════════════════════════════════════════════════════════════════
# DESIGN SYSTEM CSS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:ital,wght@0,300;0,400;0,500;0,600;0,700;0,800;0,900;1,400&display=swap');

/* ── TOKENS ── */
:root {
  --bg:          #0C0C0E;
  --surface:     #161618;
  --surface-2:   #1E1E21;
  --surface-3:   #26262A;
  --border:      rgba(255,255,255,0.07);
  --border-md:   rgba(255,255,255,0.11);

  --mint:        #3DD68C;
  --mint-dim:    rgba(61,214,140,0.12);
  --mint-text:   #3DD68C;

  --rose:        #FF8FAB;
  --rose-dim:    rgba(255,143,171,0.12);

  --amber:       #F5A623;
  --amber-dim:   rgba(245,166,35,0.12);

  --blue:        #60A5FA;
  --blue-dim:    rgba(96,165,250,0.12);

  --violet:      #A78BFA;
  --violet-dim:  rgba(167,139,250,0.12);

  --up:          #3DD68C;
  --down:        #FF6B7A;
  --up-bg:       rgba(61,214,140,0.10);
  --down-bg:     rgba(255,107,122,0.10);

  --text-1:      #F5F5F7;
  --text-2:      #A0A0AB;
  --text-3:      #5A5A66;

  --radius-sm:   10px;
  --radius-md:   16px;
  --radius-lg:   20px;
  --radius-xl:   24px;

  --shadow:      0 2px 12px rgba(0,0,0,0.45);
  --shadow-md:   0 6px 28px rgba(0,0,0,0.55);
  --shadow-lg:   0 12px 48px rgba(0,0,0,0.65);
}

/* ── RESET / GLOBAL ── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, .stApp {
  background-color: var(--bg) !important;
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
  color: var(--text-1) !important;
  -webkit-font-smoothing: antialiased;
}

#MainMenu, footer, header { visibility: hidden !important; }
.block-container {
  padding: 1.5rem 2rem 3rem !important;
  max-width: 1600px !important;
}

/* ── SCROLLBAR ── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--surface-3); border-radius: 3px; }

/* ── SIDEBAR ── */
section[data-testid="stSidebar"] {
  background: var(--surface) !important;
  border-right: 1px solid var(--border) !important;
}
section[data-testid="stSidebar"] * { color: var(--text-2) !important; }
section[data-testid="stSidebar"] label { color: var(--text-3) !important; font-size: 0.78rem !important; }

section[data-testid="stSidebar"] .stTextInput input,
section[data-testid="stSidebar"] .stDateInput input {
  background: var(--surface-2) !important;
  border: 1px solid var(--border-md) !important;
  color: var(--text-1) !important;
  border-radius: var(--radius-sm) !important;
  font-size: 0.88rem !important;
}
section[data-testid="stSidebar"] .stTextInput input:focus {
  border-color: var(--mint) !important;
  box-shadow: 0 0 0 2px rgba(61,214,140,0.15) !important;
}

section[data-testid="stSidebar"] hr {
  border: none !important;
  border-top: 1px solid var(--border) !important;
  margin: 1.1rem 0 !important;
}

/* ── TABS ── */
div[data-testid="stTabs"] {
  background: var(--surface-2);
  border-radius: var(--radius-md);
  padding: 4px;
  border: 1px solid var(--border);
  margin-bottom: 1rem;
}
div[data-testid="stTabs"] button[role="tab"] {
  font-family: 'Inter', sans-serif !important;
  font-size: 0.82rem !important;
  font-weight: 500 !important;
  color: var(--text-3) !important;
  border-radius: var(--radius-sm) !important;
  padding: 0.45rem 1.1rem !important;
  transition: all 0.18s ease !important;
}
div[data-testid="stTabs"] button[aria-selected="true"] {
  background: var(--surface-3) !important;
  color: var(--text-1) !important;
  border-bottom: none !important;
}

/* ── METRIC OVERRIDE ── */
[data-testid="stMetricLabel"] {
  font-size: 0.72rem !important;
  font-weight: 500 !important;
  color: var(--text-3) !important;
  text-transform: uppercase !important;
  letter-spacing: 0.06em !important;
}
[data-testid="stMetricValue"] {
  font-size: 1.5rem !important;
  font-weight: 700 !important;
  color: var(--text-1) !important;
  font-family: 'Inter', sans-serif !important;
}

/* ── EXPANDER ── */
details {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius-md) !important;
  overflow: hidden;
}
details summary {
  font-size: 0.84rem !important;
  font-weight: 500 !important;
  color: var(--text-2) !important;
  padding: 0.9rem 1.2rem !important;
}

/* ── DATAFRAME ── */
[data-testid="stDataFrame"] { border-radius: var(--radius-md) !important; overflow: hidden; }

/* ── SPINNER ── */
.stSpinner > div { border-top-color: var(--mint) !important; }

/* ══════════════════════════════════════════
   LAYOUT COMPONENTS
══════════════════════════════════════════ */

/* Top header bar */
.qv-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 1.4rem 2rem;
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius-xl);
  margin-bottom: 1.8rem;
}
.qv-header-left { display: flex; align-items: center; gap: 1rem; }
.qv-logo {
  font-size: 6rem;
  font-weight: 800;
  color: var(--text-1);
  letter-spacing: -0.5px;
}
.qv-logo span { color: var(--mint); }
.qv-tagline {
  font-size: 0.78rem;
  color: var(--text-3);
  font-weight: 400;
  margin-top: 2px;
}
.qv-live-badge {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  background: var(--up-bg);
  border: 1px solid rgba(61,214,140,0.2);
  color: var(--mint);
  font-size: 0.72rem;
  font-weight: 600;
  padding: 0.28rem 0.75rem;
  border-radius: 100px;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}
.qv-live-dot {
  width: 6px; height: 6px;
  background: var(--mint);
  border-radius: 50%;
  animation: blink 1.8s ease-in-out infinite;
}
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.3} }

.qv-time {
  font-size: 0.78rem;
  color: var(--text-3);
  font-weight: 400;
}

/* Section label */
.qv-section {
  font-size: 0.72rem;
  font-weight: 600;
  color: var(--text-3);
  text-transform: uppercase;
  letter-spacing: 0.1em;
  margin: 2rem 0 1rem;
  display: flex;
  align-items: center;
  gap: 0.6rem;
}
.qv-section::after {
  content: '';
  flex: 1;
  height: 1px;
  background: var(--border);
}

/* ── PRICE CARD ── */
.price-card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius-lg);
  padding: 1.5rem 1.6rem;
  position: relative;
  transition: border-color 0.2s, transform 0.2s, box-shadow 0.2s;
  cursor: default;
  overflow: hidden;
}
.price-card:hover {
  border-color: var(--border-md);
  transform: translateY(-2px);
  box-shadow: var(--shadow-md);
}
.price-card-accent {
  position: absolute;
  top: 0; left: 1.6rem; right: 1.6rem;
  height: 2px;
  border-radius: 0 0 2px 2px;
  opacity: 0.7;
}
.pc-symbol {
  font-size: 0.72rem;
  font-weight: 600;
  color: var(--text-3);
  text-transform: uppercase;
  letter-spacing: 0.1em;
  margin-bottom: 0.55rem;
}
.pc-price {
  font-size: 2rem;
  font-weight: 800;
  color: var(--text-1);
  letter-spacing: -1.5px;
  line-height: 1;
  margin-bottom: 0.55rem;
}
.pc-badge {
  display: inline-flex;
  align-items: center;
  gap: 3px;
  font-size: 0.78rem;
  font-weight: 600;
  padding: 0.2rem 0.6rem;
  border-radius: 100px;
}
.pc-badge.up   { background: var(--up-bg);   color: var(--up); }
.pc-badge.down { background: var(--down-bg); color: var(--down); }
.pc-badge.neu  { background: var(--surface-2); color: var(--text-3); }

/* ── STAT STRIP ── */
.stat-row {
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem;
  margin-bottom: 1.6rem;
}
.stat-chip {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  padding: 0.45rem 0.9rem;
  font-size: 0.75rem;
  color: var(--text-3);
  font-weight: 500;
  display: flex;
  gap: 0.4rem;
}
.stat-chip b { color: var(--text-2); font-weight: 600; }
.stat-chip b.up   { color: var(--up); }
.stat-chip b.down { color: var(--down); }

/* ── CHART SHELL ── */
.chart-card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius-lg);
  padding: 1.2rem 1.2rem 0.6rem;
  margin-bottom: 1rem;
}
.chart-card-title {
  font-size: 0.78rem;
  font-weight: 600;
  color: var(--text-3);
  text-transform: uppercase;
  letter-spacing: 0.08em;
  margin-bottom: 0.8rem;
  padding: 0 0.2rem;
}

/* ── MODEL CARD ── */
.model-card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius-lg);
  padding: 1.4rem 1.3rem;
  text-align: center;
  transition: border-color 0.2s, box-shadow 0.2s;
  position: relative;
  overflow: hidden;
}
.model-card.best {
  border-color: rgba(61,214,140,0.25);
  background: linear-gradient(160deg, rgba(61,214,140,0.04) 0%, var(--surface) 60%);
}
.mc-accent {
  position: absolute;
  top: 0; left: 0; right: 0; height: 3px;
  border-radius: var(--radius-lg) var(--radius-lg) 0 0;
}
.mc-name {
  font-size: 0.82rem;
  font-weight: 600;
  color: var(--text-2);
  margin-bottom: 0.2rem;
  margin-top: 0.3rem;
}
.mc-best-badge {
  display: inline-block;
  background: var(--up-bg);
  color: var(--mint);
  font-size: 0.62rem;
  font-weight: 700;
  padding: 0.15rem 0.5rem;
  border-radius: 100px;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  margin-left: 5px;
  vertical-align: middle;
  border: 1px solid rgba(61,214,140,0.2);
}
.mc-divider { border: none; border-top: 1px solid var(--border); margin: 0.8rem 0 0.3rem; }
.mc-label {
  font-size: 0.65rem;
  font-weight: 600;
  color: var(--text-3);
  text-transform: uppercase;
  letter-spacing: 0.1em;
  margin-top: 0.7rem;
}
.mc-value {
  font-size: 1.15rem;
  font-weight: 700;
  color: var(--text-1);
  margin-top: 0.1rem;
  letter-spacing: -0.3px;
}

/* ── FORECAST CARD ── */
.fc-card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius-lg);
  padding: 1.6rem 1.4rem;
  text-align: center;
  transition: border-color 0.2s, transform 0.2s, box-shadow 0.2s;
  position: relative;
  overflow: hidden;
}
.fc-card:hover { transform: translateY(-2px); box-shadow: var(--shadow-md); }
.fc-card.current {
  border-color: rgba(96,165,250,0.25);
  background: linear-gradient(160deg, rgba(96,165,250,0.05) 0%, var(--surface) 60%);
}
.fc-card.best {
  border-color: rgba(61,214,140,0.25);
  background: linear-gradient(160deg, rgba(61,214,140,0.05) 0%, var(--surface) 60%);
}
.fc-accent {
  position: absolute;
  top: 0; left: 0; right: 0; height: 3px;
  border-radius: var(--radius-lg) var(--radius-lg) 0 0;
}
.fc-label {
  font-size: 0.68rem;
  font-weight: 600;
  color: var(--text-3);
  text-transform: uppercase;
  letter-spacing: 0.1em;
  margin-bottom: 0.5rem;
}
.fc-model {
  font-size: 0.78rem;
  font-weight: 600;
  color: var(--text-2);
  margin-bottom: 0.6rem;
}
.fc-price {
  font-size: 1.75rem;
  font-weight: 800;
  color: var(--text-1);
  letter-spacing: -1px;
  line-height: 1;
  margin-bottom: 0.5rem;
}
.fc-price.current { color: var(--blue); }
.fc-delta {
  display: inline-flex;
  align-items: center;
  gap: 3px;
  font-size: 0.82rem;
  font-weight: 600;
  padding: 0.22rem 0.65rem;
  border-radius: 100px;
}
.fc-delta.up   { background: var(--up-bg);   color: var(--up); }
.fc-delta.down { background: var(--down-bg); color: var(--down); }

/* ── INFO BOX ── */
.info-row {
  display: flex;
  align-items: flex-start;
  gap: 0.9rem;
  background: var(--surface-2);
  border: 1px solid var(--border);
  border-radius: var(--radius-md);
  padding: 1rem 1.2rem;
  margin: 0.6rem 0 1rem;
}
.info-row.pos { border-left: 3px solid var(--mint); }
.info-row.neg { border-left: 3px solid var(--down); }
.info-row.neu { border-left: 3px solid var(--amber); }
.info-icon { font-size: 1rem; flex-shrink: 0; margin-top: 1px; }
.info-text { font-size: 0.83rem; color: var(--text-2); line-height: 1.6; }
.info-text b { color: var(--text-1); font-weight: 600; }

/* ── SIDEBAR SECTION LABEL ── */
.sb-label {
  font-size: 0.68rem;
  font-weight: 700;
  color: var(--text-3);
  text-transform: uppercase;
  letter-spacing: 0.1em;
  margin: 1.2rem 0 0.5rem;
}

/* ── FOOTER ── */
.qv-footer {
  text-align: center;
  font-size: 0.72rem;
  color: var(--text-3);
  margin-top: 2.5rem;
  padding-top: 1.2rem;
  border-top: 1px solid var(--border);
  font-weight: 400;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# Constants
# ══════════════════════════════════════════════════════════════════════════════
_MODEL_COLOR = {
    "Linear Regression": "#60A5FA",
    "Random Forest":     "#3DD68C",
    "XGBoost":           "#F5A623",
    "LightGBM":          "#A78BFA",
    "LSTM":              "#F472B6",
}
_C = {
    "green":  "#3DD68C",
    "red":    "#FF6B7A",
    "blue":   "#60A5FA",
    "orange": "#F5A623",
    "purple": "#A78BFA",
    "pink":   "#F472B6",
    "gray":   "#5A5A66",
}

_PLOTLY = dict(
    template="plotly_dark",
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color="#A0A0AB", size=11),
    legend=dict(
        bgcolor="rgba(22,22,24,0.9)",
        bordercolor="rgba(255,255,255,0.08)",
        borderwidth=1,
        font=dict(size=11),
    ),
    margin=dict(l=0, r=0, t=28, b=0),
    xaxis=dict(
        gridcolor="rgba(255,255,255,0.04)",
        zeroline=False,
        showspikes=True,
        spikecolor="rgba(255,255,255,0.15)",
        spikethickness=1,
    ),
    yaxis=dict(gridcolor="rgba(255,255,255,0.04)", zeroline=False),
    hoverlabel=dict(
        bgcolor="rgba(22,22,24,0.97)",
        bordercolor="rgba(255,255,255,0.12)",
        font=dict(family="Inter, sans-serif", size=12, color="#F5F5F7"),
    ),
)


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
    df["SMA_5"]         = close.rolling(5).mean()
    df["SMA_10"]        = close.rolling(10).mean()
    df["SMA_20"]        = close.rolling(20).mean()
    df["EMA_10"]        = close.ewm(span=10, adjust=False).mean()
    df["EMA_20"]        = close.ewm(span=20, adjust=False).mean()
    df["RSI_14"]        = compute_rsi(close, 14)

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
def compute_metrics(y_true, y_pred):
    rmse   = math.sqrt(mean_squared_error(y_true, y_pred))
    mape   = float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-9))) * 100)
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
    m = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=5,
                     subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0)
    m.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)
    return m.predict(X_te), m


def _train_lgb(X_tr, y_tr, X_te, y_te):
    m = LGBMRegressor(n_estimators=500, learning_rate=0.05, max_depth=6,
                      subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1)
    try:
        m.fit(X_tr, y_tr, eval_set=[(X_te, y_te)],
              callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(-1)])
    except Exception:
        m.fit(X_tr, y_tr)
    return m.predict(X_te), m


def sequenceify(X, y, lookback=20):
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
# Plotting
# ══════════════════════════════════════════════════════════════════════════════
def plot_price_chart(df: pd.DataFrame, show: set) -> go.Figure:
    rows, heights = 1, [0.52]
    if "Volume" in df.columns:
        rows += 1; heights.append(0.13)
    if "MACD" in show and "MACD" in df.columns:
        rows += 1; heights.append(0.18)
    if "RSI" in show and "RSI_14" in df.columns:
        rows += 1; heights.append(0.18)

    total   = sum(heights)
    heights = [h / total for h in heights]
    fig     = make_subplots(rows=rows, cols=1, shared_xaxes=True,
                             row_heights=heights, vertical_spacing=0.015)

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
        for col_name, color, dash in [
            ("SMA_20", _C["blue"],   "solid"),
            ("EMA_20", _C["orange"], "dash"),
        ]:
            if col_name in df.columns:
                fig.add_trace(go.Scatter(
                    x=df.index, y=df[col_name], name=col_name,
                    line=dict(color=color, width=1.5, dash=dash),
                ), row=1, col=1)

    if "Bollinger Bands" in show and "BB_Upper" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["BB_Upper"], name="BB Upper",
            line=dict(color="rgba(167,139,250,0.45)", width=1, dash="dot"),
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=df.index, y=df["BB_Lower"], name="BB Lower",
            line=dict(color="rgba(167,139,250,0.45)", width=1, dash="dot"),
            fill="tonexty", fillcolor="rgba(167,139,250,0.04)",
        ), row=1, col=1)

    cur = 2
    if "Volume" in df.columns:
        vc = [_C["green"] if float(c) >= float(op) else _C["red"]
              for c, op in zip(df["Close"], o)]
        fig.add_trace(go.Bar(
            x=df.index, y=df["Volume"], name="Volume",
            marker_color=vc, opacity=0.45,
        ), row=cur, col=1)
        fig.update_yaxes(title_text="Vol", title_font_size=10, row=cur, col=1)
        cur += 1

    if "MACD" in show and "MACD" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["MACD"], name="MACD",
            line=dict(color=_C["blue"], width=1.5),
        ), row=cur, col=1)
        fig.add_trace(go.Scatter(
            x=df.index, y=df["MACD_Signal"], name="Signal",
            line=dict(color=_C["orange"], width=1.5),
        ), row=cur, col=1)
        hc = [_C["green"] if v >= 0 else _C["red"]
              for v in df["MACD_Hist"].fillna(0)]
        fig.add_trace(go.Bar(
            x=df.index, y=df["MACD_Hist"], name="Hist",
            marker_color=hc, opacity=0.5,
        ), row=cur, col=1)
        fig.update_yaxes(title_text="MACD", title_font_size=10, row=cur, col=1)
        cur += 1

    if "RSI" in show and "RSI_14" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["RSI_14"], name="RSI 14",
            line=dict(color=_C["purple"], width=1.6),
        ), row=cur, col=1)
        for level, color in [(70, _C["red"]), (30, _C["green"])]:
            fig.add_trace(go.Scatter(
                x=[df.index[0], df.index[-1]], y=[level, level],
                mode="lines", line=dict(color=color, dash="dot", width=1),
                showlegend=False,
            ), row=cur, col=1)
        fig.add_hrect(y0=70, y1=100, fillcolor="rgba(255,107,122,0.04)", line_width=0,
                      row=cur, col=1)
        fig.add_hrect(y0=0,  y1=30,  fillcolor="rgba(61,214,140,0.04)",  line_width=0,
                      row=cur, col=1)
        fig.update_yaxes(title_text="RSI", range=[0, 100],
                         title_font_size=10, row=cur, col=1)

    fig.update_layout(height=640, xaxis_rangeslider_visible=False, **_PLOTLY)
    return fig


def plot_multi_stock(dfs: dict) -> go.Figure:
    palette = [_C["blue"], _C["green"], _C["orange"], _C["purple"], _C["pink"]]
    fig = go.Figure()
    for i, (sym, df) in enumerate(dfs.items()):
        if df.empty or "Close" not in df.columns:
            continue
        norm  = df["Close"] / df["Close"].iloc[0] * 100
        color = palette[i % len(palette)]
        fig.add_trace(go.Scatter(
            x=df.index, y=norm, name=sym,
            line=dict(color=color, width=2),
        ))
    fig.update_layout(yaxis_title="Normalised (base 100)", height=400, **_PLOTLY)
    return fig


def plot_predictions(results: dict, dates_test, y_test) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates_test, y=y_test, name="Actual",
        line=dict(color="#F5F5F7", width=2.2),
        fill="tozeroy", fillcolor="rgba(245,245,247,0.03)",
    ))
    for name, (preds, _) in results.items():
        n = min(len(dates_test), len(preds))
        fig.add_trace(go.Scatter(
            x=dates_test[-n:], y=preds[-n:], name=name,
            line=dict(color=_MODEL_COLOR.get(name, "#fff"), width=1.8, dash="dash"),
        ))
    fig.update_layout(xaxis_title="Date", yaxis_title="Closing Price",
                      height=430, **_PLOTLY)
    return fig


def plot_metrics_bar(results: dict, best_name: str) -> go.Figure:
    names   = list(results.keys())
    metrics = [results[n][1] for n in names]

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=["RMSE — lower is better",
                        "MAPE % — lower is better",
                        "R² — higher is better"],
    )
    for col_idx, key in enumerate(["RMSE", "MAPE (%)", "R²"], start=1):
        vals   = [m[key] for m in metrics]
        colors = [_MODEL_COLOR.get(n, "#60A5FA") if n == best_name
                  else "rgba(38,38,42,0.9)" for n in names]
        borders = [_MODEL_COLOR.get(n, "#60A5FA") if n == best_name
                   else "rgba(255,255,255,0.08)" for n in names]
        fig.add_trace(go.Bar(
            x=names, y=vals,
            marker=dict(color=colors, line=dict(color=borders, width=1.5),
                        cornerradius=6),
            text=[f"{v:.3f}" for v in vals],
            textposition="outside",
            textfont=dict(family="Inter", size=10, color="#A0A0AB"),
            showlegend=False,
        ), row=1, col=col_idx)

    fig.update_annotations(font=dict(family="Inter", size=10, color="#5A5A66"))
    fig.update_layout(height=300, **_PLOTLY)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# Sidebar
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="padding: 1.2rem 0.4rem 0.8rem; border-bottom: 1px solid rgba(255,255,255,0.07); margin-bottom:0.3rem;">
      <div style="font-size:1.15rem; font-weight:800; color:#F5F5F7; letter-spacing:-0.3px;">
        Quant<span style="color:#3DD68C;">Vision</span>
      </div>
      <div style="font-size:0.7rem; color:#5A5A66; margin-top:3px; font-weight:400;">
        Indian Markets · NSE &amp; BSE
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sb-label">Primary Stock / ETF / Fund</div>', unsafe_allow_html=True)
    st.caption("Type a name or ticker to search")
    _primary_label = st.selectbox(
        "Primary",
        options=STOCK_OPTIONS,
        index=STOCK_OPTIONS.index(_DEFAULT_LABEL) if _DEFAULT_LABEL in STOCK_OPTIONS else 0,
        label_visibility="collapsed",
    )
    primary_symbol = _TICKER_FROM_LABEL[_primary_label]

    st.markdown('<div class="sb-label">Compare With (optional)</div>', unsafe_allow_html=True)
    st.caption("Type to search, pick up to 4")
    _compare_labels = st.multiselect(
        "Compare",
        options=[o for o in STOCK_OPTIONS if o != _primary_label],
        default=[],
        max_selections=4,
        label_visibility="collapsed",
        placeholder="Search stocks, ETFs…",
    )
    compare_symbols = [_TICKER_FROM_LABEL[l] for l in _compare_labels]

    st.markdown('<div class="sb-label">Date Range</div>', unsafe_allow_html=True)
    dc1, dc2 = st.columns(2)
    with dc1:
        start_date = st.date_input("From", value=datetime.today() - timedelta(days=730),
                                   label_visibility="visible")
    with dc2:
        end_date = st.date_input("To", value=datetime.today(),
                                 label_visibility="visible")

    st.markdown('<div class="sb-label">Chart Indicators</div>', unsafe_allow_html=True)
    ic1, ic2 = st.columns(2)
    show_sma  = ic1.checkbox("SMA/EMA",   value=True)
    show_rsi  = ic2.checkbox("RSI",        value=True)
    show_macd = ic1.checkbox("MACD",       value=True)
    show_bb   = ic2.checkbox("Bollinger",  value=True)

    show_indicators = set()
    if show_sma:  show_indicators.add("SMA/EMA")
    if show_rsi:  show_indicators.add("RSI")
    if show_macd: show_indicators.add("MACD")
    if show_bb:   show_indicators.add("Bollinger Bands")

    st.markdown('<div class="sb-label">Models</div>', unsafe_allow_html=True)
    use_lr   = st.checkbox("Linear Regression", value=True)
    use_rf   = st.checkbox("Random Forest",      value=True)
    use_xgb  = st.checkbox(f"XGBoost {'✓' if XGB_AVAILABLE else '✗'}",   value=XGB_AVAILABLE)
    use_lgb  = st.checkbox(f"LightGBM {'✓' if LGB_AVAILABLE else '✗'}",  value=LGB_AVAILABLE)
    use_lstm = st.checkbox(f"LSTM (TF) {'✓' if TF_AVAILABLE else '✗'}",  value=False)

    if use_xgb  and not XGB_AVAILABLE:  use_xgb  = False
    if use_lgb  and not LGB_AVAILABLE:  use_lgb  = False
    if use_lstm and not TF_AVAILABLE:   use_lstm = False

    st.markdown('<div class="sb-label">Training</div>', unsafe_allow_html=True)
    test_size  = st.slider("Test Split",          0.1, 0.4, 0.2, 0.05)
    rf_trees   = st.slider("RF Trees",             50, 500, 200,   50) if use_rf   else 200
    lookback   = st.slider("LSTM Lookback",         10,  60,  20,    5) if use_lstm else 20
    epochs     = st.slider("LSTM Epochs",           10, 200,  60,   10) if use_lstm else 60
    batch_size = st.slider("LSTM Batch",            16, 256,  64,   16) if use_lstm else 64

    st.markdown("""
    <div style="margin-top:1.5rem; padding-top:1rem; border-top:1px solid rgba(255,255,255,0.07);
                font-size:0.68rem; color:#5A5A66; line-height:1.7;">
      For educational purposes only.<br>Not financial advice.
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# Derived settings
# ══════════════════════════════════════════════════════════════════════════════
CURRENCY = "₹"   # always Indian market

# Use the display name from our universe as the news search query
_primary_base = _primary_label.split("(")[0].strip()   # e.g. "Reliance Industries"
NEWS_QUERY    = _primary_base


# ══════════════════════════════════════════════════════════════════════════════
# TOP HEADER BAR
# ══════════════════════════════════════════════════════════════════════════════
now_str  = datetime.now().strftime("%d %b %Y  ·  %H:%M:%S")
tz_label = "IST"

st.markdown(f"""
<div class="qv-header">
  <div class="qv-header-left">
    <div>
      <div class="qv-logo">Quant<span>Vision</span></div>
      <div class="qv-tagline">Indian Markets &nbsp;·&nbsp; Technical Analysis &nbsp;·&nbsp; News Sentiment &nbsp;·&nbsp; ML Predictions</div>
    </div>
    <div class="qv-live-badge">
      <div class="qv-live-dot"></div>
      Live
    </div>
  </div>
  <div class="qv-time">{now_str} {tz_label}</div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# LIVE SNAPSHOT
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="qv-section">Live Snapshot</div>', unsafe_allow_html=True)

all_symbols = [primary_symbol] + compare_symbols
live_cols   = st.columns(len(all_symbols))

# Accent colors cycle for cards
_card_accents = ["#3DD68C", "#60A5FA", "#F5A623", "#A78BFA", "#F472B6"]

for i, (col, sym) in enumerate(zip(live_cols, all_symbols)):
    price, pct = fetch_live_price(sym)
    accent     = _card_accents[i % len(_card_accents)]

    if math.isnan(price):
        card = f"""
        <div class="price-card">
          <div class="price-card-accent" style="background:{accent};"></div>
          <div class="pc-symbol">{sym}</div>
          <div class="pc-price" style="font-size:1.4rem;color:var(--text-3);">N / A</div>
          <span class="pc-badge neu">Unavailable</span>
        </div>"""
    else:
        if not math.isnan(pct):
            arrow  = "▲" if pct >= 0 else "▼"
            cls    = "up" if pct >= 0 else "down"
            badge  = f'<span class="pc-badge {cls}">{arrow} {abs(pct):.2f}%</span>'
        else:
            badge = '<span class="pc-badge neu">— %</span>'

        card = f"""
        <div class="price-card">
          <div class="price-card-accent" style="background:{accent};"></div>
          <div class="pc-symbol">{sym}</div>
          <div class="pc-price">{CURRENCY}{price:,.2f}</div>
          {badge}
        </div>"""
    with col:
        st.markdown(card, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# Fetch & Process Data
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

with st.spinner("Analysing news sentiment…"):
    df = build_time_aligned_sentiment(df, primary_symbol, NEWS_QUERY)
    all_dfs[primary_symbol] = df

# ── Stats strip ───────────────────────────────────────────────────────────────
_cl  = df["Close"]
_hi  = float(_cl.max())
_lo  = float(_cl.min())
_ret = float((_cl.iloc[-1] - _cl.iloc[0]) / _cl.iloc[0] * 100)
_vol = float(df["Volume"].mean()) if "Volume" in df.columns else 0.0
_rsi = float(df["RSI_14"].dropna().iloc[-1]) if "RSI_14" in df.columns else float("nan")
_rows = len(df)
_ret_cls = "up" if _ret >= 0 else "down"

st.markdown(f"""
<div style="margin-top:1.2rem;">
  <div class="stat-row">
    <div class="stat-chip">Period High &nbsp;<b>{CURRENCY}{_hi:,.2f}</b></div>
    <div class="stat-chip">Period Low &nbsp;<b>{CURRENCY}{_lo:,.2f}</b></div>
    <div class="stat-chip">Return &nbsp;<b class="{_ret_cls}">{'▲' if _ret>=0 else '▼'} {abs(_ret):.2f}%</b></div>
    <div class="stat-chip">Trading Days &nbsp;<b>{_rows}</b></div>
    <div class="stat-chip">Avg Volume &nbsp;<b>{_vol:,.0f}</b></div>
    {'<div class="stat-chip">RSI(14) &nbsp;<b>' + f'{_rsi:.1f}' + '</b></div>' if not math.isnan(_rsi) else ''}
  </div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PRICE CHARTS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="qv-section">Price Chart</div>', unsafe_allow_html=True)

if len(all_dfs) > 1:
    ct1, ct2 = st.tabs([f"  {primary_symbol}  Detail  ", "  Multi-Stock Comparison  "])
else:
    ct1 = st.container()
    ct2 = None

with ct1:
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.plotly_chart(plot_price_chart(df, show_indicators), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

if ct2:
    with ct2:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.plotly_chart(plot_multi_stock(all_dfs), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        closes = pd.DataFrame({s: d["Close"] for s, d in all_dfs.items()}).dropna()
        if len(closes.columns) > 1:
            st.markdown(
                '<div style="font-size:0.78rem; font-weight:600; color:var(--text-3),'
                'text-transform:uppercase; letter-spacing:0.08em; margin-bottom:0.6rem;">'
                'Correlation Matrix</div>', unsafe_allow_html=True)
            st.dataframe(closes.corr().round(3), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# SENTIMENT
# ══════════════════════════════════════════════════════════════════════════════
if "Sentiment" in df.columns:
    avg = df["Sentiment"].mean()
    if avg > 0.05:
        cls, icon, label = "pos", "📰", "Positive"
    elif avg < -0.05:
        cls, icon, label = "neg", "📰", "Negative"
    else:
        cls, icon, label = "neu", "📰", "Neutral"

    st.markdown(f"""
    <div class="info-row {cls}">
      <div class="info-icon">{icon}</div>
      <div class="info-text">
        News sentiment for <b>{primary_symbol}</b>: <b>{label}</b>
        &nbsp;(avg polarity {avg:+.4f}) — time-aligned per trading day via
        NewsAPI (30-day window) with yfinance fallback.
      </div>
    </div>
    """, unsafe_allow_html=True)

with st.expander("  Data Preview — last 10 rows", expanded=False):
    st.markdown(f"""
    <div class="stat-row" style="margin-bottom:0.9rem;">
      <div class="stat-chip">Rows <b>{len(df)}</b></div>
      <div class="stat-chip">From <b>{df.index.min().date()}</b></div>
      <div class="stat-chip">To <b>{df.index.max().date()}</b></div>
    </div>
    """, unsafe_allow_html=True)
    st.dataframe(df.tail(10), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# MODEL TRAINING
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="qv-section">Model Training &amp; Evaluation</div>',
            unsafe_allow_html=True)

X, y, feature_cols, dates_idx = prepare_features_labels(df)

if len(X) < 200:
    st.warning(f"Only **{len(X)}** rows after feature engineering — expand date range (min ~200).")
    st.stop()

split_idx       = int(len(X) * (1 - test_size))
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]
dates_test      = dates_idx[split_idx:]

results: dict = {}
_store        = {}

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
        lstm_sc    = MinMaxScaler()
        X_train_sc = lstm_sc.fit_transform(X_train)
        X_test_sc  = lstm_sc.transform(X_test)
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
    st.warning("No models selected — enable at least one in the sidebar.")
    st.stop()

best_name = min(results, key=lambda n: results[n][1]["RMSE"])

# ── Model comparison cards ─────────────────────────────────────────────────────
card_cols = st.columns(len(results))
for col, (name, (_, metrics)) in zip(card_cols, results.items()):
    is_best = name == best_name
    color   = _MODEL_COLOR.get(name, "#60A5FA")
    badge   = '<span class="mc-best-badge">Best</span>' if is_best else ""
    cls     = "model-card best" if is_best else "model-card"
    with col:
        st.markdown(
            f'<div class="{cls}">'
            f'  <div class="mc-accent" style="background:{color};"></div>'
            f'  <div class="mc-name" style="color:{color}">{name}{badge}</div>'
            f'  <hr class="mc-divider">'
            f'  <div class="mc-label">RMSE</div>'
            f'  <div class="mc-value">{metrics["RMSE"]:.2f}</div>'
            f'  <div class="mc-label">MAPE</div>'
            f'  <div class="mc-value">{metrics["MAPE (%)"]:.2f}%</div>'
            f'  <div class="mc-label">R²</div>'
            f'  <div class="mc-value">{metrics["R²"]:.4f}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<div class="chart-card">', unsafe_allow_html=True)
st.plotly_chart(plot_metrics_bar(results, best_name), use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown("""
<div style="font-size:0.72rem; font-weight:600; color:var(--text-3);
            text-transform:uppercase; letter-spacing:0.1em; margin:1.5rem 0 0.8rem;">
  Actual vs Predicted — Test Period
</div>""", unsafe_allow_html=True)

st.markdown('<div class="chart-card">', unsafe_allow_html=True)
st.plotly_chart(plot_predictions(results, dates_test, y_test), use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# NEXT-DAY FORECAST
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="qv-section">Next-Day Forecast</div>', unsafe_allow_html=True)

X_all, y_all, _, _ = prepare_features_labels(df.dropna().copy())
X_fc_tr, y_fc_tr   = X_all[:-1], y_all[:-1]
X_last              = X_all[-1:]
current_close       = float(df["Close"].dropna().iloc[-1])

fc_cols = st.columns(len(results) + 1)

# Current price card
with fc_cols[0]:
    st.markdown(f"""
    <div class="fc-card current">
      <div class="fc-accent" style="background:#60A5FA;"></div>
      <div class="fc-label">Current Close</div>
      <div class="fc-price current">{CURRENCY}{current_close:,.2f}</div>
      <div style="font-size:0.72rem; color:var(--text-3); margin-top:0.4rem;">{primary_symbol}</div>
    </div>
    """, unsafe_allow_html=True)

# Per-model forecast
for col, name in zip(fc_cols[1:], results.keys()):
    with col:
        pred = float("nan")
        try:
            if name == "Linear Regression":
                sc   = StandardScaler().fit(X_fc_tr)
                m    = LinearRegression().fit(sc.transform(X_fc_tr), y_fc_tr)
                pred = float(m.predict(sc.transform(X_last))[0])
            elif name == "Random Forest":
                m    = RandomForestRegressor(n_estimators=rf_trees, random_state=42,
                                             n_jobs=-1).fit(X_fc_tr, y_fc_tr)
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
        color   = _MODEL_COLOR.get(name, "#60A5FA")

        if math.isnan(pred):
            card = f"""
            <div class="fc-card">
              <div class="fc-accent" style="background:{color};"></div>
              <div class="fc-model">{name}</div>
              <div class="fc-label">Predicted</div>
              <div class="fc-price" style="color:var(--text-3);font-size:1.3rem;">N / A</div>
            </div>"""
        else:
            delta     = (pred - current_close) / current_close * 100
            arrow     = "▲" if delta >= 0 else "▼"
            d_cls     = "up" if delta >= 0 else "down"
            star      = "★ " if is_best else ""
            best_cls  = " best" if is_best else ""
            card = f"""
            <div class="fc-card{best_cls}">
              <div class="fc-accent" style="background:{color};"></div>
              <div class="fc-model">{star}{name}</div>
              <div class="fc-label">Predicted Close</div>
              <div class="fc-price">{CURRENCY}{pred:,.2f}</div>
              <span class="fc-delta {d_cls}">{arrow} {abs(delta):.2f}%</span>
            </div>"""
        st.markdown(card, unsafe_allow_html=True)


# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="qv-footer">
  QuantVision &nbsp;·&nbsp; Educational demo only &nbsp;·&nbsp;
  Not financial advice &nbsp;·&nbsp;
  Data: yfinance &amp; NewsAPI &nbsp;·&nbsp; Built with Streamlit + Plotly
</div>
""", unsafe_allow_html=True)
