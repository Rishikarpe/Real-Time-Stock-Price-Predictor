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
    page_title="Stock Intelligence Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

load_dotenv()
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY", "6fe740c75d764b009015e175452500e4")

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* Main background */
  .stApp { background-color: #0d1117; }
  section[data-testid="stSidebar"] { background-color: #161b22; border-right: 1px solid #30363d; }
  section[data-testid="stSidebar"] * { color: #c9d1d9; }

  /* Header */
  .app-header {
    background: linear-gradient(135deg, #1f6feb 0%, #58a6ff 100%);
    padding: 2rem 2.5rem;
    border-radius: 12px;
    margin-bottom: 1.5rem;
  }
  .app-header h1 { color: #fff; font-size: 2rem; font-weight: 700; margin: 0; letter-spacing: -0.5px; }
  .app-header p  { color: #cce0ff; margin: 0.4rem 0 0; font-size: 0.92rem; }

  /* Section headers */
  .sec-header {
    color: #e6edf3;
    font-size: 1.05rem;
    font-weight: 600;
    padding-bottom: 0.45rem;
    border-bottom: 1px solid #30363d;
    margin: 1.4rem 0 0.9rem;
  }

  /* Model cards */
  .model-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    text-align: center;
    height: 100%;
  }
  .model-card.best { border-color: #238636; box-shadow: 0 0 0 1px #238636; }
  .model-name { font-weight: 700; font-size: 0.92rem; margin-bottom: 0.5rem; }
  .card-divider { border: none; border-top: 1px solid #30363d; margin: 0.4rem 0; }
  .card-label { color: #8b949e; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.06em; margin-top: 0.5rem; }
  .card-value { color: #e6edf3; font-size: 1.1rem; font-weight: 600; }

  /* Best badge */
  .badge-best {
    display: inline-block;
    background: #238636;
    color: #fff;
    padding: 0.15rem 0.5rem;
    border-radius: 20px;
    font-size: 0.68rem;
    font-weight: 700;
    vertical-align: middle;
    margin-left: 5px;
  }

  /* Info note */
  .info-note {
    background: #0d2137;
    border-left: 3px solid #1f6feb;
    padding: 0.65rem 1rem;
    border-radius: 0 6px 6px 0;
    color: #8cc4ff;
    font-size: 0.83rem;
    margin: 0.6rem 0 1rem;
  }

  /* Override Streamlit metric colours */
  [data-testid="stMetricLabel"] { color: #8b949e !important; font-size: 0.76rem !important; }
  [data-testid="stMetricValue"] { color: #e6edf3 !important; }

  /* Tab text */
  div[data-testid="stTabs"] button[role="tab"] { color: #8b949e; }
  div[data-testid="stTabs"] button[aria-selected="true"] {
    color: #58a6ff !important;
    border-bottom-color: #58a6ff !important;
  }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# Constants
# ══════════════════════════════════════════════════════════════════════════════
_DARK_LAYOUT = dict(
    template="plotly_dark",
    plot_bgcolor="#0d1117",
    paper_bgcolor="#0d1117",
    font=dict(color="#e6edf3", size=12),
    legend=dict(bgcolor="#161b22", bordercolor="#30363d", borderwidth=1),
    margin=dict(l=0, r=0, t=30, b=0),
)

_MODEL_COLOR = {
    "Linear Regression": "#388bfd",
    "Random Forest":     "#3fb950",
    "XGBoost":           "#f0883e",
    "LightGBM":          "#d2a8ff",
    "LSTM":              "#e3b341",
}
_C = {
    "green":  "#3fb950",
    "red":    "#f85149",
    "blue":   "#388bfd",
    "orange": "#f0883e",
    "purple": "#d2a8ff",
    "gray":   "#8b949e",
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
    """Compute all indicators unconditionally. Sidebar toggles only affect charts."""
    df = df.copy()
    if isinstance(df["Close"], pd.DataFrame):
        df["Close"] = df["Close"].squeeze()
    close = df["Close"]

    # Core momentum
    df["Return_1d"]     = close.pct_change()
    df["Return_5d"]     = close.pct_change(5)
    df["RollingStd_10"] = close.pct_change().rolling(10).std()

    # Moving averages
    df["SMA_5"]  = close.rolling(5).mean()
    df["SMA_10"] = close.rolling(10).mean()
    df["SMA_20"] = close.rolling(20).mean()
    df["EMA_10"] = close.ewm(span=10, adjust=False).mean()
    df["EMA_20"] = close.ewm(span=20, adjust=False).mean()

    # RSI
    df["RSI_14"] = compute_rsi(close, 14)

    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["MACD"]        = ema12 - ema26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"]   = df["MACD"] - df["MACD_Signal"]

    # Bollinger Bands
    std20 = close.rolling(20).std()
    df["BB_Upper"] = df["SMA_20"] + 2 * std20
    df["BB_Lower"] = df["SMA_20"] - 2 * std20
    df["BB_Width"] = (df["BB_Upper"] - df["BB_Lower"]) / (df["SMA_20"] + 1e-9)
    df["BB_Pct"]   = (close - df["BB_Lower"]) / (df["BB_Upper"] - df["BB_Lower"] + 1e-9)

    # ATR
    if "High" in df.columns and "Low" in df.columns:
        high, low = df["High"], df["Low"]
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low  - close.shift(1)).abs(),
        ], axis=1).max(axis=1)
        df["ATR_14"] = tr.rolling(14).mean()

    # OBV
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


@st.cache_data(show_spinner=False, ttl=60)          # 60 s — near-live
def fetch_live_price(symbol: str) -> tuple:
    """Returns (price, pct_change_today). 60-second TTL keeps it near-live."""
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


@st.cache_data(show_spinner=False, ttl=300)          # 5-min cache for news
def fetch_news_with_dates(symbol: str, query: str, max_items: int = 30) -> list:
    """Returns [{title, date}] for time-aligned sentiment.
    `query` is the NewsAPI search term — ticker for US, company name for Indian stocks.
    """
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

    if not articles:                                 # yfinance fallback
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
    """Per-article sentiment aligned to trading-day index via forward-fill."""
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

    # Raw Close excluded — using only derived features to avoid persistence bias
    possible = [
        "SMA_5", "SMA_10", "SMA_20", "EMA_10", "EMA_20",
        "Return_1d", "Return_5d", "RSI_14", "RollingStd_10",
        "MACD", "MACD_Signal", "MACD_Hist",
        "BB_Width", "BB_Pct", "ATR_14", "OBV",
        "Volume", "Sentiment",
    ]
    feature_cols = [f for f in possible if f in df.columns]
    if not feature_cols:
        raise KeyError("No feature columns found after processing.")

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
# Plotting — all functions return go.Figure
# ══════════════════════════════════════════════════════════════════════════════
def _col(key):
    return _C.get(key, "#e6edf3")


def plot_price_chart(df: pd.DataFrame, show: set) -> go.Figure:
    """Candlestick + optional indicator subplots."""
    rows, heights = 1, [0.5]
    if "Volume" in df.columns:
        rows += 1; heights.append(0.12)
    if "MACD" in show and "MACD" in df.columns:
        rows += 1; heights.append(0.19)
    if "RSI" in show and "RSI_14" in df.columns:
        rows += 1; heights.append(0.19)

    total = sum(heights)
    heights = [h / total for h in heights]

    fig = make_subplots(
        rows=rows, cols=1, shared_xaxes=True,
        row_heights=heights, vertical_spacing=0.02,
    )

    o = df["Open"]  if "Open"  in df.columns else df["Close"]
    h = df["High"]  if "High"  in df.columns else df["Close"]
    lo = df["Low"]  if "Low"   in df.columns else df["Close"]

    fig.add_trace(go.Candlestick(
        x=df.index, open=o, high=h, low=lo, close=df["Close"],
        name="Price",
        increasing_line_color=_col("green"),
        decreasing_line_color=_col("red"),
    ), row=1, col=1)

    if "SMA/EMA" in show:
        for col_name, color, dash in [
            ("SMA_20", _col("blue"),   "solid"),
            ("EMA_20", _col("orange"), "dash"),
        ]:
            if col_name in df.columns:
                fig.add_trace(go.Scatter(
                    x=df.index, y=df[col_name], name=col_name,
                    line=dict(color=color, width=1.5, dash=dash),
                ), row=1, col=1)

    if "Bollinger Bands" in show and "BB_Upper" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["BB_Upper"], name="BB Upper",
            line=dict(color=_col("gray"), width=1, dash="dot"),
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=df.index, y=df["BB_Lower"], name="BB Lower",
            line=dict(color=_col("gray"), width=1, dash="dot"),
            fill="tonexty", fillcolor="rgba(139,148,158,0.07)",
        ), row=1, col=1)

    cur = 2
    if "Volume" in df.columns:
        vol_colors = [
            _col("green") if float(c) >= float(op) else _col("red")
            for c, op in zip(df["Close"], o)
        ]
        fig.add_trace(go.Bar(
            x=df.index, y=df["Volume"], name="Volume",
            marker_color=vol_colors, opacity=0.5,
        ), row=cur, col=1)
        fig.update_yaxes(title_text="Vol", row=cur, col=1)
        cur += 1

    if "MACD" in show and "MACD" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["MACD"], name="MACD",
            line=dict(color=_col("blue"), width=1.5),
        ), row=cur, col=1)
        fig.add_trace(go.Scatter(
            x=df.index, y=df["MACD_Signal"], name="Signal",
            line=dict(color=_col("orange"), width=1.5),
        ), row=cur, col=1)
        hist_colors = [
            _col("green") if v >= 0 else _col("red")
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
            line=dict(color=_col("purple"), width=1.5),
        ), row=cur, col=1)
        for level, color in [(70, _col("red")), (30, _col("green"))]:
            fig.add_trace(go.Scatter(
                x=[df.index[0], df.index[-1]], y=[level, level],
                mode="lines", line=dict(color=color, dash="dot", width=1),
                showlegend=False,
            ), row=cur, col=1)
        fig.update_yaxes(title_text="RSI", range=[0, 100], row=cur, col=1)

    fig.update_layout(height=640, xaxis_rangeslider_visible=False, **_DARK_LAYOUT)
    return fig


def plot_multi_stock(dfs: dict) -> go.Figure:
    palette = [_col("blue"), _col("green"), _col("orange"), _col("purple"), _col("gray")]
    fig = go.Figure()
    for i, (sym, df) in enumerate(dfs.items()):
        if df.empty or "Close" not in df.columns:
            continue
        norm = df["Close"] / df["Close"].iloc[0] * 100
        fig.add_trace(go.Scatter(
            x=df.index, y=norm, name=sym,
            line=dict(color=palette[i % len(palette)], width=2),
        ))
    fig.update_layout(
        yaxis_title="Normalised Price (base = 100)",
        height=400, **_DARK_LAYOUT,
    )
    return fig


def plot_predictions(results: dict, dates_test, y_test) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates_test, y=y_test,
        name="Actual", line=dict(color="#e6edf3", width=2.5),
    ))
    for name, (preds, _) in results.items():
        n = min(len(dates_test), len(preds))
        fig.add_trace(go.Scatter(
            x=dates_test[-n:], y=preds[-n:],
            name=name,
            line=dict(color=_MODEL_COLOR.get(name, "#fff"), width=1.8, dash="dash"),
        ))
    fig.update_layout(
        xaxis_title="Date", yaxis_title="Closing Price",
        height=430, **_DARK_LAYOUT,
    )
    return fig


def plot_metrics_bar(results: dict, best_name: str) -> go.Figure:
    names   = list(results.keys())
    metrics = [results[n][1] for n in names]
    colors  = [_col("green") if n == best_name else "#21262d" for n in names]
    borders = [_col("green") if n == best_name else "#30363d" for n in names]

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=["RMSE  (lower ↓)", "MAPE %  (lower ↓)", "R²  (higher ↑)"],
    )
    for col_idx, key in enumerate(["RMSE", "MAPE (%)", "R²"], start=1):
        vals = [m[key] for m in metrics]
        fig.add_trace(go.Bar(
            x=names, y=vals,
            marker=dict(color=colors, line=dict(color=borders, width=1.5)),
            text=[f"{v:.3f}" for v in vals], textposition="outside",
            showlegend=False,
        ), row=1, col=col_idx)

    fig.update_layout(height=300, **_DARK_LAYOUT)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# Sidebar
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    st.markdown("---")

    st.markdown("**📌 Symbols**")
    primary_symbol = st.text_input(
        "Primary Symbol", "AAPL",
        help="US: AAPL, TSLA  |  India (NSE): TCS.NS, RELIANCE.NS  |  India (BSE): TCS.BO",
    ).upper().strip()
    compare_raw = st.text_input(
        "Compare With", "", placeholder="TSLA, MSFT, INFY.NS …",
        help="Optional comma-separated symbols for the comparison chart",
    ).upper()
    compare_symbols = [s.strip() for s in compare_raw.split(",") if s.strip()] if compare_raw else []

    st.markdown("---")
    st.markdown("**📅 Date Range**")
    dc1, dc2 = st.columns(2)
    with dc1:
        start_date = st.date_input("Start", value=datetime.today() - timedelta(days=730))
    with dc2:
        end_date = st.date_input("End", value=datetime.today())

    st.markdown("---")
    st.markdown("**📐 Chart Indicators**")
    ic1, ic2 = st.columns(2)
    show_sma  = ic1.checkbox("SMA / EMA",        value=True)
    show_rsi  = ic2.checkbox("RSI",               value=True)
    show_macd = ic1.checkbox("MACD",              value=True)
    show_bb   = ic2.checkbox("Bollinger Bands",   value=True)

    show_indicators = set()
    if show_sma:  show_indicators.add("SMA/EMA")
    if show_rsi:  show_indicators.add("RSI")
    if show_macd: show_indicators.add("MACD")
    if show_bb:   show_indicators.add("Bollinger Bands")

    st.markdown("---")
    st.markdown("**🤖 Models**")
    use_lr   = st.checkbox("Linear Regression",  value=True)
    use_rf   = st.checkbox("Random Forest",       value=True)
    _xgb_lbl = f"XGBoost {'✅' if XGB_AVAILABLE else '❌ pip install xgboost'}"
    _lgb_lbl = f"LightGBM {'✅' if LGB_AVAILABLE else '❌ pip install lightgbm'}"
    _lstm_lbl = f"LSTM (TF) {'✅' if TF_AVAILABLE else '❌ pip install tensorflow'}"
    use_xgb  = st.checkbox(_xgb_lbl,  value=XGB_AVAILABLE)
    use_lgb  = st.checkbox(_lgb_lbl,  value=LGB_AVAILABLE)
    use_lstm = st.checkbox(_lstm_lbl, value=False)

    if use_xgb  and not XGB_AVAILABLE:  use_xgb  = False
    if use_lgb  and not LGB_AVAILABLE:  use_lgb  = False
    if use_lstm and not TF_AVAILABLE:   use_lstm = False

    st.markdown("---")
    st.markdown("**⚙️ Training**")
    test_size  = st.slider("Test Split",           0.1, 0.4, 0.2, 0.05)
    rf_trees   = st.slider("RF Trees",              50, 500, 200,   50) if use_rf   else 200
    lookback   = st.slider("LSTM Lookback (days)",  10,  60,  20,    5) if use_lstm else 20
    epochs     = st.slider("LSTM Epochs",           10, 200,  60,   10) if use_lstm else 60
    batch_size = st.slider("LSTM Batch Size",       16, 256,  64,   16) if use_lstm else 64

    st.markdown("---")
    st.caption("📊 Educational demo only — not financial advice.")


# ══════════════════════════════════════════════════════════════════════════════
# Header
# ══════════════════════════════════════════════════════════════════════════════
# ── Currency & news-query derived from symbol ─────────────────────────────────
_IS_INDIAN = primary_symbol.endswith((".NS", ".BO"))
CURRENCY   = "₹" if _IS_INDIAN else "$"

@st.cache_data(show_spinner=False, ttl=3600)
def _company_name(symbol: str) -> str:
    """Look up long name from yfinance; fall back to the raw ticker."""
    try:
        name = yf.Ticker(symbol).info.get("longName", "")
        return name if name else symbol.split(".")[0]
    except Exception:
        return symbol.split(".")[0]

# For Indian stocks use the company name as the news query (better results)
NEWS_QUERY = _company_name(primary_symbol) if _IS_INDIAN else primary_symbol

st.markdown("""
<div class="app-header">
  <h1>📊 Stock Intelligence Platform</h1>
  <p>Technical indicators · News sentiment · Multi-model ML predictions · Not financial advice</p>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# Live Snapshot
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="sec-header">🕐 Live Snapshot</div>', unsafe_allow_html=True)

all_symbols = [primary_symbol] + compare_symbols
live_cols   = st.columns(len(all_symbols))
for col, sym in zip(live_cols, all_symbols):
    price, pct = fetch_live_price(sym)
    with col:
        if math.isnan(price):
            st.metric(sym, "N/A", help="Market closed or API rate limit")
        else:
            delta = f"{pct:+.2f}%" if not math.isnan(pct) else None
            st.metric(sym, f"{CURRENCY}{price:,.2f}", delta=delta)

_tz_label = "IST" if _IS_INDIAN else "local time"
st.caption(f"⏱ Refreshed at {datetime.now().strftime('%H:%M:%S')} {_tz_label} · auto-updates every 60 s")


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


# ══════════════════════════════════════════════════════════════════════════════
# Price Charts
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="sec-header">📈 Price Chart</div>', unsafe_allow_html=True)

if len(all_dfs) > 1:
    ct1, ct2 = st.tabs([f"📊 {primary_symbol} Detail", "⚖️ Multi-Stock Comparison"])
else:
    ct1 = st.container()
    ct2 = None

with ct1:
    st.plotly_chart(plot_price_chart(df, show_indicators), use_container_width=True)

if ct2:
    with ct2:
        st.plotly_chart(plot_multi_stock(all_dfs), use_container_width=True)
        closes = pd.DataFrame({s: d["Close"] for s, d in all_dfs.items()}).dropna()
        if len(closes.columns) > 1:
            st.markdown("**Correlation Matrix**")
            st.dataframe(closes.corr().round(3), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# Sentiment Note & Data Preview
# ══════════════════════════════════════════════════════════════════════════════
if "Sentiment" in df.columns:
    avg = df["Sentiment"].mean()
    icon  = "🟢" if avg > 0.05 else ("🔴" if avg < -0.05 else "🟡")
    label = "Positive" if avg > 0.05 else ("Negative" if avg < -0.05 else "Neutral")
    st.markdown(
        f'<div class="info-note">📰 News sentiment for <strong>{primary_symbol}</strong>: '
        f'{icon} {label} (avg polarity {avg:+.3f}) — '
        f'time-aligned per trading day · NewsAPI (30-day window) → yfinance fallback</div>',
        unsafe_allow_html=True,
    )

with st.expander("📋 Data preview — last 10 rows", expanded=False):
    st.write(f"Rows: **{len(df)}** | Range: **{df.index.min().date()} → {df.index.max().date()}**")
    st.dataframe(df.tail(10), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# Feature Engineering & Train / Test Split
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="sec-header">🤖 Model Training & Evaluation</div>', unsafe_allow_html=True)

X, y, feature_cols, dates_idx = prepare_features_labels(df)

if len(X) < 200:
    st.warning(f"Only **{len(X)}** rows after feature engineering. Expand the date range (min ~200 needed).")
    st.stop()

split_idx       = int(len(X) * (1 - test_size))
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]
dates_test      = dates_idx[split_idx:]


# ══════════════════════════════════════════════════════════════════════════════
# Train Selected Models
# ══════════════════════════════════════════════════════════════════════════════
results: dict = {}   # name → (preds_array, metrics_dict)
_store = {}          # keep trained objects for the forecast section

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
        X_train_sc  = lstm_sc.fit_transform(X_train)  # fit on train only — no leakage
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


# ══════════════════════════════════════════════════════════════════════════════
# Model Comparison Cards
# ══════════════════════════════════════════════════════════════════════════════
card_cols = st.columns(len(results))
for col, (name, (_, metrics)) in zip(card_cols, results.items()):
    is_best  = name == best_name
    color    = _MODEL_COLOR.get(name, "#e6edf3")
    badge    = '<span class="badge-best">★ Best</span>' if is_best else ""
    cls      = "model-card best" if is_best else "model-card"
    with col:
        st.markdown(
            f'<div class="{cls}">'
            f'<div class="model-name" style="color:{color}">{name}{badge}</div>'
            f'<hr class="card-divider">'
            f'<div class="card-label">RMSE</div>'
            f'<div class="card-value">{metrics["RMSE"]:.2f}</div>'
            f'<div class="card-label">MAPE</div>'
            f'<div class="card-value">{metrics["MAPE (%)"]:.2f}%</div>'
            f'<div class="card-label">R²</div>'
            f'<div class="card-value">{metrics["R²"]:.4f}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

st.plotly_chart(plot_metrics_bar(results, best_name), use_container_width=True)

st.markdown("**Actual vs Predicted — Test Period**")
st.plotly_chart(plot_predictions(results, dates_test, y_test), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# Next-Day Forecast  (retrain each model on ALL data → predict last row)
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="sec-header">📅 Next-Day Forecast</div>', unsafe_allow_html=True)

X_all, y_all, _, _ = prepare_features_labels(df.dropna().copy())
X_fc_tr, y_fc_tr   = X_all[:-1], y_all[:-1]
X_last              = X_all[-1:]
current_close       = float(df["Close"].dropna().iloc[-1])

fc_cols = st.columns(len(results) + 1)
with fc_cols[0]:
    st.metric("Current Close", f"{CURRENCY}{current_close:,.2f}")

for col, name in zip(fc_cols[1:], results.keys()):
    with col:
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
            else:
                pred = float("nan")

            delta_pct = (pred - current_close) / current_close * 100
            st.metric(
                f"{'★ ' if name == best_name else ''}{name}",
                f"{CURRENCY}{pred:,.2f}",
                delta=f"{delta_pct:+.2f}%",
            )
        except Exception:
            st.metric(name, "N/A")

st.caption("⚠️ Educational demo only — predictions are not financial advice.")
