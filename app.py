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
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt

# =========================
# Optional TensorFlow Import
# =========================
TF_AVAILABLE = True
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense
    from tensorflow.keras.callbacks import EarlyStopping
except Exception:
    TF_AVAILABLE = False

# =========================
# App Config
# =========================
st.set_page_config(page_title="Real-Time Stock Price Predictor", page_icon="📈", layout="wide")
load_dotenv()
NEWSAPI_KEY = "6fe740c75d764b009015e175452500e4"

# =========================
# Utility Functions
# =========================

def compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """Compute RSI safely handling 2D inputs from Yahoo Finance."""
    if isinstance(series, pd.DataFrame):
        series = series.squeeze()

    delta = series.diff().fillna(0)
    up = np.where(delta > 0, delta, 0.0)
    down = np.where(delta < 0, -delta, 0.0)
    roll_up = pd.Series(up, index=series.index).rolling(window=window).mean()
    roll_down = pd.Series(down, index=series.index).rolling(window=window).mean()
    rs = roll_up / (roll_down + 1e-9)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if isinstance(df["Close"], pd.DataFrame):
        df["Close"] = df["Close"].squeeze()

    df["SMA_5"] = df["Close"].rolling(window=5).mean()
    df["SMA_10"] = df["Close"].rolling(window=10).mean()
    df["EMA_10"] = df["Close"].ewm(span=10, adjust=False).mean()
    df["Return_1d"] = df["Close"].pct_change()
    df["Return_5d"] = df["Close"].pct_change(5)
    df["RSI_14"] = compute_rsi(df["Close"], window=14)
    df["RollingStd_10"] = df["Close"].pct_change().rolling(10).std()
    return df


@st.cache_data(show_spinner=False)
def fetch_history(symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
    df = yf.download(symbol, start=start, end=end, progress=False, auto_adjust=True)
    if df is None or df.empty:
        return pd.DataFrame()
    df.index = pd.to_datetime(df.index)
    
    # Fix MultiIndex columns if present (happens with single ticker sometimes)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    return df


@st.cache_data(show_spinner=False)
def fetch_live_price(symbol: str) -> float:
    try:
        tkr = yf.Ticker(symbol)
        info = tkr.history(period="1d", interval="1m")
        if info is not None and not info.empty and "Close" in info.columns:
            return float(info["Close"].iloc[-1])
    except Exception:
        pass
    return float("nan")


def sentiment_from_texts(texts: list[str]) -> float:
    if not texts:
        return 0.0
    scores = []
    for t in texts:
        try:
            polarity = TextBlob(t).sentiment.polarity
            scores.append(polarity)
        except Exception:
            continue
    return float(np.mean(scores)) if scores else 0.0


@st.cache_data(show_spinner=False)
def fetch_news_headlines_newsapi(symbol: str, max_items: int = 20) -> list[str]:
    """Fetch headlines using NewsAPI."""
    if not NEWSAPI_KEY:
        return []
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": symbol,
        "sortBy": "publishedAt",
        "language": "en",
        "pageSize": max_items,
        "apiKey": NEWSAPI_KEY
    }
    try:
        r = requests.get(url, params=params, timeout=10)
        data = r.json()
        articles = data.get("articles", [])
        return [a.get("title") for a in articles if a.get("title")]
    except Exception:
        return []


@st.cache_data(show_spinner=False)
def fetch_news_headlines_yf(symbol: str, max_items: int = 20) -> list[str]:
    """Fallback to yfinance's ticker news."""
    try:
        tkr = yf.Ticker(symbol)
        news = tkr.news or []
        return [n.get("title") for n in news[:max_items] if n.get("title")]
    except Exception:
        return []


def build_sentiment_feature(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    headlines = fetch_news_headlines_newsapi(symbol)
    if not headlines:
        headlines = fetch_news_headlines_yf(symbol)
    sent_score = sentiment_from_texts(headlines)
    s = pd.Series(sent_score, index=df.index)
    s = s.rolling(3, min_periods=1).mean().ffill()
    df = df.copy()
    df["Sentiment"] = s
    return df


def prepare_features_labels(df: pd.DataFrame):
    """Prepare features and labels safely — handles missing columns like Volume.

    This function is defensive: it only uses feature columns that exist in the
    DataFrame and only calls dropna with columns that are actually present to
    avoid KeyError from pandas when some features are missing.
    """
    df = df.copy()

    # Basic safety: ensure key columns exist
    if "Close" not in df.columns:
        raise KeyError("The 'Close' column is missing from Yahoo Finance data.")

    # Fill missing optional columns with defaults
    if "Volume" not in df.columns:
        df["Volume"] = 0.0
        st.warning("⚠️ Volume data unavailable — using default zeros.")
    if "Sentiment" not in df.columns:
        df["Sentiment"] = 0.0

    # Target
    df["y_next_close"] = df["Close"].shift(-1)

    # Possible features - only include those that actually exist
    possible_features = [
        "Close", "SMA_5", "SMA_10", "EMA_10",
        "Return_1d", "Return_5d", "RSI_14", "RollingStd_10",
        "Volume", "Sentiment"
    ]
    
    # Build feature_cols from only columns that exist
    feature_cols = [f for f in possible_features if f in df.columns]

    if not feature_cols:
        raise KeyError("No feature columns available after processing — check your data and indicators.")

    # Build subset for dropna - only columns that exist
    subset_cols = feature_cols.copy()
    if "y_next_close" in df.columns:
        subset_cols.append("y_next_close")
    
    # Verify all subset_cols actually exist before calling dropna
    subset_cols = [c for c in subset_cols if c in df.columns]

    # Only dropna on columns that actually exist to avoid KeyError
    df = df.dropna(subset=subset_cols).copy()

    X = df[feature_cols].values
    y = df["y_next_close"].values
    return X, y, feature_cols, df.index


def make_lr_model(X_train, y_train):
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    return lr


def sequenceify(X: np.ndarray, y: np.ndarray, lookback: int = 20):
    Xs, ys = [], []
    for i in range(lookback, len(X)):
        Xs.append(X[i - lookback:i, :])
        ys.append(y[i])
    return np.array(Xs), np.array(ys)


def make_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(64, input_shape=input_shape, return_sequences=False))
    model.add(Dense(32, activation="relu"))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mse")
    return model


def plot_actual_vs_pred(dates, actual, pred, title: str):
    fig, ax = plt.subplots()
    ax.plot(dates, actual, label="Actual")
    ax.plot(dates, pred, label="Predicted")
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    st.pyplot(fig)


def plot_mse_bar(results_dict):
    labels = list(results_dict.keys())
    values = [results_dict[k] for k in labels]
    fig, ax = plt.subplots()
    ax.bar(labels, values)
    ax.set_title("Model MSE (lower is better)")
    ax.set_xlabel("Model")
    ax.set_ylabel("MSE")
    st.pyplot(fig)

# =========================
# Sidebar Controls
# =========================
st.sidebar.title("⚙️ Controls")

default_symbol = "AAPL"
symbol = st.sidebar.text_input("Stock Symbol", value=default_symbol, help="e.g., AAPL, TSLA, MSFT, TCS.NS")

col1, col2 = st.sidebar.columns(2)
with col1:
    start_date = st.date_input("Start Date", value=(datetime.today() - timedelta(days=365)))
with col2:
    end_date = st.date_input("End Date", value=datetime.today())

use_lstm = st.sidebar.checkbox("Enable LSTM (TensorFlow)", value=False)
if use_lstm and not TF_AVAILABLE:
    st.sidebar.warning("TensorFlow not available. LSTM disabled.")
    use_lstm = False

test_size = st.sidebar.slider("Test Size (Time Split)", 0.1, 0.4, 0.2, 0.05)
lookback = st.sidebar.slider("LSTM Lookback (days)", 10, 60, 20, 5)
epochs = st.sidebar.slider("LSTM Epochs", 10, 200, 60, 10)
batch_size = st.sidebar.slider("LSTM Batch Size", 16, 256, 64, 16)

st.sidebar.markdown("---")
refresh_on = st.sidebar.checkbox("Auto-refresh live panel (30s)", value=True)

from streamlit_autorefresh import st_autorefresh
if refresh_on:
    st_autorefresh(interval=30_000, key="live-refresh")

st.sidebar.markdown("---")
st.sidebar.caption("💡 Tip: Add your NEWSAPI_KEY to .env for better sentiment results.")

# =========================
# Main Layout
# =========================
st.title("📈 Real-Time Stock Price Predictor")
st.caption("Combines technical indicators + news sentiment. Predicts **next-day closing price**. Educational demo only — not financial advice.")

# --- Live Panel ---
with st.container():
    st.subheader("🕐 Live Ticker")
    live_col1, live_col2, live_col3 = st.columns(3)
    with live_col1:
        st.metric("Symbol", symbol)
    with live_col2:
        live_price = fetch_live_price(symbol)
        if math.isnan(live_price):
            st.write("Latest price unavailable (market closed or API limit).")
        else:
            st.metric("Last Price (approx.)", f"{live_price:,.2f}")
    with live_col3:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.write(f"Updated: {now}")

# --- Data Fetch ---
st.subheader("📥 Data & Features")
with st.spinner("Fetching historical data..."):
    df = fetch_history(symbol, pd.to_datetime(start_date), pd.to_datetime(end_date) + pd.Timedelta(days=1))

if df.empty:
    st.error("No data returned. Check the symbol or date range.")
    st.stop()

st.write(f"Rows: {len(df)} | Range: {df.index.min().date()} → {df.index.max().date()}")

df = add_technical_indicators(df)
with st.spinner("Fetching news & computing sentiment..."):
    df = build_sentiment_feature(df, symbol)
st.dataframe(df.tail(10))

# --- Modeling Section ---
st.subheader("🤖 Train Models (Next-Day Close)")
X, y, feature_cols, dates = prepare_features_labels(df)
if len(X) < 200:
    st.warning("Not enough rows after feature engineering. Expand date range.")
    st.stop()

split_idx = int(len(X) * (1 - test_size))
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]
dates_test = dates[split_idx:]

tab1, tab2, tab3 = st.tabs(["Linear Regression", "LSTM (optional)", "Model Comparison"])
results = {}

# --- Linear Regression ---
with tab1:
    st.markdown("### 🔧 Linear Regression")
    scaler_lr = StandardScaler()
    X_train_lr = scaler_lr.fit_transform(X_train)
    X_test_lr = scaler_lr.transform(X_test)
    lr = make_lr_model(X_train_lr, y_train)
    preds_lr = lr.predict(X_test_lr)
    mse_lr = float(mean_squared_error(y_test, preds_lr))
    results["Linear Regression"] = mse_lr
    plot_actual_vs_pred(dates_test, y_test, preds_lr, f"LR: Actual vs Predicted (MSE={mse_lr:,.2f})")

# --- LSTM ---
with tab2:
    if not use_lstm:
        st.info("Enable LSTM in sidebar to train.")
    else:
        scaler_lstm = MinMaxScaler()
        X_scaled_all = scaler_lstm.fit_transform(X)
        X_train_scaled = X_scaled_all[:split_idx]
        X_test_scaled = X_scaled_all[split_idx:]
        X_train_seq, y_train_seq = sequenceify(X_train_scaled, y_train, lookback=lookback)
        X_test_seq, y_test_seq = sequenceify(
            np.vstack([X_train_scaled[-lookback:], X_test_scaled]),
            np.hstack([y_train[-lookback:], y_test]),
            lookback=lookback
        )
        if len(X_train_seq) >= 10:
            model = make_lstm_model((X_train_seq.shape[1], X_train_seq.shape[2]))
            es = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
            with st.spinner("Training LSTM..."):
                history = model.fit(
                    X_train_seq, y_train_seq,
                    validation_split=0.2,
                    epochs=epochs, batch_size=batch_size, verbose=0, callbacks=[es]
                )
            preds_lstm = model.predict(X_test_seq, verbose=0).flatten()
            mse_lstm = float(mean_squared_error(y_test_seq, preds_lstm))
            results["LSTM"] = mse_lstm
            plot_actual_vs_pred(dates_test[lookback:], y_test_seq, preds_lstm, f"LSTM (MSE={mse_lstm:,.2f})")

# --- Comparison ---
with tab3:
    if results:
        plot_mse_bar(results)
    else:
        st.info("Train at least one model to compare results.")

# --- Forecast ---
st.subheader("📅 Next-Day Forecast")
latest_df = df.dropna().copy()
X_all, y_all, feat_cols, idx_all = prepare_features_labels(latest_df)
X_train_all, y_train_all = X_all[:-1], y_all[:-1]
X_last = X_all[-1:].copy()

scaler_lr_all = StandardScaler()
X_train_all_sc = scaler_lr_all.fit_transform(X_train_all)
X_last_sc = scaler_lr_all.transform(X_last)
lr_all = make_lr_model(X_train_all_sc, y_train_all)
lr_next = lr_all.predict(X_last_sc)[0]
st.metric("Linear Regression — Predicted Next Close", f"{lr_next:,.2f}")
st.caption("Disclaimer: Educational demo only, not financial advice.")
