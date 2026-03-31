"""Quick test to verify core functionality without running Streamlit."""
import sys
from datetime import datetime, timedelta

print("Testing imports...")
try:
    import pandas as pd
    import numpy as np
    import yfinance as yf
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler
    import matplotlib.pyplot as plt
    print("✅ All core imports successful")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

print("\nTesting yfinance data fetch...")
try:
    symbol = "AAPL"
    start = datetime.today() - timedelta(days=60)
    end = datetime.today()
    df = yf.download(symbol, start=start, end=end, progress=False, auto_adjust=True)
    
    # Fix MultiIndex columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    print(f"✅ Fetched {len(df)} rows for {symbol}")
    print(f"   Columns: {list(df.columns)}")
    print(f"   Date range: {df.index.min().date()} to {df.index.max().date()}")
except Exception as e:
    print(f"❌ Data fetch error: {e}")
    sys.exit(1)

print("\nTesting technical indicators...")
try:
    # Add simple moving average
    df["SMA_5"] = df["Close"].rolling(window=5).mean()
    df["Return"] = df["Close"].pct_change()
    print(f"✅ Technical indicators added")
    print(f"   Latest close: ₹{df['Close'].iloc[-1]:.2f}")
    print(f"   Latest SMA_5: ₹{df['SMA_5'].iloc[-1]:.2f}")
except Exception as e:
    print(f"❌ Technical indicator error: {e}")
    sys.exit(1)

print("\nTesting Linear Regression...")
try:
    # Prepare simple features
    df = df.dropna()
    X = df[["SMA_5", "Return"]].values[:-1]
    y = df["Close"].shift(-1).dropna().values
    
    # Train-test split
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # Scale and train
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)
    
    from sklearn.metrics import mean_squared_error
    mse = mean_squared_error(y_test, preds)
    
    print(f"✅ Linear Regression trained")
    print(f"   Train samples: {len(X_train)}")
    print(f"   Test samples: {len(X_test)}")
    print(f"   MSE: {mse:.2f}")
except Exception as e:
    print(f"❌ Linear Regression error: {e}")
    sys.exit(1)

print("\n" + "="*50)
print("✅ ALL TESTS PASSED!")
print("="*50)
print("\nYou can now run the Streamlit app:")
print("  streamlit run app.py")
