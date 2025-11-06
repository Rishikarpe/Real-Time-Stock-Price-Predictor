# 📈 Real-Time Stock Price Predictor

A **Streamlit-based Machine Learning dashboard** that predicts the **next-day closing price** of a stock (e.g., AAPL, TSLA, TCS.NS) by combining **technical indicators** and **news sentiment analysis**.  
The app supports **real-time live price updates**, **model comparison (Linear Regression vs LSTM)**, and **interactive visualizations**.

---

## 🚀 Features

✅ Fetches **real-time and historical stock data** using [Yahoo Finance (`yfinance`)](https://pypi.org/project/yfinance/)  
✅ Computes **technical indicators** — SMA, EMA, RSI, returns, volatility  
✅ Performs **news sentiment analysis** using [TextBlob](https://textblob.readthedocs.io/) or [NewsAPI](https://newsapi.org/)  
✅ Trains & compares **Linear Regression** and **LSTM** models  
✅ Displays **actual vs predicted charts**, **model comparison bar plots**, and **next-day forecast**  
✅ Built with **Streamlit** for an interactive dashboard  
✅ Auto-refreshes every 30 seconds for live price updates  

---

## 🧠 Workflow Overview

1. **Data Fetching**
   - Fetches OHLCV (Open, High, Low, Close, Volume) stock data from Yahoo Finance using `yfinance`.
   - Also retrieves live prices every minute for display.

2. **Feature Engineering**
   - Calculates several technical indicators:
     - Simple Moving Average (SMA)
     - Exponential Moving Average (EMA)
     - Relative Strength Index (RSI)
     - 1-day and 5-day returns
     - Rolling standard deviation (volatility)

3. **Sentiment Analysis**
   - Fetches top news headlines about the stock from NewsAPI or Yahoo Finance.
   - Analyzes their sentiment using `TextBlob` (polarity score between -1 and +1).

4. **Model Training**
   - Linear Regression (baseline, interpretable)
   - LSTM (optional deep learning model for sequential pattern detection)
   - Time-based data split: 80% training, 20% testing.

5. **Evaluation**
   - Mean Squared Error (MSE) used for accuracy comparison.
   - Interactive charts plotted using Matplotlib.

6. **Real-Time Dashboard**
   - Displays live stock metrics and predicted next-day close.
   - Auto-refreshes every 30 seconds.

---

## ⚙️ Installation & Setup

### 1️⃣ Clone this repository
```bash
git clone https://github.com/<your-username>/stock-predictor.git
cd stock-predictor
```

2️⃣ Create a virtual environment (recommended)
python -m venv .venv
```bash
.venv\Scripts\activate    # Windows
source .venv/bin/activate # Linux/Mac
```

3️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

4️⃣ Create .env file

Create a file named .env in the project folder and add:

NEWSAPI_KEY=your_newsapi_key_here

(You can get a free key from https://newsapi.org)

5️⃣ Run the app
```bash
streamlit run app.py
```

Then open your browser at:
👉 http://localhost:8501

---

## 🧮 Models Used
Model	Type	Description
Linear Regression	Statistical ML	Predicts next-day close using weighted sum of features
LSTM (Long Short-Term Memory)	Deep Learning	Learns temporal dependencies in price sequences
Evaluation Metric:

Mean Squared Error (MSE)

Measures average squared difference between predicted and true values

Lower MSE = better model
