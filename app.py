import streamlit as st
import yfinance as yf
import pandas as pd
import requests
from datetime import datetime

API_KEY = "f8665eb595e943a7bbbe1e05ecf32730"
NEWS_URL = "https://newsapi.org/v2/everything"

# --- Helper: pip calculation ---
def pip_difference(pair, entry, exit):
    if "JPY" in pair:
        return (exit - entry) * 100   # 1 pip = 0.01
    elif "XAU" in pair or "BTC" in pair:
        return exit - entry           # treat 1 unit = 1 pip
    else:
        return (exit - entry) * 10000 # 1 pip = 0.0001

# --- Market status helper ---
def market_open(pair):
    now = datetime.utcnow()
    weekday = now.weekday()  # 0=Monday, 6=Sunday
    hour = now.hour
    if pair == "BTCUSD":
        return True  # Bitcoin trades 24/7
    else:
        # Forex closes Friday 22:00 UTC, opens Sunday 22:00 UTC
        return not ((weekday == 5 and hour >= 22) or (weekday == 6 and hour < 22))

# --- News fetch + sentiment ---
def fetch_news(query="forex OR EURUSD OR USDJPY OR GBPUSD OR XAUUSD OR BTCUSD OR bitcoin OR gold"):
    params = {
        "q": query,
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": 15,
        "apiKey": API_KEY
    }
    response = requests.get(NEWS_URL, params=params)
    if response.status_code == 200:
        return response.json().get("articles", [])
    else:
        return []

def analyze_sentiment(headline):
    text = headline.lower()
    sentiment = "Neutral"
    if any(word in text for word in ["buy", "bullish", "positive", "upside", "long"]):
        sentiment = "BUY"
    elif any(word in text for word in ["sell", "bearish", "negative", "downside", "short"]):
        sentiment = "SELL"
    if "eurusd" in text or "euro" in text:
        pair = "EURUSD"
    elif "usdjpy" in text or "yen" in text:
        pair = "USDJPY"
    elif "gbpusd" in text or "pound" in text or "sterling" in text:
        pair = "GBPUSD"
    elif "gold" in text or "xau" in text:
        pair = "XAUUSD"
    elif "bitcoin" in text or "btc" in text:
        pair = "BTCUSD"
    else:
        pair = "General"
    return sentiment, pair

# --- Streamlit UI ---
st.title("📊 Jay Market Insights")

# Market Status per Pair
st.subheader("Market Status by Pair")
for pair in ["EURUSD", "USDJPY", "GBPUSD", "XAUUSD", "BTCUSD"]:
    status = "Open" if market_open(pair) else "Closed"
    st.write(f"{pair}: Market {status}")

# Live News Sentiment
st.subheader("📰 Live Forex & Crypto News Sentiment")
articles = fetch_news()
data = []
if articles:
    for art in articles:
        sentiment, pair = analyze_sentiment(art["title"])
        data.append({
            "headline": art["title"],
            "source": art["source"]["name"],
            "published": art["publishedAt"],
            "pair": pair,
            "sentiment": sentiment
        })
    df_news = pd.DataFrame(data)
    st.dataframe(df_news, use_container_width=True)

    # Sentiment Distribution Chart
    pair_sentiment = df_news.groupby("pair")["sentiment"].value_counts().unstack().fillna(0)
    st.subheader("Sentiment Distribution by Pair")
    st.bar_chart(pair_sentiment)

    # Suggested Signals from News
    st.subheader("📌 Suggested Signals from News")
    suggested_signals = []
    ticker_map = {
        "EURUSD": "EURUSD=X",
        "USDJPY": "USDJPY=X",
        "GBPUSD": "GBPUSD=X",
        "XAUUSD": "GC=F",
        "BTCUSD": "BTC-USD"
    }
    for art in articles:
        sentiment, pair = analyze_sentiment(art["title"])
        if pair in ticker_map and sentiment in ["BUY", "SELL"]:
            df_price = yf.download(ticker_map[pair], period="1d", interval="15m")
            if not df_price.empty:
                current_price = df_price["Close"].iloc[-1]
                if pair == "BTCUSD":
                    sl = current_price * (0.98 if sentiment == "BUY" else 1.02)
                    tp = current_price * (1.02 if sentiment == "BUY" else 0.98)
                else:
                    sl = current_price * (0.99 if sentiment == "BUY" else 1.01)
                    tp = current_price * (1.01 if sentiment == "BUY" else 0.99)
                suggested_signals.append({
                    "pair": pair,
                    "signal": sentiment,
                    "entry": round(current_price, 5),
                    "stop_loss": round(sl, 5),
                    "take_profit": round(tp, 5),
                    "headline": art["title"]
                })
    if suggested_signals:
        st.dataframe(pd.DataFrame(suggested_signals), use_container_width=True)
    else:
        st.write("No actionable signals from news right now.")
else:
    st.write("No news available right now.")
    # --- Strategy Backtest Results ---
st.subheader("Strategy Backtest Results")

timeframe = st.selectbox("Select timeframe", ["1d", "15m", "5m"], index=0)

with st.form("signal_form"):
    date = st.date_input("Signal Date")
    signal = st.selectbox("Signal", ["BUY", "SELL"])
    pair = st.selectbox("Pair", ["EURUSD", "USDJPY", "GBPUSD", "XAUUSD", "BTCUSD"])
    stop_loss = st.number_input("Stop Loss", value=0.0)
    take_profit = st.number_input("Take Profit", value=0.0)
    submitted = st.form_submit_button("Add Signal")

if "signals" not in st.session_state:
    st.session_state.signals = pd.DataFrame(columns=["date","signal","pair","stop_loss","take_profit"])

if submitted:
    new_signal = pd.DataFrame([{
        "date": pd.to_datetime(date),
        "signal": signal,
        "pair": pair,
        "stop_loss": stop_loss,
        "take_profit": take_profit
    }])
    st.session_state.signals = pd.concat([st.session_state.signals, new_signal], ignore_index=True)

st.write("Signals Table", st.session_state.signals)

if st.button("Run Backtest") and not st.session_state.signals.empty:
    pairs = {
        "EURUSD": "EURUSD=X",
        "USDJPY": "USDJPY=X",
        "GBPUSD": "GBPUSD=X",
        "XAUUSD": "GC=F",
        "BTCUSD": "BTC-USD"
    }
    data = {}
    for pair, ticker in pairs.items():
        df = yf.download(ticker, start="2026-01-01", end="2026-12-31", interval=timeframe)
        df.index = pd.to_datetime(df.index)
        data[pair] = df

    lookahead = 5
    results, days_to_result, pip_results = [], [], []

    for _, row in st.session_state.signals.iterrows():
        pair_data = data.get(row['pair'])
        if row['date'] in pair_data.index:
            start_idx = pair_data.index.get_loc(row['date'])
            end_idx = min(start_idx + lookahead, len(pair_data)-1)
            window = pair_data.iloc[start_idx:end_idx+1]

            outcome, days_taken, pip_value = "HOLD", None, 0
            entry_price = pair_data.loc[row['date'], 'Open']

            for i, (idx, day) in enumerate(window.iterrows()):
                high, low = day['High'], day['Low']
                if row['signal'] == "BUY":
                    if high >= row['take_profit']:
                        outcome, days_taken = "WIN", i
                        pip_value = pip_difference(row['pair'], entry_price, row['take_profit'])
                        break
                    elif low <= row['stop_loss']:
                        outcome, days_taken = "LOSS", i
                        pip_value = pip_difference(row['pair'], entry_price, row['stop_loss'])
                        break
                elif row['signal'] == "SELL":
                    if low <= row['take_profit']:
                        outcome, days_taken = "WIN", i
                        pip_value = pip_difference(row['pair'], entry_price, row['take_profit'])
                        break
                elif high >= row['stop_loss']:
                    outcome, days_taken = "LOSS", i
                    pip_value = pip_difference(row['pair'], entry_price, row['stop_loss'])
                    break

            results.append(outcome)
            days_to_result.append(days_taken)
            pip_results.append(pip_value)
        else:
            results.append("NO DATA")
            days_to_result.append(None)
            pip_results.append(0)

    # Collect results
    st.session_state.signals["outcome"] = results
    st.session_state.signals["days_to_result"] = days_to_result
    st.session_state.signals["pip_result"] = pip_results

    st.write("Backtest Results", st.session_state.signals)

    # Performance chart
    st.subheader("Performance Chart")
    st.line_chart(st.session_state.signals["pip_result"].cumsum())
