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
    elif "XAU" in pair:
        return exit - entry           # treat 1 unit = 1 pip
    else:
        return (exit - entry) * 10000 # 1 pip = 0.0001

# --- Market status helper ---
def market_open_now():
    now = datetime.utcnow()
    weekday = now.weekday()  # 0=Monday, 6=Sunday
    hour = now.hour
    return not ((weekday == 5 and hour >= 22) or (weekday == 6 and hour < 22))

# --- News fetch + sentiment ---
def fetch_news(query="forex OR EURUSD OR USDJPY OR GBPUSD OR XAUUSD"):
    params = {
        "q": query,
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": 10,
        "apiKey": API_KEY
    }
    response = requests.get(NEWS_URL, params=params)
    if response.status_code == 200:
        return response.json().get("articles", [])
    else:
        return []

def analyze_sentiment(headline):
    headline = headline.lower()
    if any(word in headline for word in ["buy", "bullish", "positive", "upside"]):
        return "BUY"
    elif any(word in headline for word in ["sell", "bearish", "negative", "downside"]):
        return "SELL"
    else:
        return "Neutral"

# --- Streamlit UI ---
st.title("📊 Jay Market Insights")
st.subheader("Market Status")
st.write("Market Open" if market_open_now() else "Market Closed")

# Live News Sentiment
st.subheader("📰 Live Forex News Sentiment")
articles = fetch_news()
data = []
if articles:
    for art in articles:
        sentiment = analyze_sentiment(art["title"])
        data.append({
            "headline": art["title"],
            "source": art["source"]["name"],
            "published": art["publishedAt"],
            "sentiment": sentiment
        })
    df_news = pd.DataFrame(data)
    st.dataframe(df_news, use_container_width=True)

    # Sentiment Distribution Chart (bar chart instead of pie)
    sentiment_counts = df_news['sentiment'].value_counts()
    st.bar_chart(sentiment_counts)
else:
    st.write("No news available right now.")

# --- Strategy Backtest Results ---
st.subheader("Strategy Backtest Results")

timeframe = st.selectbox("Select timeframe", ["1d", "15m", "5m"], index=0)

with st.form("signal_form"):
    date = st.date_input("Signal Date")
    signal = st.selectbox("Signal", ["BUY", "SELL"])
    pair = st.selectbox("Pair", ["EURUSD", "USDJPY", "GBPUSD", "XAUUSD"])
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
        "XAUUSD": "GC=F"
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

    st.session_state.signals['result'] = results
    st.session_state.signals['days_to_result'] = days_to_result
    st.session_state.signals['pips'] = pip_results

    st.write("Backtest Results", st.session_state.signals)

    st.subheader("Backtest Statistics")
    st.write("Win rate:", (st.session_state.signals['result'] == "WIN").mean())
    st.write("Loss rate:", (st.session_state.signals['result'] == "LOSS").mean())
    st.write("Hold rate:", (st.session_state.signals['result'] == "HOLD").mean())
    st.write("Average pips per WIN:", st.session_state.signals.loc[st.session_state.signals['result']=="WIN", 'pips'].mean())
    st.write("Average pips per LOSS:", st.session_state.signals.loc[st.session_state.signals['result']=="LOSS", 'pips'].mean())
    st.write("Net average pips per trade:", st.session_state.signals['pips'].mean())

    # Guidance Layer: Signal vs News Sentiment
    if articles and not st.session_state.signals.empty:
        latest_signal = st.session_state.signals.iloc[-1]
        dominant_sentiment = sentiment_counts.idxmax()
        st.subheader("Signal vs News Sentiment")
        st.write(f"Your latest signal: {latest_signal['signal']} on {latest_signal['pair']}")
        st.write(f"Dominant news sentiment: {dominant_sentiment}")
        if latest_signal['signal'] == dominant_sentiment:
            st.success("✅ Your signal aligns with current news sentiment.")
        else:
            st.warning("⚠️ Your signal is opposite to dominant news sentiment. Trade cautiously.")

    # Performance Chart: cumulative pips
    st.subheader("Performance Chart")
    st.line_chart(st.session_state.signals['pips'].cumsum())
