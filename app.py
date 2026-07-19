import pandas as pd
from newsapi import NewsApiClient
from transformers import pipeline
import streamlit as st
import datetime

# Set page config
st.set_page_config(page_title="JAY Market Insights", layout="wide")

# Title and description
st.title("📊 JAY Market Insights")
st.markdown("Real-time sentiment analysis of market news for trading decisions")

# Initialize sentiment model
@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

sentiment_model = load_model()

# Function to check if market is open
def market_open_now():
    now = datetime.datetime.utcnow()
    weekday = now.weekday()  # 0=Monday, 6=Sunday
    hour = now.hour
    # Forex closes Friday 22:00 UTC, opens Sunday 22:00 UTC
    return not (weekday == 5 and hour >= 22 or weekday == 6 and hour < 22)

# Function to map sentiment to trading pairs
def map_to_pairs(headline, result):
    label = result[0]['label'].upper()
    pairs_map = {
        "POSITIVE": ["EURUSD", "GBPUSD", "XAUUSD"],
        "NEGATIVE": ["USDJPY", "USDCHF", "DXY"]
    }
    return pairs_map.get(label, [])

# Example logic to add Buy/Sell column
def get_trade_signal(sentiment, pairs):
    if sentiment == "POSITIVE":
        return "🟢 BUY " + ", ".join(pairs)
    elif sentiment == "NEGATIVE":
        return "🔴 SELL " + ", ".join(pairs)
    else:
        return "⏸️ HOLD"

# Get trade levels (stop loss and take profit) with better rounding
def get_trade_levels(signal, current_price):
    if signal.startswith("🟢"):  # BUY signal
        stop_loss = round(current_price * 0.99, 5)   # 1% below
        take_profit = round(current_price * 1.02, 5) # 2% above
    elif signal.startswith("🔴"):  # SELL signal
        stop_loss = round(current_price * 1.01, 5)   # 1% above
        take_profit = round(current_price * 0.98, 5) # 2% below
    else:
        stop_loss, take_profit = None, None
    return stop_loss, take_profit

# Example prices for trading pairs
example_prices = {
    "EURUSD": 1.09500,
    "GBPUSD": 1.27500,
    "XAUUSD": 2050.00000,
    "USDJPY": 145.50000,
    "USDCHF": 0.88500,
    "DXY": 103.50000
}

def get_current_price(pair):
    """Get current price for a trading pair"""
    return example_prices.get(pair, 0)

# Display market status
market_status = market_open_now()
if market_status:
    st.success("🟢 **FOREX MARKET IS OPEN**")
else:
    st.warning("🔴 **FOREX MARKET IS CLOSED**")

# Connect with your API key
try:
    newsapi = NewsApiClient(api_key="f8665eb595e943a7bbbe1e05ecf32730")

    # Pull latest financial headlines
    articles = newsapi.get_everything(
        q="Federal Reserve OR inflation OR gold OR ECB OR Bank of England OR geopolitics OR oil OR unemployment OR China OR recession OR jobs OR USD OR forex OR currency",
        language="en",
        sort_by="publishedAt",
        page_size=10
    )

    # Collect results
    results = []
    for a in articles['articles'][:10]:  # show top 10 headlines
        headline = a['title']
        result = sentiment_model(headline)
        suggestion = map_to_pairs(headline, result)
        results.append({
            "Headline": headline,
            "Sentiment": result[0]['label'],
            "Confidence": round(result[0]['score'], 2),
            "Suggested Pairs": ", ".join(suggestion) if suggestion else "N/A",
            "Source": a['source']['name']
        })

    # Create dataframe
    df = pd.DataFrame(results)
    
    # Add trading signal column
    df["Signal"] = df.apply(lambda row: get_trade_signal(row["Sentiment"], row["Suggested Pairs"].split(", ") if row["Suggested Pairs"] != "N/A" else []), axis=1)
    
    # Add current price based on first suggested pair
    def get_first_pair_price(pairs_str):
        if pairs_str == "N/A" or not pairs_str:
            return 0
        pairs = pairs_str.split(", ")
        if pairs:
            first_pair = pairs[0]
            return get_current_price(first_pair)
        return 0
    
    df["Current Price"] = df["Suggested Pairs"].apply(get_first_pair_price)
    
    # Calculate stop loss and take profit with improved rounding
    df["Stop Loss"], df["Take Profit"] = zip(*df.apply(
        lambda row: get_trade_levels(row["Signal"], row["Current Price"]),
        axis=1
    ))
    
    # Format numeric columns for display
    df["Current Price"] = df["Current Price"].apply(lambda x: f"{x:.5f}" if x else "N/A")
    df["Stop Loss"] = df["Stop Loss"].apply(lambda x: f"{x:.5f}" if x else "N/A")
    df["Take Profit"] = df["Take Profit"].apply(lambda x: f"{x:.5f}" if x else "N/A")

    # Display as beautiful table
    st.subheader("📰 Latest Market News & Trading Signals")
    st.dataframe(df, use_container_width=True)

    # Show statistics
    st.subheader("📈 Sentiment Summary")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        positive = len(df[df['Sentiment'] == 'POSITIVE'])
        st.metric("🟢 Buy Signals", positive)
    
    with col2:
        negative = len(df[df['Sentiment'] == 'NEGATIVE'])
        st.metric("🔴 Sell Signals", negative)
    
    with col3:
        avg_conf = df['Confidence'].mean()
        st.metric("Average Confidence", f"{avg_conf:.1%}")

except Exception as e:
    st.error(f"Error: {e}")
    st.info("Please check your API key is correct")
