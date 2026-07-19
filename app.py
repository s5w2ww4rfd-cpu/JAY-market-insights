import pandas as pd
from newsapi import NewsApiClient
from transformers import pipeline
import streamlit as st

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

# Get trade levels (stop loss and take profit)
def get_trade_levels(signal, current_price):
    if signal.startswith("🟢"):  # BUY signal
        stop_loss = current_price * 0.99   # 1% below
        take_profit = current_price * 1.02 # 2% above
    elif signal.startswith("🔴"):  # SELL signal
        stop_loss = current_price * 1.01   # 1% above
        take_profit = current_price * 0.98 # 2% below
    else:
        stop_loss = None
        take_profit = None
    return stop_loss, take_profit

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
    
    # Add example current prices and calculate stop loss/take profit
    # Using example prices for demonstration
    example_prices = {
        "EURUSD": 1.0950,
        "GBPUSD": 1.2750,
        "XAUUSD": 2050.0,
        "USDJPY": 145.50,
        "USDCHF": 0.8850,
        "DXY": 103.50
    }
    
    def get_price_for_pairs(pairs_str):
        if pairs_str == "N/A" or not pairs_str:
            return 0
        pairs = pairs_str.split(", ")
        if pairs:
            first_pair = pairs[0]
            return example_prices.get(first_pair, 0)
        return 0
    
    df["Current Price"] = df["Suggested Pairs"].apply(get_price_for_pairs)
    
    # Calculate stop loss and take profit
    def calc_levels(row):
        sl, tp = get_trade_levels(row["Signal"], row["Current Price"])
        return pd.Series([sl, tp])
    
    df[["Stop Loss", "Take Profit"]] = df.apply(calc_levels, axis=1)
    
    # Format the output columns
    df["Stop Loss"] = df["Stop Loss"].apply(lambda x: f"{x:.4f}" if x else "N/A")
    df["Take Profit"] = df["Take Profit"].apply(lambda x: f"{x:.4f}" if x else "N/A")

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
