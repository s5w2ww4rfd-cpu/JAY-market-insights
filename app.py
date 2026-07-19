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
