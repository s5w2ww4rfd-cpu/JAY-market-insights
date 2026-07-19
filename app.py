import streamlit as st
import pandas as pd
import time
from newsapi import NewsApiClient
from transformers import pipeline

# Initialize NewsAPI
newsapi = NewsApiClient(api_key=st.secrets["NEWSAPI_KEY"])

# Load sentiment analysis model
@st.cache_resource
def load_sentiment_model():
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

sentiment_model = load_sentiment_model()

# Function to map sentiment to trading pairs
def map_to_pairs(headline, sentiment_result):
    """Map headlines and sentiment to trading pairs"""
    label = sentiment_result[0]['label'].upper()
    
    # Mapping logic - customize based on your needs
    pairs_map = {
        "POSITIVE": ["EURUSD", "GBPUSD", "XAUUSD"],
        "NEGATIVE": ["USDJPY", "USDCHF", "DXY"]
    }
    
    return pairs_map.get(label, [])

# Streamlit App Title
st.set_page_config(page_title="JAY Market Insights", layout="wide")
st.title("📊 JAY Market Insights - Trading Sentiment Analyzer")
st.markdown("Real-time sentiment analysis of market news for trading decisions")

# Refresh button
if st.button("🔄 Refresh News"):
    st.rerun()

# Auto-refresh toggle
auto_refresh = st.checkbox("Auto-refresh every hour", value=False)

# Pull latest headlines
try:
    articles = newsapi.get_everything(
        q="Federal Reserve OR inflation OR gold OR ECB OR Bank of England OR geopolitics OR oil OR unemployment OR China OR recession OR jobs OR USD OR forex OR currency",
        language="en",
        sort_by="publishedAt",
        page_size=10
    )
    
    results = []
    for a in articles['articles'][:10]:
        headline = a['title']
        result = sentiment_model(headline)
        suggestion = map_to_pairs(headline, result)
        results.append({
            "Headline": headline,
            "Sentiment": result[0]['label'],
            "Confidence": round(result[0]['score'], 2),
            "Suggested Pairs": ", ".join(suggestion) if suggestion else "N/A",
            "Source": a['source']['name'],
            "Published": a['publishedAt'][:10]
        })
    
    # Display results
    df = pd.DataFrame(results)
    
    st.subheader("📰 Latest Market News Analysis")
    st.dataframe(df, use_container_width=True)
    
    # Summary stats
    col1, col2, col3 = st.columns(3)
    with col1:
        positive = len(df[df['Sentiment'] == 'POSITIVE'])
        st.metric("Positive Sentiment", positive)
    
    with col2:
        negative = len(df[df['Sentiment'] == 'NEGATIVE'])
        st.metric("Negative Sentiment", negative)
    
    with col3:
        avg_confidence = df['Confidence'].mean()
        st.metric("Avg Confidence", f"{avg_confidence:.2%}")
    
    # Auto-refresh logic
    if auto_refresh:
        st.info("⏳ App will auto-refresh every hour...")
        time.sleep(3600)
        st.rerun()

except Exception as e:
    st.error(f"Error fetching news: {e}")
    st.info("Make sure your NEWSAPI_KEY is set in Streamlit secrets")
