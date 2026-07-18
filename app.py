import time

while True:
    # Pull latest headlines
    articles = newsapi.get_everything(
        q="Federal Reserve OR inflation OR gold OR ECB OR Bank of England OR geopolitics OR oil OR unemployment OR China OR recession OR jobs OR USD OR forex OR currency",
        language="en",
        sort_by="publishedAt"
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
            "Suggested Pairs": suggestion
        })

    # Show table
    df = pd.DataFrame(results)
    display(df)

    # Wait 1 hour before refreshing
    print("⏳ Waiting 1 hour before next refresh...")
    time.sleep(3600)
