import sqlite3
import pandas as pd

conn = sqlite3.connect("fintech.db")

df = pd.read_sql("""
    SELECT an.company, ar.title, ar.source, ar.published_at, an.sentiment, an.sentiment_score, an.summary
    FROM analysis an
    JOIN articles ar ON an.article_id = ar.id
    ORDER BY an.company, an.sentiment_score DESC
""", conn)

conn.close()

df.to_csv("articles_analyzed.csv", index=False)
print(f"Saved {len(df)} articles to articles_analyzed.csv")
print(df.groupby("company")[["sentiment_score"]].mean().round(2))