import sqlite3
import pandas as pd

def get_connection():
    return sqlite3.connect("fintech.db")

def sentiment_by_company():
    conn = get_connection()
    df = pd.read_sql("""
        SELECT 
            company,
            COUNT(*) as total_articles,
            ROUND(AVG(sentiment_score), 3) as avg_sentiment,
            ROUND(MIN(sentiment_score), 3) as min_sentiment,
            ROUND(MAX(sentiment_score), 3) as max_sentiment,
            SUM(CASE WHEN sentiment = 'positive' THEN 1 ELSE 0 END) as positive,
            SUM(CASE WHEN sentiment = 'negative' THEN 1 ELSE 0 END) as negative,
            SUM(CASE WHEN sentiment = 'neutral' THEN 1 ELSE 0 END) as neutral,
            ROUND(100.0 * SUM(CASE WHEN sentiment = 'positive' THEN 1 ELSE 0 END) / COUNT(*), 1) as pct_positive,
            ROUND(100.0 * SUM(CASE WHEN sentiment = 'negative' THEN 1 ELSE 0 END) / COUNT(*), 1) as pct_negative
        FROM analysis
        GROUP BY company
        ORDER BY avg_sentiment DESC
    """, conn)
    conn.close()
    return df

def sentiment_over_time():
    conn = get_connection()
    df = pd.read_sql("""
        SELECT 
            an.company,
            DATE(ar.published_at) as date,
            COUNT(*) as article_count,
            ROUND(AVG(an.sentiment_score), 3) as avg_sentiment
        FROM analysis an
        JOIN articles ar ON an.article_id = ar.id
        WHERE ar.published_at IS NOT NULL
        GROUP BY an.company, DATE(ar.published_at)
        ORDER BY an.company, date
    """, conn)
    conn.close()
    return df

def coverage_spikes():
    conn = get_connection()
    df = pd.read_sql("""
        SELECT 
            an.company,
            DATE(ar.published_at) as date,
            COUNT(*) as article_count
        FROM analysis an
        JOIN articles ar ON an.article_id = ar.id
        WHERE ar.published_at IS NOT NULL
        GROUP BY an.company, DATE(ar.published_at)
        ORDER BY article_count DESC
        LIMIT 10
    """, conn)
    conn.close()
    return df

def most_negative_articles():
    conn = get_connection()
    df = pd.read_sql("""
        SELECT 
            an.company,
            ar.title,
            ar.source,
            ar.published_at,
            an.sentiment_score,
            an.summary
        FROM analysis an
        JOIN articles ar ON an.article_id = ar.id
        WHERE an.sentiment = 'negative'
        ORDER BY an.sentiment_score ASC
        LIMIT 10
    """, conn)
    conn.close()
    return df

def most_positive_articles():
    conn = get_connection()
    df = pd.read_sql("""
        SELECT 
            an.company,
            ar.title,
            ar.source,
            ar.published_at,
            an.sentiment_score,
            an.summary
        FROM analysis an
        JOIN articles ar ON an.article_id = ar.id
        WHERE an.sentiment = 'positive'
        ORDER BY an.sentiment_score DESC
        LIMIT 10
    """, conn)
    conn.close()
    return df

def top_themes_by_company():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT company, key_themes FROM analysis")
    rows = cursor.fetchall()
    conn.close()

    import json
    from collections import Counter

    company_themes = {}
    for company, themes_json in rows:
        if not themes_json:
            continue
        try:
            themes = json.loads(themes_json)
            if company not in company_themes:
                company_themes[company] = []
            company_themes[company].extend(themes)
        except:
            continue

    print("\n--- Top Themes by Company ---")
    for company, themes in sorted(company_themes.items()):
        top = Counter(themes).most_common(5)
        print(f"\n{company}:")
        for theme, count in top:
            print(f"  {theme:<30} {count} mentions")

def run_all():
    print("=" * 60)
    print("FINTECH COMPETITIVE INTELLIGENCE — ANALYTICS REPORT")
    print("=" * 60)

    print("\n--- Sentiment by Company ---")
    df = sentiment_by_company()
    print(df.to_string(index=False))

    print("\n--- Coverage Spikes (Top 10 Busiest Days) ---")
    df = coverage_spikes()
    print(df.to_string(index=False))

    print("\n--- Most Negative Articles ---")
    df = most_negative_articles()
    for _, row in df.iterrows():
        print(f"\n  [{row['company']}] {row['title'][:70]}")
        print(f"  Score: {row['sentiment_score']} | {row['summary'][:100]}")

    print("\n--- Most Positive Articles ---")
    df = most_positive_articles()
    for _, row in df.iterrows():
        print(f"\n  [{row['company']}] {row['title'][:70]}")
        print(f"  Score: {row['sentiment_score']} | {row['summary'][:100]}")

    top_themes_by_company()

if __name__ == "__main__":
    run_all()