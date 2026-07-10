import anthropic
import sqlite3
import json
from datetime import datetime
from dotenv import load_dotenv
import os

load_dotenv()

CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")
client = anthropic.Anthropic(api_key=CLAUDE_API_KEY)

def get_batch_size():
    conn = sqlite3.connect("fintech.db")
    cursor = conn.cursor()
    try:
        cursor.execute("""
            SELECT COUNT(*) FROM articles a
            LEFT JOIN analysis an ON a.id = an.article_id
            WHERE an.article_id IS NULL
            AND a.description IS NOT NULL
        """)
        count = cursor.fetchone()[0]
    except:
        cursor.execute("SELECT COUNT(*) FROM articles WHERE description IS NOT NULL")
        count = cursor.fetchone()[0]
    conn.close()
    return count

BATCH_SIZE = get_batch_size()

def get_unanalyzed_articles():
    conn = sqlite3.connect("fintech.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS analysis (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            article_id INTEGER,
            company TEXT,
            sentiment TEXT,
            sentiment_score REAL,
            key_themes TEXT,
            summary TEXT,
            analyzed_at TEXT,
            relevant INTEGER DEFAULT 1,
            FOREIGN KEY (article_id) REFERENCES articles(id)
        )
    """)
    conn.commit()
    cursor.execute(f"""
        SELECT a.id, a.company, a.title, a.description
        FROM articles a
        LEFT JOIN analysis an ON a.id = an.article_id
        WHERE an.article_id IS NULL
        AND a.description IS NOT NULL
        LIMIT {BATCH_SIZE}
    """)
    articles = cursor.fetchall()
    conn.close()
    print(f"Found {len(articles)} unanalyzed articles.")
    return articles

def analyze_article(article_id, company, title, description):
    prompt = f"""You are a financial news analyst. Analyze this news article about {company} and respond ONLY with a JSON object, no extra text.

    Title: {title}
    Description: {description}

    Return exactly this JSON structure:
    {{
        "relevant": true,
        "sentiment": "positive" or "negative" or "neutral",
        "sentiment_score": a number from -1.0 (very negative) to 1.0 (very positive),
        "key_themes": ["theme1", "theme2", "theme3"],
        "summary": "one sentence summary of the article"
    }}"""

    message = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}]
        )

    if not message.content or not message.content[0].text:
            raise ValueError("Empty response from Claude")

    raw = message.content[0].text.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()
    result = json.loads(raw)
    return result

def save_analysis(article_id, company, result):
    conn = sqlite3.connect("fintech.db")
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO analysis (article_id, company, sentiment, sentiment_score, key_themes, summary, analyzed_at, relevant)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        article_id,
        company,
        result["sentiment"],
        result["sentiment_score"],
        json.dumps(result["key_themes"]),
        result["summary"],
        datetime.now().isoformat(),
        1 if result.get("relevant", True) else 0
    ))
    conn.commit()
    conn.close()

def run_analysis():
    articles = get_unanalyzed_articles()
    if not articles:
        print("No new articles to analyze.")
        conn = sqlite3.connect("fintech.db")
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM articles")
        total_articles = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM analysis")
        total_analyzed = cursor.fetchone()[0]
        cursor.execute("""
            SELECT company, COUNT(*) as analyzed,
                ROUND(AVG(sentiment_score), 2) as avg_sentiment,
                SUM(CASE WHEN sentiment = 'positive' THEN 1 ELSE 0 END) as positive,
                SUM(CASE WHEN sentiment = 'negative' THEN 1 ELSE 0 END) as negative,
                SUM(CASE WHEN sentiment = 'neutral' THEN 1 ELSE 0 END) as neutral
            FROM analysis GROUP BY company ORDER BY avg_sentiment DESC
        """)
        rows = cursor.fetchall()
        conn.close()
        print(f"\nTotal articles collected: {total_articles}")
        print(f"Total articles analyzed:  {total_analyzed}")
        print(f"Remaining:                {total_articles - total_analyzed}")
        print("\n--- Sentiment Summary ---")
        print(f"{'Company':<12} {'Articles':<10} {'Avg Score':<12} {'Pos':<6} {'Neg':<6} {'Neu':<6}")
        print("-" * 52)
        for row in rows:
            print(f"{row[0]:<12} {row[1]:<10} {row[2]:<12} {row[3]:<6} {row[4]:<6} {row[5]:<6}")
        return

    success = 0
    errors = 0

    for i, (article_id, company, title, description) in enumerate(articles):
        if not title or not description:
            print(f"  Skipping — missing title or description")
            continue
        try:
            print(f"[{i+1}/{len(articles)}] Analyzing {company}: {title[:60]}...")
            result = analyze_article(article_id, company, title, description)
            save_analysis(article_id, company, result)
            success += 1
        except Exception as e:
            print(f"  Error: {e}")
            errors += 1

    print(f"\nDone! {success} analyzed, {errors} errors.")

    conn = sqlite3.connect("fintech.db")
    cursor = conn.cursor()
    cursor.execute("""
        SELECT company, 
               COUNT(*) as analyzed,
               SUM(CASE WHEN relevant = 0 THEN 1 ELSE 0 END) as filtered_out,
               ROUND(AVG(sentiment_score), 2) as avg_sentiment,
               SUM(CASE WHEN sentiment = 'positive' THEN 1 ELSE 0 END) as positive,
               SUM(CASE WHEN sentiment = 'negative' THEN 1 ELSE 0 END) as negative,
               SUM(CASE WHEN sentiment = 'neutral' THEN 1 ELSE 0 END) as neutral
        FROM analysis
        GROUP BY company
        ORDER BY avg_sentiment DESC
    """)
    rows = cursor.fetchall()
    cursor.execute("SELECT COUNT(*) FROM articles")
    total_articles = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM analysis")
    total_analyzed = cursor.fetchone()[0]

    conn.close()

    print(f"\nTotal articles collected: {total_articles}")
    print(f"Total articles analyzed:  {total_analyzed}")
    print(f"Remaining:                {total_articles - total_analyzed}")

    print("\n--- Sentiment Summary ---")
    print(f"{'Company':<12} {'Articles':<10} {'Filtered':<10} {'Avg Score':<12} {'Pos':<6} {'Neg':<6} {'Neu':<6}")
    print("-" * 62)
    for row in rows:
        print(f"{row[0]:<12} {row[1]:<10} {row[2]:<10} {row[3]:<12} {row[4]:<6} {row[5]:<6} {row[6]:<6}")
if __name__ == "__main__":
    run_analysis()