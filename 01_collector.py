import requests
import sqlite3
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv
import os

load_dotenv()

NEWS_API_KEY = os.getenv("NEWS_API_KEY")

COMPANIES = ["Stripe", "Wise", "Square", "PayPal", "Klarna"]

def create_database():
    conn = sqlite3.connect("fintech.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS articles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            company TEXT,
            title TEXT,
            description TEXT,
            url TEXT,
            published_at TEXT,
            source TEXT,
            collected_at TEXT
        )
    """)
    conn.commit()
    conn.close()
    print("Database ready.")

def fetch_articles(company):
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": f'"{company}" AND (payments OR fintech OR finance OR banking OR crypto OR lending OR "financial services" OR startup OR investors OR revenue OR earnings)',
        "language": "en",
        "sortBy": "publishedAt",
        "from": (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d"),
        "apiKey": NEWS_API_KEY
    }
    response = requests.get(url, params=params)
    data = response.json()

    if data["status"] != "ok":
        print(f"Error fetching {company}: {data.get('message')}")
        return []

    articles = []
    for article in data["articles"]:
        articles.append({
            "company": company,
            "title": article["title"],
            "description": article["description"],
            "url": article["url"],
            "published_at": article["publishedAt"],
            "source": article["source"]["name"],
            "collected_at": datetime.now().isoformat()
        })
    print(f"Fetched {len(articles)} articles for {company}")
    return articles

FINTECH_KEYWORDS = [
    "payment", "fintech", "finance", "banking", "crypto", "lending",
    "revenue", "earnings", "funding", "investors", "startup", "acquisition",
    "partnership", "product", "launch", "app", "platform", "service",
    "transaction", "wallet", "card", "loan", "invest", "stock", "market",
    "regulatory", "compliance", "IPO", "valuation", "billion", "million"
]

def is_relevant(title, description):
    title_lower = (title or "").lower()
    description_lower = (description or "").lower()
    full_text = f"{title_lower} {description_lower}"
    return any(keyword in full_text for keyword in FINTECH_KEYWORDS)

def save_articles(articles):
    if not articles:
        return
    conn = sqlite3.connect("fintech.db")
    cursor = conn.cursor()
    saved = 0
    skipped = 0
    for a in articles:
        if not is_relevant(a["title"] or "", a["description"] or ""):
            skipped += 1
            continue
        cursor.execute("""
            INSERT INTO articles (company, title, description, url, published_at, source, collected_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (a["company"], a["title"], a["description"], a["url"], a["published_at"], a["source"], a["collected_at"]))
        saved += 1
    conn.commit()
    conn.close()
    print(f"  Saved {saved}, skipped {skipped} irrelevant articles")

def run_collection():
    print("Starting data collection...")
    create_database()
    for company in COMPANIES:
        articles = fetch_articles(company)
        save_articles(articles)
    print("Done! Checking database...")
    conn = sqlite3.connect("fintech.db")
    df = pd.read_sql("SELECT company, COUNT(*) as article_count FROM articles GROUP BY company", conn)
    conn.close()
    print(df)

if __name__ == "__main__":
    run_collection()