# Fintech Competitive Intelligence Dashboard

An automated data pipeline that tracks news coverage across 5 major fintech companies. Uses the Claude AI API to perform sentiment analysis and theme extraction, stores structured results in a SQL database, and surfaces insights in an interactive Streamlit dashboard.

**Companies tracked:** Stripe, Wise, Square, PayPal, Klarna

---

## Features

- **Automated news collection:** pulls the latest articles from 150,000+ sources via NewsAPI
- **AI-powered analysis:** uses the Claude API to score sentiment (-1.0 to 1.0), extract key themes, and generate plain-English summaries for each article
- **SQL data layer:** stores all articles and analysis results in a structured SQLite database
- **Interactive dashboard:** built with Streamlit and Plotly, featuring:
  - Sentiment scores and positive/negative breakdown by company
  - Daily sentiment trends over time with company filtering
  - Top themes per company
  - Browsable article feed with AI summaries, filterable by company and sentiment

---

## Tech Stack

| Layer | Tools |
|---|---|
| Data collection | Python, NewsAPI, `requests` |
| AI analysis | Claude API (`claude-sonnet-4-6`) |
| Data storage | SQLite, `pandas` |
| Dashboard | Streamlit, Plotly |
| Environment | `python-dotenv` |

---

## Project Structure

| File | Description |
│---|---|
| 01_collector.py | Pulls articles from NewsAPI and stores in SQLite |
| 02_analyzer.py | Sends articles to Claude API for sentiment analysis |
| 03_analytics.py | SQL queries for insights (terminal output) |
| 04_dashboard.py | Streamlit dashboard |
| .env.example | Template for required environment variables |
| README.md | Project Overview |

---

## Setup & Installation

**1. Clone the repository**
```bash
git clone https://github.com/petall-io/fintech-intelligence.git
cd fintech-intelligence
```

**2. Create a virtual environment**
```bash
python -m venv venv

venv\Scripts\activate
```

**3. Install dependencies**
```bash
pip install requests pandas anthropic streamlit plotly python-dotenv schedule
```

**4. Set up environment variables**

Copy `.env.example` to `.env` and fill in your API keys:
```bash
cp .env.example .env
```

```
CLAUDE_API_KEY=your-claude-api-key
NEWS_API_KEY=your-newsapi-key
```

- NewsAPI: [newsapi.org](https://newsapi.org)
- Claude API key: [console.anthropic.com](https://console.anthropic.com)

---

## Running the Project

Run each script in order:

**Step 1 — Collect articles**
```bash
python 01_collector.py
```

**Step 2 — Analyze with Claude AI**
```bash
python 02_analyzer.py
```
*(Run multiple times if needed, will pick up where you left off)*

**Step 3 — (Optional) View analytics report in terminal**
```bash
python 03_analytics.py
```

**Step 4 — Launch the dashboard**
```bash
streamlit run 04_dashboard.py
```

---

## Known Limitations

- **NewsAPI free tier** returns up to 100 articles per query and limits requests to 100/day. Upgrading to a paid plan would increase coverage significantly.
- **Data noise** — because NewsAPI matches keywords anywhere in an article, some results may not be directly about the tracked company's core business. A paid news API with more targeted filtering would improve signal quality.
- **7-day window** — the free NewsAPI tier only returns articles from the past 7 days. Historical trend analysis would require a paid plan or alternative data source.
- **Klarna coverage** — Klarna consistently returns fewer articles than other companies due to lower overall English-language news volume.

---

## Potential Improvements

- Schedule `01_collector.py` and `02_analyzer.py` to run automatically on a weekly cadence
- Add more companies or expand to other industries
- Integrate a paid news API for cleaner, more targeted data
- Add email or Slack alerts when a company's sentiment drops significantly
- Deploy the dashboard to Streamlit Cloud for public access
