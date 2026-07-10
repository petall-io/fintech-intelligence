import streamlit as st
import sqlite3
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
from collections import Counter

st.cache_data.clear()

st.set_page_config(
    page_title="Fintech Competitive Intelligence",
    page_icon="📊",
    layout="wide"
)

def get_connection():
    return sqlite3.connect("fintech.db")

@st.cache_data
def load_sentiment_by_company():
    conn = get_connection()
    df = pd.read_sql("""
        SELECT 
            company,
            COUNT(*) as total_articles,
            ROUND(AVG(sentiment_score), 3) as avg_sentiment,
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

@st.cache_data
def load_sentiment_over_time():
    conn = get_connection()
    df = pd.read_sql("""
        SELECT 
            an.company,
            DATE(ar.published_at) as date,
            COUNT(*) as article_count,
            ROUND(AVG(an.sentiment_score), 3) as avg_sentiment
        FROM analysis an
        JOIN articles ar ON an.article_id = ar.id
        WHERE an.relevant = 1
        AND ar.published_at IS NOT NULL
        GROUP BY an.company, DATE(ar.published_at)
        ORDER BY date
    """, conn)
    conn.close()
    return df

@st.cache_data
def load_recent_articles(company=None, sentiment=None):
    conn = get_connection()
    query = """
        SELECT 
            an.company,
            ar.title,
            ar.source,
            ar.published_at,
            an.sentiment,
            an.sentiment_score,
            an.summary,
            ar.url
        FROM analysis an
        JOIN articles ar ON an.article_id = ar.id
    """
    if company and company != "All":
        query += f" AND an.company = '{company}'"
    if sentiment and sentiment != "All":
        query += f" AND an.sentiment = '{sentiment.lower()}'"
    query += " ORDER BY ar.published_at DESC LIMIT 100"
    df = pd.read_sql(query, conn)
    conn.close()
    return df

@st.cache_data
def load_themes():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT company, key_themes FROM analysis WHERE relevant = 1")
    rows = cursor.fetchall()
    conn.close()

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
    return company_themes

st.title("📊 Fintech Competitive Intelligence Dashboard")
st.caption("Powered by NewsAPI + Claude AI — tracking Stripe, Wise, Square, PayPal, Klarna")

st.markdown("---")

df_company = load_sentiment_by_company()

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Companies Tracked", len(df_company))
with col2:
    st.metric("Total Articles Analyzed", int(df_company["total_articles"].sum()))
with col3:
    top = df_company.iloc[0]["company"]
    st.metric("Most Positive Coverage", top)
with col4:
    bottom = df_company.iloc[-1]["company"]
    st.metric("Most Negative Coverage", bottom)

st.markdown("---")

st.subheader("Sentiment by Company")
col1, col2 = st.columns(2)

with col1:
    fig = px.bar(
        df_company,
        x="company",
        y="avg_sentiment",
        color="avg_sentiment",
        color_continuous_scale=["#ef4444", "#f97316", "#22c55e"],
        title="Average Sentiment Score",
        labels={"avg_sentiment": "Avg Sentiment Score", "company": "Company"}
    )
    fig.update_layout(coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    fig = px.bar(
        df_company,
        x="company",
        y=["pct_positive", "pct_negative"],
        title="Positive vs Negative Coverage (%)",
        labels={"value": "Percentage", "company": "Company", "variable": "Sentiment"},
        color_discrete_map={"pct_positive": "#22c55e", "pct_negative": "#ef4444"},
        barmode="group"
    )
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.subheader("Sentiment Over Time")

df_time = load_sentiment_over_time()
companies = df_company["company"].tolist()
selected_companies = st.multiselect("Filter by company", companies, default=companies)

df_filtered = df_time[df_time["company"].isin(selected_companies)]
fig = px.line(
    df_filtered,
    x="date",
    y="avg_sentiment",
    color="company",
    title="Daily Sentiment Score by Company",
    labels={"avg_sentiment": "Avg Sentiment", "date": "Date", "company": "Company"},
    markers=True
)
fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.subheader("Top Themes by Company")

company_themes = load_themes()
selected_company = st.selectbox("Select a company", companies)

if selected_company in company_themes:
    top_themes = Counter(company_themes[selected_company]).most_common(10)
    theme_df = pd.DataFrame(top_themes, columns=["Theme", "Mentions"])
    fig = px.bar(
        theme_df,
        x="Mentions",
        y="Theme",
        orientation="h",
        title=f"Top Themes — {selected_company}",
        color="Mentions",
        color_continuous_scale=["#93c5fd", "#1d4ed8"]
    )
    fig.update_layout(coloraxis_showscale=False, yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.subheader("Recent Articles")

col1, col2 = st.columns(2)
with col1:
    company_filter = st.selectbox("Filter by company", ["All"] + companies, key="article_company")
with col2:
    sentiment_filter = st.selectbox("Filter by sentiment", ["All", "Positive", "Negative", "Neutral"])

df_articles = load_recent_articles(company_filter, sentiment_filter)

for _, row in df_articles.iterrows():
    score = row["sentiment_score"]
    color = "🟢" if row["sentiment"] == "positive" else "🔴" if row["sentiment"] == "negative" else "⚪"
    with st.expander(f"{color} [{row['company']}] {row['title']}"):
        st.write(f"**Source:** {row['source']} | **Published:** {row['published_at'][:10]}")
        st.write(f"**Sentiment Score:** {score}")
        st.write(f"**AI Summary:** {row['summary']}")
        if row["url"]:
            st.markdown(f"[Read full article]({row['url']})")