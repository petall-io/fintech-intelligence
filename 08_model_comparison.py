import sqlite3
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings('ignore')

def load_ml_model():
    print("Loading saved ML sentiment model...")
    model = joblib.load("sentiment_model.pkl")
    tfidf = joblib.load("tfidf_vectorizer.pkl")
    le = joblib.load("label_encoder.pkl")
    print(f"  Classes: {le.classes_}")
    return model, tfidf, le

def load_claude_scores():
    print("\nLoading Claude sentiment scores from database...")
    conn = sqlite3.connect("fintech.db")
    df = pd.read_sql("""
        SELECT 
            an.company,
            ar.title,
            ar.description,
            an.sentiment as claude_sentiment,
            an.sentiment_score as claude_score
        FROM analysis an
        JOIN articles ar ON an.article_id = ar.id
        WHERE ar.description IS NOT NULL
        AND ar.title IS NOT NULL
    """, conn)
    conn.close()
    print(f"  Loaded {len(df)} articles with Claude scores")
    return df

def score_with_ml_model(df, model, tfidf, le):
    print("\nScoring same articles with ML model...")
    texts = (df['title'].fillna('') + ' ' + df['description'].fillna('')).tolist()
    
    X = tfidf.transform(texts)
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)
    
    ml_sentiments = []
    ml_scores = []
    
    for pred, prob in zip(predictions, probabilities):
        label = le.classes_[pred]
        ml_sentiments.append(label)
        if label == 'positive':
            score = prob[list(le.classes_).index('positive')]
        elif label == 'negative':
            score = -prob[list(le.classes_).index('negative')]
        else:
            score = 0.0
        ml_scores.append(score)
    
    df['ml_sentiment'] = ml_sentiments
    df['ml_score'] = ml_scores
    print(f"  Scored {len(df)} articles")
    return df

def analyze_agreement(df):
    # Overall agreement
    df['agree'] = df['claude_sentiment'] == df['ml_sentiment']
    overall_agreement = df['agree'].mean()
    
    print(f"\n--- Agreement Analysis ---")
    print(f"Overall agreement: {overall_agreement:.1%}")
    
    # Agreement by company
    print(f"\nAgreement by company:")
    company_agreement = df.groupby('company')['agree'].mean()
    for company, rate in company_agreement.items():
        print(f"  {company:<12} {rate:.1%}")
    
    # Agreement by sentiment class
    print(f"\nAgreement by sentiment class:")
    for sentiment in ['positive', 'negative', 'neutral']:
        subset = df[df['claude_sentiment'] == sentiment]
        if len(subset) > 0:
            rate = (subset['ml_sentiment'] == sentiment).mean()
            print(f"  {sentiment:<12} {rate:.1%} ({len(subset)} articles)")
    
    return overall_agreement

def plot_comparison(df, overall_agreement):
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle("Claude API vs ML Model — Sentiment Comparison", 
                 fontsize=16, fontweight='bold')
    
    # 1. Score correlation scatter plot
    ax = axes[0, 0]
    colors = {'positive': '#22c55e', 'negative': '#ef4444', 'neutral': '#94a3b8'}
    for sentiment, color in colors.items():
        mask = df['claude_sentiment'] == sentiment
        ax.scatter(df[mask]['claude_score'], df[mask]['ml_score'],
                  alpha=0.5, c=color, label=sentiment, s=20)
    
    # Add diagonal line (perfect agreement)
    lims = [-1, 1]
    ax.plot(lims, lims, 'k--', alpha=0.5, label='Perfect agreement')
    ax.set_xlabel("Claude Sentiment Score")
    ax.set_ylabel("ML Model Sentiment Score")
    ax.set_title(f"Score Correlation\n(Overall agreement: {overall_agreement:.1%})")
    ax.legend(fontsize=8)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    
    # 2. Agreement by company bar chart
    ax = axes[0, 1]
    company_agreement = df.groupby('company')['agree'].mean().sort_values(ascending=False)
    bars = ax.bar(company_agreement.index, company_agreement.values,
                  color=['#3b82f6', '#22c55e', '#f97316', '#a855f7', '#ec4899'])
    ax.set_title("Agreement Rate by Company")
    ax.set_ylabel("Agreement Rate")
    ax.set_ylim(0, 1)
    ax.axhline(y=overall_agreement, color='red', linestyle='--', 
               alpha=0.7, label=f'Overall: {overall_agreement:.1%}')
    ax.legend()
    for bar, val in zip(bars, company_agreement.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.1%}', ha='center', fontsize=9)
    ax.set_xticklabels(company_agreement.index, rotation=15, ha='right')

    # 3. Confusion matrix
    ax = axes[0, 2]
    labels = ['negative', 'neutral', 'positive']
    cm = confusion_matrix(df['claude_sentiment'], df['ml_sentiment'], labels=labels)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
               xticklabels=labels, yticklabels=labels)
    ax.set_title("Confusion Matrix\n(Claude = Actual, ML = Predicted)")
    ax.set_ylabel("Claude Label")
    ax.set_xlabel("ML Model Label")

    # 4. Score distribution comparison
    ax = axes[1, 0]
    ax.hist(df['claude_score'], bins=30, alpha=0.6, color='#3b82f6', label='Claude')
    ax.hist(df['ml_score'], bins=30, alpha=0.6, color='#f97316', label='ML Model')
    ax.set_title("Score Distribution Comparison")
    ax.set_xlabel("Sentiment Score")
    ax.set_ylabel("Count")
    ax.legend()
    ax.axvline(x=0, color='black', linestyle='--', alpha=0.5)

    # 5. Sentiment label distribution
    ax = axes[1, 1]
    x = np.arange(3)
    width = 0.35
    labels = ['negative', 'neutral', 'positive']
    claude_counts = [len(df[df['claude_sentiment'] == l]) for l in labels]
    ml_counts = [len(df[df['ml_sentiment'] == l]) for l in labels]
    
    ax.bar(x - width/2, claude_counts, width, label='Claude', color='#3b82f6', alpha=0.8)
    ax.bar(x + width/2, ml_counts, width, label='ML Model', color='#f97316', alpha=0.8)
    ax.set_title("Sentiment Label Distribution")
    ax.set_ylabel("Count")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    # 6. Disagreement examples table
    ax = axes[1, 2]
    ax.axis('off')
    disagreements = df[df['agree'] == False][['company', 'claude_sentiment', 'ml_sentiment', 'claude_score', 'ml_score']].head(8)
    
    if len(disagreements) > 0:
        table_data = []
        for _, row in disagreements.iterrows():
            table_data.append([
                row['company'],
                row['claude_sentiment'],
                row['ml_sentiment'],
                f"{row['claude_score']:.2f}",
                f"{row['ml_score']:.2f}"
            ])
        
        table = ax.table(
            cellText=table_data,
            colLabels=['Company', 'Claude', 'ML Model', 'C Score', 'ML Score'],
            cellLoc='center',
            loc='center'
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1.2, 1.8)
    ax.set_title("Sample Disagreements", pad=20)

    plt.tight_layout()
    plt.savefig("model_comparison_claude_vs_ml.png", dpi=150, bbox_inches='tight')
    print("\nSaved model_comparison_claude_vs_ml.png")
    plt.show()

def print_interesting_findings(df):
    print("\n--- Interesting Findings ---")
    
    # Biggest score differences
    df['score_diff'] = abs(df['claude_score'] - df['ml_score'])
    
    print(f"\nAvg Claude score:    {df['claude_score'].mean():.3f}")
    print(f"Avg ML model score:  {df['ml_score'].mean():.3f}")
    print(f"Avg score difference: {df['score_diff'].mean():.3f}")
    
    # Most disagreed company
    company_agreement = df.groupby('company')['agree'].mean()
    most_disagreed = company_agreement.idxmin()
    most_agreed = company_agreement.idxmax()
    print(f"\nMost agreed on:    {most_agreed} ({company_agreement[most_agreed]:.1%})")
    print(f"Most disagreed on: {most_disagreed} ({company_agreement[most_disagreed]:.1%})")
    
    # When they disagree what does each say
    disagreements = df[df['agree'] == False]
    print(f"\nWhen they disagree ({len(disagreements)} articles):")
    print(f"  Claude says positive, ML says negative: {len(disagreements[(disagreements['claude_sentiment']=='positive') & (disagreements['ml_sentiment']=='negative')])}")
    print(f"  Claude says negative, ML says positive: {len(disagreements[(disagreements['claude_sentiment']=='negative') & (disagreements['ml_sentiment']=='positive')])}")
    print(f"  Claude says neutral, ML disagrees:      {len(disagreements[disagreements['claude_sentiment']=='neutral'])}")

if __name__ == "__main__":
    print("=" * 60)
    print("CLAUDE API vs ML MODEL — SENTIMENT COMPARISON")
    print("=" * 60)

    # Load models and data
    model, tfidf, le = load_ml_model()
    df = load_claude_scores()

    # Score with ML model
    df = score_with_ml_model(df, model, tfidf, le)

    # Analyze agreement
    overall_agreement = analyze_agreement(df)

    # Print interesting findings
    print_interesting_findings(df)

    # Plot everything
    print("\nGenerating comparison charts...")
    plot_comparison(df, overall_agreement)

    print("\nDone!")