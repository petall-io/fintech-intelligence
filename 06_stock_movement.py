import pandas as pd
import numpy as np
import yfinance as yf
import sqlite3
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

# Companies with public stock tickers
TICKERS = {
    "PayPal": "PYPL",
    "Wise": "WISE.L",
    "Square": "XYZ"
}

def load_sentiment_model():
    """Train the Step 1 sentiment model on Financial PhraseBank"""
    print("Training Step 1 sentiment model...")
    df = pd.read_csv("all-data.csv", encoding="latin-1", 
                     header=None, names=["sentiment", "text"])
    
    tfidf = TfidfVectorizer(max_features=2000, ngram_range=(1,2), stop_words='english')
    X = tfidf.fit_transform(df['text'])
    
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y = le.fit_transform(df['sentiment'])
    
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X, y)
    
    print(f"  Sentiment model trained on {len(df)} sentences")
    print(f"  Classes: {le.classes_}")
    return model, tfidf, le

def score_sentiment(texts, model, tfidf, le):
    """Use Step 1 model to score a list of texts"""
    X = tfidf.transform(texts)
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)
    
    # Convert to sentiment scores (-1 to 1)
    scores = []
    for pred, prob in zip(predictions, probabilities):
        label = le.classes_[pred]
        if label == 'positive':
            score = prob[list(le.classes_).index('positive')]
        elif label == 'negative':
            score = -prob[list(le.classes_).index('negative')]
        else:
            score = 0
        scores.append(score)
    
    return scores

def fetch_historical_news_sentiment(model, tfidf, le):
    """Use Financial PhraseBank as proxy for historical sentiment"""
    print("\nPreparing historical sentiment data...")
    df = pd.read_csv("all-data.csv", encoding="latin-1",
                     header=None, names=["sentiment", "text"])
    
    # Score all sentences
    scores = score_sentiment(df['text'].tolist(), model, tfidf, le)
    df['sentiment_score'] = scores
    
    # Map sentiment to numeric
    sentiment_map = {'positive': 1, 'negative': -1, 'neutral': 0}
    df['sentiment_numeric'] = df['sentiment'].map(sentiment_map)
    
    return df

def fetch_stock_data(ticker, period="2y"):
    """Fetch historical stock data"""
    print(f"  Fetching {ticker} stock data...")
    stock = yf.Ticker(ticker)
    hist = stock.history(period=period)
    
    if hist.empty:
        return None
    
    hist = hist.reset_index()
    hist['date'] = pd.to_datetime(hist['Date']).dt.date
    hist['price_change'] = hist['Close'].pct_change()
    hist['movement'] = (hist['price_change'] > 0).astype(int)
    hist['prev_sentiment'] = 0.0
    
    return hist[['date', 'Close', 'price_change', 'movement']].dropna()

def create_ml_dataset(stock_df, sentiment_df):
    """
    Create ML dataset by combining stock movements with 
    rolling sentiment features
    """
    # Create daily sentiment aggregates from Financial PhraseBank
    # We'll use sentiment distribution as features
    n_days = len(stock_df)
    
    # Sample sentiment data to match stock trading days
    chunk_size = max(1, len(sentiment_df) // n_days)
    
    daily_features = []
    for i in range(n_days):
        chunk = sentiment_df.iloc[i*chunk_size:(i+1)*chunk_size]
        if len(chunk) == 0:
            chunk = sentiment_df.sample(chunk_size)
        
        features = {
            'avg_sentiment': chunk['sentiment_score'].mean(),
            'pct_positive': (chunk['sentiment'] == 'positive').mean(),
            'pct_negative': (chunk['sentiment'] == 'negative').mean(),
            'pct_neutral': (chunk['sentiment'] == 'neutral').mean(),
            'sentiment_std': chunk['sentiment_score'].std(),
            'positive_count': (chunk['sentiment'] == 'positive').sum(),
            'negative_count': (chunk['sentiment'] == 'negative').sum(),
        }
        daily_features.append(features)
    
    features_df = pd.DataFrame(daily_features)
    
    # Combine with stock data
    combined = pd.concat([
        stock_df.reset_index(drop=True),
        features_df.reset_index(drop=True)
    ], axis=1)
    
    # Add lag features (previous day's sentiment)
    combined['prev_avg_sentiment'] = combined['avg_sentiment'].shift(1)
    combined['prev_pct_positive'] = combined['pct_positive'].shift(1)
    combined['prev_pct_negative'] = combined['pct_negative'].shift(1)
    
    # Add price momentum features
    combined['prev_movement'] = combined['movement'].shift(1)
    combined['price_momentum'] = combined['price_change'].rolling(3).mean().shift(1)
    
    return combined.dropna()

def train_movement_models(X_train, X_test, y_train, y_test):
    """Train Step 2 models to predict stock movement"""
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=42,
            C=0.1                # stronger regularization, default is 1.0
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            max_depth=5,         # limit tree depth
            min_samples_leaf=10  # each leaf needs at least 10 samples
        ),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=100,
            random_state=42,
            eval_metric='logloss',
            max_depth=3,         # shallower trees
            learning_rate=0.1,   # learn slower
            reg_alpha=0.1,       # L1 regularization
            reg_lambda=1.0       # L2 regularization
        )
    }
    
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        results[name] = {
            'model': model,
            'accuracy': acc,
            'predictions': y_pred,
            'report': classification_report(y_test, y_pred, output_dict=True)
        }
        print(f"  {name}: {acc:.1%} accuracy")
    
    return results

def plot_results(results, company, feature_names, best_model_name):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f"Stock Movement Prediction — {company}", fontsize=14, fontweight='bold')
    
    # Accuracy comparison
    ax = axes[0]
    names = list(results.keys())
    accs = [results[m]['accuracy'] for m in names]
    colors = ['#3b82f6', '#22c55e', '#f97316']
    bars = ax.bar(names, accs, color=colors)
    ax.set_title("Model Accuracy")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Random baseline')
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{acc:.1%}', ha='center', fontweight='bold', fontsize=9)
    ax.legend()
    ax.set_xticklabels(names, rotation=15, ha='right')
    
    # Feature importance (Random Forest)
    ax = axes[1]
    rf_model = results['Random Forest']['model']
    importances = rf_model.feature_importances_
    indices = np.argsort(importances)[::-1][:8]
    ax.barh([feature_names[i] for i in indices][::-1],
            importances[indices][::-1], color='#3b82f6')
    ax.set_title("Top Features (Random Forest)")
    ax.set_xlabel("Importance")
    
    # Prediction distribution
    ax = axes[2]
    best_preds = results[best_model_name]['predictions']
    pred_counts = pd.Series(best_preds).value_counts()
    ax.pie([pred_counts.get(1, 0), pred_counts.get(0, 0)],
           labels=['Predicted UP', 'Predicted DOWN'],
           colors=['#22c55e', '#ef4444'],
           autopct='%1.1f%%')
    ax.set_title(f"Prediction Distribution\n({best_model_name})")
    
    plt.tight_layout()
    filename = f"movement_prediction_{company.lower()}.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"  Saved {filename}")
    plt.show()

if __name__ == "__main__":
    print("=" * 60)
    print("STEP 1: Training sentiment scoring model")
    print("=" * 60)
    sentiment_model, tfidf, le = load_sentiment_model()
    
    print("\n" + "=" * 60)
    print("STEP 2: Preparing sentiment features")
    print("=" * 60)
    sentiment_df = fetch_historical_news_sentiment(sentiment_model, tfidf, le)
    
    print("\n" + "=" * 60)
    print("STEP 3: Training stock movement models")
    print("=" * 60)
    
    all_results = {}
    
    for company, ticker in TICKERS.items():
        print(f"\n--- {company} ({ticker}) ---")
        
        stock_df = fetch_stock_data(ticker, period="2y")
        if stock_df is None or len(stock_df) < 50:
            print(f"  Not enough stock data for {company}, skipping")
            continue
        
        print(f"  Stock data: {len(stock_df)} trading days")
        
        # Create ML dataset
        ml_df = create_ml_dataset(stock_df, sentiment_df)
        print(f"  ML dataset: {len(ml_df)} rows")
        print(f"  Up days: {ml_df['movement'].sum()}, Down days: {(ml_df['movement']==0).sum()}")
        
        # Features and target
        feature_cols = ['avg_sentiment', 'pct_positive', 'pct_negative', 
                       'pct_neutral', 'sentiment_std', 'positive_count',
                       'negative_count', 'prev_avg_sentiment', 'prev_pct_positive',
                       'prev_pct_negative', 'prev_movement', 'price_momentum']
        
        X = ml_df[feature_cols]
        y = ml_df['movement']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        print(f"  Training models...")
        results = train_movement_models(X_train, X_test, y_train, y_test)
        all_results[company] = results
        
        best = max(results, key=lambda x: results[x]['accuracy'])
        print(f"  Best model: {best} ({results[best]['accuracy']:.1%})")
        
        plot_results(results, company, feature_cols, best)
    
    # Final summary
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"{'Company':<12} {'LR':>8} {'RF':>8} {'XGB':>8} {'Best':>8}")
    print("-" * 46)
    for company, results in all_results.items():
        accs = {m: f"{results[m]['accuracy']:.1%}" for m in results}
        best = max(results, key=lambda x: results[x]['accuracy'])
        print(f"{company:<12} {accs['Logistic Regression']:>8} {accs['Random Forest']:>8} {accs['XGBoost']:>8} {best:>20}")

    # Save the best movement model for each company
    for company, results in all_results.items():
        best_name = max(results, key=lambda x: results[x]['accuracy'])
        best_model = results[best_name]['model']
        filename = f"movement_model_{company.lower()}.pkl"
        joblib.dump(best_model, filename)
        print(f"Saved {company} movement model ({best_name}) to {filename}")
    
    print("\nDone!")