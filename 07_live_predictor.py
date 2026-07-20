import sqlite3
import pandas as pd
import numpy as np
import yfinance as yf
import joblib
from datetime import datetime, date

# Companies with saved models and stock tickers
COMPANIES = {
    "PayPal": {"ticker": "PYPL", "model": "movement_model_paypal.pkl"},
    "Wise": {"ticker": "WISE.L", "model": "movement_model_wise.pkl"},
    "Square": {"ticker": "XYZ", "model": "movement_model_square.pkl"}
}

def load_models():
    """Load the saved sentiment model and movement models"""
    print("Loading saved models...")
    sentiment_model = joblib.load("sentiment_model.pkl")
    tfidf = joblib.load("tfidf_vectorizer.pkl")
    le = joblib.load("label_encoder.pkl")
    print("  Sentiment model loaded")
    return sentiment_model, tfidf, le

def get_live_sentiment(company, sentiment_model, tfidf, le):
    """Pull recent articles from database and score them"""
    conn = sqlite3.connect("fintech.db")
    df = pd.read_sql(f"""
        SELECT ar.title, ar.description, ar.published_at
        FROM articles ar
        WHERE ar.company = '{company}'
        AND ar.description IS NOT NULL
        ORDER BY ar.published_at DESC
        LIMIT 20
    """, conn)
    conn.close()

    if df.empty:
        return None

    # Combine title and description
    texts = (df['title'].fillna('') + ' ' + df['description'].fillna('')).tolist()

    # Score sentiment using saved model
    X = tfidf.transform(texts)
    predictions = sentiment_model.predict(X)
    probabilities = sentiment_model.predict_proba(X)

    scores = []
    for pred, prob in zip(predictions, probabilities):
        label = le.classes_[pred]
        if label == 'positive':
            score = prob[list(le.classes_).index('positive')]
        elif label == 'negative':
            score = -prob[list(le.classes_).index('negative')]
        else:
            score = 0.0
        scores.append(score)

    return {
        'avg_sentiment': np.mean(scores),
        'pct_positive': sum(1 for p in predictions if le.classes_[p] == 'positive') / len(predictions),
        'pct_negative': sum(1 for p in predictions if le.classes_[p] == 'negative') / len(predictions),
        'pct_neutral': sum(1 for p in predictions if le.classes_[p] == 'neutral') / len(predictions),
        'sentiment_std': np.std(scores),
        'positive_count': sum(1 for p in predictions if le.classes_[p] == 'positive'),
        'negative_count': sum(1 for p in predictions if le.classes_[p] == 'negative'),
        'article_count': len(texts)
    }

def get_price_features(ticker):
    """Get recent price momentum from stock data"""
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="10d")
        if hist.empty or len(hist) < 3:
            return None
        hist['price_change'] = hist['Close'].pct_change()
        hist['movement'] = (hist['price_change'] > 0).astype(int)
        return {
            'prev_movement': int(hist['movement'].iloc[-2]),
            'price_momentum': hist['price_change'].iloc[-3:].mean(),
            'current_price': hist['Close'].iloc[-1],
            'prev_price_change': hist['price_change'].iloc[-1]
        }
    except Exception as e:
        print(f"  Error fetching price data: {e}")
        return None

def predict_movement(company, sentiment_features, price_features, movement_model):
    """Use saved movement model to predict UP or DOWN"""
    feature_cols = ['avg_sentiment', 'pct_positive', 'pct_negative',
                   'pct_neutral', 'sentiment_std', 'positive_count',
                   'negative_count', 'prev_avg_sentiment', 'prev_pct_positive',
                   'prev_pct_negative', 'prev_movement', 'price_momentum']

    features = {
        'avg_sentiment': sentiment_features['avg_sentiment'],
        'pct_positive': sentiment_features['pct_positive'],
        'pct_negative': sentiment_features['pct_negative'],
        'pct_neutral': sentiment_features['pct_neutral'],
        'sentiment_std': sentiment_features['sentiment_std'],
        'positive_count': sentiment_features['positive_count'],
        'negative_count': sentiment_features['negative_count'],
        'prev_avg_sentiment': sentiment_features['avg_sentiment'],
        'prev_pct_positive': sentiment_features['pct_positive'],
        'prev_pct_negative': sentiment_features['pct_negative'],
        'prev_movement': price_features['prev_movement'],
        'price_momentum': price_features['price_momentum']
    }

    X = pd.DataFrame([features])[feature_cols]
    prediction = movement_model.predict(X)[0]
    probability = movement_model.predict_proba(X)[0]

    return {
        'prediction': 'UP ⬆️' if prediction == 1 else 'DOWN ⬇️',
        'confidence': max(probability),
        'prob_up': probability[1] if len(probability) > 1 else probability[0],
        'prob_down': probability[0]
    }

def run_live_predictions():
    print("=" * 60)
    print("FINTECH LIVE STOCK MOVEMENT PREDICTOR")
    print(f"Run date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)

    # Load sentiment model
    sentiment_model, tfidf, le = load_models()

    results = []

    for company, config in COMPANIES.items():
        print(f"\n--- {company} ---")

        # Get live sentiment from database
        sentiment_features = get_live_sentiment(company, sentiment_model, tfidf, le)
        if sentiment_features is None:
            print(f"  No articles found for {company}, skipping")
            continue
        print(f"  Scored {sentiment_features['article_count']} articles")
        print(f"  Avg sentiment: {sentiment_features['avg_sentiment']:.3f}")
        print(f"  Positive: {sentiment_features['pct_positive']:.1%} | Negative: {sentiment_features['pct_negative']:.1%}")

        # Get price features
        price_features = get_price_features(config['ticker'])
        if price_features is None:
            print(f"  No price data available, skipping")
            continue
        print(f"  Current price: ${price_features['current_price']:.2f}")
        print(f"  Price momentum: {price_features['price_momentum']:.3f}")

        # Load movement model
        movement_model = joblib.load(config['model'])

        # Make prediction
        pred = predict_movement(company, sentiment_features, price_features, movement_model)

        print(f"\n  🎯 PREDICTION: {pred['prediction']}")
        print(f"  Confidence: {pred['confidence']:.1%}")
        print(f"  Prob UP: {pred['prob_up']:.1%} | Prob DOWN: {pred['prob_down']:.1%}")

        results.append({
            'company': company,
            'ticker': config['ticker'],
            'current_price': price_features['current_price'],
            'avg_sentiment': sentiment_features['avg_sentiment'],
            'prediction': pred['prediction'],
            'confidence': pred['confidence'],
            'prob_up': pred['prob_up'],
            'prob_down': pred['prob_down'],
            'run_date': datetime.now().strftime('%Y-%m-%d')
        })

    # Final summary
    print("\n" + "=" * 60)
    print("PREDICTION SUMMARY")
    print("=" * 60)
    print(f"{'Company':<12} {'Price':>10} {'Sentiment':>12} {'Prediction':>12} {'Confidence':>12}")
    print("-" * 60)
    for r in results:
        print(f"{r['company']:<12} ${r['current_price']:>9.2f} {r['avg_sentiment']:>12.3f} {r['prediction']:>12} {r['confidence']:>11.1%}")

    # Save predictions to CSV
    if results:
        df = pd.DataFrame(results)
        df.to_csv("live_predictions.csv", index=False)
        print(f"\nSaved predictions to live_predictions.csv")

    print("\nDone!")

if __name__ == "__main__":
    run_live_predictions()