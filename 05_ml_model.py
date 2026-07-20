import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

def load_data():
    print("Loading Financial PhraseBank dataset...")
    df = pd.read_csv("all-data.csv", encoding="latin-1", header=None, names=["sentiment", "text"])
    print(f"Loaded {len(df)} rows")
    print(f"\nSentiment distribution:")
    print(df['sentiment'].value_counts())
    return df

def prepare_features(df):
    print("\nPreparing features with TF-IDF...")
    
    # Convert text to numerical features using TF-IDF
    tfidf = TfidfVectorizer(
        max_features=2000,
        ngram_range=(1, 2),
        stop_words='english'
    )
    
    X = tfidf.fit_transform(df['text'])
    
    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(df['sentiment'])
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Classes: {le.classes_}")
    
    return X, y, tfidf, le

def train_models(X_train, X_test, y_train, y_test, le):
    results = {}
    
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost": xgb.XGBClassifier(n_estimators=100, random_state=42, eval_metric='mlogloss')
    }
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)
        
        results[name] = {
            "model": model,
            "accuracy": accuracy,
            "report": report,
            "confusion_matrix": cm,
            "predictions": y_pred
        }

        # Save the best model and vectorizer
        best_name = max(results, key=lambda x: results[x]['accuracy'])
        best_model = results[best_name]['model']
        joblib.dump(best_model, 'sentiment_model.pkl')
        joblib.dump(tfidf, 'tfidf_vectorizer.pkl')
        joblib.dump(le, 'label_encoder.pkl')
        print(f"\nSaved best sentiment model ({best_name}) to sentiment_model.pkl")
        
        print(f"  Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
    
    return results

def plot_results(results, le):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Financial Sentiment Model Comparison", fontsize=16, fontweight='bold')
    
    model_names = list(results.keys())
    accuracies = [results[m]['accuracy'] for m in model_names]
    
    # Accuracy comparison bar chart
    ax = axes[0, 0]
    bars = ax.bar(model_names, accuracies, color=['#3b82f6', '#22c55e', '#f97316'])
    ax.set_title("Model Accuracy Comparison")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)
    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{acc:.1%}', ha='center', fontweight='bold')
    
    # Confusion matrices
    positions = [(0,1), (0,2), (1,0)]
    for i, (name, pos) in enumerate(zip(model_names, positions)):
        ax = axes[pos]
        cm = results[name]['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=le.classes_, yticklabels=le.classes_)
        ax.set_title(f"{name}\nConfusion Matrix")
        ax.set_ylabel("Actual")
        ax.set_xlabel("Predicted")
    
    # F1 scores comparison
    ax = axes[1, 1]
    classes = le.classes_
    x = np.arange(len(classes))
    width = 0.25
    
    for i, name in enumerate(model_names):
        f1_scores = [results[name]['report'][c]['f1-score'] for c in classes]
        ax.bar(x + i*width, f1_scores, width, label=name,
               color=['#3b82f6', '#22c55e', '#f97316'][i])
    
    ax.set_title("F1 Score by Sentiment Class")
    ax.set_ylabel("F1 Score")
    ax.set_xticks(x + width)
    ax.set_xticklabels(classes)
    ax.legend()
    
    # Summary table
    ax = axes[1, 2]
    ax.axis('off')
    table_data = []
    for name in model_names:
        r = results[name]['report']
        table_data.append([
            name.replace(" ", "\n"),
            f"{results[name]['accuracy']:.1%}",
            f"{r['macro avg']['precision']:.3f}",
            f"{r['macro avg']['recall']:.3f}",
            f"{r['macro avg']['f1-score']:.3f}"
        ])
    
    table = ax.table(
        cellText=table_data,
        colLabels=["Model", "Accuracy", "Precision", "Recall", "F1"],
        cellLoc='center',
        loc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 2)
    ax.set_title("Summary Metrics", pad=20)
    
    plt.tight_layout()
    plt.savefig("model_comparison.png", dpi=150, bbox_inches='tight')
    print("\nSaved model_comparison.png")
    plt.show()

def print_summary(results):
    print("\n" + "=" * 60)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 60)
    print(f"{'Model':<25} {'Accuracy':<12} {'F1 (macro)':<12}")
    print("-" * 49)
    for name, r in results.items():
        acc = r['accuracy']
        f1 = r['report']['macro avg']['f1-score']
        print(f"{name:<25} {acc:.1%}{'':>5} {f1:.3f}")
    
    best = max(results, key=lambda x: results[x]['accuracy'])
    print(f"\nBest model: {best} ({results[best]['accuracy']:.1%} accuracy)")

if __name__ == "__main__":
    # Load data
    df = load_data()
    
    # Prepare features
    X, y, tfidf, le = prepare_features(df)
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nTrain set: {X_train.shape[0]} rows")
    print(f"Test set: {X_test.shape[0]} rows")
    
    # Train and evaluate all 3 models
    results = train_models(X_train, X_test, y_train, y_test, le)
    
    # Print summary
    print_summary(results)
    
    # Plot results
    print("\nGenerating comparison charts...")
    plot_results(results, le)
    
    print("\nDone!")