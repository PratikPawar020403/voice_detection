"""
Model Evaluation Script for Documentation
Generates:
1. Per-language accuracy table
2. Confusion matrix (saved as image)
3. Calibration reliability curve (saved as image)
4. Latency benchmarks
"""

import os
import sys
import io

# Fix Windows console encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
import time
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.calibration import calibration_curve
from sklearn.model_selection import StratifiedShuffleSplit

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.config import DATA_DIR

MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
DOCS_DIR = os.path.join(os.path.dirname(__file__), '..', 'docs')

if not os.path.exists(DOCS_DIR):
    os.makedirs(DOCS_DIR)

# Language code to name mapping
LANG_NAMES = {
    'en': 'English',
    'ta': 'Tamil',
    'hi': 'Hindi',
    'ml': 'Malayalam',
    'te': 'Telugu'
}

def load_data():
    """Load feature data with language info"""
    dsp_path = os.path.join(DATA_DIR, 'features', 'dsp_features.csv')
    master_path = os.path.join(DATA_DIR, 'master_dataset.csv')
    
    dsp_df = pd.read_csv(dsp_path)
    master_df = pd.read_csv(master_path)
    
    # Merge to get language info
    merged = pd.merge(dsp_df, master_df[['filename', 'language']], on='filename', how='left')
    return merged

def evaluate_model():
    """Main evaluation function"""
    print("=" * 60)
    print("MODEL EVALUATION FOR DOCUMENTATION")
    print("=" * 60)
    
    # Load data
    data = load_data()
    print(f"\nTotal samples: {len(data)}")
    
    # Load model
    model = joblib.load(os.path.join(MODELS_DIR, 'dsp_model.pkl'))
    dsp_cols = joblib.load(os.path.join(MODELS_DIR, 'dsp_cols.pkl'))
    
    # Prepare data
    X = data[dsp_cols].values
    y = (data['label'] == 'ai').astype(int).values
    languages = data['language'].values
    
    # Train/Test Split (same as training)
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(splitter.split(np.zeros(len(y)), y))
    
    X_test = X[test_idx]
    y_test = y[test_idx]
    langs_test = languages[test_idx]
    
    # Predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # 1. OVERALL METRICS
    print("\n" + "=" * 40)
    print("OVERALL PERFORMANCE")
    print("=" * 40)
    overall_acc = accuracy_score(y_test, y_pred)
    print(f"Overall Accuracy: {overall_acc:.4f} ({overall_acc*100:.2f}%)")
    
    # 2. PER-LANGUAGE ACCURACY
    print("\n" + "=" * 40)
    print("PER-LANGUAGE ACCURACY")
    print("=" * 40)
    
    lang_results = []
    for lang_code in ['en', 'ta', 'hi', 'ml', 'te']:
        mask = langs_test == lang_code
        if mask.sum() > 0:
            acc = accuracy_score(y_test[mask], y_pred[mask])
            n_samples = mask.sum()
            n_correct = (y_test[mask] == y_pred[mask]).sum()
            lang_results.append({
                'Language': LANG_NAMES[lang_code],
                'Code': lang_code,
                'Samples': n_samples,
                'Correct': n_correct,
                'Accuracy': acc
            })
            print(f"{LANG_NAMES[lang_code]:12s} ({lang_code}): {acc*100:.1f}% ({n_correct}/{n_samples})")
    
    # Save per-language results
    lang_df = pd.DataFrame(lang_results)
    lang_df.to_csv(os.path.join(DOCS_DIR, 'per_language_accuracy.csv'), index=False)
    
    # 3. CONFUSION MATRIX
    print("\n" + "=" * 40)
    print("CONFUSION MATRIX")
    print("=" * 40)
    
    cm = confusion_matrix(y_test, y_pred)
    print(f"               Predicted")
    print(f"              HUMAN  AI")
    print(f"Actual HUMAN    {cm[0,0]:3d}   {cm[0,1]:3d}")
    print(f"Actual AI       {cm[1,0]:3d}   {cm[1,1]:3d}")
    
    # Plot confusion matrix
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax)
    
    classes = ['HUMAN', 'AI_GENERATED']
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           xlabel='Predicted Label',
           ylabel='True Label',
           title='Confusion Matrix - AI Voice Detection')
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=20)
    
    plt.tight_layout()
    plt.savefig(os.path.join(DOCS_DIR, 'confusion_matrix.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {os.path.join(DOCS_DIR, 'confusion_matrix.png')}")
    
    # 4. CALIBRATION CURVE
    print("\n" + "=" * 40)
    print("CALIBRATION RELIABILITY")
    print("=" * 40)
    
    prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=10)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([0, 1], [0, 1], 'k--', label='Perfectly Calibrated')
    ax.plot(prob_pred, prob_true, 'b-o', label='Model Calibration')
    ax.set_xlabel('Predicted Probability')
    ax.set_ylabel('True Probability')
    ax.set_title('Calibration Reliability Curve')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(DOCS_DIR, 'calibration_curve.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {os.path.join(DOCS_DIR, 'calibration_curve.png')}")
    
    # Calculate calibration error
    ece = np.mean(np.abs(prob_true - prob_pred))
    print(f"Expected Calibration Error (ECE): {ece:.4f}")
    
    # 5. LATENCY BENCHMARKS
    print("\n" + "=" * 40)
    print("LATENCY BENCHMARKS")
    print("=" * 40)
    
    # Benchmark prediction time
    latencies = []
    for _ in range(100):
        sample = X_test[np.random.randint(len(X_test))].reshape(1, -1)
        start = time.perf_counter()
        _ = model.predict_proba(sample)
        latencies.append((time.perf_counter() - start) * 1000)  # ms
    
    latencies = np.array(latencies)
    print(f"Prediction latency (model only):")
    print(f"  Mean: {latencies.mean():.2f} ms")
    print(f"  P50:  {np.percentile(latencies, 50):.2f} ms")
    print(f"  P95:  {np.percentile(latencies, 95):.2f} ms")
    print(f"  P99:  {np.percentile(latencies, 99):.2f} ms")
    
    # Save latency results
    latency_stats = {
        'metric': ['Mean', 'P50', 'P95', 'P99'],
        'latency_ms': [latencies.mean(), np.percentile(latencies, 50), 
                       np.percentile(latencies, 95), np.percentile(latencies, 99)]
    }
    pd.DataFrame(latency_stats).to_csv(os.path.join(DOCS_DIR, 'latency_benchmarks.csv'), index=False)
    
    # 6. GENERATE MARKDOWN SUMMARY
    print("\n" + "=" * 40)
    print("GENERATING MARKDOWN SUMMARY")
    print("=" * 40)
    
    # Calculate precision/recall
    from sklearn.metrics import precision_score, recall_score, f1_score
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    markdown = f"""## 📈 Model Performance Metrics

### Overall Performance

| Metric | Value |
|--------|-------|
| **Overall Accuracy** | {overall_acc*100:.1f}% |
| **Precision** | {precision*100:.1f}% |
| **Recall** | {recall*100:.1f}% |
| **F1 Score** | {f1*100:.1f}% |
| **Test Samples** | {len(y_test)} |

---

### Per-Language Accuracy

| Language | Samples | Accuracy |
|----------|---------|----------|
"""
    for r in lang_results:
        markdown += f"| {r['Language']} | {r['Samples']} | {r['Accuracy']*100:.1f}% |\n"
    
    markdown += f"""
---

### Confusion Matrix

![Confusion Matrix](docs/confusion_matrix.png)

|  | Predicted HUMAN | Predicted AI |
|--|-----------------|--------------|
| **Actual HUMAN** | {cm[0,0]} | {cm[0,1]} |
| **Actual AI** | {cm[1,0]} | {cm[1,1]} |

---

### Calibration Reliability

![Calibration Curve](docs/calibration_curve.png)

- **Expected Calibration Error (ECE)**: {ece:.4f}
- The closer to the diagonal line, the better calibrated the model is

---

### Latency Benchmarks

| Metric | Latency (ms) |
|--------|--------------|
| Mean | {latencies.mean():.2f} |
| P50 (Median) | {np.percentile(latencies, 50):.2f} |
| P95 | {np.percentile(latencies, 95):.2f} |
| P99 | {np.percentile(latencies, 99):.2f} |

> **Note**: These are model-only prediction times. Full API latency includes audio decoding, feature extraction, and network overhead (typically ~500-1500ms total).

---
"""
    
    # Save markdown
    with open(os.path.join(DOCS_DIR, 'performance_metrics.md'), 'w') as f:
        f.write(markdown)
    print(f"Saved: {os.path.join(DOCS_DIR, 'performance_metrics.md')}")
    
    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE!")
    print("=" * 60)
    print(f"\nOutput files in: {DOCS_DIR}")
    print("  - confusion_matrix.png")
    print("  - calibration_curve.png") 
    print("  - per_language_accuracy.csv")
    print("  - latency_benchmarks.csv")
    print("  - performance_metrics.md")
    
    return {
        'overall_accuracy': overall_acc,
        'per_language': lang_results,
        'confusion_matrix': cm,
        'ece': ece,
        'latency_mean': latencies.mean()
    }

if __name__ == "__main__":
    evaluate_model()
