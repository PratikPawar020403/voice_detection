import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.config import DATA_DIR

MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)

def load_data():
    dsp_path = os.path.join(DATA_DIR, 'features', 'dsp_features.csv')
    emb_path = os.path.join(DATA_DIR, 'features', 'embeddings.csv')
    
    if not os.path.exists(dsp_path):
        raise FileNotFoundError("DSP Feature file not found. Run feature extraction first.")
        
    dsp_df = pd.read_csv(dsp_path)
    
    if os.path.exists(emb_path):
        print("Loading Embeddings...")
        emb_df = pd.read_csv(emb_path)
        if 'label' in emb_df.columns:
            emb_df = emb_df.drop(columns=['label'])
        merged_df = pd.merge(dsp_df, emb_df, on='filename', how='inner')
        return merged_df, True
    else:
        print("Embeddings not found. Using DSP features only.")
        return dsp_df, False

def train_dsp_model(X_train, y_train):
    print("Training DSP Classifier (Random Forest)...")
    clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    calibrated_clf = CalibratedClassifierCV(clf, method='sigmoid', cv=5)
    calibrated_clf.fit(X_train, y_train)
    return calibrated_clf

def train_embedding_model(X_train, y_train):
    print("Training Embedding Classifier (XGBoost)...")
    clf = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42, eval_metric='logloss')
    calibrated_clf = CalibratedClassifierCV(clf, method='sigmoid', cv=5)
    calibrated_clf.fit(X_train, y_train)
    return calibrated_clf

def main():
    try:
        data, has_embeddings = load_data()
    except Exception as e:
        print(f"Skipping training for now: {e}")
        return

    print(f"Loaded {len(data)} samples.")
    
    # Prepare Features
    dsp_cols = [c for c in data.columns if c not in ['filename', 'label', 'path'] and not c.startswith('emb_')]
    X_dsp = data[dsp_cols].values
    y = (data['label'] == 'ai').astype(int).values
    
    # Train/Test Split
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(splitter.split(np.zeros(len(y)), y))
    
    X_dsp_train, X_dsp_test = X_dsp[train_idx], X_dsp[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    # 1. Train DSP Model
    dsp_model = train_dsp_model(X_dsp_train, y_train)
    
    # Save DSP Model
    joblib.dump(dsp_model, os.path.join(MODELS_DIR, 'dsp_model.pkl'))
    joblib.dump(dsp_cols, os.path.join(MODELS_DIR, 'dsp_cols.pkl'))
    
    # Eval
    y_pred_dsp = dsp_model.predict(X_dsp_test)
    print(f"DSP Accuracy: {accuracy_score(y_test, y_pred_dsp):.4f}")
    
    emb_model = None
    if has_embeddings:
        emb_cols = [c for c in data.columns if c.startswith('emb_')]
        X_emb = data[emb_cols].values
        X_emb_train, X_emb_test = X_emb[train_idx], X_emb[test_idx]
        
        # 2. Train Embedding Model
        emb_model = train_embedding_model(X_emb_train, y_train)
        
        # Eval
        y_pred_emb = emb_model.predict(X_emb_test)
        print(f"Embedding Accuracy: {accuracy_score(y_test, y_pred_emb):.4f}")
        
        # Ensemble
        y_prob_dsp = dsp_model.predict_proba(X_dsp_test)[:, 1]
        y_prob_emb = emb_model.predict_proba(X_emb_test)[:, 1]
        y_prob_ensemble = (y_prob_dsp + y_prob_emb) / 2
        y_pred_ensemble = (y_prob_ensemble > 0.5).astype(int)
        print(f"Ensemble Accuracy: {accuracy_score(y_test, y_pred_ensemble):.4f}")
        
        joblib.dump(emb_model, os.path.join(MODELS_DIR, 'emb_model.pkl'))
    else:
        print("Skipping Embedding Model training.")
        # Remove old model if exists
        e_path = os.path.join(MODELS_DIR, 'emb_model.pkl')
        if os.path.exists(e_path):
            os.remove(e_path)
            
    print(f"\nModels saved to {MODELS_DIR}")

if __name__ == "__main__":
    main()
