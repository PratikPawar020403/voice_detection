"""
DSP Classifier v2 — Training Pipeline
=======================================
Extracts v2 features from raw audio, trains multiple classifiers,
selects the best, calibrates probabilities, and saves the model.

Run: python src/train_dsp_v2.py
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
import joblib
import warnings
from pathlib import Path
from tqdm import tqdm

from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, roc_auc_score, precision_score, recall_score
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.config import DATA_DIR, SAMPLE_RATE
from src.features.extract_dsp_v2 import extract_all_features_v2

# Try importing optional dependencies
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

try:
    import optuna
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# Step 1: Scan raw data and extract v2 features
# ============================================================

def scan_raw_data():
    """Scan raw data directories for audio files with proper labels."""
    raw_human = os.path.join(DATA_DIR, 'raw', 'human')
    raw_ai = os.path.join(DATA_DIR, 'raw', 'ai')

    records = []

    # Scan human files
    if os.path.exists(raw_human):
        for root, dirs, files in os.walk(raw_human):
            for f in files:
                if f.endswith(('.flac', '.wav', '.mp3', '.ogg')):
                    full_path = os.path.join(root, f)
                    # Extract language from filename or path
                    lang = 'unknown'
                    for code in ['en', 'ta', 'hi', 'ml', 'te']:
                        if f'_{code}_' in f or f'/{code}/' in root or f'\\{code}\\' in root:
                            lang = code
                            break
                    records.append({
                        'path': full_path,
                        'filename': f,
                        'label': 'human',
                        'language': lang,
                    })

    # Scan AI files
    if os.path.exists(raw_ai):
        for root, dirs, files in os.walk(raw_ai):
            for f in files:
                if f.endswith(('.flac', '.wav', '.mp3', '.ogg')):
                    full_path = os.path.join(root, f)
                    lang = 'unknown'
                    for code in ['en', 'ta', 'hi', 'ml', 'te']:
                        if f'_{code}_' in f or f'/{code}/' in root or f'\\{code}\\' in root:
                            lang = code
                            break

                    # Detect TTS source from filename
                    source = 'unknown'
                    if 'edge' in f.lower():
                        source = 'edge_tts'
                    elif 'gtts' in f.lower():
                        source = 'gtts'

                    records.append({
                        'path': full_path,
                        'filename': f,
                        'label': 'ai',
                        'language': lang,
                        'source': source,
                    })

    df = pd.DataFrame(records)
    print(f"Found {len(df)} raw audio files:")
    print(f"  Human: {(df['label']=='human').sum()}")
    print(f"  AI:    {(df['label']=='ai').sum()}")
    print(f"  Languages: {df['language'].value_counts().to_dict()}")
    return df


def extract_features_batch(df, cache_path=None):
    """Extract v2 DSP features from all audio files."""

    # Check cache
    if cache_path and os.path.exists(cache_path):
        print(f"Loading cached features from {cache_path}")
        return pd.read_csv(cache_path)

    feature_list = []
    failed = []

    print(f"\nExtracting v2 DSP features from {len(df)} files...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        features = extract_all_features_v2(row['path'])
        if features is not None:
            features['filename'] = row['filename']
            features['label'] = row['label']
            features['language'] = row['language']
            feature_list.append(features)
        else:
            failed.append(row['path'])

    feature_df = pd.DataFrame(feature_list)

    print(f"  Extracted: {len(feature_list)} / {len(df)}")
    print(f"  Failed: {len(failed)}")

    # Cache results
    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        feature_df.to_csv(cache_path, index=False)
        print(f"  Cached to: {cache_path}")

    return feature_df


# ============================================================
# Step 2: Prepare data for training
# ============================================================

def prepare_data(feature_df):
    """Split features into X, y and train/test sets."""
    # Feature columns = everything except metadata
    meta_cols = ['filename', 'label', 'language', 'source']
    feature_cols = [c for c in feature_df.columns if c not in meta_cols]

    X = feature_df[feature_cols].values
    y = (feature_df['label'] == 'ai').astype(int).values
    languages = feature_df['language'].values

    # Replace any remaining NaN/Inf
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # Stratified split
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(splitter.split(X, y))

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    langs_test = languages[test_idx]

    print(f"\nData split:")
    print(f"  Train: {len(X_train)} (human={sum(y_train==0)}, ai={sum(y_train==1)})")
    print(f"  Test:  {len(X_test)} (human={sum(y_test==0)}, ai={sum(y_test==1)})")
    print(f"  Features: {X_train.shape[1]}")

    return X_train, X_test, y_train, y_test, feature_cols, langs_test


# ============================================================
# Step 3: Train and compare classifiers
# ============================================================

def train_classifiers(X_train, y_train, X_test, y_test):
    """Train multiple classifiers and return results."""

    # Calculate class weights for imbalanced data
    n_human = sum(y_train == 0)
    n_ai = sum(y_train == 1)
    scale_pos_weight = n_human / max(n_ai, 1)
    class_weight = {0: 1.0, 1: scale_pos_weight}

    print(f"\nClass weight for AI: {scale_pos_weight:.2f} (compensating for {n_human} human vs {n_ai} ai)")

    candidates = {}

    # 1. Random Forest (with class weight)
    print("\n--- Random Forest ---")
    rf = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight=class_weight,
            random_state=42,
            n_jobs=-1
        ))
    ])
    rf.fit(X_train, y_train)
    candidates['RandomForest'] = rf

    # 2. Gradient Boosting
    print("--- Gradient Boosting ---")
    gb = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', GradientBoostingClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            min_samples_split=5,
            subsample=0.8,
            random_state=42
        ))
    ])
    gb.fit(X_train, y_train)
    candidates['GradientBoosting'] = gb

    # 3. XGBoost (if available)
    if HAS_XGB:
        print("--- XGBoost ---")
        xgb_clf = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', xgb.XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                scale_pos_weight=scale_pos_weight,
                eval_metric='logloss',
                random_state=42,
                n_jobs=-1
            ))
        ])
        xgb_clf.fit(X_train, y_train)
        candidates['XGBoost'] = xgb_clf

    # 4. LightGBM (if available)
    if HAS_LGB:
        print("--- LightGBM ---")
        lgb_clf = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', lgb.LGBMClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                scale_pos_weight=scale_pos_weight,
                random_state=42,
                n_jobs=-1,
                verbose=-1
            ))
        ])
        lgb_clf.fit(X_train, y_train)
        candidates['LightGBM'] = lgb_clf

    # Evaluate all
    results = []
    for name, model in candidates.items():
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob) if len(np.unique(y_test)) > 1 else 0
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)

        results.append({
            'model': name,
            'accuracy': acc,
            'f1': f1,
            'auc': auc,
            'precision': prec,
            'recall': rec,
        })
        print(f"  {name:20s} | Acc: {acc:.4f} | F1: {f1:.4f} | AUC: {auc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f}")

    results_df = pd.DataFrame(results)

    # Pick best by F1
    best_name = results_df.loc[results_df['f1'].idxmax(), 'model']
    best_model = candidates[best_name]

    print(f"\n* Best model: {best_name} (F1: {results_df.loc[results_df['f1'].idxmax(), 'f1']:.4f})")

    return best_model, best_name, candidates, results_df


# ============================================================
# Step 4: Hyperparameter tuning (optional, if Optuna available)
# ============================================================

def tune_best_model(X_train, y_train, X_test, y_test, best_name):
    """Tune the best model with Optuna."""
    if not HAS_OPTUNA:
        print("Optuna not available, skipping hyperparameter tuning.")
        return None

    print(f"\nTuning {best_name} with Optuna (30 trials)...")

    n_human = sum(y_train == 0)
    n_ai = sum(y_train == 1)
    scale_pos_weight = n_human / max(n_ai, 1)

    def objective(trial):
        if best_name in ('XGBoost', 'LightGBM', 'GradientBoosting'):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            }

            if best_name == 'XGBoost' and HAS_XGB:
                clf = xgb.XGBClassifier(
                    **params,
                    scale_pos_weight=scale_pos_weight,
                    eval_metric='logloss',
                    random_state=42,
                    n_jobs=-1
                )
            elif best_name == 'LightGBM' and HAS_LGB:
                clf = lgb.LGBMClassifier(
                    **{k: v for k, v in params.items() if k != 'min_child_weight'},
                    scale_pos_weight=scale_pos_weight,
                    random_state=42,
                    n_jobs=-1,
                    verbose=-1
                )
            else:
                clf = GradientBoostingClassifier(
                    n_estimators=params['n_estimators'],
                    max_depth=params['max_depth'],
                    learning_rate=params['learning_rate'],
                    subsample=params['subsample'],
                    random_state=42
                )
        else:
            # Random Forest
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'max_depth': trial.suggest_int('max_depth', 5, 25),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 15),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 8),
            }
            clf = RandomForestClassifier(
                **params,
                class_weight={0: 1.0, 1: scale_pos_weight},
                random_state=42,
                n_jobs=-1
            )

        pipe = Pipeline([('scaler', StandardScaler()), ('clf', clf)])

        # Cross-validation on training data
        scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring='f1', n_jobs=-1)
        return scores.mean()

    # Suppress Optuna logs
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=30, show_progress_bar=True)

    print(f"  Best trial F1: {study.best_value:.4f}")
    print(f"  Best params: {study.best_params}")

    return study.best_params


# ============================================================
# Step 5: Calibrate and save
# ============================================================

def calibrate_and_save(best_model, best_name, X_train, y_train, X_test, y_test,
                       feature_cols, langs_test, results_df, tuned_params=None):
    """Calibrate the best model's probabilities and save everything."""

    # If we have tuned params, retrain with them
    if tuned_params is not None:
        print(f"\nRetraining {best_name} with tuned parameters...")
        n_human = sum(y_train == 0)
        n_ai = sum(y_train == 1)
        scale_pos_weight = n_human / max(n_ai, 1)

        if best_name == 'XGBoost' and HAS_XGB:
            clf = xgb.XGBClassifier(
                **tuned_params, scale_pos_weight=scale_pos_weight,
                eval_metric='logloss', random_state=42, n_jobs=-1
            )
        elif best_name == 'LightGBM' and HAS_LGB:
            clf = lgb.LGBMClassifier(
                **tuned_params, scale_pos_weight=scale_pos_weight,
                random_state=42, n_jobs=-1, verbose=-1
            )
        elif best_name == 'RandomForest':
            clf = RandomForestClassifier(
                **tuned_params, class_weight={0: 1.0, 1: scale_pos_weight},
                random_state=42, n_jobs=-1
            )
        else:
            clf = GradientBoostingClassifier(**tuned_params, random_state=42)

        best_model = Pipeline([('scaler', StandardScaler()), ('clf', clf)])
        best_model.fit(X_train, y_train)

    # Calibrate probabilities
    print("\nCalibrating probabilities (isotonic)...")
    calibrated = CalibratedClassifierCV(best_model, method='isotonic', cv=5)
    calibrated.fit(X_train, y_train)

    # Final evaluation
    y_pred = calibrated.predict(X_test)
    y_prob = calibrated.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob) if len(np.unique(y_test)) > 1 else 0

    print(f"\n{'='*60}")
    print(f"FINAL CALIBRATED MODEL RESULTS")
    print(f"{'='*60}")
    print(f"  Model:     {best_name} (calibrated)")
    print(f"  Accuracy:  {acc:.4f} ({acc*100:.1f}%)")
    print(f"  F1:        {f1:.4f}")
    print(f"  AUC:       {auc:.4f}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['HUMAN', 'AI'])}")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"Confusion Matrix:")
    print(f"               Predicted")
    print(f"              HUMAN  AI")
    print(f"Actual HUMAN    {cm[0,0]:3d}   {cm[0,1]:3d}")
    print(f"Actual AI       {cm[1,0]:3d}   {cm[1,1]:3d}")

    # Per-language accuracy
    lang_names = {'en': 'English', 'ta': 'Tamil', 'hi': 'Hindi', 'ml': 'Malayalam', 'te': 'Telugu'}
    print(f"\nPer-Language Accuracy:")
    for lang_code in ['en', 'ta', 'hi', 'ml', 'te']:
        mask = langs_test == lang_code
        if mask.sum() > 0:
            lang_acc = accuracy_score(y_test[mask], y_pred[mask])
            print(f"  {lang_names.get(lang_code, lang_code):12s}: {lang_acc*100:.1f}% ({mask.sum()} samples)")

    # Save model
    model_path = os.path.join(MODELS_DIR, 'dsp_model_v2.pkl')
    cols_path = os.path.join(MODELS_DIR, 'dsp_cols_v2.pkl')

    joblib.dump(calibrated, model_path)
    joblib.dump(feature_cols, cols_path)

    print(f"\n  Model saved: {model_path}")
    print(f"  Columns saved: {cols_path}")

    # Save results
    results_path = os.path.join(RESULTS_DIR, 'dsp_v2_results.json')
    results_data = {
        'model': best_name,
        'accuracy': float(acc),
        'f1': float(f1),
        'auc': float(auc),
        'n_features': len(feature_cols),
        'n_train': len(y_train),
        'n_test': len(y_test),
        'confusion_matrix': cm.tolist(),
        'feature_columns': feature_cols,
    }
    with open(results_path, 'w') as f:
        json.dump(results_data, f, indent=2)

    # Save comparison results
    results_df.to_csv(os.path.join(RESULTS_DIR, 'dsp_v2_model_comparison.csv'), index=False)

    print(f"  Results saved: {results_path}")

    return calibrated


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 60)
    print("DSP CLASSIFIER v2 — TRAINING PIPELINE")
    print("=" * 60)

    # Step 1: Scan raw data
    print("\n[1/5] Scanning raw data...")
    raw_df = scan_raw_data()

    if len(raw_df) == 0:
        print("No audio files found! Make sure data/raw/ has human/ and ai/ subdirs.")
        return

    # Step 2: Extract v2 features (with caching)
    print("\n[2/5] Extracting v2 DSP features...")
    cache_path = os.path.join(DATA_DIR, 'features', 'dsp_features_v2.csv')
    feature_df = extract_features_batch(raw_df, cache_path=cache_path)

    # Step 3: Prepare data
    print("\n[3/5] Preparing train/test split...")
    X_train, X_test, y_train, y_test, feature_cols, langs_test = prepare_data(feature_df)

    # Step 4: Train and compare classifiers
    print("\n[4/5] Training classifiers...")
    best_model, best_name, all_models, results_df = train_classifiers(
        X_train, y_train, X_test, y_test
    )

    # Step 4b: Hyperparameter tuning (optional)
    tuned_params = tune_best_model(X_train, y_train, X_test, y_test, best_name)

    # Step 5: Calibrate and save
    print("\n[5/5] Calibrating and saving...")
    final_model = calibrate_and_save(
        best_model, best_name, X_train, y_train, X_test, y_test,
        feature_cols, langs_test, results_df, tuned_params
    )

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
