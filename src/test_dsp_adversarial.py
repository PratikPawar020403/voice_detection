"""
Stress test DSP model v2 on ElevenLabs samples.
"""
import os
import sys
import glob
import pandas as pd
import joblib

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.features.extract_dsp_v2 import extract_all_features_v2

def main():
    model_path = 'models/dsp_model_v2.pkl'
    cols_path = 'models/dsp_cols_v2.pkl'
    
    if not os.path.exists(model_path):
        print("Model not found.")
        return
        
    model = joblib.load(model_path)
    feature_cols = joblib.load(cols_path)
    
    test_files = glob.glob('data/ElevenLabs*.mp3') + glob.glob('data/voice_preview*.mp3')
    
    if not test_files:
        print("No test files found.")
        return
        
    print(f"Testing {len(test_files)} high-quality AI samples (ElevenLabs)...\n")
    
    results = []
    
    for f in test_files:
        feats = extract_all_features_v2(f)
        if feats is None:
            continue
            
        df = pd.DataFrame([feats])
        X = df[feature_cols].values
        
        pred = model.predict(X)[0]
        prob = model.predict_proba(X)[0]
        
        # 1 is AI, 0 is HUMAN
        predicted_class = "AI" if pred == 1 else "HUMAN"
        ai_prob = prob[1]
        
        name = os.path.basename(f)[:30] + "..."
        results.append({
            'file': name,
            'prediction': predicted_class,
            'confidence_ai': ai_prob
        })
        
        marker = "CORRECT" if pred == 1 else "FAILED (Missed AI)"
        print(f"{name:35s} -> Predicted: {predicted_class:5s} (AI Conf: {ai_prob:.2f}) {marker}")
        
    correct = sum(1 for r in results if r['prediction'] == 'AI')
    total = len(results)
    
    print("\n" + "="*50)
    print(f"Adversarial Accuracy: {correct}/{total} ({correct/total*100:.1f}%)")
    print("="*50)

if __name__ == "__main__":
    main()
