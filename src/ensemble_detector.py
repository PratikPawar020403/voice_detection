import os
import sys
import torch
import torch.nn as nn
import librosa
import numpy as np
import pandas as pd
import joblib
from transformers import AutoFeatureExtractor, AutoModel

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, "..")
sys.path.append(project_root)

from src.features.extract_dsp_v2 import extract_all_features_v2

# ============================================================
# Neural Model Architecture definition
# ============================================================
class AudioClassifier(nn.Module):
    def __init__(self, encoder, hidden_size):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 2),
        )

    def forward(self, input_values):
        outputs = self.encoder(input_values)
        hidden = outputs.last_hidden_state.mean(dim=1)
        logits = self.classifier(hidden)
        return logits

# ============================================================
# Ensemble Detector Class
# ============================================================
class EnsembleDetector:
    def __init__(self, neural_model_path, dsp_model_path, dsp_cols_path, device="cpu"):
        self.device = torch.device(device)
        print(f"Initializing Ensemble Detector on {self.device}...")
        
        # Load Neural Model
        print("1. Loading Neural Model (wav2vec2)...")
        base_model = "facebook/wav2vec2-base"
        self.processor = AutoFeatureExtractor.from_pretrained(base_model)
        encoder = AutoModel.from_pretrained(base_model)
        self.neural_model = AudioClassifier(encoder, encoder.config.hidden_size)
        self.neural_model.load_state_dict(torch.load(neural_model_path, map_location=self.device))
        self.neural_model.to(self.device)
        self.neural_model.eval()
        
        # Load DSP Model
        print("2. Loading DSP Model (Random Forest v2)...")
        self.dsp_model = joblib.load(dsp_model_path)
        self.dsp_cols = joblib.load(dsp_cols_path)
        
        print("Ensemble Ready!\n")

    def predict_neural(self, audio_path):
        """Get prediction probability from Neural Model"""
        sr = 16000
        max_len = sr * 5
        
        y, _ = librosa.load(audio_path, sr=sr, mono=True)
        if len(y) > max_len:
            y = y[:max_len]
        elif len(y) < max_len:
            y = np.pad(y, (0, max_len - len(y)), mode='constant')
            
        waveform = torch.FloatTensor(y).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            logits = self.neural_model(waveform)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
            
        return probs[1] # Return probability of AI

    def predict_dsp(self, audio_path):
        """Get prediction probability from DSP Model"""
        # Extract features
        features = extract_all_features_v2(audio_path)
        if features is None:
            return 0.5 # Default to uncertain if extraction fails
            
        # Convert to DataFrame
        df = pd.DataFrame([features])
        
        # Keep only the features that the model was trained on
        X = pd.DataFrame(0, index=np.arange(1), columns=self.dsp_cols)
        for col in self.dsp_cols:
            if col in df.columns:
                X[col] = df[col]
                
        # Predict probability
        probs = self.dsp_model.predict_proba(X)[0]
        # Assuming class 1 is AI. The classes_ attribute usually helps but let's assume index 1 is 'ai'
        # Let's check classes_ if we can. Usually ['human', 'ai'] or [0, 1].
        # In extract_dsp_v2.py, labels are 'human' and 'ai'. Alphabetical: 'ai' is 0, 'human' is 1.
        # Wait, if AI is 0, then probs[0] is AI. Let's dynamically check:
        if hasattr(self.dsp_model, 'classes_'):
            classes = list(self.dsp_model.classes_)
            if 'ai' in classes:
                ai_idx = classes.index('ai')
            elif 1 in classes: # fallback
                ai_idx = 1
            else:
                ai_idx = 0
            return probs[ai_idx]
        
        return probs[1]

    def predict(self, audio_path):
        """
        Combine both models using a confidence-based routing strategy.
        """
        neural_ai_prob = self.predict_neural(audio_path)
        dsp_ai_prob = self.predict_dsp(audio_path)
        
        # Calculate confidences (distance from 0.5)
        neural_conf = abs(neural_ai_prob - 0.5) * 2
        dsp_conf = abs(dsp_ai_prob - 0.5) * 2
        
        # Weighted Ensemble Logic
        # We generally trust Neural more, but if Neural is uncertain and DSP is confident, we blend.
        if neural_conf > 0.90:
            # Neural is highly confident (>95% or <5%), trust it 90%
            final_prob = (0.9 * neural_ai_prob) + (0.1 * dsp_ai_prob)
            reason = "High Neural Confidence"
        elif dsp_conf > 0.80 and neural_conf < 0.50:
            # Neural is uncertain, but DSP found strong evidence
            final_prob = (0.4 * neural_ai_prob) + (0.6 * dsp_ai_prob)
            reason = "DSP Overrode Neural Uncertainty"
        else:
            # Standard blend
            final_prob = (0.7 * neural_ai_prob) + (0.3 * dsp_ai_prob)
            reason = "Standard Blend"
            
        prediction = "AI" if final_prob > 0.5 else "Human"
        
        return {
            "prediction": prediction,
            "confidence": float(abs(final_prob - 0.5) * 2), # 0 to 1 confidence scale
            "final_ai_prob": float(final_prob),
            "neural_ai_prob": float(neural_ai_prob),
            "dsp_ai_prob": float(dsp_ai_prob),
            "routing_reason": reason
        }

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ensemble_detector.py <path_to_audio>")
        sys.exit(1)
        
    audio_file = sys.argv[1]
    
    # Paths
    NEURAL_PATH = os.path.join(project_root, "voice_detection_v2", "voice_detector_neural.pt")
    DSP_MODEL_PATH = os.path.join(project_root, "models", "dsp_model_v2.pkl")
    DSP_COLS_PATH = os.path.join(project_root, "models", "dsp_cols_v2.pkl")
    
    detector = EnsembleDetector(NEURAL_PATH, DSP_MODEL_PATH, DSP_COLS_PATH)
    
    print(f"Analyzing: {os.path.basename(audio_file)}")
    result = detector.predict(audio_file)
    
    print("\n" + "="*50)
    print("ENSEMBLE RESULTS")
    print("="*50)
    print(f"Final Prediction : {result['prediction']}")
    print(f"Confidence       : {result['confidence']:.2%}")
    print(f"Routing Reason   : {result['routing_reason']}")
    print("-" * 50)
    print(f"Final AI Prob    : {result['final_ai_prob']:.2%}")
    print(f"Neural AI Prob   : {result['neural_ai_prob']:.2%}")
    print(f"DSP AI Prob      : {result['dsp_ai_prob']:.2%}")
    print("="*50)
