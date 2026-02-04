import os
import joblib
import numpy as np
import librosa
import torch
import base64
import io
import soundfile as sf
# from transformers import Wav2Vec2Model # Lazy import instead
from src.api.lid import identify_language

# Adjust paths as needed
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(BASE_DIR))
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')

# Load Resources (Global for caching)
_dsp_model = None
_emb_model = None
_dsp_cols = None
_processor = None
_wav2vec = None
_device = None

def load_resources():
    global _dsp_model, _emb_model, _dsp_cols, _processor, _wav2vec, _device
    
    if _dsp_model is not None:
        return

    print("Loading models...")
    models_found = False
    
    # DSP Model (Core)
    if os.path.exists(os.path.join(MODELS_DIR, 'dsp_model.pkl')):
        _dsp_model = joblib.load(os.path.join(MODELS_DIR, 'dsp_model.pkl'))
        _dsp_cols = joblib.load(os.path.join(MODELS_DIR, 'dsp_cols.pkl'))
        models_found = True
    
    # Embedding Model (Optional)
    if os.path.exists(os.path.join(MODELS_DIR, 'emb_model.pkl')):
        try:
            _emb_model = joblib.load(os.path.join(MODELS_DIR, 'emb_model.pkl'))
            
            # Load Wav2Vec2 only if we have the embedding model
            model_id = "facebook/wav2vec2-large-xlsr-53"
            # Try loading processor with fallback
            from transformers import AutoFeatureExtractor, Wav2Vec2Model
            _processor = AutoFeatureExtractor.from_pretrained(model_id)
            _wav2vec = Wav2Vec2Model.from_pretrained(model_id)
            _device = "cuda" if torch.cuda.is_available() else "cpu"
            _wav2vec.to(_device)
            print("Embedding model resources loaded.")
        except Exception as e:
            print(f"Failed to load embedding resources: {e}")
            _emb_model = None
            _processor = None
            _wav2vec = None
    
    if not models_found:
        print("Models not found. Inference will fail.")

def extract_dsp_features_single(audio_array, sr):
    # This must match training logic EXACTLY
    
    # ... (Keep existing logic)
    y = audio_array
    features = {}
    
    # 1. MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    features['mfcc_mean'] = np.mean(mfcc)
    features['mfcc_var'] = np.var(mfcc)
    for i in range(1, 14):
        features[f'mfcc_{i}_mean'] = np.mean(mfcc[i-1])
        features[f'mfcc_{i}_var'] = np.var(mfcc[i-1])

    # 2. Spectral
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    features['spec_cent_mean'] = np.mean(spectral_centroid)
    features['spec_cent_var'] = np.var(spectral_centroid)
    
    spectral_flatness = librosa.feature.spectral_flatness(y=y)
    features['spec_flat_mean'] = np.mean(spectral_flatness)
    features['spec_flat_var'] = np.var(spectral_flatness)
    
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    features['spec_roll_mean'] = np.mean(spectral_rolloff)
    
    # 3. RMS
    rms = librosa.feature.rms(y=y)
    features['rms_mean'] = np.mean(rms)
    features['rms_var'] = np.var(rms)
    
    # 4. ZCR
    zcr = librosa.feature.zero_crossing_rate(y)
    features['zcr_mean'] = np.mean(zcr)
    features['zcr_var'] = np.var(zcr)
    
    # 5. Chroma
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    features['chroma_mean'] = np.mean(chroma)
    
    # 6. Pitch
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitches_filtered = pitches[magnitudes > np.median(magnitudes)]
    if len(pitches_filtered) > 0:
        features['pitch_mean'] = np.mean(pitches_filtered)
        features['pitch_std'] = np.std(pitches_filtered)
    else:
        features['pitch_mean'] = 0
        features['pitch_std'] = 0
        
    return features

def extract_embedding_single(audio_array, sr):
    if _processor is None or _wav2vec is None:
        return None
        
    # Resample to 16k if needed (Wav2Vec2 requirement)
    if sr != 16000:
        audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=16000)
        
    inputs = _processor(audio_array, sampling_rate=16000, return_tensors="pt", padding=True)
    input_values = inputs.input_values.to(_device)
    
    with torch.no_grad():
        outputs = _wav2vec(input_values)
        hidden_states = outputs.last_hidden_state
        pooled_output = torch.mean(hidden_states, dim=1)
        
    return pooled_output.cpu().numpy().flatten()

def predict_pipeline(audio_bytes):
    ensure_resources()
    
    if _dsp_model is None:
        return {"result": "ERROR", "confidence": 0, "explanation": "Model not loaded"}
    
    # 1. Decode Audio
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name
    
    try:
        y, sr = librosa.load(tmp_path, sr=16000)
        
        # Run LID
        detected_lang = identify_language(tmp_path)
    finally:
        try:
            os.remove(tmp_path)
        except:
            pass
    
    # 2. Extract Features
    dsp_feats = extract_dsp_features_single(y, sr)
    
    # Prepare DFs for models
    import pandas as pd
    dsp_df = pd.DataFrame([dsp_feats])
    # Ensure columns match training order using _dsp_cols
    dsp_df = dsp_df.reindex(columns=_dsp_cols, fill_value=0)
    
    # 3. Predict DSP
    prob_dsp = _dsp_model.predict_proba(dsp_df)[0][1]
    
    prob_emb = None
    emb_feats = None
    
    if _emb_model is not None:
        try:
            emb_feats = extract_embedding_single(y, sr)
            if emb_feats is not None:
                emb_df = pd.DataFrame([emb_feats], columns=[f'emb_{i}' for i in range(len(emb_feats))])
                prob_emb = _emb_model.predict_proba(emb_df)[0][1]
        except Exception as e:
            print(f"Embedding inference failed: {e}")
    
    # Ensemble
    if prob_emb is not None:
        prob_ensemble = (prob_dsp + prob_emb) / 2
    else:
        prob_ensemble = prob_dsp
    
    result = "AI_GENERATED" if prob_ensemble > 0.5 else "HUMAN"
    
    # Explanation
    explanation = "Audio shows consistency with human speech patterns."
    if result == "AI_GENERATED":
        explanation = f"Detected synthetic signatures in spectral flatness ({dsp_feats.get('spec_flat_mean',0):.2f}) and pitch stability."
        
    return {
        "result": result,
        "confidence": float(prob_ensemble) if result == "AI_GENERATED" else float(1 - prob_ensemble),
        "explanation": explanation,
        "detected_language": detected_lang,
        "details": {
            "dsp_prob": float(prob_dsp),
            "emb_prob": float(prob_emb) if prob_emb is not None else -1
        }
    }

def ensure_resources():
    if _dsp_model is None:
        load_resources()
