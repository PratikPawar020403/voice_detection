import os
import tempfile
import base64
import io
from src.ensemble_detector import EnsembleDetector

# Adjust paths as needed
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(BASE_DIR))

NEURAL_PATH = os.path.join(PROJECT_ROOT, "voice_detection_v2", "voice_detector_neural.pt")
DSP_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "dsp_model_v2.pkl")
DSP_COLS_PATH = os.path.join(PROJECT_ROOT, "models", "dsp_cols_v2.pkl")

# Global ensemble detector instance
_detector = None

def load_resources():
    global _detector
    if _detector is None:
        print("Loading v2 Ensemble Resources...")
        _detector = EnsembleDetector(NEURAL_PATH, DSP_MODEL_PATH, DSP_COLS_PATH)
        print("v2 Ensemble loaded successfully!")

def ensure_resources():
    if _detector is None:
        load_resources()

def predict_pipeline(audio_bytes):
    ensure_resources()
    
    # Write bytes to temporary file for EnsembleDetector
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name
        
    try:
        # Run v2 Ensemble Prediction
        ensemble_result = _detector.predict(tmp_path)
        
        # Map Ensemble output to expected API format
        result_label = "AI_GENERATED" if ensemble_result['prediction'] == "AI" else "HUMAN"
        
        # Improved Explanation Logic
        if result_label == "AI_GENERATED":
            explanation = f"Audio is flagged as AI-generated. "
            if ensemble_result['neural_ai_prob'] > 0.8:
                explanation += "Deep neural representations strongly match known synthetic voice models. "
            if ensemble_result['dsp_ai_prob'] > 0.8:
                explanation += "Acoustic features (like micro-tremors and spectral flatness) lack natural human variation. "
        else:
            explanation = f"Audio appears to be natural Human speech. "
            if ensemble_result['neural_ai_prob'] < 0.2:
                explanation += "Neural characteristics align smoothly with authentic speech recordings. "
            if ensemble_result['dsp_ai_prob'] < 0.2:
                explanation += "Vocal tract features, breathing patterns, and pitch variations are consistent with human biology. "
                
        explanation += f"(Primary Decision Driver: {ensemble_result['routing_reason']})"
        
        return {
            "result": result_label,
            "confidence": ensemble_result['confidence'],
            "explanation": explanation,
            "details": {
                "final_ai_prob": ensemble_result['final_ai_prob'],
                "neural_ai_prob": ensemble_result['neural_ai_prob'],
                "dsp_ai_prob": ensemble_result['dsp_ai_prob']
            }
        }
    finally:
        try:
            os.remove(tmp_path)
        except:
            pass
