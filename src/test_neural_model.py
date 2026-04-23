import os
import sys
import torch
import torch.nn as nn
import torchaudio
import librosa
import numpy as np
from transformers import AutoFeatureExtractor, AutoModel

# Recreate the architecture used during training
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

def load_neural_model(model_path, base_model="facebook/wav2vec2-base", device="cpu"):
    """Load the trained model weights."""
    print(f"Loading processor and base model from {base_model}...")
    processor = AutoFeatureExtractor.from_pretrained(base_model)
    encoder = AutoModel.from_pretrained(base_model)
    
    hidden_size = encoder.config.hidden_size
    model = AudioClassifier(encoder, hidden_size)
    
    print(f"Loading custom weights from {model_path}...")
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    
    model.to(device)
    model.eval()
    return processor, model

def predict_audio(audio_path, processor, model, device="cpu"):
    """Run inference on a single audio file."""
    print(f"Processing audio: {audio_path}")
    
    # Same preprocessing as training (16kHz, max 5 seconds)
    sr = 16000
    max_len = sr * 5
    
    y, _ = librosa.load(audio_path, sr=sr, mono=True)
    
    # Pad or trim
    if len(y) > max_len:
        y = y[:max_len]
    elif len(y) < max_len:
        y = np.pad(y, (0, max_len - len(y)), mode='constant')
        
    waveform = torch.FloatTensor(y).unsqueeze(0).to(device)
    
    with torch.no_grad():
        logits = model(waveform)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
        
    # Map labels (0 = Human, 1 = AI)
    prediction = "AI" if probs[1] > 0.5 else "Human"
    confidence = probs[1] if prediction == "AI" else probs[0]
    
    return {
        "prediction": prediction,
        "confidence": float(confidence),
        "probs": {"human": float(probs[0]), "ai": float(probs[1])}
    }

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_neural_model.py <path_to_audio_file>")
        sys.exit(1)
        
    audio_file = sys.argv[1]
    model_weights = r"C:\Users\prati\OneDrive\Desktop\deployed&running\v-detection\voice_detection_v2\voice_detector_neural.pt"
    
    if not os.path.exists(model_weights):
        print(f"Error: Model file not found at {model_weights}")
        sys.exit(1)
        
    if not os.path.exists(audio_file):
        print(f"Error: Audio file not found at {audio_file}")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    processor, model = load_neural_model(model_weights, device=device)
    
    result = predict_audio(audio_file, processor, model, device=device)
    
    print("\n" + "="*40)
    print("RESULTS:")
    print("="*40)
    print(f"File:       {os.path.basename(audio_file)}")
    print(f"Prediction: {result['prediction']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"Human Prob: {result['probs']['human']:.2%}")
    print(f"AI Prob:    {result['probs']['ai']:.2%}")
    print("="*40)
