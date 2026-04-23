import os
import sys
import torch
import torch.nn as nn
from transformers import AutoModel

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

def convert_to_onnx():
    model_path = r"C:\Users\prati\OneDrive\Desktop\deployed&running\v-detection\voice_detection_v2\voice_detector_neural.pt"
    onnx_path = r"C:\Users\prati\OneDrive\Desktop\deployed&running\v-detection\voice_detection_v2\voice_detector_neural.onnx"
    
    if not os.path.exists(model_path):
        print(f"Error: Could not find {model_path}")
        return
        
    print("Loading base model architecture...")
    encoder = AutoModel.from_pretrained("facebook/wav2vec2-base")
    model = AudioClassifier(encoder, encoder.config.hidden_size)
    
    print("Loading custom weights...")
    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict)
    
    model.eval()
    model.to("cpu")
    
    print("Generating dummy input...")
    # wav2vec2 expects (batch_size, sequence_length)
    # 5 seconds of audio at 16kHz
    dummy_input = torch.randn(1, 16000 * 5)
    
    print(f"Exporting ONNX model to {onnx_path}...")
    try:
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=18,
            do_constant_folding=True,
            input_names=['audio_input'],
            output_names=['logits'],
            dynamic_axes={
                'audio_input': {0: 'batch_size', 1: 'audio_length'},
                'logits': {0: 'batch_size'}
            }
        )
        print("ONNX export successful!")
        print(f"Size: {os.path.getsize(onnx_path) / 1024 / 1024:.2f} MB")
    except Exception as e:
        print(f"ONNX export failed: {e}")
        print("Make sure you have 'onnx' and 'onnxscript' installed via pip.")

if __name__ == "__main__":
    convert_to_onnx()
