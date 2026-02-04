import os
import pandas as pd
import numpy as np
import torch
import librosa
from transformers import AutoFeatureExtractor, Wav2Vec2Model
from tqdm import tqdm
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from src.config import DATA_DIR, SAMPLE_RATE

# Model Checkpoint
MODEL_ID = "facebook/wav2vec2-large-xlsr-53"

def extract_embeddings(df, output_path):
    print(f"Loading Wav2Vec2 Model: {MODEL_ID}...")
    try:
        processor = AutoFeatureExtractor.from_pretrained(MODEL_ID)
    except Exception as e:
        print(f"Failed to load AutoFeatureExtractor: {e}")
        return
        
    model = Wav2Vec2Model.from_pretrained(MODEL_ID)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    model = model.to(device)
    
    embeddings = []
    labels = []
    filenames = []
    
    print("Extracting Embeddings...")
    for index, row in tqdm(df.iterrows(), total=len(df)):
        file_path = row['path']
        
        try:
            # Load Audio
            y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
            
            # Process input
            # Wav2Vec2 expects input_values (raw audio)
            inputs = processor(y, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True)
            input_values = inputs.input_values.to(device)
            
            # Inference
            with torch.no_grad():
                outputs = model(input_values)
                # outputs.last_hidden_state shape: (batch, sequence_length, hidden_size)
                # We need a fixed vector per audio. Mean pooling is standard.
                hidden_states = outputs.last_hidden_state
                pooled_output = torch.mean(hidden_states, dim=1) # Average over time dimension
                
            emb = pooled_output.cpu().numpy().flatten()
            embeddings.append(emb)
            labels.append(row['label'])
            filenames.append(row['filename'])
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            
    # Save as DataFrame
    # Create columns for each dimension
    if len(embeddings) > 0:
        emb_matrix = np.array(embeddings)
        col_names = [f'emb_{i}' for i in range(emb_matrix.shape[1])]
        
        emb_df = pd.DataFrame(emb_matrix, columns=col_names)
        emb_df['filename'] = filenames
        emb_df['label'] = labels
        
        emb_df.to_csv(output_path, index=False)
        print(f"Embedding Extraction Complete! Saved to {output_path}")
        print(f"Embedding Shape: {emb_matrix.shape}")
    else:
        print("No embeddings extracted.")

def main():
    master_csv = os.path.join(DATA_DIR, 'master_dataset.csv')
    if not os.path.exists(master_csv):
        print("Master dataset not found. Run preprocessing first.")
        return

    df = pd.read_csv(master_csv)
    output_path = os.path.join(DATA_DIR, 'features', 'embeddings.csv')
    
    extract_embeddings(df, output_path)

if __name__ == "__main__":
    main()
