import os
import pandas as pd
import librosa
import soundfile as sf
import numpy as np
from tqdm import tqdm
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.config import RAW_HUMAN_DIR, RAW_AI_DIR, PROCESSED_DIR, SAMPLE_RATE, DURATION_LIMIT, DATA_DIR

def preprocess_audio(file_path, output_path):
    """
    Standardize audio:
    - Load as Mono
    - Resample to 16kHz
    - Trim silence
    - Normalize amplitude
    - Pad/Trim to fixed duration (optional, but good for batching, let's just ensure min length for now)
    """
    try:
        # Load audio (librosa handles resampling and mono conversion)
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)
        
        # Trim silence (top_db=20 is a standard threshold)
        y_trimmed, _ = librosa.effects.trim(y, top_db=20)
        
        # Skip if too short (less than 0.5s)
        if len(y_trimmed) < 0.5 * SAMPLE_RATE:
            return False, "Too short"
        
        # Normalize amplitude (Peak normalization)
        y_norm = librosa.util.normalize(y_trimmed)
        
        # Save processed file
        sf.write(output_path, y_norm, SAMPLE_RATE)
        
        return True, "Success"
        
    except Exception as e:
        return False, str(e)

def process_dataset(input_csv, source_type):
    """
    Process all files listed in the CSV
    source_type: 'human' or 'ai'
    """
    if not os.path.exists(input_csv):
        print(f"Dataset CSV not found: {input_csv}")
        return []

    df = pd.read_csv(input_csv)
    processed_records = []
    
    output_dir = os.path.join(PROCESSED_DIR, source_type)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    print(f"Processing {source_type} samples...")
    
    for _, row in tqdm(df.iterrows(), total=len(df)):
        file_path = row['path']
        filename = row['filename']
        lang = row['language']
        
        # Create language subdir in processed
        lang_dir = os.path.join(output_dir, lang)
        if not os.path.exists(lang_dir):
            os.makedirs(lang_dir)
            
        output_filename = f"proc_{filename}"
        if not output_filename.endswith('.wav'):
            # Enforce wav for processed data usually, or keep original extension if flac/mp3 is fine.
            # wav is safer for downstream processing.
            output_filename = os.path.splitext(output_filename)[0] + ".wav"
            
        output_path = os.path.join(lang_dir, output_filename)
        
        success, msg = preprocess_audio(file_path, output_path)
        
        if success:
            processed_records.append({
                'filename': output_filename,
                'original_filename': filename,
                'path': output_path,
                'label': source_type, # 'human' or 'ai'
                'language': lang,
                'split': 'train' # Default, will split later
            })
            
    return processed_records

def main():
    if not os.path.exists(PROCESSED_DIR):
        os.makedirs(PROCESSED_DIR)
        
    all_processed = []
    
    # Process Human Data
    human_csv = os.path.join(RAW_HUMAN_DIR, 'human_samples.csv')
    human_data = process_dataset(human_csv, 'human')
    all_processed.extend(human_data)
    
    # Process AI Data
    ai_csv = os.path.join(RAW_AI_DIR, 'ai_samples.csv')
    ai_data = process_dataset(ai_csv, 'ai')
    all_processed.extend(ai_data)
    
    # Save Master Dataset
    master_df = pd.DataFrame(all_processed)
    master_csv = os.path.join(DATA_DIR, 'master_dataset.csv')
    master_df.to_csv(master_csv, index=False)
    
    # Print Stats
    print("\nProcessing Complete!")
    print(f"Total Processed Samples: {len(master_df)}")
    print(master_df['label'].value_counts())
    print(master_df['language'].value_counts())

if __name__ == "__main__":
    main()
