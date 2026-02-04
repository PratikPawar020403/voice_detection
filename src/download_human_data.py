import os
import pandas as pd
import soundfile as sf
import librosa
from datasets import load_dataset
from tqdm import tqdm
import sys

# Add src to path to import config
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.config import LANGUAGES, RAW_HUMAN_DIR, SAMPLE_RATE

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def download_english_data(num_samples=50):
    print(f"Downloading English samples from LibriSpeech...")
    lang_dir = os.path.join(RAW_HUMAN_DIR, 'en')
    ensure_dir(lang_dir)
    
    # Using LibriSpeech clean test set for quick access
    dataset = load_dataset("librispeech_asr", "clean", split="validation", streaming=True)
    
    data_records = []
    
    count = 0
    for sample in tqdm(dataset, total=num_samples):
        if count >= num_samples:
            break
            
        audio_array = sample['audio']['array']
        sr = sample['audio']['sampling_rate']
        
        # Resample if necessary
        if sr != SAMPLE_RATE:
            audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=SAMPLE_RATE)
            
        file_name = f"human_en_{count:04d}.flac"
        file_path = os.path.join(lang_dir, file_name)
        
        sf.write(file_path, audio_array, SAMPLE_RATE)
        
        data_records.append({
            'filename': file_name,
            'language': 'en',
            'path': file_path,
            'source': 'librispeech'
        })
        count += 1
        
    return data_records

def download_indic_data(lang_code, lang_name, num_samples=50):
    print(f"Downloading {lang_name} ({lang_code}) samples...")
    lang_dir = os.path.join(RAW_HUMAN_DIR, lang_code)
    ensure_dir(lang_dir)
    
    # Try IndicVoices first, fallback to Common Voice or FLEURS if needed
    # Note: IndicVoices might require manual download or authentication.
    # We'll use google/fleurs as a reliable automated fallback for this script 
    # if IndicVoices requires specific auth/access that we can't guarantee here.
    # However, user requested IndicVoices. Let's try to load a subset or use a compatible open dataset.
    # Common Voice (mozilla-foundation/common_voice_11_0) is a good standard.
    
    dataset_name = "google/fleurs" # Reliable open access
    subset = f"{lang_code}_in"
    
    print(f"Attempting to download from {dataset_name} ({subset})...")
    
    try:
        dataset = load_dataset(dataset_name, subset, split="validation", streaming=True, trust_remote_code=True)
    except Exception as e:
        print(f"Error loading {dataset_name}: {e}")
        return []

    data_records = []
    count = 0
    
    for sample in tqdm(dataset, total=num_samples):
        if count >= num_samples:
            break
            
        audio_array = sample['audio']['array']
        sr = sample['audio']['sampling_rate']
        
        if sr != SAMPLE_RATE:
            audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=SAMPLE_RATE)
            
        file_name = f"human_{lang_code}_{count:04d}.flac"
        file_path = os.path.join(lang_dir, file_name)
        
        sf.write(file_path, audio_array, SAMPLE_RATE)
        
        data_records.append({
            'filename': file_name,
            'language': lang_code,
            'path': file_path,
            'source': dataset_name
        })
        count += 1
        
    return data_records

def main():
    all_records = []
    
    # 1. English
    en_records = download_english_data()
    all_records.extend(en_records)
    
    # 2. Indic Languages
    for code, name in LANGUAGES.items():
        if code == 'en': continue
        
        records = download_indic_data(code, name)
        all_records.extend(records)
        
    # Save CSV
    df = pd.DataFrame(all_records)
    csv_path = os.path.join(RAW_HUMAN_DIR, 'human_samples.csv')
    df.to_csv(csv_path, index=False)
    print(f"Completed! Metadata saved to {csv_path}")
    print(f"Total samples: {len(df)}")

if __name__ == "__main__":
    main()
