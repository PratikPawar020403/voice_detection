import os
import pandas as pd
import numpy as np
import librosa
from tqdm import tqdm
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from src.config import DATA_DIR, SAMPLE_RATE

def extract_dsp_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
        
        features = {}
        
        # 1. MFCC (Mel-frequency cepstral coefficients)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        features['mfcc_mean'] = np.mean(mfcc)
        features['mfcc_var'] = np.var(mfcc)
        
        # Add individual MFCC stats if needed, but mean/var aggregation is common for baselines
        for i in range(1, 14):
            features[f'mfcc_{i}_mean'] = np.mean(mfcc[i-1])
            features[f'mfcc_{i}_var'] = np.var(mfcc[i-1])

        # 2. Spectral Features
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        features['spec_cent_mean'] = np.mean(spectral_centroid)
        features['spec_cent_var'] = np.var(spectral_centroid)
        
        spectral_flatness = librosa.feature.spectral_flatness(y=y)
        features['spec_flat_mean'] = np.mean(spectral_flatness)
        features['spec_flat_var'] = np.var(spectral_flatness)
        
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        features['spec_roll_mean'] = np.mean(spectral_rolloff)
        
        # 3. Energy / RMS
        rms = librosa.feature.rms(y=y)
        features['rms_mean'] = np.mean(rms)
        features['rms_var'] = np.var(rms)
        
        # 4. Zero Crossing Rate
        zcr = librosa.feature.zero_crossing_rate(y)
        features['zcr_mean'] = np.mean(zcr)
        features['zcr_var'] = np.var(zcr)
        
        # 5. Chroma
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        features['chroma_mean'] = np.mean(chroma)
        
        # 6. Pitch (using simple piptrack)
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        # Select pitches with high magnitude
        pitches_filtered = pitches[magnitudes > np.median(magnitudes)]
        if len(pitches_filtered) > 0:
            features['pitch_mean'] = np.mean(pitches_filtered)
            features['pitch_std'] = np.std(pitches_filtered)
        else:
            features['pitch_mean'] = 0
            features['pitch_std'] = 0
            
        return features

    except Exception as e:
        print(f"Error extracting features for {file_path}: {e}")
        return None

def main():
    master_csv = os.path.join(DATA_DIR, 'master_dataset.csv')
    if not os.path.exists(master_csv):
        print("Master dataset not found. Run preprocessing first.")
        return

    df = pd.read_csv(master_csv)
    feature_list = []
    
    print("Extracting DSP Features...")
    for index, row in tqdm(df.iterrows(), total=len(df)):
        file_path = row['path']
        features = extract_dsp_features(file_path)
        
        if features:
            # Combine meta info with features
            features['filename'] = row['filename']
            features['label'] = row['label'] # Target
            feature_list.append(features)
            
    # Save Feature Dataset
    feature_df = pd.DataFrame(feature_list)
    output_path = os.path.join(DATA_DIR, 'features', 'dsp_features.csv')
    feature_df.to_csv(output_path, index=False)
    print(f"DSP Feature Extraction Complete! Saved to {output_path}")

if __name__ == "__main__":
    main()
