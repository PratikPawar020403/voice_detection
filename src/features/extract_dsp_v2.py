"""
DSP Feature Extraction v2 — Expanded feature set for AI Voice Detection
========================================================================
Features: ~85 total (compared to 37 in v1)

New in v2:
  - Delta & Delta-Delta MFCC (temporal dynamics)
  - Spectral bandwidth, contrast, tonnetz
  - Jitter & Shimmer (micro pitch/amplitude perturbations)
  - Harmonic-to-Noise Ratio (HNR)
  - Formant frequencies (F1-F4)
  - Spectral skewness, kurtosis, entropy
  - Silence ratio & pause patterns
  - Temporal envelope modulation

Requires: librosa, numpy, scipy
Optional: parselmouth (for jitter, shimmer, HNR, formants)
"""

import os
import numpy as np
import librosa
import pandas as pd
from scipy import stats as scipy_stats
from scipy.signal import hilbert
from tqdm import tqdm
import sys
import warnings

warnings.filterwarnings("ignore")

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from src.config import DATA_DIR, SAMPLE_RATE

# Try importing parselmouth for voice quality features
try:
    import parselmouth
    from parselmouth.praat import call
    HAS_PARSELMOUTH = True
except ImportError:
    HAS_PARSELMOUTH = False
    print("WARNING: parselmouth not installed. Jitter/Shimmer/HNR/Formant features will be zeros.")
    print("  Install with: pip install praat-parselmouth")


# ============================================================
# Feature Extraction Functions
# ============================================================

def extract_mfcc_features(y, sr, n_mfcc=13):
    """
    MFCC + Delta + Delta-Delta
    Returns: 80 features (13 * 2 * 3 + 2 overall)
    """
    features = {}

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    delta_mfcc = librosa.feature.delta(mfcc, order=1)
    delta2_mfcc = librosa.feature.delta(mfcc, order=2)

    # Overall MFCC stats
    features['mfcc_mean'] = np.mean(mfcc)
    features['mfcc_var'] = np.var(mfcc)

    # Per-coefficient stats for MFCC, Delta, Delta-Delta
    for i in range(n_mfcc):
        features[f'mfcc_{i+1}_mean'] = np.mean(mfcc[i])
        features[f'mfcc_{i+1}_var'] = np.var(mfcc[i])
        features[f'delta_mfcc_{i+1}_mean'] = np.mean(delta_mfcc[i])
        features[f'delta_mfcc_{i+1}_var'] = np.var(delta_mfcc[i])
        features[f'delta2_mfcc_{i+1}_mean'] = np.mean(delta2_mfcc[i])
        features[f'delta2_mfcc_{i+1}_var'] = np.var(delta2_mfcc[i])

    return features


def extract_spectral_features(y, sr):
    """
    Spectral: centroid, bandwidth, flatness, rolloff, contrast, tonnetz
    Returns: ~24 features
    """
    features = {}

    # Spectral Centroid
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    features['spec_cent_mean'] = np.mean(spec_cent)
    features['spec_cent_var'] = np.var(spec_cent)

    # Spectral Bandwidth (NEW in v2)
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    features['spec_bw_mean'] = np.mean(spec_bw)
    features['spec_bw_var'] = np.var(spec_bw)

    # Spectral Flatness
    spec_flat = librosa.feature.spectral_flatness(y=y)
    features['spec_flat_mean'] = np.mean(spec_flat)
    features['spec_flat_var'] = np.var(spec_flat)

    # Spectral Rolloff
    spec_roll = librosa.feature.spectral_rolloff(y=y, sr=sr)
    features['spec_roll_mean'] = np.mean(spec_roll)
    features['spec_roll_var'] = np.var(spec_roll)

    # Spectral Contrast — 7 bands (NEW in v2)
    spec_contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_bands=6)
    for i in range(7):  # 6 bands + 1 valley
        features[f'spec_contrast_{i}_mean'] = np.mean(spec_contrast[i])

    # Tonnetz — 6 tonal features (NEW in v2)
    # Requires harmonic component
    y_harmonic = librosa.effects.harmonic(y)
    tonnetz = librosa.feature.tonnetz(y=y_harmonic, sr=sr)
    for i in range(6):
        features[f'tonnetz_{i}_mean'] = np.mean(tonnetz[i])

    return features


def extract_energy_rhythm_features(y, sr):
    """
    RMS energy, ZCR, Chroma, Tempo
    Returns: ~6 features
    """
    features = {}

    # RMS Energy
    rms = librosa.feature.rms(y=y)
    features['rms_mean'] = np.mean(rms)
    features['rms_var'] = np.var(rms)

    # Zero Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(y)
    features['zcr_mean'] = np.mean(zcr)
    features['zcr_var'] = np.var(zcr)

    # Chroma
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    features['chroma_mean'] = np.mean(chroma)
    features['chroma_var'] = np.var(chroma)

    return features


def extract_pitch_features(y, sr):
    """
    Pitch (F0) statistics using librosa piptrack
    Returns: 4 features
    """
    features = {}

    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitches_filtered = pitches[magnitudes > np.median(magnitudes)]

    if len(pitches_filtered) > 0:
        pitches_nonzero = pitches_filtered[pitches_filtered > 0]
        if len(pitches_nonzero) > 0:
            features['pitch_mean'] = np.mean(pitches_nonzero)
            features['pitch_std'] = np.std(pitches_nonzero)
            features['pitch_range'] = np.ptp(pitches_nonzero)  # max - min
            features['pitch_cv'] = np.std(pitches_nonzero) / (np.mean(pitches_nonzero) + 1e-8)  # coefficient of variation
        else:
            features['pitch_mean'] = 0
            features['pitch_std'] = 0
            features['pitch_range'] = 0
            features['pitch_cv'] = 0
    else:
        features['pitch_mean'] = 0
        features['pitch_std'] = 0
        features['pitch_range'] = 0
        features['pitch_cv'] = 0

    return features


def extract_voice_quality_features(y, sr):
    """
    Jitter, Shimmer, HNR, Formants via parselmouth/Praat
    Returns: 10 features (or zeros if parselmouth not available)

    These are CRITICAL for AI voice detection:
    - Jitter: micro pitch perturbations (humans have them, AI doesn't)
    - Shimmer: micro amplitude perturbations (same)
    - HNR: how clean the voice is (AI is too clean)
    - Formants: vocal tract resonances (AI has unnatural transitions)
    """
    features = {
        'jitter_local': 0.0,
        'jitter_rap': 0.0,
        'jitter_ppq5': 0.0,
        'shimmer_local': 0.0,
        'shimmer_apq3': 0.0,
        'shimmer_apq5': 0.0,
        'hnr_mean': 0.0,
        'formant_f1_mean': 0.0,
        'formant_f2_mean': 0.0,
        'formant_f3_mean': 0.0,
    }

    if not HAS_PARSELMOUTH:
        return features

    try:
        # Create Praat Sound object
        snd = parselmouth.Sound(y, sampling_frequency=sr)

        # --- Pitch Object (needed for jitter/shimmer) ---
        pitch = call(snd, "To Pitch", 0.0, 75, 600)

        # --- Point Process (needed for jitter/shimmer) ---
        point_process = call(snd, "To PointProcess (periodic, cc)", 75, 600)

        # --- Jitter ---
        try:
            features['jitter_local'] = call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
        except Exception:
            pass

        try:
            features['jitter_rap'] = call(point_process, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3)
        except Exception:
            pass

        try:
            features['jitter_ppq5'] = call(point_process, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3)
        except Exception:
            pass

        # --- Shimmer ---
        try:
            features['shimmer_local'] = call([snd, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        except Exception:
            pass

        try:
            features['shimmer_apq3'] = call([snd, point_process], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        except Exception:
            pass

        try:
            features['shimmer_apq5'] = call([snd, point_process], "Get shimmer (apq5)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        except Exception:
            pass

        # --- HNR (Harmonics-to-Noise Ratio) ---
        try:
            harmonicity = call(snd, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
            features['hnr_mean'] = call(harmonicity, "Get mean", 0, 0)
            if np.isnan(features['hnr_mean']):
                features['hnr_mean'] = 0.0
        except Exception:
            pass

        # --- Formants (F1-F3) ---
        try:
            formant = call(snd, "To Formant (burg)", 0.0, 5, 5500, 0.025, 50)
            duration = snd.get_total_duration()

            f1_values, f2_values, f3_values = [], [], []
            n_frames = call(formant, "Get number of frames")

            for frame in range(1, n_frames + 1):
                t = call(formant, "Get time from frame number", frame)
                f1 = call(formant, "Get value at time", 1, t, "Hertz", "Linear")
                f2 = call(formant, "Get value at time", 2, t, "Hertz", "Linear")
                f3 = call(formant, "Get value at time", 3, t, "Hertz", "Linear")

                if not np.isnan(f1): f1_values.append(f1)
                if not np.isnan(f2): f2_values.append(f2)
                if not np.isnan(f3): f3_values.append(f3)

            features['formant_f1_mean'] = np.mean(f1_values) if f1_values else 0.0
            features['formant_f2_mean'] = np.mean(f2_values) if f2_values else 0.0
            features['formant_f3_mean'] = np.mean(f3_values) if f3_values else 0.0
        except Exception:
            pass

    except Exception as e:
        # If parselmouth fails entirely, all features stay at 0
        pass

    # Replace any NaN with 0
    for k, v in features.items():
        if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
            features[k] = 0.0

    return features


def extract_spectral_stats(y, sr):
    """
    Higher-order spectral statistics: skewness, kurtosis, entropy
    Returns: 3 features
    """
    features = {}

    # Compute magnitude spectrum
    S = np.abs(librosa.stft(y))
    S_mean = np.mean(S, axis=1)  # Average across time

    # Normalize to probability distribution
    S_norm = S_mean / (np.sum(S_mean) + 1e-8)

    # Spectral Skewness — asymmetry of spectral distribution
    features['spec_skewness'] = float(scipy_stats.skew(S_norm))

    # Spectral Kurtosis — peakedness of spectral distribution
    features['spec_kurtosis'] = float(scipy_stats.kurtosis(S_norm))

    # Spectral Entropy — randomness/complexity of spectrum
    S_entropy = S_norm[S_norm > 0]
    features['spec_entropy'] = float(-np.sum(S_entropy * np.log2(S_entropy + 1e-12)))

    return features


def extract_silence_features(y, sr, threshold_db=30):
    """
    Silence/pause analysis — AI voices have mechanical pauses
    Returns: 3 features
    """
    features = {}

    # Split audio into non-silent intervals
    intervals = librosa.effects.split(y, top_db=threshold_db)

    total_duration = len(y) / sr

    if len(intervals) > 0:
        # Total non-silent duration
        voiced_duration = sum((end - start) for start, end in intervals) / sr
        silence_duration = total_duration - voiced_duration

        features['silence_ratio'] = silence_duration / (total_duration + 1e-8)

        # Pause analysis (gaps between voiced segments)
        if len(intervals) > 1:
            pauses = []
            for i in range(1, len(intervals)):
                pause = (intervals[i][0] - intervals[i-1][1]) / sr
                pauses.append(pause)

            features['pause_count'] = len(pauses)
            features['pause_mean_duration'] = np.mean(pauses)
        else:
            features['pause_count'] = 0
            features['pause_mean_duration'] = 0.0
    else:
        features['silence_ratio'] = 1.0
        features['pause_count'] = 0
        features['pause_mean_duration'] = 0.0

    return features


def extract_modulation_features(y, sr):
    """
    Temporal envelope modulation — natural speech has ~4Hz modulation
    AI voices often lack this natural rhythm
    Returns: 2 features
    """
    features = {}

    try:
        # Get amplitude envelope using Hilbert transform
        analytic_signal = hilbert(y)
        envelope = np.abs(analytic_signal)

        # Compute spectrum of the envelope
        n_fft = min(len(envelope), 4096)
        env_fft = np.abs(np.fft.rfft(envelope, n=n_fft))
        freqs = np.fft.rfftfreq(n_fft, d=1.0/sr)

        # Energy in 2-8 Hz band (speech modulation range)
        mask_speech = (freqs >= 2) & (freqs <= 8)
        # Energy in 0-2 Hz band (baseline)
        mask_low = (freqs >= 0.1) & (freqs < 2)

        speech_mod_energy = np.mean(env_fft[mask_speech]) if np.any(mask_speech) else 0
        low_energy = np.mean(env_fft[mask_low]) if np.any(mask_low) else 1e-8

        # Modulation index: ratio of speech-rate modulation to baseline
        features['mod_index_4hz'] = float(speech_mod_energy / (low_energy + 1e-8))

        # Peak modulation frequency
        if np.any(mask_speech):
            speech_freqs = freqs[mask_speech]
            speech_fft = env_fft[mask_speech]
            features['mod_peak_freq'] = float(speech_freqs[np.argmax(speech_fft)])
        else:
            features['mod_peak_freq'] = 0.0

    except Exception:
        features['mod_index_4hz'] = 0.0
        features['mod_peak_freq'] = 0.0

    return features


# ============================================================
# Main Feature Extraction
# ============================================================

def extract_all_features_v2(file_path, sr=SAMPLE_RATE):
    """
    Extract all v2 DSP features from a single audio file.
    Returns: dict of ~85 features, or None on error
    """
    try:
        y, sr = librosa.load(file_path, sr=sr)

        if len(y) < int(0.5 * sr):
            return None  # Too short

        features = {}

        # 1. MFCC + Deltas (~80 features)
        features.update(extract_mfcc_features(y, sr))

        # 2. Spectral (~24 features)
        features.update(extract_spectral_features(y, sr))

        # 3. Energy & Rhythm (~6 features)
        features.update(extract_energy_rhythm_features(y, sr))

        # 4. Pitch (~4 features)
        features.update(extract_pitch_features(y, sr))

        # 5. Voice Quality: Jitter, Shimmer, HNR, Formants (~10 features)
        features.update(extract_voice_quality_features(y, sr))

        # 6. Spectral Statistics (~3 features)
        features.update(extract_spectral_stats(y, sr))

        # 7. Silence Analysis (~3 features)
        features.update(extract_silence_features(y, sr))

        # 8. Temporal Modulation (~2 features)
        features.update(extract_modulation_features(y, sr))

        # Sanitize: replace NaN/Inf with 0
        for k, v in features.items():
            if isinstance(v, (float, np.floating)):
                if np.isnan(v) or np.isinf(v):
                    features[k] = 0.0

        return features

    except Exception as e:
        print(f"Error extracting features for {file_path}: {e}")
        return None


def get_feature_names():
    """
    Returns the ordered list of all v2 feature names.
    Useful for ensuring consistent column ordering.
    """
    # Generate a dummy extraction to get all feature names
    dummy_y = np.random.randn(SAMPLE_RATE * 2)  # 2s of noise
    features = extract_all_features_v2.__wrapped__(dummy_y, SAMPLE_RATE) if hasattr(extract_all_features_v2, '__wrapped__') else None

    # Fallback: manually list all expected feature names
    names = []

    # MFCC (80 features)
    names.extend(['mfcc_mean', 'mfcc_var'])
    for i in range(1, 14):
        names.extend([f'mfcc_{i}_mean', f'mfcc_{i}_var'])
        names.extend([f'delta_mfcc_{i}_mean', f'delta_mfcc_{i}_var'])
        names.extend([f'delta2_mfcc_{i}_mean', f'delta2_mfcc_{i}_var'])

    # Spectral (24 features)
    names.extend(['spec_cent_mean', 'spec_cent_var'])
    names.extend(['spec_bw_mean', 'spec_bw_var'])
    names.extend(['spec_flat_mean', 'spec_flat_var'])
    names.extend(['spec_roll_mean', 'spec_roll_var'])
    for i in range(7):
        names.append(f'spec_contrast_{i}_mean')
    for i in range(6):
        names.append(f'tonnetz_{i}_mean')

    # Energy & Rhythm (6 features)
    names.extend(['rms_mean', 'rms_var', 'zcr_mean', 'zcr_var', 'chroma_mean', 'chroma_var'])

    # Pitch (4 features)
    names.extend(['pitch_mean', 'pitch_std', 'pitch_range', 'pitch_cv'])

    # Voice Quality (10 features)
    names.extend(['jitter_local', 'jitter_rap', 'jitter_ppq5'])
    names.extend(['shimmer_local', 'shimmer_apq3', 'shimmer_apq5'])
    names.extend(['hnr_mean'])
    names.extend(['formant_f1_mean', 'formant_f2_mean', 'formant_f3_mean'])

    # Spectral Stats (3 features)
    names.extend(['spec_skewness', 'spec_kurtosis', 'spec_entropy'])

    # Silence (3 features)
    names.extend(['silence_ratio', 'pause_count', 'pause_mean_duration'])

    # Modulation (2 features)
    names.extend(['mod_index_4hz', 'mod_peak_freq'])

    return names


# ============================================================
# Batch Extraction (from master dataset)
# ============================================================

def main():
    """
    Extract v2 features from all samples in master_dataset.csv
    """
    master_csv = os.path.join(DATA_DIR, 'master_dataset.csv')
    if not os.path.exists(master_csv):
        print("Master dataset not found. Run preprocessing first.")
        return

    df = pd.read_csv(master_csv)
    feature_list = []
    failed = []

    print(f"Extracting v2 DSP Features from {len(df)} samples...")
    print(f"  Parselmouth available: {HAS_PARSELMOUTH}")

    for index, row in tqdm(df.iterrows(), total=len(df)):
        file_path = row['path']
        features = extract_all_features_v2(file_path)

        if features:
            features['filename'] = row['filename']
            features['label'] = row['label']
            feature_list.append(features)
        else:
            failed.append(file_path)

    # Save
    feature_df = pd.DataFrame(feature_list)

    output_dir = os.path.join(DATA_DIR, 'features')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'dsp_features_v2.csv')
    feature_df.to_csv(output_path, index=False)

    print(f"\nv2 Feature Extraction Complete!")
    print(f"  Saved to: {output_path}")
    print(f"  Total features per sample: {len(feature_df.columns) - 2}")  # minus filename, label
    print(f"  Successful: {len(feature_list)}")
    print(f"  Failed: {len(failed)}")

    if failed:
        print(f"\nFailed files:")
        for f in failed[:10]:
            print(f"  - {f}")
        if len(failed) > 10:
            print(f"  ... and {len(failed) - 10} more")


if __name__ == "__main__":
    main()
